"""Report assembly: run a comparison method over a snapshot and freeze it.

A report is computed once and never modified; publishing and adopting only
flip lifecycle flags.  Revisions (supplementary annotation, label fixes) are
new snapshots in the same series, and their reports take the next
``report_version`` — published/adopted conclusions of earlier versions stay
exactly as they were.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import eval_match, eval_truth
from .eval_models import (
    EvalAnnotationRange,
    EvalLabel,
    EvalMatch,
    EvalNoiseLayer,
    EvalReport,
    EvalReportGroup,
    EvalScope,
    EvalSegment,
    EvalSnapshot,
)
from .eval_snapshot import content_hash
from .timeutil import now_us


def _segments_by_cell(segments: list[EvalSegment]) -> dict[tuple[str, str], list[eval_match.Segment]]:
    cells: dict[tuple[str, str], list[eval_match.Segment]] = defaultdict(list)
    for row in segments:
        cells[(row.station_id, row.class_label)].append(
            eval_match.Segment(
                segment_key=row.segment_key,
                version_role=row.version_role,
                station_id=row.station_id,
                class_label=row.class_label,
                start_us=row.start_us,
                end_us=row.end_us,
                features=tuple(float(x) for x in row.features),
            )
        )
    return cells


def create_report(
    db: Session, snapshot: EvalSnapshot, method: str, params: dict
) -> tuple[EvalReport, bool]:
    """Compute a report; returns (report, created).

    A report is a pure function of (snapshot, method, params): asking for the
    same combination again replays the stored report instead of adding a
    duplicate version.  New versions only appear when the evaluation inputs
    actually change (a revised snapshot, another method or other costs).
    """
    params_full = {"method": method, **params}
    params_hash = content_hash(params_full)
    existing = db.scalar(
        select(EvalReport).where(
            EvalReport.snapshot_id == snapshot.id,
            EvalReport.method == method,
            EvalReport.params_hash == params_hash,
        )
    )
    if existing is not None:
        return existing, False

    match_params = eval_match.MatchParams.from_dict(params)
    ci_level = float(params.get("ci_level", 0.95))

    segments = list(
        db.execute(
            select(EvalSegment).where(EvalSegment.snapshot_id == snapshot.id)
        ).scalars()
    )
    labels = list(
        db.execute(select(EvalLabel).where(EvalLabel.snapshot_id == snapshot.id)).scalars()
    )
    scopes = list(
        db.execute(select(EvalScope).where(EvalScope.snapshot_id == snapshot.id)).scalars()
    )
    noise_rows = list(
        db.execute(
            select(EvalNoiseLayer).where(EvalNoiseLayer.snapshot_id == snapshot.id)
        ).scalars()
    )
    ann_rows = list(
        db.execute(
            select(EvalAnnotationRange).where(
                EvalAnnotationRange.snapshot_id == snapshot.id
            )
        ).scalars()
    )

    stations = sorted({row.station_id for row in scopes})
    layer_rows_by_station: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in noise_rows:
        layer_rows_by_station[row.station_id].append((row.layer, row.start_us, row.end_us))
    annotation = {
        st: eval_truth._merge_ranges(
            [(r.start_us, r.end_us) for r in ann_rows if r.station_id == st]
        )
        for st in stations
    }
    scope_map = {
        st: eval_truth._merge_ranges(
            [(r.start_us, r.end_us) for r in scopes if r.station_id == st]
        )
        for st in stations
    }

    # --- old↔new comparison, per (station, class) cell -------------------- #
    cells = _segments_by_cell(segments)
    label_rows = [
        eval_truth.Label(
            label_key=row.label_key,
            station_id=row.station_id,
            class_label=row.class_label,
            start_us=row.start_us,
            end_us=row.end_us,
        )
        for row in labels
    ]
    labels_by_cell: dict[tuple[str, str], list[eval_truth.Label]] = defaultdict(list)
    for lab in label_rows:
        labels_by_cell[(lab.station_id, lab.class_label)].append(lab)

    # Per-version detection state, restricted to segments inside the fully
    # annotated layer — the same rule the grouped rates use, so provenance
    # flags and interval estimates never disagree.
    version_truths: dict[tuple[str, str, str], eval_truth.VersionTruth] = {}
    for (station, class_label), cell_segs in cells.items():
        cell_labels = labels_by_cell.get((station, class_label), [])
        ann = annotation.get(station, [])
        for role in ("old", "new"):
            judged = [
                s
                for s in cell_segs
                if s.version_role == role
                and eval_truth._in_ranges(ann, s.midpoint_us)
            ]
            version_truths[(station, class_label, role)] = eval_truth.version_truth(
                cell_labels, judged
            )

    all_groups: list[dict] = []
    meta_cells = {}
    for (station, class_label), cell_segs in sorted(cells.items()):
        groups, meta = eval_match.compare(cell_segs, method, match_params)
        meta_cells[f"{station}/{class_label}"] = meta
        for group in groups:
            midpoint = (group["start_us"] + group["end_us"]) // 2
            group["noise_layer"] = eval_truth.layer_of(
                layer_rows_by_station.get(station, []), midpoint
            )
            cell_labels = labels_by_cell.get((station, class_label), [])
            overlapping = [
                lab
                for lab in cell_labels
                if eval_truth._overlap_us(lab.start_us, lab.end_us, group["start_us"], group["end_us"])
                > 0
            ]
            group["label_keys"] = sorted(lab.label_key for lab in overlapping)
            group["truth"] = {
                "labels": [
                    {
                        "label_key": lab.label_key,
                        "detected_old": lab.label_key
                        in version_truths[(station, class_label, "old")].detected,
                        "detected_new": lab.label_key
                        in version_truths[(station, class_label, "new")].detected,
                    }
                    for lab in sorted(overlapping, key=lambda x: x.label_key)
                ]
            }
            all_groups.append(group)

    # --- truth metrics grouped by station × class × noise layer ----------- #
    all_segments = [
        seg for cell_segs in cells.values() for seg in cell_segs
    ]
    group_metrics = eval_truth.compute_groups(
        stations=stations,
        labels=label_rows,
        segments=all_segments,
        annotation=annotation,
        scopes=scope_map,
        noise=layer_rows_by_station,
        ci_level=ci_level,
    )

    # --- summary ---------------------------------------------------------- #
    relation_counts: dict[str, int] = {}
    ambiguous_groups = 0
    for group in all_groups:
        if group["ambiguous"]:
            ambiguous_groups += 1
        else:
            relation_counts[group["relation"]] = relation_counts.get(group["relation"], 0) + 1
    optimal_alignments = 1
    for m in meta_cells.values():
        optimal_alignments *= m["optimal_alignments"] or 1
    unannotated_layers = eval_truth.unannotated_by_layer(
        stations, annotation, scope_map, layer_rows_by_station
    )
    summary = {
        "method": method,
        "stations": len(stations),
        "labels": len(label_rows),
        "segments_old": sum(1 for s in all_segments if s.version_role == "old"),
        "segments_new": sum(1 for s in all_segments if s.version_role == "new"),
        "relation_counts": relation_counts,
        "ambiguous_groups": ambiguous_groups,
        "optimal_alignments": optimal_alignments if method == "alignment" else None,
        "alignment_count_truncated": any(
            m["alignment_count_truncated"] for m in meta_cells.values()
        ),
        "additions": sum(g["additions"] for g in group_metrics.values()),
        "missed_old": sum(g["missed_old"] for g in group_metrics.values()),
        "missed_new": sum(g["missed_new"] for g in group_metrics.values()),
        "false_positives_old": sum(g["false_positives_old"] for g in group_metrics.values()),
        "false_positives_new": sum(g["false_positives_new"] for g in group_metrics.values()),
        # Station-layer quantity, summed once per (station, layer) — the
        # per-class group rows repeat it and must not be summed.
        "unannotated_seconds": round(sum(unannotated_layers.values()), 6),
        "unannotated_by_layer": [
            {"station_id": st, "noise_layer": layer, "seconds": round(sec, 6)}
            for (st, layer), sec in sorted(unannotated_layers.items())
        ],
        "unannotated_segments_old": sum(
            g["unannotated_segments_old"] for g in group_metrics.values()
        ),
        "unannotated_segments_new": sum(
            g["unannotated_segments_new"] for g in group_metrics.values()
        ),
    }

    version = (
        db.scalar(
            select(func.max(EvalReport.report_version)).where(
                EvalReport.series_id == snapshot.series_id
            )
        )
        or 0
    ) + 1
    report = EvalReport(
        snapshot_id=snapshot.id,
        series_id=snapshot.series_id,
        report_version=version,
        method=method,
        params_json=params_full,
        params_hash=params_hash,
        status="draft",
        summary_json=summary,
        created_at_us=now_us(),
    )
    db.add(report)
    db.flush()

    for idx, group in enumerate(all_groups, start=1):
        db.add(
            EvalMatch(
                report_id=report.id,
                group_key=f"G{idx:05d}",
                station_id=group["station_id"],
                class_label=group["class_label"],
                noise_layer=group["noise_layer"],
                relation=group["relation"],
                ambiguous=group["ambiguous"],
                old_keys=group["old_keys"],
                new_keys=group["new_keys"],
                label_keys=group["label_keys"],
                truth_json=group["truth"],
                alternatives=group["alternatives"],
                start_us=group["start_us"],
                end_us=group["end_us"],
            )
        )
    for (station, class_label, layer), metrics in sorted(group_metrics.items()):
        db.add(
            EvalReportGroup(
                report_id=report.id,
                station_id=station,
                class_label=class_label,
                noise_layer=layer,
                metrics_json=metrics,
                unannotated_seconds=round(metrics["unannotated_seconds"], 6),
                unannotated_segments_old=metrics["unannotated_segments_old"],
                unannotated_segments_new=metrics["unannotated_segments_new"],
            )
        )
    db.commit()
    db.refresh(report)
    return report, True


def publish_report(db: Session, report: EvalReport) -> EvalReport:
    if report.status == "published":
        return report
    report.status = "published"
    report.published_at_us = now_us()
    db.commit()
    db.refresh(report)
    return report


def adopt_report(db: Session, report: EvalReport) -> EvalReport:
    """Mark a published report as the adopted conclusion of its series.

    Adoption never clears the flag on earlier reports: the adopted history of
    the series stays intact and the *current* conclusion is simply the latest
    adopted report.
    """
    if report.status != "published":
        raise ReportNotPublished(report.id)
    if not report.adopted:
        report.adopted = True
        report.adopted_at_us = now_us()
        db.commit()
        db.refresh(report)
    return report


class ReportNotPublished(Exception):
    def __init__(self, report_id: int):
        super().__init__(f"report {report_id} must be published before it can be adopted")
        self.report_id = report_id
