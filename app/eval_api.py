"""API routes for the recognizer-upgrade evaluation subsystem."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import eval_report as eval_report_mod
from . import eval_schemas as es
from .db import get_db
from .eval_models import (
    EvalLabel,
    EvalMatch,
    EvalNoiseLayer,
    EvalReport,
    EvalReportGroup,
    EvalSegment,
    EvalSnapshot,
)
from .eval_snapshot import (
    SnapshotConflict,
    SnapshotParentMissing,
    create_snapshot,
    noise_layer_at,
)
from .timeutil import iso

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _snapshot_or_404(db: Session, key: str) -> EvalSnapshot:
    snapshot = db.scalar(select(EvalSnapshot).where(EvalSnapshot.snapshot_key == key))
    if snapshot is None:
        raise HTTPException(404, f"evaluation snapshot {key!r} not found")
    return snapshot


def _report_or_404(db: Session, report_id: int) -> EvalReport:
    report = db.get(EvalReport, report_id)
    if report is None:
        raise HTTPException(404, f"evaluation report {report_id} not found")
    return report


def _snapshot_out(db: Session, snapshot: EvalSnapshot, replay: bool = False) -> es.SnapshotOut:
    parent_key = None
    if snapshot.revision_of_id is not None:
        parent = db.get(EvalSnapshot, snapshot.revision_of_id)
        parent_key = parent.snapshot_key if parent else None
    return es.SnapshotOut(
        id=snapshot.id,
        snapshot_key=snapshot.snapshot_key,
        series_id=snapshot.series_id,
        revision_of=parent_key,
        name=snapshot.name,
        parser_version=snapshot.parser_version,
        old_recognizer_version=snapshot.old_recognizer_version,
        new_recognizer_version=snapshot.new_recognizer_version,
        content_hash=snapshot.content_hash,
        counts=snapshot.counts_json,
        coverage=snapshot.coverage_json,
        created_at=iso(snapshot.created_at_us),
        idempotent_replay=replay,
    )


@router.post(
    "/snapshots",
    response_model=es.SnapshotOut,
    status_code=201,
    summary="Freeze an immutable evaluation snapshot",
)
def post_snapshot(body: es.SnapshotCreate, response: Response, db: Session = Depends(get_db)):
    try:
        snapshot, created = create_snapshot(db, body)
    except SnapshotConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except SnapshotParentMissing as exc:
        raise HTTPException(422, str(exc)) from exc
    if not created:
        response.status_code = 200
    return _snapshot_out(db, snapshot, replay=not created)


@router.get("/snapshots", response_model=list[es.SnapshotOut], summary="List evaluation snapshots")
def list_snapshots(db: Session = Depends(get_db)):
    rows = list(db.execute(select(EvalSnapshot).order_by(EvalSnapshot.id)).scalars())
    return [_snapshot_out(db, row) for row in rows]


@router.get(
    "/snapshots/{key}",
    response_model=es.SnapshotOut,
    summary="Snapshot detail incl. annotated/unannotated coverage",
)
def get_snapshot(key: str, db: Session = Depends(get_db)):
    return _snapshot_out(db, _snapshot_or_404(db, key))


@router.get(
    "/snapshots/{key}/segments",
    response_model=list[es.SegmentOut],
    summary="List recognizer segments of a snapshot",
)
def list_segments(
    key: str,
    version_role: str | None = Query(None, pattern="^(old|new)$"),
    station_id: str | None = None,
    class_label: str | None = None,
    db: Session = Depends(get_db),
):
    snapshot = _snapshot_or_404(db, key)
    stmt = select(EvalSegment).where(EvalSegment.snapshot_id == snapshot.id)
    if version_role:
        stmt = stmt.where(EvalSegment.version_role == version_role)
    if station_id:
        stmt = stmt.where(EvalSegment.station_id == station_id)
    if class_label:
        stmt = stmt.where(EvalSegment.class_label == class_label)
    layers = list(
        db.execute(
            select(EvalNoiseLayer).where(EvalNoiseLayer.snapshot_id == snapshot.id)
        ).scalars()
    )
    out = []
    for row in db.execute(stmt.order_by(EvalSegment.start_us, EvalSegment.segment_key)).scalars():
        out.append(
            es.SegmentOut(
                segment_key=row.segment_key,
                version_role=row.version_role,
                station_id=row.station_id,
                class_label=row.class_label,
                start=iso(row.start_us),
                end=iso(row.end_us),
                features=list(row.features),
                confidence=row.confidence,
                noise_layer=noise_layer_at(layers, row.station_id, (row.start_us + row.end_us) // 2),
            )
        )
    return out


@router.get(
    "/snapshots/{key}/labels",
    response_model=list[es.LabelOut],
    summary="List human labels of a snapshot",
)
def list_labels(key: str, db: Session = Depends(get_db)):
    snapshot = _snapshot_or_404(db, key)
    layers = list(
        db.execute(
            select(EvalNoiseLayer).where(EvalNoiseLayer.snapshot_id == snapshot.id)
        ).scalars()
    )
    rows = list(
        db.execute(
            select(EvalLabel)
            .where(EvalLabel.snapshot_id == snapshot.id)
            .order_by(EvalLabel.start_us, EvalLabel.label_key)
        ).scalars()
    )
    return [
        es.LabelOut(
            label_key=row.label_key,
            station_id=row.station_id,
            class_label=row.class_label,
            start=iso(row.start_us),
            end=iso(row.end_us),
            annotator=row.annotator,
            note=row.note,
            noise_layer=noise_layer_at(layers, row.station_id, (row.start_us + row.end_us) // 2),
        )
        for row in rows
    ]


def _report_out(db: Session, report: EvalReport) -> es.ReportOut:
    snapshot = db.get(EvalSnapshot, report.snapshot_id)
    return es.ReportOut(
        id=report.id,
        snapshot_id=report.snapshot_id,
        snapshot_key=snapshot.snapshot_key if snapshot else "?",
        series_id=report.series_id,
        report_version=report.report_version,
        method=report.method,
        params=report.params_json,
        status=report.status,
        adopted=report.adopted,
        adopted_at=iso(report.adopted_at_us),
        summary=report.summary_json,
        created_at=iso(report.created_at_us),
        published_at=iso(report.published_at_us),
    )


@router.post(
    "/snapshots/{key}/reports",
    response_model=es.ReportOut,
    status_code=201,
    summary="Compute a comparison report (next version in the series)",
)
def post_report(
    key: str, body: es.ReportCreate, response: Response, db: Session = Depends(get_db)
):
    snapshot = _snapshot_or_404(db, key)
    params = body.params.model_dump()
    report, created = eval_report_mod.create_report(db, snapshot, body.method, params)
    out = _report_out(db, report)
    if not created:
        response.status_code = 200
        out.idempotent_replay = True
    return out


@router.get("/reports", response_model=list[es.ReportOut], summary="List evaluation reports")
def list_reports(
    series_id: int | None = None,
    snapshot_key: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(EvalReport).order_by(EvalReport.series_id, EvalReport.report_version)
    if series_id is not None:
        stmt = stmt.where(EvalReport.series_id == series_id)
    if snapshot_key:
        snapshot = _snapshot_or_404(db, snapshot_key)
        stmt = stmt.where(EvalReport.snapshot_id == snapshot.id)
    return [_report_out(db, row) for row in db.execute(stmt).scalars()]


@router.get(
    "/reports/{report_id}",
    response_model=es.ReportDetail,
    summary="Report detail with per-station/class/noise-layer interval estimates",
)
def get_report(report_id: int, db: Session = Depends(get_db)):
    report = _report_or_404(db, report_id)
    groups = list(
        db.execute(
            select(EvalReportGroup)
            .where(EvalReportGroup.report_id == report.id)
            .order_by(EvalReportGroup.station_id, EvalReportGroup.class_label, EvalReportGroup.noise_layer)
        ).scalars()
    )
    out = _report_out(db, report)
    return es.ReportDetail(
        **out.model_dump(),
        groups=[
            es.GroupOut(
                id=row.id,
                station_id=row.station_id,
                class_label=row.class_label,
                noise_layer=row.noise_layer,
                metrics=row.metrics_json,
                unannotated_seconds=row.unannotated_seconds,
                unannotated_segments_old=row.unannotated_segments_old,
                unannotated_segments_new=row.unannotated_segments_new,
            )
            for row in groups
        ],
    )


def _match_out(row: EvalMatch) -> es.MatchOut:
    return es.MatchOut(
        group_key=row.group_key,
        station_id=row.station_id,
        class_label=row.class_label,
        noise_layer=row.noise_layer,
        relation=row.relation,
        ambiguous=row.ambiguous,
        old_keys=list(row.old_keys),
        new_keys=list(row.new_keys),
        label_keys=list(row.label_keys),
        truth=row.truth_json,
        alternatives=list(row.alternatives),
        start=iso(row.start_us),
        end=iso(row.end_us),
    )


@router.get(
    "/reports/{report_id}/matches",
    response_model=list[es.MatchOut],
    summary="Provenance: relation groups behind every reported number",
)
def list_matches(
    report_id: int,
    station_id: str | None = None,
    class_label: str | None = None,
    noise_layer: str | None = None,
    relation: str | None = None,
    ambiguous: bool | None = None,
    db: Session = Depends(get_db),
):
    report = _report_or_404(db, report_id)
    stmt = select(EvalMatch).where(EvalMatch.report_id == report.id)
    if station_id:
        stmt = stmt.where(EvalMatch.station_id == station_id)
    if class_label:
        stmt = stmt.where(EvalMatch.class_label == class_label)
    if noise_layer:
        stmt = stmt.where(EvalMatch.noise_layer == noise_layer)
    if relation:
        stmt = stmt.where(EvalMatch.relation == relation)
    if ambiguous is not None:
        stmt = stmt.where(EvalMatch.ambiguous == ambiguous)
    rows = db.execute(stmt.order_by(EvalMatch.start_us, EvalMatch.group_key)).scalars()
    return [_match_out(row) for row in rows]


@router.get(
    "/reports/{report_id}/matches/{group_key}",
    response_model=es.MatchOut,
    summary="One relation group with its equal-cost alternatives",
)
def get_match(report_id: int, group_key: str, db: Session = Depends(get_db)):
    report = _report_or_404(db, report_id)
    row = db.scalar(
        select(EvalMatch).where(
            EvalMatch.report_id == report.id, EvalMatch.group_key == group_key
        )
    )
    if row is None:
        raise HTTPException(404, f"match group {group_key!r} not found in report {report_id}")
    return _match_out(row)


@router.post(
    "/reports/{report_id}/publish",
    response_model=es.ReportOut,
    summary="Publish a report (immutable conclusion)",
)
def publish_report(report_id: int, db: Session = Depends(get_db)):
    report = _report_or_404(db, report_id)
    return _report_out(db, eval_report_mod.publish_report(db, report))


@router.post(
    "/reports/{report_id}/adopt",
    response_model=es.ReportOut,
    summary="Adopt a published report as the series conclusion",
)
def adopt_report(report_id: int, db: Session = Depends(get_db)):
    report = _report_or_404(db, report_id)
    try:
        report = eval_report_mod.adopt_report(db, report)
    except eval_report_mod.ReportNotPublished as exc:
        raise HTTPException(409, str(exc)) from exc
    return _report_out(db, report)


@router.get(
    "/series/{series_id}/adopted",
    response_model=list[es.ReportOut],
    summary="Adopted conclusions of a series, latest last",
)
def adopted_reports(series_id: int, db: Session = Depends(get_db)):
    rows = list(
        db.execute(
            select(EvalReport)
            .where(EvalReport.series_id == series_id, EvalReport.adopted.is_(True))
            .order_by(EvalReport.report_version)
        ).scalars()
    )
    return [_report_out(db, row) for row in rows]
