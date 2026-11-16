"""Truth estimation against the fully human-annotated layer.

Truth rates (misses, false positives, splits, merges, additions) are only
estimable where a human annotated *everything*; segments and scope time
outside the annotated ranges are counted separately and never silently folded
into the rates.  All rates carry a Wilson score interval so a small annotated
sample does not read as a precise estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from .eval_match import Segment


@dataclass(frozen=True)
class Label:
    label_key: str
    station_id: str
    class_label: str
    start_us: int
    end_us: int

    @property
    def midpoint_us(self) -> int:
        return (self.start_us + self.end_us) // 2


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf((1.0 + level) / 2.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_entry(k: int, n: int, level: float) -> dict:
    lo, hi = wilson_interval(k, n, level)
    return {
        "count": k,
        "total": n,
        "rate": (k / n) if n else None,
        "ci_low": lo if n else None,
        "ci_high": hi if n else None,
        "ci_level": level,
    }


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def _subtract(ranges: list[tuple[int, int]], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """ranges minus cuts (both merged)."""
    out: list[tuple[int, int]] = []
    for start, end in ranges:
        cursor = start
        for cs, ce in cuts:
            if ce <= cursor or cs >= end:
                continue
            if cs > cursor:
                out.append((cursor, min(cs, end)))
            cursor = max(cursor, ce)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return out


def _overlap_us(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _in_ranges(ranges: list[tuple[int, int]], instant_us: int) -> bool:
    return any(s <= instant_us < e for s, e in ranges)


def version_label_components(
    labels: list[Label], segments: list[Segment]
) -> list[tuple[list[Label], list[Segment]]]:
    """Connected components of the label↔segment overlap bipartite graph."""
    nodes: list[tuple[str, int]] = [("label", i) for i in range(len(labels))] + [
        ("seg", j) for j in range(len(segments))
    ]
    parent = {node: node for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, lab in enumerate(labels):
        for j, seg in enumerate(segments):
            if _overlap_us(lab.start_us, lab.end_us, seg.start_us, seg.end_us) > 0:
                union(("label", i), ("seg", j))

    comps: dict[tuple, dict[str, list]] = {}
    for kind, idx in nodes:
        comp = comps.setdefault(find((kind, idx)), {"label": [], "seg": []})
        comp[kind].append(idx)
    out = []
    for comp in comps.values():
        out.append(
            (
                [labels[i] for i in comp["label"]],
                [segments[j] for j in comp["seg"]],
            )
        )
    return out


@dataclass
class VersionTruth:
    """Per-label detection state for one recognizer version."""

    detected: set[str]
    missed: set[str]
    split_labels: set[str]
    merge_labels: set[str]
    false_positive_keys: set[str]


def version_truth(labels: list[Label], segments: list[Segment]) -> VersionTruth:
    detected: set[str] = set()
    missed: set[str] = set()
    split_labels: set[str] = set()
    merge_labels: set[str] = set()
    fps: set[str] = set()
    for comp_labels, comp_segs in version_label_components(labels, segments):
        if not comp_labels:
            fps.update(s.segment_key for s in comp_segs)
            continue
        if not comp_segs:
            missed.update(lab.label_key for lab in comp_labels)
            continue
        detected.update(lab.label_key for lab in comp_labels)
        if len(comp_segs) >= 2:
            split_labels.update(lab.label_key for lab in comp_labels)
        if len(comp_labels) >= 2:
            merge_labels.update(lab.label_key for lab in comp_labels)
    return VersionTruth(detected, missed, split_labels, merge_labels, fps)


def layer_of(layers: list[tuple[str, int, int]], instant_us: int) -> str:
    for name, s, e in layers:
        if s <= instant_us < e:
            return name
    return "unlayered"


def unannotated_by_layer(
    stations: list[str],
    annotation: dict[str, list[tuple[int, int]]],
    scopes: dict[str, list[tuple[int, int]]],
    noise: dict[str, list[tuple[str, int, int]]],
) -> dict[tuple[str, str], float]:
    """Unannotated scope seconds per (station, noise layer).

    This is a station-level quantity: it must be listed separately and never
    folded into any rate, and it is *not* per-class — summing it over class
    cells would double-count.
    """
    out: dict[tuple[str, str], float] = {}
    for station in stations:
        ann = annotation.get(station, [])
        unannotated = _subtract(_merge_ranges(scopes.get(station, [])), ann)
        st_noise = noise.get(station, [])
        for start, end in unannotated:
            for layer_name, ls, le in st_noise:
                ov = _overlap_us(start, end, ls, le)
                if ov > 0:
                    key = (station, layer_name)
                    out[key] = out.get(key, 0.0) + ov / 1e6
            for s, e in _subtract([(start, end)], [(ls, le) for _, ls, le in st_noise]):
                key = (station, "unlayered")
                out[key] = out.get(key, 0.0) + (e - s) / 1e6
    return out


def compute_groups(
    stations: list[str],
    labels: list[Label],
    segments: list[Segment],
    annotation: dict[str, list[tuple[int, int]]],
    scopes: dict[str, list[tuple[int, int]]],
    noise: dict[str, list[tuple[str, int, int]]],
    ci_level: float = 0.95,
) -> dict[tuple[str, str, str], dict]:
    """Interval estimates per (station, class, noise layer).

    ``annotation``/``scopes`` map station -> merged ranges; ``noise`` maps
    station -> (layer, start, end) rows.  Returns a dict keyed by
    (station_id, class_label, noise_layer).
    """
    groups: dict[tuple[str, str, str], dict] = {}

    def cell(station: str, class_label: str, layer: str) -> dict:
        key = (station, class_label, layer)
        if key not in groups:
            groups[key] = {
                "labels_total": 0,
                "detected_old": 0,
                "detected_new": 0,
                "missed_old": 0,
                "missed_new": 0,
                "additions": 0,
                "regressions": 0,
                "splits_old": 0,
                "splits_new": 0,
                "merges_old": 0,
                "merges_new": 0,
                "segments_old": 0,
                "segments_new": 0,
                "false_positives_old": 0,
                "false_positives_new": 0,
                "unannotated_seconds": 0.0,
                "unannotated_segments_old": 0,
                "unannotated_segments_new": 0,
            }
        return groups[key]

    classes_by_station: dict[str, set[str]] = {st: set() for st in stations}
    truths: dict[tuple[str, str, str], VersionTruth] = {}
    for station in stations:
        ann = annotation.get(station, [])
        st_labels = [lab for lab in labels if lab.station_id == station]
        for class_label in {lab.class_label for lab in st_labels} | {
            s.class_label for s in segments if s.station_id == station
        }:
            classes_by_station[station].add(class_label)
            cls_labels = [lab for lab in st_labels if lab.class_label == class_label]
            for role in ("old", "new"):
                cls_segs = [
                    s
                    for s in segments
                    if s.station_id == station
                    and s.class_label == class_label
                    and s.version_role == role
                    and _in_ranges(ann, s.midpoint_us)
                ]
                truths[(station, class_label, role)] = version_truth(cls_labels, cls_segs)

    # Labels and their detection state, attributed to the layer at each
    # label's midpoint.
    for lab in labels:
        layer = layer_of(noise.get(lab.station_id, []), lab.midpoint_us)
        c = cell(lab.station_id, lab.class_label, layer)
        c["labels_total"] += 1
        old_t = truths[(lab.station_id, lab.class_label, "old")]
        new_t = truths[(lab.station_id, lab.class_label, "new")]
        in_old = lab.label_key in old_t.detected
        in_new = lab.label_key in new_t.detected
        c["detected_old"] += int(in_old)
        c["detected_new"] += int(in_new)
        c["missed_old"] += int(not in_old)
        c["missed_new"] += int(not in_new)
        c["additions"] += int(in_new and not in_old)
        c["regressions"] += int(in_old and not in_new)
        c["splits_old"] += int(lab.label_key in old_t.split_labels)
        c["splits_new"] += int(lab.label_key in new_t.split_labels)
        c["merges_old"] += int(lab.label_key in old_t.merge_labels)
        c["merges_new"] += int(lab.label_key in new_t.merge_labels)

    # Judgeable segments (midpoint inside the annotated layer).
    for seg in segments:
        ann = annotation.get(seg.station_id, [])
        layer = layer_of(noise.get(seg.station_id, []), seg.midpoint_us)
        c = cell(seg.station_id, seg.class_label, layer)
        if not _in_ranges(ann, seg.midpoint_us):
            c[f"unannotated_segments_{seg.version_role}"] += 1
            continue
        c[f"segments_{seg.version_role}"] += 1
        t = truths[(seg.station_id, seg.class_label, seg.version_role)]
        if seg.segment_key in t.false_positive_keys:
            c[f"false_positives_{seg.version_role}"] += 1

    # Unannotated scope time per (station, layer): listed separately, never
    # folded into any rate.  The value is a station-layer quantity; it is
    # repeated into each class cell of that station-layer for convenience and
    # must not be summed across cells (the report summary dedupes it).
    for (station, layer_name), seconds in unannotated_by_layer(
        stations, annotation, scopes, noise
    ).items():
        for class_label in sorted(classes_by_station[station]):
            cell(station, class_label, layer_name)["unannotated_seconds"] = seconds

    # Attach interval estimates.
    out: dict[tuple[str, str, str], dict] = {}
    for key, c in groups.items():
        metrics = dict(c)
        total = c["labels_total"]
        for name, num in (
            ("miss_rate_old", c["missed_old"]),
            ("miss_rate_new", c["missed_new"]),
            ("split_rate_old", c["splits_old"]),
            ("split_rate_new", c["splits_new"]),
            ("merge_rate_old", c["merges_old"]),
            ("merge_rate_new", c["merges_new"]),
            ("addition_rate", c["additions"]),
            ("regression_rate", c["regressions"]),
        ):
            metrics[name] = rate_entry(num, total, ci_level)
        metrics["false_positive_rate_old"] = rate_entry(
            c["false_positives_old"], c["segments_old"], ci_level
        )
        metrics["false_positive_rate_new"] = rate_entry(
            c["false_positives_new"], c["segments_new"], ci_level
        )
        out[key] = metrics
    return out
