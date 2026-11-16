"""Immutable evaluation snapshots: hashing, ingestion, coverage.

A snapshot freezes recording scopes, human labels, noise layers, both
recognizer versions' segments and the parser version.  The canonical JSON of
the payload is hashed; the hash decides whether a re-POST is an idempotent
replay (same key, same content) or a conflict (same key, different content).
Nothing in a stored snapshot is ever updated — supplementary annotation or
label revision creates a new snapshot in the same series.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import eval_schemas as es
from .eval_models import (
    EvalAnnotationRange,
    EvalLabel,
    EvalNoiseLayer,
    EvalScope,
    EvalSegment,
    EvalSnapshot,
)
from .timeutil import now_us


def canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(data: dict) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _us(value: str) -> int:
    return es._us(value)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent [start, end) ranges, sorted by start."""
    out: list[list[int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def _covered(ranges: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in _merge_ranges(ranges))


def compute_coverage(body: es.SnapshotCreate) -> dict:
    """Per-station recording/annotated/unannotated durations.

    ``unannotated`` is scope time with no full human-annotation coverage; it
    must be reported separately because truth rates are only estimable inside
    the annotated layer.
    """
    per_station: dict[str, dict] = {}
    for scope in body.recording_scopes:
        per_station.setdefault(scope.station_id, {"scopes": [], "annotation": [], "noise": {}})
    for rng in body.annotation_ranges:
        per_station[rng.station_id]["annotation"].append((_us(rng.start), _us(rng.end)))
    for scope in body.recording_scopes:
        per_station[scope.station_id]["scopes"].append((_us(scope.start), _us(scope.end)))
    for layer in body.noise_layers:
        per_station[layer.station_id]["noise"].setdefault(layer.layer, []).append(
            (_us(layer.start), _us(layer.end))
        )

    out: dict[str, dict] = {}
    for station_id, data in per_station.items():
        scope_us = _covered(data["scopes"])
        annotated_us = _covered(data["annotation"])
        noise = {
            layer: round(_covered(ranges) / 1e6, 6) for layer, ranges in data["noise"].items()
        }
        out[station_id] = {
            "recorded_seconds": round(scope_us / 1e6, 6),
            "annotated_seconds": round(annotated_us / 1e6, 6),
            "unannotated_seconds": round(max(scope_us - annotated_us, 0) / 1e6, 6),
            "noise_layer_seconds": noise,
        }
    return out


def noise_layer_at(layers: list[EvalNoiseLayer], station_id: str, instant_us: int) -> str:
    """The noise layer covering an instant on a station ('unlayered' if none)."""
    for row in layers:
        if row.station_id == station_id and row.start_us <= instant_us < row.end_us:
            return row.layer
    return "unlayered"


def create_snapshot(db: Session, body: es.SnapshotCreate) -> tuple[EvalSnapshot, bool]:
    """Insert a snapshot; returns (snapshot, created).

    Replays are idempotent: same key + same content returns the stored row
    with created=False.  Same key + different content raises SnapshotConflict.
    """
    payload = body.model_dump(mode="json")
    digest = content_hash(payload)
    existing = db.scalar(select(EvalSnapshot).where(EvalSnapshot.snapshot_key == body.snapshot_key))
    if existing is not None:
        if existing.content_hash == digest:
            return existing, False
        raise SnapshotConflict(body.snapshot_key)

    revision_of_id: int | None = None
    series_id: int | None = None
    if body.revision_of is not None:
        parent = db.scalar(
            select(EvalSnapshot).where(EvalSnapshot.snapshot_key == body.revision_of)
        )
        if parent is None:
            raise SnapshotParentMissing(body.revision_of)
        revision_of_id = parent.id
        series_id = parent.series_id

    counts = {
        "stations": len({s.station_id for s in body.recording_scopes}),
        "recording_scopes": len(body.recording_scopes),
        "noise_layers": len(body.noise_layers),
        "annotation_ranges": len(body.annotation_ranges),
        "labels": len(body.labels),
        "segments_old": sum(1 for s in body.segments if s.version_role == "old"),
        "segments_new": sum(1 for s in body.segments if s.version_role == "new"),
    }
    coverage = compute_coverage(body)

    snapshot = EvalSnapshot(
        snapshot_key=body.snapshot_key,
        series_id=0,  # patched below for roots
        revision_of_id=revision_of_id,
        name=body.name,
        parser_version=body.parser_version,
        old_recognizer_version=body.recognizer_versions.old,
        new_recognizer_version=body.recognizer_versions.new,
        content_hash=digest,
        payload=payload,
        counts_json=counts,
        coverage_json=coverage,
        created_at_us=now_us(),
    )
    db.add(snapshot)
    db.flush()
    snapshot.series_id = series_id if series_id is not None else snapshot.id

    for scope in body.recording_scopes:
        db.add(
            EvalScope(
                snapshot_id=snapshot.id,
                station_id=scope.station_id,
                start_us=_us(scope.start),
                end_us=_us(scope.end),
            )
        )
    for layer in body.noise_layers:
        db.add(
            EvalNoiseLayer(
                snapshot_id=snapshot.id,
                station_id=layer.station_id,
                layer=layer.layer,
                start_us=_us(layer.start),
                end_us=_us(layer.end),
            )
        )
    for rng in body.annotation_ranges:
        db.add(
            EvalAnnotationRange(
                snapshot_id=snapshot.id,
                station_id=rng.station_id,
                annotator=rng.annotator,
                start_us=_us(rng.start),
                end_us=_us(rng.end),
            )
        )
    for lab in body.labels:
        db.add(
            EvalLabel(
                snapshot_id=snapshot.id,
                label_key=lab.label_key,
                station_id=lab.station_id,
                class_label=lab.class_label,
                start_us=_us(lab.start),
                end_us=_us(lab.end),
                annotator=lab.annotator,
                note=lab.note,
            )
        )
    for seg in body.segments:
        db.add(
            EvalSegment(
                snapshot_id=snapshot.id,
                segment_key=seg.segment_key,
                version_role=seg.version_role,
                station_id=seg.station_id,
                class_label=seg.class_label,
                start_us=_us(seg.start),
                end_us=_us(seg.end),
                features=list(seg.features),
                confidence=seg.confidence,
            )
        )
    db.commit()
    db.refresh(snapshot)
    return snapshot, True


class SnapshotConflict(Exception):
    def __init__(self, key: str):
        super().__init__(f"snapshot key {key!r} already exists with different content")
        self.key = key


class SnapshotParentMissing(Exception):
    def __init__(self, key: str):
        super().__init__(f"revision_of snapshot {key!r} does not exist")
        self.key = key
