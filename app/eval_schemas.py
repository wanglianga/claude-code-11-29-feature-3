"""Pydantic schemas for the recognizer-upgrade evaluation subsystem."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .schemas import JsonModel
from .timeutil import parse_iso


def _us(value: str) -> int:
    return int(round(parse_iso(value).timestamp() * 1_000_000))


class _Range(JsonModel):
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _iso(cls, v: str) -> str:
        parse_iso(v)
        return v

    @model_validator(mode="after")
    def _ordered(self):
        if _us(self.end) <= _us(self.start):
            raise ValueError("end must be after start")
        return self


class ScopeIn(_Range):
    station_id: str = Field(min_length=1, max_length=64)


class NoiseLayerIn(_Range):
    station_id: str = Field(min_length=1, max_length=64)
    layer: str = Field(min_length=1, max_length=64)


class AnnotationRangeIn(_Range):
    station_id: str = Field(min_length=1, max_length=64)
    annotator: str = Field(min_length=1, max_length=128)


class LabelIn(_Range):
    label_key: str = Field(min_length=1, max_length=128)
    station_id: str = Field(min_length=1, max_length=64)
    class_label: str = Field(min_length=1, max_length=64)
    annotator: str | None = Field(default=None, max_length=128)
    note: str | None = None


class SegmentIn(_Range):
    segment_key: str = Field(min_length=1, max_length=128)
    version_role: Literal["old", "new"]
    station_id: str = Field(min_length=1, max_length=64)
    class_label: str = Field(min_length=1, max_length=64)
    features: list[float] = Field(min_length=1, max_length=128)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RecognizerVersions(JsonModel):
    old: str = Field(min_length=1, max_length=128)
    new: str = Field(min_length=1, max_length=128)


class SnapshotCreate(JsonModel):
    """One immutable evaluation snapshot.

    The whole payload is content-hashed; re-posting identical content under
    the same ``snapshot_key`` is an idempotent replay, different content under
    a used key is rejected.  Label revisions are a *new* snapshot with
    ``revision_of`` pointing at the snapshot they revise.
    """

    snapshot_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    name: str = Field(min_length=1, max_length=200)
    parser_version: str = Field(min_length=1, max_length=128)
    recognizer_versions: RecognizerVersions
    revision_of: str | None = Field(default=None, max_length=128)
    recording_scopes: list[ScopeIn] = Field(min_length=1)
    noise_layers: list[NoiseLayerIn] = []
    annotation_ranges: list[AnnotationRangeIn] = Field(min_length=1)
    labels: list[LabelIn] = []
    segments: list[SegmentIn] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_snapshot(self):
        stations = {s.station_id for s in self.recording_scopes}
        for coll, what in (
            (self.noise_layers, "noise layer"),
            (self.annotation_ranges, "annotation range"),
            (self.labels, "label"),
            (self.segments, "segment"),
        ):
            for item in coll:
                if item.station_id not in stations:
                    raise ValueError(
                        f"{what} references station {item.station_id!r} which has no recording scope"
                    )
        seen: set[tuple[str, str]] = set()
        for seg in self.segments:
            key = (seg.version_role, seg.segment_key)
            if key in seen:
                raise ValueError(f"duplicate segment key {seg.segment_key!r} in {seg.version_role}")
            seen.add(key)
        roles = {seg.version_role for seg in self.segments}
        if roles != {"old", "new"}:
            raise ValueError("snapshot needs segments of both version roles 'old' and 'new'")
        label_keys = [lab.label_key for lab in self.labels]
        if len(label_keys) != len(set(label_keys)):
            raise ValueError("duplicate label_key in labels")
        dims: dict[str, int] = {}
        for seg in self.segments:
            dim = len(seg.features)
            if seg.class_label in dims and dims[seg.class_label] != dim:
                raise ValueError(
                    f"class {seg.class_label!r} mixes feature dims {dims[seg.class_label]} and {dim}"
                )
            dims[seg.class_label] = dim
        # A label is only meaningful inside a fully annotated range of its station.
        for lab in self.labels:
            ls, le = _us(lab.start), _us(lab.end)
            ok = any(
                r.station_id == lab.station_id and _us(r.start) <= ls and le <= _us(r.end)
                for r in self.annotation_ranges
            )
            if not ok:
                raise ValueError(
                    f"label {lab.label_key!r} lies outside the fully annotated ranges of "
                    f"station {lab.station_id!r}"
                )
        # Everything sits inside the recording scopes of its station.
        for coll, what, key in (
            (self.annotation_ranges, "annotation range", None),
            (self.noise_layers, "noise layer", None),
            (self.labels, "label", "label_key"),
            (self.segments, "segment", "segment_key"),
        ):
            for item in coll:
                s, e = _us(item.start), _us(item.end)
                ok = any(
                    sc.station_id == item.station_id and _us(sc.start) <= s and e <= _us(sc.end)
                    for sc in self.recording_scopes
                )
                if not ok:
                    name = f" {getattr(item, key)!r}" if key else ""
                    raise ValueError(
                        f"{what}{name} lies outside the recording scopes of "
                        f"station {item.station_id!r}"
                    )
        # Noise layers do not overlap on a station, so layer attribution is
        # unambiguous for every segment, label and unannotated stretch.
        for station in stations:
            ranges = sorted(
                (_us(l.start), _us(l.end)) for l in self.noise_layers if l.station_id == station
            )
            for (_, e1), (s2, _) in zip(ranges, ranges[1:]):
                if s2 < e1:
                    raise ValueError(f"noise layers overlap on station {station!r}")
        return self


class SnapshotOut(BaseModel):
    id: int
    snapshot_key: str
    series_id: int
    revision_of: str | None
    name: str
    parser_version: str
    old_recognizer_version: str
    new_recognizer_version: str
    content_hash: str
    counts: dict
    coverage: dict
    created_at: str
    idempotent_replay: bool = False


class SegmentOut(BaseModel):
    segment_key: str
    version_role: str
    station_id: str
    class_label: str
    start: str
    end: str
    features: list[float]
    confidence: float | None
    noise_layer: str


class LabelOut(BaseModel):
    label_key: str
    station_id: str
    class_label: str
    start: str
    end: str
    annotator: str | None
    note: str | None
    noise_layer: str


class ReportParams(JsonModel):
    """Tunable costs/thresholds of a comparison run (frozen into the report)."""

    cluster_gap_us: int = Field(default=250_000, ge=0)
    feature_distance_limit: float = Field(default=0.25, ge=0.0)
    skip_cost_us: int = Field(default=400_000, ge=1)
    reseg_penalty_us: int = Field(default=150_000, ge=0)
    align_max_gap_us: int = Field(default=250_000, ge=0)
    max_warp_us: int = Field(default=2_000_000, ge=0)
    max_block: int = Field(default=4, ge=1, le=8)
    max_alignments: int = Field(default=16, ge=1, le=256)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)


class ReportCreate(JsonModel):
    method: Literal["cluster", "alignment"]
    params: ReportParams = ReportParams()


class ReportOut(BaseModel):
    id: int
    snapshot_id: int
    snapshot_key: str
    series_id: int
    report_version: int
    method: str
    params: dict
    status: str
    adopted: bool
    adopted_at: str | None
    summary: dict
    created_at: str
    published_at: str | None
    idempotent_replay: bool = False


class GroupOut(BaseModel):
    id: int
    station_id: str
    class_label: str
    noise_layer: str
    metrics: dict
    unannotated_seconds: float
    unannotated_segments_old: int
    unannotated_segments_new: int


class ReportDetail(ReportOut):
    groups: list[GroupOut]


class MatchOut(BaseModel):
    group_key: str
    station_id: str
    class_label: str
    noise_layer: str
    relation: str
    ambiguous: bool
    old_keys: list[str]
    new_keys: list[str]
    label_keys: list[str]
    truth: dict
    alternatives: list
    start: str
    end: str
