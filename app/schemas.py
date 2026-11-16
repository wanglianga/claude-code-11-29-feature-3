"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .timeutil import parse_iso, parse_local_us


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
class StationCreate(JsonModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=200)
    x: float
    y: float
    hearing_radius_m: float | None = Field(default=None, gt=0, le=100_000)
    active: bool = True
    notes: str | None = None


class StationOut(BaseModel):
    id: str
    name: str
    x: float
    y: float
    hearing_radius_m: float
    active: bool
    notes: str | None
    created_at: str
    boot_count: int = 0


class ClassCreate(JsonModel):
    id: str = Field(min_length=1, max_length=64)
    description: str | None = None
    feature_dim: int | None = Field(default=None, ge=1, le=128)
    feature_distance_limit: float | None = Field(default=None, ge=0, le=2)


class ClassOut(BaseModel):
    id: str
    description: str | None
    feature_dim: int | None
    feature_distance_limit: float | None
    observation_count: int = 0


class CalibrationSourceCreate(JsonModel):
    id: str = Field(min_length=1, max_length=64)
    name: str | None = None
    x: float
    y: float
    emissions: list[str] = Field(min_length=1)
    notes: str | None = None

    @field_validator("emissions")
    @classmethod
    def _validate_emissions(cls, v: list[str]) -> list[str]:
        for item in v:
            parse_iso(item)
        return v


class CalibrationSourceOut(BaseModel):
    id: str
    name: str | None
    x: float
    y: float
    emissions: list[str]
    notes: str | None


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class CalibrationRef(JsonModel):
    source_id: str = Field(min_length=1, max_length=64)
    emission_reference_at: str


class ObservationRecord(JsonModel):
    source_seq: int = Field(ge=1)
    kind: Literal["sound", "calibration"] = "sound"
    class_label: str = Field(min_length=1, max_length=64)
    local_capture_at: str
    features: list[float] = Field(min_length=1, max_length=128)
    uncertainty_ms: float = Field(gt=0, le=5000)
    calibration: CalibrationRef | None = None
    detector_note: str | None = None

    @field_validator("local_capture_at")
    @classmethod
    def _local_iso(cls, v: str) -> str:
        parse_local_us(v)
        return v.strip()

    @model_validator(mode="after")
    def _calibration_required(self):
        if self.kind == "calibration" and self.calibration is None:
            raise ValueError("calibration records require calibration.source_id and emission_reference_at")
        if self.kind == "sound" and self.calibration is not None:
            raise ValueError("sound records must not carry calibration metadata")
        return self


class ProgressUpload(JsonModel):
    complete_from_local_at: str | None = None
    complete_through_local_at: str
    complete: bool = True
    gaps: list[dict] = Field(default_factory=list)
    note: str | None = None

    @field_validator("complete_from_local_at", "complete_through_local_at")
    @classmethod
    def _local_iso(cls, v: str | None) -> str | None:
        if v is None:
            return v
        parse_local_us(v)
        return v.strip()


class ObservationBatch(JsonModel):
    station_id: str
    boot_id: str = Field(min_length=1, max_length=64)
    batch_key: str = Field(min_length=1, max_length=128, description="Station-chosen unique batch identifier")
    uploaded_at: str | None = Field(
        default=None,
        description="Station-side upload timestamp (UTC). Informational only; never used for association.",
    )
    records: list[ObservationRecord] = Field(min_length=1, max_length=5000)
    progress: ProgressUpload | None = None
    notes: str | None = None


class IngestionCounts(BaseModel):
    received: int
    accepted: int
    duplicates: int
    conflicts: int
    rejected: list[dict] = Field(default_factory=list)


class BatchResponse(BaseModel):
    batch_id: int | None
    replayed: bool
    boot_id: str
    counts: IngestionCounts
    accepted_observation_ids: list[int]
    duplicate_sequences: list[int]
    conflict_sequences: list[int]
    progress_recorded: bool
    replay_of_batch_id: int | None = None
    original_observation_ids: list[int] = Field(default_factory=list)


class ProgressRequest(JsonModel):
    boot_id: str
    progress: ProgressUpload


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
class RunRequest(JsonModel):
    mode: Literal["full", "revision"] = "full"
    wait: bool = True
    debug_stop_after_phase: Literal[
        "calibrate", "collect", "candidates", "localize", "solve", "finalize"
    ] | None = None
    class_labels: list[str] | None = None
    created_by: str | None = None


class RunOut(BaseModel):
    id: int
    batch_id: str
    mode: str
    status: str
    phase: str | None
    cutoff: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    calibration_hash: str | None
    summary: dict | None
    error: str | None
    resume_of_run_id: int | None
    revision_of_run_id: int | None


class RunOption(BaseModel):
    option_id: str
    explained_observations: int
    total_residual_us: int
    events: list[dict]


class RunOptions(BaseModel):
    run_id: int
    status: str
    options: list[RunOption]
    proven_optimal: bool
    notes: list[str]


class SelectOptionRequest(JsonModel):
    option_id: str


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
class EventObservationOut(BaseModel):
    observation_id: int
    station_id: str
    boot_id: str
    source_seq: int
    class_label: str
    local_capture_at: str
    corrected_arrival: str
    uncertainty_ms: float
    residual_ms: float
    calibration_model_id: int | None
    anchor_ids: list[str]


class RegionOut(BaseModel):
    region_id: str
    bounds: dict
    area_m2: float
    cell_count: int
    representative: dict
    cells: list[list[float]] = Field(default_factory=list)
    cells_truncated: bool = False


class EventOut(BaseModel):
    id: int
    run_id: int
    batch_id: str
    class_label: str
    status: str
    revision: int
    reason: str | None
    position: dict
    emission: dict
    residual: dict
    station_count: int
    observation_count: int
    region_count: int
    regions: list[RegionOut]
    finalization: dict | None
    calibration_hash: str
    supersedes_events: list[int]
    superseded_by_event_id: int | None
    window: dict
    created_at: str
    updated_at: str


class EventDetail(EventOut):
    observations: list[EventObservationOut]
    exclusion_reasons: list[dict]
    revisions: list[dict]


class EventList(BaseModel):
    total: int
    events: list[EventOut]


class StationExclusionCreate(JsonModel):
    reason: str = Field(min_length=3, max_length=2000)
    from_ref_at: str | None = None
    to_ref_at: str | None = None
    created_by: str = Field(default="investigator", min_length=1, max_length=128)


class EventExclusionCreate(JsonModel):
    station_id: str
    reason: str = Field(min_length=3, max_length=2000)
    created_by: str = Field(default="investigator", min_length=1, max_length=128)


class ExclusionOut(BaseModel):
    id: int
    station_id: str
    reason: str
    from_ref_at: str | None
    to_ref_at: str | None
    active: bool
    created_by: str
    created_at: str


class ObservationOut(BaseModel):
    id: int
    station_id: str
    boot_id: str
    source_seq: int
    kind: str
    class_label: str
    local_capture_at: str
    uploaded_at: str
    ingested_at: str
    features: list[float]
    uncertainty_ms: float
    calibration_status: str
    calibration_reason: str | None
    calibration_model_id: int | None
    corrected_arrival: str | None
    current_event_id: int | None


class ObservationList(BaseModel):
    total: int
    observations: list[ObservationOut]


class ConflictOut(BaseModel):
    id: int
    station_id: str
    boot_id: str
    source_seq: int
    existing_observation_id: int | None
    content_hash: str
    payload: dict
    received_at: str
    resolved: bool
    resolution_note: str | None


class CalibrationSegmentOut(BaseModel):
    index: int
    anchor_count: int
    anchor_ids: list[str]
    local_from: str
    local_to: str
    slope: float
    intercept_ms: float
    rms_residual_ms: float
    max_residual_ms: float


class CalibrationOut(BaseModel):
    id: int
    station_id: str
    boot_id: str
    version_hash: str
    anchor_count: int
    anchor_ids: list[str]
    valid: bool
    invalid_reason: str | None
    rms_residual_ms: float | None
    max_residual_ms: float | None
    uncertainty_ms: float | None
    anchor_local_span_s: float | None
    segments: list[CalibrationSegmentOut]
    created_at: str


class CalibrationList(BaseModel):
    total: int
    calibrations: list[CalibrationOut]


class StatsSummary(BaseModel):
    stations: int
    observations: int
    uncalibrated_observations: int
    conflicts: int
    runs: dict
    events: dict
    latest_run: RunOut | None
