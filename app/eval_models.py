"""Tables for the recognizer-upgrade evaluation subsystem.

Design notes
------------
* An *evaluation snapshot* freezes everything an upgrade comparison may ever
  need: recording scopes, human labels, noise layers, the two recognizer
  versions' segments and the parser version that produced them.  Snapshots are
  immutable: the payload is content-hashed, a replay with identical content is
  an idempotent no-op, and the same key with different content is rejected
  with a conflict instead of overwriting.
* Supplementary annotation or label revision never edits a snapshot.  A new
  snapshot is created with ``revision_of`` pointing at its predecessor; both
  belong to the same *series*, and reports computed from them take increasing
  ``report_version`` numbers inside that series.  Published/adopted reports
  are therefore never overwritten by later annotation work.
* Every report keeps its match rows (with the segment and label keys behind
  each classification) so any published number can be traced back to the exact
  evidence it came from.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class EvalSnapshot(Base):
    __tablename__ = "eval_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_key", name="uq_eval_snapshot_key"),
        Index("ix_eval_snapshots_series", "series_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_key: Mapped[str] = mapped_column(String(128), nullable=False)
    # All revisions of one evaluation share a series; the root snapshot's id
    # is the series id.
    series_id: Mapped[int] = mapped_column(Integer, nullable=False)
    revision_of_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    old_recognizer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    new_recognizer_version: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    counts_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    coverage_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at_us: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvalScope(Base):
    """Recording scope: the station was recording during this range."""

    __tablename__ = "eval_scopes"
    __table_args__ = (
        Index("ix_eval_scopes_snapshot_station", "snapshot_id", "station_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvalNoiseLayer(Base):
    __tablename__ = "eval_noise_layers"
    __table_args__ = (
        Index("ix_eval_noise_snapshot_station", "snapshot_id", "station_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    layer: Mapped[str] = mapped_column(String(64), nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvalAnnotationRange(Base):
    """A fully human-annotated range: truth is only estimable inside these."""

    __tablename__ = "eval_annotation_ranges"
    __table_args__ = (
        Index("ix_eval_annotation_snapshot_station", "snapshot_id", "station_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    annotator: Mapped[str] = mapped_column(String(128), nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvalLabel(Base):
    __tablename__ = "eval_labels"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "label_key", name="uq_eval_label_key"),
        Index("ix_eval_labels_snapshot_station", "snapshot_id", "station_id", "class_label"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    label_key: Mapped[str] = mapped_column(String(128), nullable=False)
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_label: Mapped[str] = mapped_column(String(64), nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    annotator: Mapped[str | None] = mapped_column(String(128))
    note: Mapped[str | None] = mapped_column(Text)


class EvalSegment(Base):
    __tablename__ = "eval_segments"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "version_role", "segment_key", name="uq_eval_segment_key"),
        Index(
            "ix_eval_segments_snapshot_station",
            "snapshot_id",
            "station_id",
            "class_label",
            "version_role",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    segment_key: Mapped[str] = mapped_column(String(128), nullable=False)
    version_role: Mapped[str] = mapped_column(String(8), nullable=False)  # old | new
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_label: Mapped[str] = mapped_column(String(64), nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    features: Mapped[list] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)


class EvalReport(Base):
    __tablename__ = "eval_reports"
    __table_args__ = (
        UniqueConstraint("series_id", "report_version", name="uq_eval_report_version"),
        Index("ix_eval_reports_snapshot", "snapshot_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("eval_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    series_id: Mapped[int] = mapped_column(Integer, nullable=False)
    report_version: Mapped[int] = mapped_column(Integer, nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False)  # cluster | alignment
    params_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    adopted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    adopted_at_us: Mapped[int | None] = mapped_column(BigInteger)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    published_at_us: Mapped[int | None] = mapped_column(BigInteger)


class EvalMatch(Base):
    """One relation group from the old↔new comparison, with provenance.

    For the cluster method a row is one connected cluster.  For the alignment
    method a row is one ambiguity group: segments whose optimal decompositions
    never disagree form an unambiguous group (``alternatives`` has one entry);
    segments caught in an equal-cost tie keep every optimal decomposition in
    ``alternatives`` and are flagged ``ambiguous``.
    """

    __tablename__ = "eval_matches"
    __table_args__ = (
        UniqueConstraint("report_id", "group_key", name="uq_eval_match_group"),
        Index("ix_eval_matches_report_station", "report_id", "station_id", "class_label"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("eval_reports.id", ondelete="CASCADE"), nullable=False
    )
    group_key: Mapped[str] = mapped_column(String(64), nullable=False)
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_label: Mapped[str] = mapped_column(String(64), nullable=False)
    noise_layer: Mapped[str] = mapped_column(String(64), nullable=False)
    relation: Mapped[str] = mapped_column(String(16), nullable=False)
    ambiguous: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    old_keys: Mapped[list] = mapped_column(JSONB, nullable=False)
    new_keys: Mapped[list] = mapped_column(JSONB, nullable=False)
    label_keys: Mapped[list] = mapped_column(JSONB, nullable=False)
    truth_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    alternatives: Mapped[list] = mapped_column(JSONB, nullable=False)
    start_us: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_us: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvalReportGroup(Base):
    """Interval estimates for one station × class × noise-layer cell."""

    __tablename__ = "eval_report_groups"
    __table_args__ = (
        UniqueConstraint(
            "report_id", "station_id", "class_label", "noise_layer", name="uq_eval_report_group"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("eval_reports.id", ondelete="CASCADE"), nullable=False
    )
    station_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_label: Mapped[str] = mapped_column(String(64), nullable=False)
    noise_layer: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    unannotated_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unannotated_segments_old: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unannotated_segments_new: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
