"""Unit tests for truth estimation against the human-annotated layer."""

from __future__ import annotations

from app.eval_match import Segment
from app.eval_truth import (
    Label,
    compute_groups,
    unannotated_by_layer,
    version_label_components,
    version_truth,
    wilson_interval,
)

REF = 1_794_700_000_000_000
S = 1_000_000


def lab(key, start_s, end_s, cls="croak", station="S1"):
    return Label(
        label_key=key,
        station_id=station,
        class_label=cls,
        start_us=REF + int(start_s * S),
        end_us=REF + int(end_s * S),
    )


def seg(key, role, start_s, end_s, cls="croak", station="S1"):
    return Segment(
        segment_key=key,
        version_role=role,
        station_id=station,
        class_label=cls,
        start_us=REF + int(start_s * S),
        end_us=REF + int(end_s * S),
        features=(0.5, 0.5),
    )


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(1, 2, 0.95)
    assert 0.0 < lo < 0.5 < hi <= 1.0
    lo0, hi0 = wilson_interval(0, 10, 0.95)
    assert lo0 < 1e-12 and hi0 < 0.5
    lo_all, hi_all = wilson_interval(10, 10, 0.95)
    assert lo_all > 0.5 and hi_all > 1.0 - 1e-12
    assert wilson_interval(0, 0, 0.95) == (0.0, 0.0)


def test_version_truth_components():
    labels = [lab("L1", 100.0, 101.0), lab("L2", 200.0, 201.0), lab("L3", 300.0, 301.0)]
    segs = [
        seg("N1", "new", 100.1, 100.5),
        seg("N2", "new", 100.6, 100.9),  # L1 split into two
        seg("N3", "new", 400.0, 401.0),  # false positive
    ]
    truth = version_truth(labels, segs)
    assert truth.detected == {"L1"}
    assert truth.missed == {"L2", "L3"}
    assert truth.split_labels == {"L1"}
    assert truth.merge_labels == set()
    assert truth.false_positive_keys == {"N3"}


def test_merge_detection():
    labels = [lab("L1", 100.0, 101.0), lab("L2", 101.5, 102.5)]
    segs = [seg("N1", "new", 100.0, 102.5)]
    truth = version_truth(labels, segs)
    assert truth.detected == {"L1", "L2"}
    assert truth.merge_labels == {"L1", "L2"}
    assert truth.split_labels == set()


def test_components_are_connected_components():
    labels = [lab("L1", 100.0, 101.0), lab("L2", 100.5, 101.5)]
    segs = [seg("N1", "new", 100.2, 100.8), seg("N2", "new", 500.0, 501.0)]
    comps = version_label_components(labels, segs)
    sizes = sorted((len(ls), len(ss)) for ls, ss in comps)
    assert sizes == [(0, 1), (2, 1)]  # FP segment alone; two labels sharing one segment


def _groups(**overrides):
    labels = overrides.get("labels", [])
    segments = overrides.get("segments", [])
    annotation = overrides.get("annotation", {"S1": [(REF, REF + 600 * S)]})
    scopes = overrides.get("scopes", {"S1": [(REF, REF + 600 * S)]})
    noise = overrides.get("noise", {"S1": [("quiet", REF, REF + 600 * S)]})
    return compute_groups(
        stations=["S1"],
        labels=labels,
        segments=segments,
        annotation=annotation,
        scopes=scopes,
        noise=noise,
        ci_level=0.95,
    )


def test_groups_additions_and_misses():
    labels = [lab("L1", 100.0, 101.0), lab("L2", 200.0, 201.0)]
    segments = [
        seg("O1", "old", 100.1, 100.9),
        seg("N1", "new", 100.1, 100.9),
        seg("N2", "new", 200.1, 200.9),  # addition: L2 only found by the new version
    ]
    groups = _groups(labels=labels, segments=segments)
    cell = groups[("S1", "croak", "quiet")]
    assert cell["labels_total"] == 2
    assert cell["missed_old"] == 1
    assert cell["missed_new"] == 0
    assert cell["additions"] == 1
    assert cell["regressions"] == 0
    assert cell["addition_rate"]["count"] == 1
    assert cell["addition_rate"]["total"] == 2
    assert cell["addition_rate"]["ci_low"] < 0.5 < cell["addition_rate"]["ci_high"]


def test_unannotated_segments_and_time_are_listed_separately():
    # Annotated [0, 300] of a 600 s scope; one new segment sits in the
    # unannotated tail and must not be judged as a false positive.
    labels = [lab("L1", 100.0, 101.0)]
    segments = [
        seg("N1", "new", 100.1, 100.9),
        seg("N2", "new", 400.0, 401.0),  # outside the annotated layer
    ]
    groups = _groups(
        labels=labels,
        segments=segments,
        annotation={"S1": [(REF, REF + 300 * S)]},
        noise={"S1": [("rain", REF + 300 * S, REF + 600 * S)]},
    )
    quiet = groups[("S1", "croak", "unlayered")]
    assert quiet["labels_total"] == 1
    assert quiet["false_positives_new"] == 0
    assert quiet["unannotated_segments_new"] == 0
    rain = groups[("S1", "croak", "rain")]
    assert rain["labels_total"] == 0
    assert rain["unannotated_segments_new"] == 1
    assert rain["unannotated_seconds"] == 300.0


def test_false_positive_rate_interval():
    labels = [lab("L1", 100.0, 101.0)]
    segments = [seg("N1", "new", 100.1, 100.9), seg("N2", "new", 200.0, 201.0)]
    groups = _groups(labels=labels, segments=segments)
    cell = groups[("S1", "croak", "quiet")]
    fp = cell["false_positive_rate_new"]
    assert fp["count"] == 1 and fp["total"] == 2
    assert fp["ci_low"] < 0.5 < fp["ci_high"]


def test_split_and_merge_rates_per_version():
    labels = [lab("L1", 100.0, 102.0), lab("L2", 103.0, 104.0), lab("L3", 104.2, 105.0)]
    segments = [
        seg("N1", "new", 100.0, 101.0),
        seg("N2", "new", 101.2, 102.0),  # L1 split
        seg("N3", "new", 103.0, 105.0),  # L2+L3 merged
    ]
    groups = _groups(labels=labels, segments=segments)
    cell = groups[("S1", "croak", "quiet")]
    assert cell["splits_new"] == 1
    assert cell["merges_new"] == 2
    assert cell["split_rate_new"]["count"] == 1
    assert cell["merge_rate_new"]["count"] == 2


def test_unannotated_seconds_is_a_station_layer_quantity():
    # Two classes share the station: the 300 s unannotated tail is one
    # quantity, repeated per class cell but summed once per (station, layer).
    labels = [lab("L1", 100.0, 101.0), lab("L2", 110.0, 111.0, cls="whoop")]
    segments = [seg("N1", "new", 100.1, 100.9), seg("N2", "new", 110.1, 110.9, cls="whoop")]
    annotation = {"S1": [(REF, REF + 300 * S)]}
    scopes = {"S1": [(REF, REF + 600 * S)]}
    noise = {"S1": [("rain", REF + 300 * S, REF + 600 * S)]}
    by_layer = unannotated_by_layer(["S1"], annotation, scopes, noise)
    assert by_layer == {("S1", "rain"): 300.0}
    groups = compute_groups(
        stations=["S1"],
        labels=labels,
        segments=segments,
        annotation=annotation,
        scopes=scopes,
        noise=noise,
    )
    assert groups[("S1", "croak", "rain")]["unannotated_seconds"] == 300.0
    assert groups[("S1", "whoop", "rain")]["unannotated_seconds"] == 300.0
