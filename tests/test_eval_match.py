"""Unit tests for the old↔new comparison methods (cluster and alignment)."""

from __future__ import annotations

from app.eval_match import (
    MatchParams,
    Segment,
    align_sequences,
    alignment_groups,
    cluster_groups,
)

REF = 1_794_700_000_000_000  # fixed reference instant, microseconds
S = 1_000_000


def seg(key, role, start_s, end_s, cls="croak", features=(0.5, 0.5)):
    return Segment(
        segment_key=key,
        version_role=role,
        station_id="S1",
        class_label=cls,
        start_us=REF + int(start_s * S),
        end_us=REF + int(end_s * S),
        features=tuple(features),
    )


def relations(groups):
    return sorted(g["relation"] for g in groups)


def test_cluster_detects_split_one_to_three():
    segs = [
        seg("O1", "old", 100.0, 112.0),
        seg("N1", "new", 100.1, 104.0),
        seg("N2", "new", 104.2, 108.0),
        seg("N3", "new", 108.2, 111.9),
    ]
    groups = cluster_groups(segs, MatchParams())
    assert len(groups) == 1
    assert groups[0]["relation"] == "split"
    assert groups[0]["old_keys"] == ["O1"]
    assert groups[0]["new_keys"] == ["N1", "N2", "N3"]
    assert groups[0]["ambiguous"] is False


def test_alignment_detects_split_one_to_three():
    old = [seg("O1", "old", 100.0, 112.0)]
    new = [seg("N1", "new", 100.1, 104.0), seg("N2", "new", 104.2, 108.0), seg("N3", "new", 108.2, 111.9)]
    alignments, count, truncated = align_sequences(old, new, MatchParams())
    assert count == 1 and not truncated
    groups = alignment_groups(old, new, alignments)
    assert len(groups) == 1
    assert groups[0]["relation"] == "split"
    op = groups[0]["alternatives"][0]["ops"][0]
    assert op["old"] == ["O1"] and op["new"] == ["N1", "N2", "N3"]


def test_cluster_merges_adjacent_calls_but_alignment_separates():
    # Three calls 0.2 s apart: inside the cluster gap tolerance, so the
    # transitive cluster method pulls them into one 3:3 cluster, while the
    # alignment keeps three clean one-to-one pairs.
    old = [seg(f"O{i}", "old", 100.0 + i, 100.8 + i, features=(0.3, 0.1)) for i in range(3)]
    new = [seg(f"N{i}", "new", 100.05 + i, 100.85 + i, features=(0.3, 0.1)) for i in range(3)]
    clusters = cluster_groups(old + new, MatchParams())
    assert len(clusters) == 1
    assert clusters[0]["relation"] == "complex"
    alignments, count, _ = align_sequences(old, new, MatchParams())
    assert count == 1
    groups = alignment_groups(old, new, alignments)
    assert relations(groups) == ["one_to_one"] * 3


def test_alignment_block_spanning_silence_is_rejected():
    # Two old and two new segments separated by 100 s of silence must not be
    # glued into one cheap "complex" block: the internal-gap rule rejects it.
    old = [seg("O1", "old", 100.0, 101.0), seg("O2", "old", 200.0, 201.0)]
    new = [seg("N1", "new", 100.4, 101.4), seg("N2", "new", 200.1, 201.1)]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    groups = alignment_groups(old, new, alignments)
    for group in groups:
        assert group["relation"] != "complex" or group["ambiguous"]


def test_equal_cost_tie_is_preserved_as_ambiguity():
    # match cost = 0.4+0.4 s == skip old (0.4 s) + skip new (0.4 s)
    old = [seg("O1", "old", 100.0, 101.0), seg("O2", "old", 200.0, 201.0)]
    new = [seg("N1", "new", 100.4, 101.4), seg("N2", "new", 200.1, 201.1)]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    assert count == 2  # {O1~N1, O2~N2} and {skip O1, skip N1, O2~N2}
    groups = alignment_groups(old, new, alignments)
    by_old = {tuple(g["old_keys"]): g for g in groups}
    tied = by_old[("O1",)]
    assert tied["ambiguous"] is True
    assert tied["relation"] == "ambiguous"
    alt_relations = sorted(alt["relation"] for alt in tied["alternatives"])
    assert alt_relations == ["mixed", "one_to_one"]
    clean = by_old[("O2",)]
    assert clean["ambiguous"] is False
    assert clean["relation"] == "one_to_one"


def test_two_independent_ties_multiply_optima():
    old = [seg("O1", "old", 100.0, 101.0), seg("O2", "old", 200.0, 201.0), seg("O3", "old", 300.0, 301.0)]
    new = [seg("N1", "new", 100.4, 101.4), seg("N2", "new", 200.1, 201.1), seg("N3", "new", 300.4, 301.4)]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    assert count == 4  # 2 x 2 independent ties
    groups = alignment_groups(old, new, alignments)
    assert sum(1 for g in groups if g["ambiguous"]) == 2
    assert sum(1 for g in groups if not g["ambiguous"]) == 1


def test_new_only_and_old_only():
    old = [seg("O1", "old", 100.0, 101.0)]
    new = [seg("N1", "new", 100.1, 100.9), seg("N2", "new", 300.0, 301.0, features=(0.9, 0.9))]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    assert count == 1
    groups = alignment_groups(old, new, alignments)
    assert relations(groups) == ["new_only", "one_to_one"]
    clusters = cluster_groups(old + new, MatchParams())
    assert relations(clusters) == ["new_only", "one_to_one"]


def test_merge_many_old_to_one_new():
    old = [seg("O1", "old", 100.0, 104.0), seg("O2", "old", 104.2, 108.0)]
    new = [seg("N1", "new", 100.1, 107.9)]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    groups = alignment_groups(old, new, alignments)
    assert len(groups) == 1
    assert groups[0]["relation"] == "merge"
    clusters = cluster_groups(old + new, MatchParams())
    assert clusters[0]["relation"] == "merge"


def test_feature_distance_separates_dissimilar_segments():
    # Same time range but acoustically different: no cluster link, no block.
    old = [seg("O1", "old", 100.0, 101.0, features=(0.1, 0.1))]
    new = [seg("N1", "new", 100.0, 101.0, features=(0.9, 0.9))]
    clusters = cluster_groups(old + new, MatchParams())
    assert relations(clusters) == ["new_only", "old_only"]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    groups = alignment_groups(old, new, alignments)
    assert relations(groups) == ["new_only", "old_only"]


def test_alignment_respects_class_cells():
    # Different classes are compared in separate cells by the report layer;
    # here we just confirm sequences of one class align independently.
    old = [seg("O1", "old", 100.0, 101.0, cls="croak")]
    new = [seg("N1", "new", 100.1, 101.1, cls="croak")]
    alignments, count, _ = align_sequences(old, new, MatchParams())
    assert count == 1
    groups = alignment_groups(old, new, alignments)
    assert groups[0]["relation"] == "one_to_one"


def test_alignment_count_cap_flags_truncation():
    # Eight independent ties -> 2**8 = 256 optima, above the storage cap.
    old = [seg(f"O{i}", "old", 100.0 + 100 * i, 101.0 + 100 * i) for i in range(8)]
    new = [seg(f"N{i}", "new", 100.4 + 100 * i, 101.4 + 100 * i) for i in range(8)]
    params = MatchParams(max_alignments=16)
    alignments, count, truncated = align_sequences(old, new, params)
    assert count == 256
    assert len(alignments) == 16
    assert truncated is True
