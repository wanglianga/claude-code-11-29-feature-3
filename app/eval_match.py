"""Old↔new segment comparison: connected clusters and sequence alignment.

Two interchangeable ways to classify how the upgraded recognizer's segments
relate to the previous version's:

* **cluster** — link every pair of segments (either version) whose time ranges
  overlap or nearly touch and whose acoustic features are close, then take
  connected components.  This naturally recognises re-segmentation (one old
  segment landing in a cluster with several new ones) but, because links are
  transitive, adjacent same-class sounds can be pulled into one cluster.
* **alignment** — align the two ordered boundary sequences with a dynamic
  program whose moves are skip-old, skip-new and block matches (1:1, 1:k
  splits, k:1 merges, k:l complex).  It never merges two new segments into
  one old one unless the block move genuinely wins on cost, which controls
  adjacent-event mis-merges — at the price of having to choose the time-warp
  and skip costs.

Both preserve equal-cost ambiguity: when several optimal alignments exist,
every one of them is kept and the affected segments are reported as an
ambiguous group instead of silently picking a winner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

# Relation names shared by both methods.
ONE_TO_ONE = "one_to_one"
SPLIT = "split"
MERGE = "merge"
COMPLEX = "complex"
NEW_ONLY = "new_only"
OLD_ONLY = "old_only"
AMBIGUOUS = "ambiguous"

RELATIONS = (ONE_TO_ONE, SPLIT, MERGE, COMPLEX, NEW_ONLY, OLD_ONLY)


@dataclass(frozen=True)
class Segment:
    segment_key: str
    version_role: str  # "old" | "new"
    station_id: str
    class_label: str
    start_us: int
    end_us: int
    features: tuple[float, ...]

    @property
    def midpoint_us(self) -> int:
        return (self.start_us + self.end_us) // 2


@dataclass(frozen=True)
class MatchParams:
    cluster_gap_us: int = 250_000
    feature_distance_limit: float = 0.25
    skip_cost_us: int = 400_000
    reseg_penalty_us: int = 150_000
    align_max_gap_us: int = 250_000
    max_warp_us: int = 2_000_000
    max_block: int = 4
    max_alignments: int = 16

    @classmethod
    def from_dict(cls, data: dict) -> "MatchParams":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


def feature_distance(a: Segment, b: Segment) -> float:
    return math.dist(a.features, b.features)


def relation_of(n_old: int, n_new: int) -> str:
    if n_old == 0:
        return NEW_ONLY
    if n_new == 0:
        return OLD_ONLY
    if n_old == 1 and n_new == 1:
        return ONE_TO_ONE
    if n_old == 1:
        return SPLIT
    if n_new == 1:
        return MERGE
    return COMPLEX


def _group(
    station_id: str,
    class_label: str,
    old_segs: list[Segment],
    new_segs: list[Segment],
    alternatives: list[dict],
) -> dict:
    segs = list(old_segs) + list(new_segs)
    ambiguous = len(alternatives) > 1
    relation = alternatives[0]["relation"] if len(alternatives) == 1 else AMBIGUOUS
    return {
        "station_id": station_id,
        "class_label": class_label,
        "relation": relation,
        "ambiguous": ambiguous,
        "old_keys": sorted(s.segment_key for s in old_segs),
        "new_keys": sorted(s.segment_key for s in new_segs),
        "start_us": min(s.start_us for s in segs),
        "end_us": max(s.end_us for s in segs),
        "alternatives": alternatives,
    }


# --------------------------------------------------------------------------- #
# Connected clusters
# --------------------------------------------------------------------------- #
def cluster_groups(segments: list[Segment], params: MatchParams) -> list[dict]:
    """Connected components over time overlap/nearness + acoustic similarity."""
    segs = sorted(segments, key=lambda s: (s.start_us, s.end_us, s.segment_key))
    parent = list(range(len(segs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if b.start_us > a.end_us + params.cluster_gap_us:
                break
            gap = max(0, max(a.start_us, b.start_us) - min(a.end_us, b.end_us))
            if gap > params.cluster_gap_us:
                continue
            if feature_distance(a, b) <= params.feature_distance_limit:
                union(i, j)

    components: dict[int, list[Segment]] = {}
    for idx, seg in enumerate(segs):
        components.setdefault(find(idx), []).append(seg)

    groups = []
    for members in components.values():
        old = [s for s in members if s.version_role == "old"]
        new = [s for s in members if s.version_role == "new"]
        relation = relation_of(len(old), len(new))
        alternatives = [
            {
                "relation": relation,
                "ops": [
                    {
                        "relation": relation,
                        "old": sorted(s.segment_key for s in old),
                        "new": sorted(s.segment_key for s in new),
                    }
                ],
            }
        ]
        groups.append(
            _group(members[0].station_id, members[0].class_label, old, new, alternatives)
        )
    groups.sort(key=lambda g: (g["start_us"], g["end_us"]))
    return groups


# --------------------------------------------------------------------------- #
# Sequence alignment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Op:
    relation: str
    old_idxs: tuple[int, ...]
    new_idxs: tuple[int, ...]
    cost_us: int


def _union_range(segs: list[Segment], idxs: Iterable[int]) -> tuple[int, int]:
    idxs = list(idxs)
    return (
        min(segs[i].start_us for i in idxs),
        max(segs[i].end_us for i in idxs),
    )


def _block_compatible(
    old: list[Segment], i: int, k: int, new: list[Segment], j: int, l: int, params: MatchParams
) -> int | None:
    """Cost of aligning old[i:i+k] with new[j:j+l], or None if not allowed.

    A block is one acoustic event re-segmented, so its members must be
    temporally adjacent on *both* sides: without that check the union
    boundaries would hide arbitrary internal silence and distant segments
    could be glued into a cheap "complex" block.
    """
    for seq, start, size in ((old, i, k), (new, j, l)):
        for x in range(start, start + size - 1):
            if seq[x + 1].start_us - seq[x].end_us > params.align_max_gap_us:
                return None
    o_start, o_end = _union_range(old, range(i, i + k))
    n_start, n_end = _union_range(new, range(j, j + l))
    gap = max(0, max(o_start, n_start) - min(o_end, n_end))
    if gap > params.align_max_gap_us:
        return None
    warp = abs(o_start - n_start) + abs(o_end - n_end)
    if warp > params.max_warp_us:
        return None
    dists = [
        feature_distance(old[i + a], new[j + b]) for a in range(k) for b in range(l)
    ]
    if sum(dists) / len(dists) > params.feature_distance_limit:
        return None
    return warp + params.reseg_penalty_us * (k - 1 + l - 1)


def _block_relation(k: int, l: int) -> str:
    if k == 1 and l == 1:
        return ONE_TO_ONE
    if k == 1:
        return SPLIT
    if l == 1:
        return MERGE
    return COMPLEX


def align_sequences(
    old: list[Segment], new: list[Segment], params: MatchParams
) -> tuple[list[list[Op]], int, bool]:
    """All minimum-cost alignments of the two ordered boundary sequences.

    Returns (alignments, optimal_count, count_truncated).  ``alignments`` is
    capped at ``params.max_alignments``; ``optimal_count`` counts every
    optimum found (capped at _COUNT_CAP) so callers can report ambiguity even
    when not all of them are stored.
    """

    old = sorted(old, key=lambda s: (s.start_us, s.end_us, s.segment_key))
    new = sorted(new, key=lambda s: (s.start_us, s.end_us, s.segment_key))
    m, n = len(old), len(new)
    K = params.max_block
    INF = math.inf

    block_cost: dict[tuple[int, int, int, int], int | None] = {}

    def bcost(i: int, k: int, j: int, l: int) -> int | None:
        key = (i, k, j, l)
        if key not in block_cost:
            block_cost[key] = _block_compatible(old, i, k, new, j, l, params)
        return block_cost[key]

    # Suffix DP: dp[i][j] = min cost of aligning old[i:] with new[j:].
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[m][n] = 0
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m and j == n:
                continue
            best = INF
            if i < m:
                best = min(best, params.skip_cost_us + dp[i + 1][j])
            if j < n:
                best = min(best, params.skip_cost_us + dp[i][j + 1])
            for k in range(1, K + 1):
                if i + k > m:
                    break
                for l in range(1, K + 1):
                    if j + l > n:
                        break
                    c = bcost(i, k, j, l)
                    if c is not None:
                        best = min(best, c + dp[i + k][j + l])
            dp[i][j] = best

    def optimal_moves(i: int, j: int) -> Iterator[tuple[Op, int, int]]:
        target = dp[i][j]
        if i < m and target == params.skip_cost_us + dp[i + 1][j]:
            yield Op(OLD_ONLY, (i,), (), params.skip_cost_us), i + 1, j
        if j < n and target == params.skip_cost_us + dp[i][j + 1]:
            yield Op(NEW_ONLY, (), (j,), params.skip_cost_us), i, j + 1
        for k in range(1, K + 1):
            if i + k > m:
                break
            for l in range(1, K + 1):
                if j + l > n:
                    break
                c = bcost(i, k, j, l)
                if c is not None and target == c + dp[i + k][j + l]:
                    op = Op(
                        _block_relation(k, l),
                        tuple(range(i, i + k)),
                        tuple(range(j, j + l)),
                        c,
                    )
                    yield op, i + k, j + l

    _COUNT_CAP = 100_000
    alignments: list[list[Op]] = []
    seen_signatures: set[tuple] = set()
    count = 0
    leaves = 0
    truncated = False

    def backtrack(i: int, j: int, ops: list[Op]) -> None:
        nonlocal count, leaves, truncated
        if truncated:
            return
        if i == m and j == n:
            leaves += 1
            if leaves >= _COUNT_CAP:
                truncated = True
                return
            # Two paths that differ only in the *order* of skips are the same
            # decomposition; only genuinely different decompositions count as
            # equal-cost ambiguity.
            signature = tuple(sorted((op.relation, op.old_idxs, op.new_idxs) for op in ops))
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)
            count += 1
            if len(alignments) < params.max_alignments:
                alignments.append(list(ops))
            return
        for op, ni, nj in optimal_moves(i, j):
            ops.append(op)
            backtrack(ni, nj, ops)
            ops.pop()

    backtrack(0, 0, [])
    # truncated means: there are more optimal alignments than we stored (or
    # the count itself hit the safety cap).
    truncated = truncated or count > len(alignments)
    return alignments, count, truncated


def alignment_groups(
    old: list[Segment],
    new: list[Segment],
    alignments: list[list[Op]],
) -> list[dict]:
    """Collapse optimal alignments into ambiguity groups.

    Segments that every stored optimum decomposes the same way form an
    unambiguous group; segments caught between equal-cost decompositions keep
    all of their alternatives.
    """
    old = sorted(old, key=lambda s: (s.start_us, s.end_us, s.segment_key))
    new = sorted(new, key=lambda s: (s.start_us, s.end_us, s.segment_key))
    nodes = [("old", i) for i in range(len(old))] + [("new", j) for j in range(len(new))]
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

    for aln in alignments:
        for op in aln:
            members = [("old", i) for i in op.old_idxs] + [("new", j) for j in op.new_idxs]
            for other in members[1:]:
                union(members[0], other)

    comps: dict[tuple, list[tuple]] = {}
    for node in nodes:
        comps.setdefault(find(node), []).append(node)

    groups = []
    for members in comps.values():
        member_set = set(members)
        # Per alignment, the ops restricted to this component.
        seen: dict[tuple, dict] = {}
        for aln in alignments:
            ops = []
            for op in aln:
                op_nodes = {("old", i) for i in op.old_idxs} | {("new", j) for j in op.new_idxs}
                if op_nodes & member_set:
                    ops.append(op)
            signature = tuple(
                (op.relation, op.old_idxs, op.new_idxs) for op in ops
            )
            if signature not in seen:
                seen[signature] = {
                    "ops": [
                        {
                            "relation": op.relation,
                            "old": [old[i].segment_key for i in op.old_idxs],
                            "new": [new[j].segment_key for j in op.new_idxs],
                            "cost_us": op.cost_us,
                        }
                        for op in ops
                    ],
                }
        alternatives = list(seen.values())
        # Alternative-level relation: the op's own relation when the
        # alternative is a single op, "mixed" when one optimum decomposes the
        # group into several ops (the ops list keeps the exact relations).
        for alt in alternatives:
            alt["relation"] = (
                alt["ops"][0]["relation"] if len(alt["ops"]) == 1 else "mixed"
            )
        old_members = [old[i] for role, i in members if role == "old"]
        new_members = [new[i] for role, i in members if role == "new"]
        groups.append(
            _group(
                old[0].station_id if old else new[0].station_id,
                old[0].class_label if old else new[0].class_label,
                old_members,
                new_members,
                alternatives,
            )
        )
    groups.sort(key=lambda g: (g["start_us"], g["end_us"]))
    return groups


def compare(
    segments: list[Segment], method: str, params: MatchParams
) -> tuple[list[dict], dict]:
    """Compare one (station, class) cell; returns (groups, meta)."""
    old = [s for s in segments if s.version_role == "old"]
    new = [s for s in segments if s.version_role == "new"]
    if method == "cluster":
        groups = cluster_groups(segments, params)
        meta = {"optimal_alignments": None, "alignment_count_truncated": False}
    elif method == "alignment":
        alignments, count, truncated = align_sequences(old, new, params)
        groups = alignment_groups(old, new, alignments)
        meta = {"optimal_alignments": count, "alignment_count_truncated": truncated}
    else:  # pragma: no cover - guarded by the API schema
        raise ValueError(f"unknown method {method!r}")
    return groups, meta
