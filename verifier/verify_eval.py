"""Independent verification of the recognizer-upgrade evaluation subsystem.

This module deliberately shares *no* code with the service.  It re-implements
the comparison methods from their documented definitions:

* cluster: connected components over time overlap/nearness + feature distance;
* alignment: exhaustive enumeration of every alignment of the two ordered
  boundary sequences (skip old / skip new / block moves), keeping every
  minimum-cost decomposition as ambiguity;
* truth: label<->segment overlap components inside the fully annotated layer
  only, Wilson score intervals for every rate.

Given a ``samples/eval/<scenario>/truth.json`` it recomputes the expected
report from the snapshot payload and compares it with what the service
actually published, including provenance (every reported group traces back to
segment and label keys) and immutability (snapshot replay is idempotent,
conflicting content is rejected, a revised annotation layer starts a new
report version and never rewrites the published one).

Usage::

    python -m verifier.verify_eval --scenario reseg3 --base-url http://127.0.0.1:8000

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.eval_scenarios import build_eval_scenario, scenario_names  # noqa: E402


# --------------------------------------------------------------------------- #
# Small independent helpers
# --------------------------------------------------------------------------- #
def us(value: str) -> int:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1_000_000))


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def subtract(ranges, cuts):
    out = []
    for s, e in ranges:
        cur = s
        for cs, ce in cuts:
            if ce <= cur or cs >= e:
                continue
            if cs > cur:
                out.append((cur, min(cs, e)))
            cur = max(cur, ce)
        if cur < e:
            out.append((cur, e))
    return out


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf((1.0 + level) / 2.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


DEFAULTS = dict(
    cluster_gap_us=250_000,
    feature_distance_limit=0.25,
    skip_cost_us=400_000,
    reseg_penalty_us=150_000,
    align_max_gap_us=250_000,
    max_warp_us=2_000_000,
    max_block=4,
    max_alignments=16,
)


# --------------------------------------------------------------------------- #
# Independent re-implementation of the two comparison methods
# --------------------------------------------------------------------------- #
def relation_of(n_old: int, n_new: int) -> str:
    if n_old == 0:
        return "new_only"
    if n_new == 0:
        return "old_only"
    if n_old == 1 and n_new == 1:
        return "one_to_one"
    if n_old == 1:
        return "split"
    if n_new == 1:
        return "merge"
    return "complex"


def cluster(segments: list[dict], params: dict) -> list[dict]:
    segs = sorted(segments, key=lambda s: (us(s["start"]), us(s["end"]), s["segment_key"]))
    parent = list(range(len(segs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if us(b["start"]) > us(a["end"]) + params["cluster_gap_us"]:
                break
            gap = max(0, max(us(a["start"]), us(b["start"])) - min(us(a["end"]), us(b["end"])))
            if gap <= params["cluster_gap_us"] and dist(a["features"], b["features"]) <= params[
                "feature_distance_limit"
            ]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    comps: dict[int, list[dict]] = {}
    for idx, seg in enumerate(segs):
        comps.setdefault(find(idx), []).append(seg)
    groups = []
    for members in comps.values():
        old = [s for s in members if s["version_role"] == "old"]
        new = [s for s in members if s["version_role"] == "new"]
        groups.append(
            {
                "relation": relation_of(len(old), len(new)),
                "old_keys": sorted(s["segment_key"] for s in old),
                "new_keys": sorted(s["segment_key"] for s in new),
                "ambiguous": False,
                "alternatives": 1,
            }
        )
    return groups


def align(old: list[dict], new: list[dict], params: dict) -> tuple[list[dict], int]:
    """Exhaustive enumeration of all alignments; returns (ambiguity groups, #optima)."""
    old = sorted(old, key=lambda s: (us(s["start"]), us(s["end"]), s["segment_key"]))
    new = sorted(new, key=lambda s: (us(s["start"]), us(s["end"]), s["segment_key"]))
    m, n = len(old), len(new)
    K = params["max_block"]

    def block_cost(i, k, j, l):
        for seq, st, size in ((old, i, k), (new, j, l)):
            for x in range(st, st + size - 1):
                if us(seq[x + 1]["start"]) - us(seq[x]["end"]) > params["align_max_gap_us"]:
                    return None
        os_ = min(us(old[x]["start"]) for x in range(i, i + k))
        oe = max(us(old[x]["end"]) for x in range(i, i + k))
        ns = min(us(new[x]["start"]) for x in range(j, j + l))
        ne = max(us(new[x]["end"]) for x in range(j, j + l))
        gap = max(0, max(os_, ns) - min(oe, ne))
        if gap > params["align_max_gap_us"]:
            return None
        warp = abs(os_ - ns) + abs(oe - ne)
        if warp > params["max_warp_us"]:
            return None
        d = [
            dist(old[i + a]["features"], new[j + b]["features"])
            for a in range(k)
            for b in range(l)
        ]
        if sum(d) / len(d) > params["feature_distance_limit"]:
            return None
        return warp + params["reseg_penalty_us"] * (k - 1 + l - 1)

    def rel(k, l):
        if k == 1 and l == 1:
            return "one_to_one"
        if k == 1:
            return "split"
        if l == 1:
            return "merge"
        return "complex"

    best = [math.inf]
    optima: list[tuple] = []

    def rec(i, j, cost, ops):
        if cost > best[0]:
            return
        if i == m and j == n:
            sig = tuple(sorted(ops))
            if cost < best[0]:
                best[0] = cost
                optima.clear()
                optima.append(sig)
            elif cost == best[0] and sig not in optima:
                optima.append(sig)
            return
        if i < m:
            rec(i + 1, j, cost + params["skip_cost_us"], ops + [("old_only", (i,), ())])
        if j < n:
            rec(i, j + 1, cost + params["skip_cost_us"], ops + [("new_only", (), (j,))])
        for k in range(1, K + 1):
            if i + k > m:
                break
            for l in range(1, K + 1):
                if j + l > n:
                    break
                c = block_cost(i, k, j, l)
                if c is not None:
                    rec(
                        i + k,
                        j + l,
                        cost + c,
                        ops + [(rel(k, l), tuple(range(i, i + k)), tuple(range(j, j + l)))],
                    )

    rec(0, 0, 0, [])

    # ambiguity groups over the optimal decompositions
    nodes = [("old", i) for i in range(m)] + [("new", j) for j in range(n)]
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sig in optima:
        for _, oidx, nidx in sig:
            members = [("old", i) for i in oidx] + [("new", j) for j in nidx]
            for other in members[1:]:
                ra, rb = find(members[0]), find(other)
                if ra != rb:
                    parent[rb] = ra
    comps: dict[tuple, list[tuple]] = {}
    for node in nodes:
        comps.setdefault(find(node), []).append(node)

    groups = []
    for members in comps.values():
        member_set = set(members)
        alts = set()
        for sig in optima:
            alts.add(
                tuple(
                    op
                    for op in sig
                    if ({("old", i) for i in op[1]} | {("new", j) for j in op[2]}) & member_set
                )
            )
        old_keys = sorted(old[i]["segment_key"] for role, i in members if role == "old")
        new_keys = sorted(new[j]["segment_key"] for role, j in members if role == "new")
        ambiguous = len(alts) > 1
        if ambiguous:
            relation = "ambiguous"
        else:
            only = next(iter(alts))
            relation = only[0][0] if len(only) == 1 else "mixed"
        groups.append(
            {
                "relation": relation,
                "old_keys": old_keys,
                "new_keys": new_keys,
                "ambiguous": ambiguous,
                "alternatives": len(alts),
            }
        )
    return groups, len(optima)


def unannotated_by_layer(snapshot: dict) -> dict[tuple[str, str], float]:
    """Unannotated scope seconds per (station, noise layer), independently."""
    stations = sorted({s["station_id"] for s in snapshot["recording_scopes"]})
    out: dict[tuple[str, str], float] = {}
    for st in stations:
        scopes = merge_ranges(
            [(us(r["start"]), us(r["end"])) for r in snapshot["recording_scopes"] if r["station_id"] == st]
        )
        ann = merge_ranges(
            [(us(r["start"]), us(r["end"])) for r in snapshot["annotation_ranges"] if r["station_id"] == st]
        )
        layers = [
            (r["layer"], us(r["start"]), us(r["end"]))
            for r in snapshot["noise_layers"]
            if r["station_id"] == st
        ]
        for s, e in subtract(scopes, ann):
            for name, ls, le in layers:
                ov = max(0, min(e, le) - max(s, ls))
                if ov > 0:
                    out[(st, name)] = out.get((st, name), 0.0) + ov / 1e6
            for us_, ue in subtract([(s, e)], [(ls, le) for _, ls, le in layers]):
                out[(st, "unlayered")] = out.get((st, "unlayered"), 0.0) + (ue - us_) / 1e6
    return out


# --------------------------------------------------------------------------- #
# Independent truth metrics
# --------------------------------------------------------------------------- #
def truth_groups(snapshot: dict, ci_level: float = 0.95) -> dict[tuple[str, str, str], dict]:
    stations = sorted({s["station_id"] for s in snapshot["recording_scopes"]})
    labels = snapshot["labels"]
    segments = snapshot["segments"]
    ann = {
        st: merge_ranges(
            [(us(r["start"]), us(r["end"])) for r in snapshot["annotation_ranges"] if r["station_id"] == st]
        )
        for st in stations
    }
    scopes = {
        st: merge_ranges(
            [(us(r["start"]), us(r["end"])) for r in snapshot["recording_scopes"] if r["station_id"] == st]
        )
        for st in stations
    }
    noise = {
        st: [(r["layer"], us(r["start"]), us(r["end"])) for r in snapshot["noise_layers"] if r["station_id"] == st]
        for st in stations
    }

    def layer_at(st, instant):
        for name, s, e in noise.get(st, []):
            if s <= instant < e:
                return name
        return "unlayered"

    def mid(row):
        return (us(row["start"]) + us(row["end"])) // 2

    def in_ann(st, instant):
        return any(s <= instant < e for s, e in ann.get(st, []))

    def version_truth(cls_labels, cls_segs):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, lab in enumerate(cls_labels):
            for j, seg in enumerate(cls_segs):
                if min(us(lab["end"]), us(seg["end"])) - max(us(lab["start"]), us(seg["start"])) > 0:
                    union(("l", i), ("s", j))
        comps: dict[tuple, dict[str, set]] = {}
        for i in range(len(cls_labels)):
            comps.setdefault(find(("l", i)), {"l": set(), "s": set()})["l"].add(i)
        for j in range(len(cls_segs)):
            comps.setdefault(find(("s", j)), {"l": set(), "s": set()})["s"].add(j)
        detected, missed, splits, merges, fps = set(), set(), set(), set(), set()
        for comp in comps.values():
            if not comp["l"]:
                fps.update(cls_segs[j]["segment_key"] for j in comp["s"])
            elif not comp["s"]:
                missed.update(cls_labels[i]["label_key"] for i in comp["l"])
            else:
                detected.update(cls_labels[i]["label_key"] for i in comp["l"])
                if len(comp["s"]) >= 2:
                    splits.update(cls_labels[i]["label_key"] for i in comp["l"])
                if len(comp["l"]) >= 2:
                    merges.update(cls_labels[i]["label_key"] for i in comp["l"])
        return {"detected": detected, "missed": missed, "splits": splits, "merges": merges, "fps": fps}

    cells: dict[tuple[str, str, str], dict] = {}

    def cell(st, cls, layer):
        key = (st, cls, layer)
        if key not in cells:
            cells[key] = {
                "labels_total": 0, "detected_old": 0, "detected_new": 0,
                "missed_old": 0, "missed_new": 0, "additions": 0, "regressions": 0,
                "splits_old": 0, "splits_new": 0, "merges_old": 0, "merges_new": 0,
                "segments_old": 0, "segments_new": 0,
                "false_positives_old": 0, "false_positives_new": 0,
                "unannotated_seconds": 0.0,
                "unannotated_segments_old": 0, "unannotated_segments_new": 0,
            }
        return cells[key]

    truths = {}
    classes = {st: set() for st in stations}
    for st in stations:
        st_labels = [l for l in labels if l["station_id"] == st]
        st_classes = {l["class_label"] for l in st_labels} | {
            s["class_label"] for s in segments if s["station_id"] == st
        }
        classes[st] = st_classes
        for cls in st_classes:
            cls_labels = [l for l in st_labels if l["class_label"] == cls]
            for role in ("old", "new"):
                cls_segs = [
                    s for s in segments
                    if s["station_id"] == st and s["class_label"] == cls
                    and s["version_role"] == role and in_ann(st, mid(s))
                ]
                truths[(st, cls, role)] = version_truth(cls_labels, cls_segs)

    for lab in labels:
        c = cell(lab["station_id"], lab["class_label"], layer_at(lab["station_id"], mid(lab)))
        c["labels_total"] += 1
        ot = truths[(lab["station_id"], lab["class_label"], "old")]
        nt = truths[(lab["station_id"], lab["class_label"], "new")]
        in_o = lab["label_key"] in ot["detected"]
        in_n = lab["label_key"] in nt["detected"]
        c["detected_old"] += in_o
        c["detected_new"] += in_n
        c["missed_old"] += not in_o
        c["missed_new"] += not in_n
        c["additions"] += in_n and not in_o
        c["regressions"] += in_o and not in_n
        c["splits_old"] += lab["label_key"] in ot["splits"]
        c["splits_new"] += lab["label_key"] in nt["splits"]
        c["merges_old"] += lab["label_key"] in ot["merges"]
        c["merges_new"] += lab["label_key"] in nt["merges"]

    for seg in segments:
        st = seg["station_id"]
        c = cell(st, seg["class_label"], layer_at(st, mid(seg)))
        if not in_ann(st, mid(seg)):
            c[f"unannotated_segments_{seg['version_role']}"] += 1
            continue
        c[f"segments_{seg['version_role']}"] += 1
        if seg["segment_key"] in truths[(st, seg["class_label"], seg["version_role"])]["fps"]:
            c[f"false_positives_{seg['version_role']}"] += 1

    for st in stations:
        if not classes[st]:
            continue
        for s, e in subtract(scopes.get(st, []), ann.get(st, [])):
            for name, ls, le in noise.get(st, []):
                ov = max(0, min(e, le) - max(s, ls))
                if ov > 0:
                    for cls in classes[st]:
                        cell(st, cls, name)["unannotated_seconds"] += ov / 1e6
            for us_, ue in subtract([(s, e)], [(ls, le) for _, ls, le in noise.get(st, [])]):
                for cls in classes[st]:
                    cell(st, cls, "unlayered")["unannotated_seconds"] += (ue - us_) / 1e6

    # attach rates
    for c in cells.values():
        total = c["labels_total"]
        for name, num in (
            ("miss_rate_old", c["missed_old"]), ("miss_rate_new", c["missed_new"]),
            ("split_rate_old", c["splits_old"]), ("split_rate_new", c["splits_new"]),
            ("merge_rate_old", c["merges_old"]), ("merge_rate_new", c["merges_new"]),
            ("addition_rate", c["additions"]), ("regression_rate", c["regressions"]),
        ):
            lo, hi = wilson(num, total, ci_level)
            c[name] = {"count": num, "total": total, "rate": num / total if total else None,
                       "ci_low": lo if total else None, "ci_high": hi if total else None}
        for role in ("old", "new"):
            n = c[f"segments_{role}"]
            k = c[f"false_positives_{role}"]
            lo, hi = wilson(k, n, ci_level)
            c[f"false_positive_rate_{role}"] = {
                "count": k, "total": n, "rate": k / n if n else None,
                "ci_low": lo if n else None, "ci_high": hi if n else None,
            }
    return cells


# --------------------------------------------------------------------------- #
# Checking machinery
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def __str__(self):
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


class Verifier:
    def __init__(self, base_url: str, api_key: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "Content-Type": "application/json"})
        self.checks: list[Check] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.checks.append(Check(name, ok, detail))

    def req(self, method: str, path: str, expect: int | tuple = (200,), **kwargs):
        resp = self.session.request(method, f"{self.base}{path}", timeout=60, **kwargs)
        if not (resp.status_code in expect if isinstance(expect, tuple) else resp.status_code == expect):
            raise RuntimeError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp

    def get(self, path: str):
        return self.req("GET", path).json()


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


def verify_scenario(name: str, base_url: str, api_key: str) -> list[Check]:
    data = build_eval_scenario(name)
    v = Verifier(base_url, api_key)
    snap = data["snapshot"]
    key = snap["snapshot_key"]

    # -- immutability: replay is idempotent, conflicting content is rejected
    replay = v.req("POST", "/eval/snapshots", expect=(200, 201), data=json.dumps(snap))
    replay_body = replay.json()
    v.check(
        "snapshot replay idempotent",
        replay.status_code == 200 and replay_body.get("idempotent_replay") is True,
        f"status={replay.status_code}",
    )
    conflict_payload = dict(snap, name=snap["name"] + " (tampered)")
    conflict = v.req("POST", "/eval/snapshots", expect=(409,), data=json.dumps(conflict_payload))
    v.check("conflicting content rejected with 409", conflict.status_code == 409)

    # -- snapshot coverage recomputed independently
    detail = v.get(f"/eval/snapshots/{key}")
    stations = sorted({s["station_id"] for s in snap["recording_scopes"]})
    for st in stations:
        scopes = merge_ranges([(us(r["start"]), us(r["end"])) for r in snap["recording_scopes"] if r["station_id"] == st])
        ann = merge_ranges([(us(r["start"]), us(r["end"])) for r in snap["annotation_ranges"] if r["station_id"] == st])
        scope_s = sum(e - s for s, e in scopes) / 1e6
        ann_s = sum(e - s for s, e in ann) / 1e6
        cov = detail["coverage"][st]
        v.check(
            f"coverage {st}",
            close(cov["recorded_seconds"], scope_s, 1e-6)
            and close(cov["annotated_seconds"], ann_s, 1e-6)
            and close(cov["unannotated_seconds"], max(scope_s - ann_s, 0.0), 1e-6),
            f"recorded={cov['recorded_seconds']} annotated={cov['annotated_seconds']} "
            f"unannotated={cov['unannotated_seconds']}",
        )

    # -- reports
    reports = v.get(f"/eval/reports?snapshot_key={key}")
    expected_reports = data["expect"]["reports"]
    v.check(
        "report count",
        len(reports) == len(data["reports"]),
        f"{len(reports)} vs {len(data['reports'])}",
    )
    params = dict(DEFAULTS)
    cells_expected = truth_groups(snap)
    seg_keys = {(s["version_role"], s["segment_key"]) for s in snap["segments"]}
    label_keys = {l["label_key"] for l in snap["labels"]}

    for idx, exp in enumerate(expected_reports):
        method = exp["method"]
        report = reports[idx]
        v.check(
            f"report[{idx}] method/version",
            report["method"] == method and report["report_version"] == idx + 1,
            f"method={report['method']} v{report['report_version']}",
        )
        rid = report["id"]
        full = v.get(f"/eval/reports/{rid}")
        matches = v.get(f"/eval/reports/{rid}/matches")

        # independent recompute of the comparison
        by_cell: dict[tuple[str, str], list[dict]] = {}
        for s in snap["segments"]:
            by_cell.setdefault((s["station_id"], s["class_label"]), []).append(s)
        expected_groups: list[dict] = []
        n_optima = None
        for cell_key, segs in sorted(by_cell.items()):
            old = [s for s in segs if s["version_role"] == "old"]
            new = [s for s in segs if s["version_role"] == "new"]
            if method == "cluster":
                expected_groups.extend(cluster(segs, params))
            else:
                groups, count = align(old, new, params)
                expected_groups.extend(groups)
                n_optima = count if n_optima is None else n_optima * count

        summary = full["summary"]
        rel_counts: dict[str, int] = {}
        amb = 0
        for g in expected_groups:
            if g["ambiguous"]:
                amb += 1
            else:
                rel_counts[g["relation"]] = rel_counts.get(g["relation"], 0) + 1
        v.check(
            f"report[{idx}] relation counts (independent recompute)",
            summary["relation_counts"] == rel_counts and summary["ambiguous_groups"] == amb,
            f"{summary['relation_counts']} amb={summary['ambiguous_groups']}",
        )
        if "relation_counts" in exp:
            v.check(
                f"report[{idx}] relation counts (acceptance)",
                summary["relation_counts"] == exp["relation_counts"]
                and summary["ambiguous_groups"] == exp["ambiguous_groups"],
                f"{summary['relation_counts']} amb={summary['ambiguous_groups']}",
            )
        if method == "alignment" and "optimal_alignments" in exp:
            v.check(
                f"report[{idx}] optimal alignments",
                summary["optimal_alignments"] == exp["optimal_alignments"] == n_optima,
                f"{summary['optimal_alignments']}",
            )

        # matches, order-insensitive, incl. provenance keys
        def mkey(m):
            return (tuple(m["old_keys"]), tuple(m["new_keys"]))

        got = {mkey(m): m for m in matches}
        want = {mkey(m): m for m in exp.get("matches", [])}
        v.check(f"report[{idx}] match groups", set(got) == set(want), f"{len(got)} groups")
        for mk, wm in want.items():
            gm = got.get(mk)
            if gm is None:
                continue
            v.check(
                f"report[{idx}] match {mk}",
                gm["relation"] == wm["relation"]
                and gm["ambiguous"] == wm["ambiguous"]
                and sorted(gm["label_keys"]) == sorted(wm["label_keys"])
                and len(gm["alternatives"]) == wm["alternatives"],
                f"relation={gm['relation']} amb={gm['ambiguous']} labels={gm['label_keys']}",
            )
        prov_ok = all(
            all(("old", k) in seg_keys for k in m["old_keys"])
            and all(("new", k) in seg_keys for k in m["new_keys"])
            and all(k in label_keys for k in m["label_keys"])
            for m in matches
        )
        v.check(f"report[{idx}] provenance keys resolve", prov_ok)
        covered = sorted(
            (role, k)
            for m in matches
            for role, keys in (("old", m["old_keys"]), ("new", m["new_keys"]))
            for k in keys
        )
        v.check(
            f"report[{idx}] matches partition all segments",
            covered == sorted(seg_keys),
            f"{len(covered)} vs {len(seg_keys)}",
        )

        # groups: counts and interval estimates recomputed independently
        got_groups = {
            (g["station_id"], g["class_label"], g["noise_layer"]): g for g in full["groups"]
        }
        for gkey, cexp in cells_expected.items():
            gg = got_groups.get(gkey)
            if gg is None:
                v.check(f"report[{idx}] group {gkey}", False, "missing")
                continue
            m = gg["metrics"]
            ok = all(
                m[field] == cexp[field]
                for field in (
                    "labels_total", "detected_old", "detected_new", "missed_old", "missed_new",
                    "additions", "regressions", "splits_old", "splits_new", "merges_old",
                    "merges_new", "segments_old", "segments_new",
                    "false_positives_old", "false_positives_new",
                    "unannotated_segments_old", "unannotated_segments_new",
                )
            )
            rate_ok = True
            for rate_name in (
                "miss_rate_old", "miss_rate_new", "split_rate_old", "split_rate_new",
                "merge_rate_old", "merge_rate_new", "addition_rate", "regression_rate",
                "false_positive_rate_old", "false_positive_rate_new",
            ):
                got_rate = m[rate_name]
                exp_rate = cexp[rate_name]
                rate_ok &= (
                    got_rate["count"] == exp_rate["count"]
                    and got_rate["total"] == exp_rate["total"]
                    and close(got_rate["rate"], exp_rate["rate"])
                    and close(got_rate["ci_low"], exp_rate["ci_low"])
                    and close(got_rate["ci_high"], exp_rate["ci_high"])
                )
            v.check(
                f"report[{idx}] group {gkey} metrics+intervals",
                ok and rate_ok and close(gg["unannotated_seconds"], cexp["unannotated_seconds"], 1e-6),
                f"labels={m['labels_total']} add={m['additions']} "
                f"unannotated={gg['unannotated_seconds']}",
            )
        for gexp in exp.get("groups", []):
            gkey = (gexp["station_id"], gexp["class_label"], gexp["noise_layer"])
            gg = got_groups.get(gkey)
            ok = gg is not None and all(
                gg["metrics"][field] == value
                for field, value in gexp.items()
                if field not in ("station_id", "class_label", "noise_layer", "unannotated_seconds")
            ) and close(gg["unannotated_seconds"], gexp.get("unannotated_seconds", 0.0), 1e-6)
            v.check(f"report[{idx}] group {gkey} (acceptance)", ok)
        for skey, sval in exp.get("summary", {}).items():
            v.check(
                f"report[{idx}] summary {skey}",
                close(summary.get(skey), sval, 1e-6),
                f"{summary.get(skey)} vs {sval}",
            )
        # unannotated duration: station-layer quantity, deduped in the summary
        indep_unann = unannotated_by_layer(snap)
        got_unann = {
            (row["station_id"], row["noise_layer"]): row["seconds"]
            for row in summary.get("unannotated_by_layer", [])
        }
        v.check(
            f"report[{idx}] unannotated listed separately",
            all(close(got_unann.get(k, 0.0), val, 1e-6) for k, val in indep_unann.items())
            and close(summary["unannotated_seconds"], sum(indep_unann.values()), 1e-6),
            f"total={summary['unannotated_seconds']} by_layer={got_unann}",
        )

    # -- lifecycle: publish/adopt flags as scripted
    for idx, spec in enumerate(data["reports"]):
        report = reports[idx]
        want_status = "published" if spec.get("publish") else "draft"
        v.check(
            f"report[{idx}] lifecycle",
            report["status"] == want_status and report["adopted"] == bool(spec.get("adopt")),
            f"status={report['status']} adopted={report['adopted']}",
        )
    adopted = v.get(f"/eval/series/{reports[0]['series_id']}/adopted")
    want_adopted = sum(1 for s in data["reports"] if s.get("adopt")) + sum(
        1 for s in data.get("revision", {}).get("reports", []) if s.get("adopt")
    )
    v.check(
        "adopted history length",
        len(adopted) == want_adopted,
        f"{len(adopted)} adopted reports",
    )

    # -- revision phase (supplementary annotation -> new report version)
    revision = data.get("revision")
    if revision:
        rev_key = revision["snapshot"]["snapshot_key"]
        rev_reports = v.get(f"/eval/reports?snapshot_key={rev_key}")
        base_versions = [r["report_version"] for r in reports]
        rev_versions = [r["report_version"] for r in rev_reports]
        v.check(
            "revision report versions continue the series",
            rev_versions == [max(base_versions) + i + 1 for i in range(len(rev_versions))],
            f"{rev_versions} after {base_versions}",
        )
        first = v.get(f"/eval/reports/{reports[0]['id']}")
        v.check(
            "published report unchanged after revision",
            first["summary"] == reports[0]["summary"] and first["status"] == "published",
            f"v{first['report_version']} summary intact",
        )
        rev_full = v.get(f"/eval/reports/{rev_reports[0]['id']}")
        rev_cells = truth_groups(revision["snapshot"])
        for gexp in revision["expect"]["reports"][0].get("groups", []):
            gkey = (gexp["station_id"], gexp["class_label"], gexp["noise_layer"])
            gg = next(
                (g for g in rev_full["groups"]
                 if (g["station_id"], g["class_label"], g["noise_layer"]) == gkey),
                None,
            )
            ok = gg is not None and all(
                gg["metrics"][field] == value
                for field, value in gexp.items()
                if field not in ("station_id", "class_label", "noise_layer", "unannotated_seconds")
            ) and close(gg["unannotated_seconds"], gexp.get("unannotated_seconds", 0.0), 1e-6)
            indep = rev_cells.get(gkey)
            ok = ok and indep is not None and all(
                gg["metrics"][f] == indep[f]
                for f in ("labels_total", "additions", "missed_old", "missed_new",
                          "false_positives_new", "unannotated_segments_new")
            )
            v.check(f"revision group {gkey}", ok)
        for skey, sval in revision["expect"]["reports"][0].get("summary", {}).items():
            v.check(
                f"revision summary {skey}",
                close(rev_full["summary"].get(skey), sval, 1e-6),
                f"{rev_full['summary'].get(skey)} vs {sval}",
            )
    return v.checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=scenario_names() + ["all"], default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-api-key")
    args = parser.parse_args()

    names = scenario_names() if args.scenario == "all" else [args.scenario]
    all_ok = True
    for name in names:
        checks = verify_scenario(name, args.base_url, args.api_key)
        ok = all(c.ok for c in checks)
        all_ok &= ok
        print(f"=== {name}: {'PASS' if ok else 'FAIL'} ({sum(c.ok for c in checks)}/{len(checks)}) ===")
        for c in checks:
            print(c)
        out = Path(__file__).resolve().parent.parent / "samples" / "eval" / name / "verification.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "scenario": name,
                    "ok": ok,
                    "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
