"""API tests for the evaluation subsystem: immutability, versioning, provenance."""

from __future__ import annotations

import copy

import pytest

from simulator.eval_scenarios import build_eval_scenario, scenario_names

T0 = "2026-11-10T00:00:00Z"
T1 = "2026-11-10T00:10:00Z"


def minimal_snapshot(key="snap-1"):
    return {
        "snapshot_key": key,
        "name": "minimal",
        "parser_version": "p/1",
        "recognizer_versions": {"old": "r/1", "new": "r/2"},
        "recording_scopes": [{"station_id": "S1", "start": T0, "end": T1}],
        "noise_layers": [],
        "annotation_ranges": [{"station_id": "S1", "start": T0, "end": T1, "annotator": "a"}],
        "labels": [],
        "segments": [
            {"segment_key": "O1", "version_role": "old", "station_id": "S1",
             "class_label": "c", "start": "2026-11-10T00:01:00Z", "end": "2026-11-10T00:01:01Z",
             "features": [0.1, 0.2]},
            {"segment_key": "N1", "version_role": "new", "station_id": "S1",
             "class_label": "c", "start": "2026-11-10T00:01:00.1Z", "end": "2026-11-10T00:01:01.1Z",
             "features": [0.1, 0.2]},
        ],
    }


def test_snapshot_replay_and_conflict(client):
    snap = minimal_snapshot()
    r1 = client.post("/eval/snapshots", json=snap)
    assert r1.status_code == 201, r1.text
    r2 = client.post("/eval/snapshots", json=snap)
    assert r2.status_code == 200
    assert r2.json()["idempotent_replay"] is True
    assert r2.json()["id"] == r1.json()["id"]
    tampered = dict(snap, name="different content")
    r3 = client.post("/eval/snapshots", json=tampered)
    assert r3.status_code == 409


def test_snapshot_validation(client):
    bad = minimal_snapshot("bad-label")
    bad["labels"] = [
        {"label_key": "L1", "station_id": "S1", "class_label": "c",
         "start": "2026-11-10T00:20:00Z", "end": "2026-11-10T00:20:01Z"}
    ]
    r = client.post("/eval/snapshots", json=bad)
    assert r.status_code == 422
    assert "annotated" in r.text

    overlap = minimal_snapshot("bad-noise")
    overlap["noise_layers"] = [
        {"station_id": "S1", "start": T0, "end": "2026-11-10T00:05:00Z", "layer": "quiet"},
        {"station_id": "S1", "start": "2026-11-10T00:04:00Z", "end": T1, "layer": "rain"},
    ]
    r = client.post("/eval/snapshots", json=overlap)
    assert r.status_code == 422
    assert "overlap" in r.text

    dup = minimal_snapshot("bad-dup")
    dup["segments"].append(dict(dup["segments"][0]))
    r = client.post("/eval/snapshots", json=dup)
    assert r.status_code == 422

    orphan = minimal_snapshot("bad-parent")
    orphan["revision_of"] = "does-not-exist"
    r = client.post("/eval/snapshots", json=orphan)
    assert r.status_code == 422

    outside = minimal_snapshot("bad-scope")
    outside["segments"][0]["end"] = "2026-11-10T00:30:00Z"
    r = client.post("/eval/snapshots", json=outside)
    assert r.status_code == 422
    assert "recording scopes" in r.text


def test_report_versioning_and_revision(client):
    client.post("/eval/snapshots", json=minimal_snapshot("rev-1"))
    r1 = client.post("/eval/snapshots/rev-1/reports", json={"method": "cluster"})
    assert r1.status_code == 201
    assert r1.json()["report_version"] == 1
    assert r1.json()["status"] == "draft"

    # adopt requires publish first
    r = client.post(f"/eval/reports/{r1.json()['id']}/adopt")
    assert r.status_code == 409
    client.post(f"/eval/reports/{r1.json()['id']}/publish")
    r = client.post(f"/eval/reports/{r1.json()['id']}/adopt")
    assert r.status_code == 200
    assert r.json()["adopted"] is True

    # supplementary annotation -> new snapshot in the same series -> v2
    revised = minimal_snapshot("rev-2")
    revised["revision_of"] = "rev-1"
    revised["labels"] = [
        {"label_key": "L9", "station_id": "S1", "class_label": "c",
         "start": "2026-11-10T00:05:00Z", "end": "2026-11-10T00:05:01Z"}
    ]
    r2 = client.post("/eval/snapshots", json=revised)
    assert r2.status_code == 201
    assert r2.json()["series_id"] == r1.json()["series_id"]
    r3 = client.post("/eval/snapshots/rev-2/reports", json={"method": "cluster"})
    assert r3.json()["report_version"] == 2

    # the published v1 report is untouched by the revision
    again = client.get(f"/eval/reports/{r1.json()['id']}")
    assert again.json()["summary"] == r1.json()["summary"]
    assert again.json()["status"] == "published"

    adopted = client.get(f"/eval/series/{r1.json()['series_id']}/adopted")
    assert [r["report_version"] for r in adopted.json()] == [1]


def test_provenance_endpoints(client):
    client.post("/eval/snapshots", json=minimal_snapshot("prov-1"))
    report = client.post("/eval/snapshots/prov-1/reports", json={"method": "alignment"}).json()
    rid = report["id"]
    matches = client.get(f"/eval/reports/{rid}/matches").json()
    assert len(matches) == 1
    match = matches[0]
    assert match["relation"] == "one_to_one"
    assert match["old_keys"] == ["O1"] and match["new_keys"] == ["N1"]

    detail = client.get(f"/eval/reports/{rid}/matches/{match['group_key']}").json()
    assert detail["group_key"] == match["group_key"]
    assert detail["alternatives"][0]["ops"][0]["old"] == ["O1"]

    filtered = client.get(f"/eval/reports/{rid}/matches?relation=new_only").json()
    assert filtered == []
    filtered = client.get(f"/eval/reports/{rid}/matches?relation=one_to_one").json()
    assert len(filtered) == 1

    full = client.get(f"/eval/reports/{rid}").json()
    assert full["groups"][0]["metrics"]["segments_old"] == 1
    assert full["groups"][0]["metrics"]["false_positive_rate_old"]["ci_level"] == 0.95


def test_report_replay_is_idempotent(client):
    client.post("/eval/snapshots", json=minimal_snapshot("idem-1"))
    r1 = client.post("/eval/snapshots/idem-1/reports", json={"method": "cluster"})
    assert r1.status_code == 201
    r2 = client.post("/eval/snapshots/idem-1/reports", json={"method": "cluster"})
    assert r2.status_code == 200
    assert r2.json()["id"] == r1.json()["id"]
    assert r2.json()["idempotent_replay"] is True
    # different costs -> a genuinely different report -> next version
    r3 = client.post(
        "/eval/snapshots/idem-1/reports",
        json={"method": "cluster", "params": {"cluster_gap_us": 500_000}},
    )
    assert r3.status_code == 201
    assert r3.json()["report_version"] == 2


def test_snapshot_detail_and_lists(client):
    client.post("/eval/snapshots", json=minimal_snapshot("list-1"))
    detail = client.get("/eval/snapshots/list-1").json()
    assert detail["counts"]["segments_old"] == 1
    assert detail["coverage"]["S1"]["unannotated_seconds"] == 0.0
    segs = client.get("/eval/snapshots/list-1/segments?version_role=new").json()
    assert [s["segment_key"] for s in segs] == ["N1"]
    assert segs[0]["noise_layer"] == "unlayered"
    labels = client.get("/eval/snapshots/list-1/labels").json()
    assert labels == []
    listing = client.get("/eval/snapshots").json()
    assert any(s["snapshot_key"] == "list-1" for s in listing)
    assert client.get("/eval/snapshots/nope").status_code == 404


@pytest.mark.parametrize("name", scenario_names())
def test_acceptance_scenarios(client, name):
    """Each bundled scenario: snapshot -> reports -> expected classifications."""
    data = build_eval_scenario(name)
    r = client.post("/eval/snapshots", json=data["snapshot"])
    assert r.status_code == 201, r.text
    key = data["snapshot"]["snapshot_key"]
    for spec, expect in zip(data["reports"], data["expect"]["reports"]):
        report = client.post(
            f"/eval/snapshots/{key}/reports",
            json={"method": spec["method"], "params": spec.get("params", {})},
        ).json()
        summary = report["summary"]
        assert summary["relation_counts"] == expect["relation_counts"]
        assert summary["ambiguous_groups"] == expect["ambiguous_groups"]
        matches = client.get(f"/eval/reports/{report['id']}/matches").json()
        got = {(tuple(m["old_keys"]), tuple(m["new_keys"])): m for m in matches}
        for want in expect["matches"]:
            mk = (tuple(want["old_keys"]), tuple(want["new_keys"]))
            assert mk in got, f"{name}/{spec['method']}: missing {mk}"
            gm = got[mk]
            assert gm["relation"] == want["relation"]
            assert gm["ambiguous"] == want["ambiguous"]
            assert sorted(gm["label_keys"]) == sorted(want["label_keys"])
            assert len(gm["alternatives"]) == want["alternatives"]
        full = client.get(f"/eval/reports/{report['id']}").json()
        groups = {(g["station_id"], g["class_label"], g["noise_layer"]): g for g in full["groups"]}
        for gexp in expect.get("groups", []):
            gkey = (gexp["station_id"], gexp["class_label"], gexp["noise_layer"])
            assert gkey in groups
            metrics = groups[gkey]["metrics"]
            for field, value in gexp.items():
                if field in ("station_id", "class_label", "noise_layer", "unannotated_seconds"):
                    continue
                assert metrics[field] == value, f"{name} {gkey} {field}: {metrics[field]} != {value}"
            assert groups[gkey]["unannotated_seconds"] == gexp.get("unannotated_seconds", 0.0)
        for skey, sval in expect.get("summary", {}).items():
            assert summary[skey] == sval

    revision = data.get("revision")
    if revision:
        r = client.post("/eval/snapshots", json=revision["snapshot"])
        assert r.status_code == 201
        rev_key = revision["snapshot"]["snapshot_key"]
        for spec in revision["reports"]:
            report = client.post(
                f"/eval/snapshots/{rev_key}/reports",
                json={"method": spec["method"], "params": spec.get("params", {})},
            ).json()
            assert report["report_version"] == len(data["reports"]) + 1
