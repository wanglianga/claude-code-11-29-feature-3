"""Acceptance scenarios for the recognizer-upgrade evaluation subsystem.

Each scenario is a full immutable snapshot payload plus the expected
classifications.  The four scenarios are the acceptance set:

* ``reseg3``   — one long call re-sliced into three by the new recognizer;
* ``adjacent`` — three adjacent repeated calls: the cluster method merges
  them into one cluster (its documented weakness) while the alignment method
  keeps them apart;
* ``rain``     — a rain noise layer with false positives, a genuine addition,
  and an unannotated tail that must be listed separately; a second snapshot
  revises the annotation layer and must produce a new report version without
  touching the published one;
* ``tied``     — equal-cost alignments that must be preserved as ambiguity.

All timestamps are reference time (the parser version recorded in the
snapshot is what normalised the recognizer output).
"""

from __future__ import annotations

BASE = "2026-11-10T"


def T(hms: str) -> str:
    return f"{BASE}{hms}Z"


def _seg(key, role, station, cls, start, end, features, confidence=None):
    row = {
        "segment_key": key,
        "version_role": role,
        "station_id": station,
        "class_label": cls,
        "start": start,
        "end": end,
        "features": features,
    }
    if confidence is not None:
        row["confidence"] = confidence
    return row


def _label(key, station, cls, start, end, annotator="annotator-1", note=None):
    row = {
        "label_key": key,
        "station_id": station,
        "class_label": cls,
        "start": start,
        "end": end,
        "annotator": annotator,
    }
    if note:
        row["note"] = note
    return row


def reseg3() -> dict:
    snapshot = {
        "snapshot_key": "eval-reseg3",
        "name": "长声被三段切开",
        "parser_version": "segment-parser/1.2.0",
        "recognizer_versions": {"old": "recognizer/1.4.0", "new": "recognizer/2.0.0"},
        "recording_scopes": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00")}
        ],
        "noise_layers": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "layer": "quiet"}
        ],
        "annotation_ranges": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "annotator": "annotator-1"}
        ],
        "labels": [
            _label("L1", "S1", "whoop", T("00:01:40.0"), T("00:01:52.0"), note="one long call"),
        ],
        "segments": [
            _seg("O1", "old", "S1", "whoop", T("00:01:40.1"), T("00:01:51.9"), [0.5, 0.5]),
            _seg("N1", "new", "S1", "whoop", T("00:01:40.1"), T("00:01:44.0"), [0.5, 0.5]),
            _seg("N2", "new", "S1", "whoop", T("00:01:44.2"), T("00:01:48.0"), [0.5, 0.5]),
            _seg("N3", "new", "S1", "whoop", T("00:01:48.2"), T("00:01:51.9"), [0.5, 0.5]),
        ],
    }
    expect_matches = [
        {
            "relation": "split",
            "old_keys": ["O1"],
            "new_keys": ["N1", "N2", "N3"],
            "label_keys": ["L1"],
            "ambiguous": False,
            "alternatives": 1,
        }
    ]
    expect_group = {
        "station_id": "S1",
        "class_label": "whoop",
        "noise_layer": "quiet",
        "labels_total": 1,
        "detected_old": 1,
        "detected_new": 1,
        "splits_old": 0,
        "splits_new": 1,
        "additions": 0,
        "unannotated_seconds": 0.0,
    }
    return {
        "scenario": "reseg3",
        "title": "One long call re-sliced into three segments by the new recognizer",
        "snapshot": snapshot,
        "reports": [
            {"method": "cluster", "publish": True, "adopt": False},
            {"method": "alignment", "publish": True, "adopt": True},
        ],
        "expect": {
            "reports": [
                {
                    "method": "cluster",
                    "relation_counts": {"split": 1},
                    "ambiguous_groups": 0,
                    "matches": expect_matches,
                    "groups": [expect_group],
                },
                {
                    "method": "alignment",
                    "relation_counts": {"split": 1},
                    "ambiguous_groups": 0,
                    "optimal_alignments": 1,
                    "matches": expect_matches,
                    "groups": [expect_group],
                },
            ]
        },
    }


def adjacent() -> dict:
    def ts(seconds: float) -> str:
        return f"{BASE}00:{int(seconds // 60):02d}:{seconds % 60:05.2f}Z"

    feats = [[0.30, 0.10], [0.31, 0.10], [0.30, 0.11]]
    labels, segments = [], []
    for i in range(3):
        labels.append(_label(f"L{i+1}", "S1", "chirp", ts(120.0 + i), ts(120.8 + i)))
        segments.append(_seg(f"O{i+1}", "old", "S1", "chirp", ts(120.0 + i), ts(120.8 + i), feats[i]))
        segments.append(_seg(f"N{i+1}", "new", "S1", "chirp", ts(120.05 + i), ts(120.85 + i), feats[i]))

    snapshot = {
        "snapshot_key": "eval-adjacent",
        "name": "相邻重复叫声",
        "parser_version": "segment-parser/1.2.0",
        "recognizer_versions": {"old": "recognizer/1.4.0", "new": "recognizer/2.0.0"},
        "recording_scopes": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00")}
        ],
        "noise_layers": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "layer": "quiet"}
        ],
        "annotation_ranges": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "annotator": "annotator-1"}
        ],
        "labels": labels,
        "segments": segments,
    }
    expect_group = {
        "station_id": "S1",
        "class_label": "chirp",
        "noise_layer": "quiet",
        "labels_total": 3,
        "detected_old": 3,
        "detected_new": 3,
        "splits_old": 0,
        "splits_new": 0,
        "merges_old": 0,
        "merges_new": 0,
        "additions": 0,
        "unannotated_seconds": 0.0,
    }
    return {
        "scenario": "adjacent",
        "title": "Three adjacent repeated calls: cluster merges them, alignment separates them",
        "snapshot": snapshot,
        "reports": [
            {"method": "cluster", "publish": True, "adopt": False},
            {"method": "alignment", "publish": True, "adopt": True},
        ],
        "expect": {
            "reports": [
                {
                    "method": "cluster",
                    "relation_counts": {"complex": 1},
                    "ambiguous_groups": 0,
                    "matches": [
                        {
                            "relation": "complex",
                            "old_keys": ["O1", "O2", "O3"],
                            "new_keys": ["N1", "N2", "N3"],
                            "label_keys": ["L1", "L2", "L3"],
                            "ambiguous": False,
                            "alternatives": 1,
                        }
                    ],
                    "groups": [expect_group],
                },
                {
                    "method": "alignment",
                    "relation_counts": {"one_to_one": 3},
                    "ambiguous_groups": 0,
                    "optimal_alignments": 1,
                    "matches": [
                        {
                            "relation": "one_to_one",
                            "old_keys": [f"O{i}"],
                            "new_keys": [f"N{i}"],
                            "label_keys": [f"L{i}"],
                            "ambiguous": False,
                            "alternatives": 1,
                        }
                        for i in (1, 2, 3)
                    ],
                    "groups": [expect_group],
                },
            ]
        },
    }


def rain() -> dict:
    snapshot = {
        "snapshot_key": "eval-rain",
        "name": "雨天噪声（含未标注时段）",
        "parser_version": "segment-parser/1.2.0",
        "recognizer_versions": {"old": "recognizer/1.4.0", "new": "recognizer/2.0.0"},
        "recording_scopes": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:20:00")}
        ],
        "noise_layers": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "layer": "quiet"},
            {"station_id": "S1", "start": T("00:10:00"), "end": T("00:20:00"), "layer": "rain"},
        ],
        "annotation_ranges": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:15:00"), "annotator": "annotator-1"}
        ],
        "labels": [
            _label("L1", "S1", "croak", T("00:01:40.0"), T("00:01:41.5")),
            _label("L2", "S1", "croak", T("00:03:20.0"), T("00:03:21.5")),
            _label("L3", "S1", "croak", T("00:11:40.0"), T("00:11:41.5"), note="in rain"),
        ],
        "segments": [
            _seg("O1", "old", "S1", "croak", T("00:01:40.1"), T("00:01:41.4"), [0.6, 0.2]),
            _seg("O3", "old", "S1", "croak", T("00:11:40.2"), T("00:11:41.4"), [0.6, 0.2]),
            _seg("N1", "new", "S1", "croak", T("00:01:40.1"), T("00:01:41.4"), [0.6, 0.2]),
            _seg("N2", "new", "S1", "croak", T("00:03:19.9"), T("00:03:21.6"), [0.6, 0.2]),
            _seg("N3", "new", "S1", "croak", T("00:11:40.2"), T("00:11:41.4"), [0.6, 0.2]),
            _seg("N4", "new", "S1", "croak", T("00:12:30.0"), T("00:12:30.8"), [0.9, 0.4]),
            _seg("N5", "new", "S1", "croak", T("00:12:40.0"), T("00:12:40.6"), [0.9, 0.4]),
            _seg("N6", "new", "S1", "croak", T("00:15:50.0"), T("00:15:51.0"), [0.6, 0.2]),
        ],
    }
    quiet_group = {
        "station_id": "S1",
        "class_label": "croak",
        "noise_layer": "quiet",
        "labels_total": 2,
        "detected_old": 1,
        "detected_new": 2,
        "missed_old": 1,
        "missed_new": 0,
        "additions": 1,
        "false_positives_new": 0,
        "unannotated_seconds": 0.0,
        "unannotated_segments_new": 0,
    }
    rain_group = {
        "station_id": "S1",
        "class_label": "croak",
        "noise_layer": "rain",
        "labels_total": 1,
        "detected_old": 1,
        "detected_new": 1,
        "missed_old": 0,
        "missed_new": 0,
        "additions": 0,
        "false_positives_new": 2,
        "unannotated_seconds": 300.0,
        "unannotated_segments_new": 1,
    }
    revision_snapshot = {
        **snapshot,
        "snapshot_key": "eval-rain-r2",
        "name": "雨天噪声（补标修订）",
        "revision_of": "eval-rain",
        "annotation_ranges": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:20:00"), "annotator": "annotator-2"}
        ],
        "labels": [
            *snapshot["labels"],
            _label("L4", "S1", "croak", T("00:15:49.8"), T("00:15:51.2"), annotator="annotator-2",
                   note="supplementary annotation of the former unannotated tail"),
        ],
    }
    rain_group_r2 = {
        **rain_group,
        "labels_total": 2,
        "detected_new": 2,
        "missed_old": 1,
        "additions": 1,
        "unannotated_seconds": 0.0,
        "unannotated_segments_new": 0,
    }
    return {
        "scenario": "rain",
        "title": "Rain noise layer: false positives grouped by layer, unannotated time listed separately",
        "snapshot": snapshot,
        "reports": [{"method": "alignment", "publish": True, "adopt": True}],
        "expect": {
            "reports": [
                {
                    "method": "alignment",
                    "relation_counts": {"one_to_one": 2, "new_only": 4},
                    "ambiguous_groups": 0,
                    "optimal_alignments": 1,
                    "matches": [
                        {"relation": "one_to_one", "old_keys": ["O1"], "new_keys": ["N1"], "label_keys": ["L1"], "ambiguous": False, "alternatives": 1},
                        {"relation": "new_only", "old_keys": [], "new_keys": ["N2"], "label_keys": ["L2"], "ambiguous": False, "alternatives": 1},
                        {"relation": "one_to_one", "old_keys": ["O3"], "new_keys": ["N3"], "label_keys": ["L3"], "ambiguous": False, "alternatives": 1},
                        {"relation": "new_only", "old_keys": [], "new_keys": ["N4"], "label_keys": [], "ambiguous": False, "alternatives": 1},
                        {"relation": "new_only", "old_keys": [], "new_keys": ["N5"], "label_keys": [], "ambiguous": False, "alternatives": 1},
                        {"relation": "new_only", "old_keys": [], "new_keys": ["N6"], "label_keys": [], "ambiguous": False, "alternatives": 1},
                    ],
                    "groups": [quiet_group, rain_group],
                    "summary": {"additions": 1, "missed_old": 1, "false_positives_new": 2, "unannotated_seconds": 300.0},
                }
            ]
        },
        "revision": {
            "snapshot": revision_snapshot,
            "reports": [{"method": "alignment", "publish": True, "adopt": True}],
            "expect": {
                "reports": [
                    {
                        "method": "alignment",
                        "groups": [quiet_group, rain_group_r2],
                        "summary": {"additions": 2, "unannotated_seconds": 0.0},
                    }
                ]
            },
        },
    }


def tied() -> dict:
    snapshot = {
        "snapshot_key": "eval-tied",
        "name": "并列对齐（同代价歧义）",
        "parser_version": "segment-parser/1.2.0",
        "recognizer_versions": {"old": "recognizer/1.4.0", "new": "recognizer/2.0.0"},
        "recording_scopes": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00")}
        ],
        "noise_layers": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "layer": "quiet"}
        ],
        "annotation_ranges": [
            {"station_id": "S1", "start": T("00:00:00"), "end": T("00:10:00"), "annotator": "annotator-1"}
        ],
        "labels": [
            _label("LA", "S1", "tick", T("00:01:39.9"), T("00:01:41.1")),
            _label("LB", "S1", "tick", T("00:03:19.9"), T("00:03:21.1")),
            _label("LC", "S1", "tick", T("00:06:39.9"), T("00:06:41.1")),
        ],
        "segments": [
            # tie: match cost 0.4+0.4 s == skip old + skip new (2 x 0.4 s)
            _seg("OA", "old", "S1", "tick", T("00:01:40.0"), T("00:01:41.0"), [0.1, 0.2]),
            _seg("NA", "new", "S1", "tick", T("00:01:40.4"), T("00:01:41.4"), [0.1, 0.2]),
            # clean pair: match cost 0.2 s < skip cost
            _seg("OB", "old", "S1", "tick", T("00:03:20.0"), T("00:03:21.0"), [0.1, 0.2]),
            _seg("NB", "new", "S1", "tick", T("00:03:20.1"), T("00:03:21.1"), [0.1, 0.2]),
            # second tie -> 2 x 2 = 4 equal-cost decompositions overall
            _seg("OC", "old", "S1", "tick", T("00:06:40.0"), T("00:06:41.0"), [0.1, 0.2]),
            _seg("NC", "new", "S1", "tick", T("00:06:40.4"), T("00:06:41.4"), [0.1, 0.2]),
        ],
    }
    expect_group = {
        "station_id": "S1",
        "class_label": "tick",
        "noise_layer": "quiet",
        "labels_total": 3,
        "detected_old": 3,
        "detected_new": 3,
        "additions": 0,
        "unannotated_seconds": 0.0,
    }
    return {
        "scenario": "tied",
        "title": "Equal-cost alignments preserved as ambiguity (2 ties -> 4 optimal decompositions)",
        "snapshot": snapshot,
        "reports": [
            {"method": "alignment", "publish": True, "adopt": True},
            {"method": "cluster", "publish": True, "adopt": False},
        ],
        "expect": {
            "reports": [
                {
                    "method": "alignment",
                    "relation_counts": {"one_to_one": 1},
                    "ambiguous_groups": 2,
                    "optimal_alignments": 4,
                    "matches": [
                        {"relation": "ambiguous", "old_keys": ["OA"], "new_keys": ["NA"], "label_keys": ["LA"], "ambiguous": True, "alternatives": 2},
                        {"relation": "one_to_one", "old_keys": ["OB"], "new_keys": ["NB"], "label_keys": ["LB"], "ambiguous": False, "alternatives": 1},
                        {"relation": "ambiguous", "old_keys": ["OC"], "new_keys": ["NC"], "label_keys": ["LC"], "ambiguous": True, "alternatives": 2},
                    ],
                    "groups": [expect_group],
                },
                {
                    "method": "cluster",
                    "relation_counts": {"one_to_one": 3},
                    "ambiguous_groups": 0,
                    "matches": [
                        {"relation": "one_to_one", "old_keys": ["OA"], "new_keys": ["NA"], "label_keys": ["LA"], "ambiguous": False, "alternatives": 1},
                        {"relation": "one_to_one", "old_keys": ["OB"], "new_keys": ["NB"], "label_keys": ["LB"], "ambiguous": False, "alternatives": 1},
                        {"relation": "one_to_one", "old_keys": ["OC"], "new_keys": ["NC"], "label_keys": ["LC"], "ambiguous": False, "alternatives": 1},
                    ],
                    "groups": [expect_group],
                },
            ]
        },
    }


SCENARIOS = {
    "reseg3": reseg3,
    "adjacent": adjacent,
    "rain": rain,
    "tied": tied,
}


def scenario_names() -> list[str]:
    return list(SCENARIOS)


def build_eval_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise KeyError(f"unknown eval scenario {name!r}; choices: {sorted(SCENARIOS)}")
    return SCENARIOS[name]()
