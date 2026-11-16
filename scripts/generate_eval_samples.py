"""Regenerate the bundled evaluation scenarios under ``samples/eval/<name>/``.

Nothing here talks to the service, so the files are reproducible and safe to
regenerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.eval_scenarios import build_eval_scenario, scenario_names  # noqa: E402


def main() -> int:
    out_root = ROOT / "samples" / "eval"
    summary = {}
    for name in scenario_names():
        data = build_eval_scenario(name)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "truth.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        snap = data["snapshot"]
        summary[name] = {
            "title": data["title"],
            "labels": len(snap["labels"]),
            "segments_old": sum(1 for s in snap["segments"] if s["version_role"] == "old"),
            "segments_new": sum(1 for s in snap["segments"] if s["version_role"] == "new"),
            "reports": len(data["reports"]) + len(data.get("revision", {}).get("reports", [])),
        }
        print(f"{name}: {summary[name]}")
    (out_root / "index.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
