#!/usr/bin/env bash
# One-command demonstration of the recognizer-upgrade evaluation subsystem.
#
#   ./scripts/demo_eval.sh                 # uses http://127.0.0.1:8000
#   BASE=http://host:8000 ./scripts/demo_eval.sh
#
# For every bundled evaluation scenario it freezes the immutable snapshot,
# computes the scenario's reports (cluster and/or alignment method),
# publishes/adopts them as scripted, and then runs the independent verifier,
# which recomputes every classification, interval estimate and provenance
# link from the snapshot payload alone.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
KEY="${WETLAND_API_KEY:-dev-api-key}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

cd "$ROOT"

echo "service: $BASE"
"$PY" - <<PY
import json, urllib.request
with urllib.request.urlopen("$BASE/health", timeout=5) as r:
    print("health:", json.load(r))
PY

for scenario in reseg3 adjacent rain tied; do
  echo
  echo "=== $scenario ==="
  "$PY" -m simulator.run_eval_simulator --scenario "$scenario" --base-url "$BASE" --api-key "$KEY"
  "$PY" -m verifier.verify_eval --scenario "$scenario" --base-url "$BASE" --api-key "$KEY"
done

echo
echo "All evaluation scenarios verified."
