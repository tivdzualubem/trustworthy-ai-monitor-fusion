#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
for P in phase11 phase13 phase14 phase15; do
  case "$P" in
    phase11) tests=(test_phase11_benchmark.py);;
    phase13) tests=(test_phase13.py test_phase13_reference.py);;
    phase14) tests=(test_phase14.py test_phase14_reference.py);;
    phase15) tests=(test_phase15.py);;
  esac
  echo "=== $P ==="
  (cd "$ROOT/analysis/$P" && "$PY" -m unittest -q "${tests[@]}")
done
"$PY" "$ROOT/verify_results.py"
