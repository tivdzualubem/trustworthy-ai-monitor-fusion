#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${1:-$PWD}"
OUTPUT_DIR="${2:-$PWD/final_v3_policy_timing_results}"

python -m pip install -q -r "$PACKAGE_ROOT/requirements.txt"

python "$PACKAGE_ROOT/benchmark_final_v3_policy_timing_gpu.py" \
  --package-root "$PACKAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --expected-gpu-substring T4 \
  --warmup-rows 8 \
  --max-length 512 \
  --max-new-tokens 128
