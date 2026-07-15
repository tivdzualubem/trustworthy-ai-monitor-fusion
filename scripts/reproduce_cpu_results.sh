#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONHASHSEED=0

echo "1/5 Train and serialize v3 fusion models"
python scripts/train_serialize_fusion_models_v3.py

echo "2/5 Evaluate all prespecified v3 policies"
python scripts/run_policy_evaluation_v3.py

echo "3/5 Run nested leave-group-out evaluation"
python scripts/run_nested_leave_group_out_v1.py

echo "4/5 Run nested Learn-then-Test risk control"
python scripts/run_nested_ltt_risk_control_v1.py

echo "5/5 Write final stop/go decision"
python scripts/write_stop_go_decision_v3.py

echo
echo "Structural verification after reproduction"
python scripts/verify_reproducibility.py
