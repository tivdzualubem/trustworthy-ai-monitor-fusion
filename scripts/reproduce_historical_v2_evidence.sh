#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "ERROR: no usable Python interpreter"
    exit 1
fi

REPORT_DIR="reports/exact_cost_risk_cascade_v2"

HISTORICAL_TABLES=(
    "$REPORT_DIR/across_seed_summary.csv"
    "$REPORT_DIR/cost_equivalence_pass_counts.csv"
    "$REPORT_DIR/endpoint_metrics.csv"
    "$REPORT_DIR/latency_and_tail_summary.csv"
    "$REPORT_DIR/outer_fold_threshold_calibration_and_evaluation.csv"
    "$REPORT_DIR/per_seed_metrics.csv"
    "$REPORT_DIR/primary_pairwise_development_comparisons.csv"
    "$REPORT_DIR/recall_stability_pass_counts.csv"
    "$REPORT_DIR/signed_value_family_selection.csv"
)

echo "=== HISTORICAL V2 REPRODUCTION / PROVENANCE CHECK ==="

for path in "${HISTORICAL_TABLES[@]}"; do
    [[ -f "$path" ]] || {
        echo "ERROR: missing historical table: $path"
        exit 1
    }
done

# Historical published tables are immutable snapshots.
# Refuse to run if any tracked historical CSV is already modified.
if ! git diff --quiet -- "${HISTORICAL_TABLES[@]}"; then
    echo "ERROR: historical v2 CSV snapshot has local modifications"
    git diff --stat -- "${HISTORICAL_TABLES[@]}"
    exit 1
fi

echo
echo "1/3 Regenerate canonical provenance-status artifact"
"$PY" scripts/check_v2_evidence_provenance.py --write

echo
echo "2/3 Validate provenance contract"
"$PY" -m pytest tests/test_v2_evidence_provenance.py -q

echo
echo "3/3 Confirm historical tables remained untouched"
if ! git diff --quiet -- "${HISTORICAL_TABLES[@]}"; then
    echo "ERROR: reproduction check modified historical v2 CSVs"
    git diff --stat -- "${HISTORICAL_TABLES[@]}"
    exit 1
fi

echo
echo "historical_v2_csv_regeneration=false"
echo "historical_v2_reason=missing_raw_timing_provenance"
echo "historical_v2_tables_immutable=true"
echo "corrected_measurement_requires_new_namespace=true"
echo "HISTORICAL_V2_REPRODUCTION_STATUS=PASS"
