#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/trustworthy-ai-monitor-fusion"
cd "$REPO"

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "external-validation-precollection-freeze" ]]; then
  echo "ERROR: expected branch external-validation-precollection-freeze, got: $BRANCH"
  exit 1
fi

SRC_BASE="/mnt/c/Users/user/Downloads"
INDEP_DIR="$SRC_BASE/external_validation_independent_block_feasibility_v1"
BAL_DIR="$SRC_BASE/external_validation_balanced_human_block_feasibility_v1"
EB_DIR="$SRC_BASE/external_validation_empirical_bernstein_block_feasibility_v1"

required=(
  "$INDEP_DIR/independent_block_feasibility_v1.json"
  "$INDEP_DIR/independent_block_power_grid_v1.csv"
  "$BAL_DIR/balanced_human_block_feasibility_v1.json"
  "$BAL_DIR/balanced_human_block_screen_v1.csv"
  "$EB_DIR/empirical_bernstein_block_feasibility_v1.json"
  "$EB_DIR/empirical_bernstein_block_power_grid_v1.csv"
)

for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required artifact missing: $f"
    exit 1
  fi
done

mkdir -p results docs

cp "$INDEP_DIR/independent_block_feasibility_v1.json" \
   results/external_validation_independent_block_feasibility_v1.json
cp "$INDEP_DIR/independent_block_power_grid_v1.csv" \
   results/external_validation_independent_block_power_grid_v1.csv

cp "$BAL_DIR/balanced_human_block_feasibility_v1.json" \
   results/external_validation_balanced_human_block_feasibility_v1.json
cp "$BAL_DIR/balanced_human_block_screen_v1.csv" \
   results/external_validation_balanced_human_block_screen_v1.csv

cp "$EB_DIR/empirical_bernstein_block_feasibility_v1.json" \
   results/external_validation_empirical_bernstein_block_feasibility_v1.json
cp "$EB_DIR/empirical_bernstein_block_power_grid_v1.csv" \
   results/external_validation_empirical_bernstein_block_power_grid_v1.csv

python - <<'PY'
from __future__ import annotations
import json
from pathlib import Path

repo = Path.home() / "trustworthy-ai-monitor-fusion"
res = repo / "results"
docs = repo / "docs"

ind = json.loads((res / "external_validation_independent_block_feasibility_v1.json").read_text())
bal = json.loads((res / "external_validation_balanced_human_block_feasibility_v1.json").read_text())
eb = json.loads((res / "external_validation_empirical_bernstein_block_feasibility_v1.json").read_text())

# Fail closed if the uploaded evidence does not match the result state already reviewed.
assert ind["W0_authorized"] is False
assert ind["repo_modified"] is False
assert ind["first_n_meeting_80pct_lower_MC_bound"]["full_preservation_n"] == 1400

assert bal["clear_typeI_failure_count"] > 0
assert bal["interpretation"]["statistical_candidate_survives_targeted_screen"] is False
assert bal["W0_authorized"] is False
assert bal["repo_modified"] is False

assert eb["clear_typeI_failure_count"] == 0
assert eb["decision_boundary"]["W0_authorized"] is False
assert eb["decision_boundary"]["quotas_frozen"] is False
assert eb["repo_modified"] is False

screen = eb["first_screen_candidate"]
primary = eb["candidate_primary_lower_bounds"]
stress = eb["candidate_rho0p20_sensitivity_lower_bounds"]

# Identify first targeted G that clears lower-MC 0.80 for full preservation at rho=.10
# across both tested generators.
rows = eb["targeted_power_rows"]
candidate_Gs = sorted({r["G"] for r in rows})
g_nominal_pass = None
for g in candidate_Gs:
    rr = [r for r in rows if r["G"] == g and r["human_rho"] == 0.10]
    if rr and min(r["full_mc95_lower"] for r in rr) >= 0.80:
        g_nominal_pass = g
        break

rho20_best_g = max(candidate_Gs)
rho20_rows = [r for r in rows if r["G"] == rho20_best_g and r["human_rho"] == 0.20]
rho20_best_lower = min(r["full_mc95_lower"] for r in rho20_rows)

checkpoint = {
    "checkpoint_id": "external_validation_confirmatory_redesign_checkpoint_v1",
    "date": "2026-09-19",
    "status": "supervisor_review_checkpoint_no_W0_authorization",
    "repo_head_observed_by_feasibility_runs": ind["repo_head_observed"],
    "frozen_claim_inputs": {
        "relative_FNR_NI_margin": 0.03,
        "absolute_FNR_ceiling": 0.10,
        "safety_alpha": 0.025,
        "FPR_ceiling": 0.05,
        "FPR_alpha": 0.05,
        "planning_power_target": 0.80,
    },
    "evidence": {
        "singleton_independent_primary_units": {
            "calibration": "no clear anti-conservative boundary rows in the tested screen",
            "first_full_preservation_n_per_cell_with_MC95_lower_at_least_0_80": 1400,
            "interpretation": "statistically viable candidate but operationally requires roughly the same order of distinct human authors in each human validation cell/label stratum",
        },
        "balanced_repeated_author_welch_t": {
            "screen_candidate": bal["screen_candidate"],
            "clear_typeI_failure_count": bal["clear_typeI_failure_count"],
            "survives_targeted_screen": False,
            "interpretation": "reject as primary confirmatory inference candidate",
        },
        "empirical_bernstein_repeated_author": {
            "first_point_estimate_screen_candidate": screen,
            "clear_typeI_failure_count": eb["clear_typeI_failure_count"],
            "screen_candidate_targeted_full_lower95_at_rho_0_10": primary["full_preservation"],
            "first_targeted_G_with_full_lower95_at_least_0_80_at_rho_0_10": g_nominal_pass,
            "largest_tested_G": rho20_best_g,
            "largest_tested_G_full_lower95_at_rho_0_20": rho20_best_lower,
            "interpretation": "assumption-light route is calibrated in the tested screen but requires thousands of independent human author blocks; robustness at rho=0.20 remains below the 0.80 planning-power target even at the largest tested G",
        },
    },
    "scientific_conclusion": {
        "valid_and_operationally_feasible_primary_design_found": False,
        "independent_unit_bottleneck_supported": True,
        "another_ad_hoc_estimator_search_authorized": False,
        "next_decision": "supervisor-level choice between narrowing the confirmatory claim/scope and adopting a stronger explicitly justified model-based primary analysis; beta-binomial remains diagnostic/secondary under the current instruction",
    },
    "execution_boundary": {
        "W0_collection_authorized": False,
        "fresh_monitor_scoring_authorized": False,
        "quotas_frozen": False,
        "sampling_frozen": False,
        "primary_inference_frozen": False,
    },
}

(res / "external_validation_confirmatory_redesign_checkpoint_v1.json").write_text(
    json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
)

md = f"""# External-validation confirmatory redesign checkpoint — 2026-09-19

## Status

This is a **pre-W0 supervisor-review checkpoint**, not a protocol freeze.

- W0 collection: **NOT authorized**
- Fresh monitor scoring: **NOT authorized**
- Validation quotas/caps: **NOT frozen**
- Primary inference: **NOT frozen**
- Sampling mechanism: **NOT frozen**

## Frozen scientific claim inputs retained

- Primary relative FNR non-inferiority margin: **3 percentage points**
- Study-specific absolute FNR ceiling: **10%**
- One-sided safety alpha: **0.025**
- FPR operating ceiling: **5%** at one-sided alpha **0.05**
- Planning joint-power target: **80%**

## Redesign evidence

### 1. Singleton independent primary units

The clean independent-unit candidate produced no clear anti-conservative boundary rows in the tested screen.

The first per-cell size whose Monte Carlo 95% lower bound reached 0.80 for the full preservation claim was:

- **n = 1400 independent primary units per validation cell**

For human cells, this implies roughly the same order of distinct independent authors yielding eligible retained observations. This is statistically viable in the tested screen but operationally demanding.

Source:
`results/external_validation_independent_block_feasibility_v1.json`

### 2. Balanced repeated-author design with Welch/t inference

The repeated-author screen found a nominal-power candidate, but targeted calibration produced:

- **{bal["clear_typeI_failure_count"]} clear anti-conservative type-I rows**

Therefore this inference route is rejected as the primary confirmatory procedure.

Source:
`results/external_validation_balanced_human_block_feasibility_v1.json`

### 3. Assumption-light empirical-Bernstein author-block design

The finite-sample bounded-mean candidate produced:

- **0 clear anti-conservative type-I rows** in the targeted tested domain.
- First coarse point-power candidate: **G={screen["G_authors"]} authors/cell, m={screen["m_rows_per_author"]} retained eligible rows/author**
- At that candidate, targeted full-preservation lower-MC power at rho=0.10 was only **{primary["full_preservation"]:.3f}**, below 0.80.
- The first targeted G that cleared 0.80 lower-MC power at rho=0.10 across the tested generators was **G={g_nominal_pass} authors/cell**.
- Under the rho=0.20 sensitivity, even the largest tested **G={rho20_best_g}** reached only **{rho20_best_lower:.3f}** lower-MC full-preservation power.

This route is assumption-light and calibrated in the tested screen, but it requires thousands of independent human author blocks and remains underpowered in the higher-dependence sensitivity.

Source:
`results/external_validation_empirical_bernstein_block_feasibility_v1.json`

## Scientific synthesis

Across the redesign work now completed:

1. Row-level independence is not permitted by the protocol.
2. The clean singleton-independent design is statistically workable but requires on the order of 1,400 eligible independent human units per cell.
3. Reusing multiple observations per author with Welch/t inference reduces apparent recruitment needs but fails type-I calibration.
4. Reusing multiple observations per author with an assumption-light empirical-Bernstein analysis restores calibration in the tested domain, but the required number of independent authors rises into the thousands.
5. The earlier CR2/WCR and restricted-model candidates remain unsuitable as a frozen primary procedure under the current evidence.

## Decision boundary

The redesign has reached a genuine scientific fork rather than another estimator-tuning step.

Under the current claim structure, no primary design has yet been shown to be both:

- adequately calibrated, and
- operationally feasible for prospective human-source collection.

The next decision should therefore be made at the study-scope/assumption level:

- **narrow the confirmatory claim/scope**, or
- **adopt a stronger explicitly justified model-based primary analysis** with valid nuisance treatment.

Under the supervisor's existing instruction, beta-binomial Monte Carlo remains diagnostic/model-based secondary evidence unless a justified model and valid nuisance treatment support stronger use.

No additional ad-hoc estimator search is justified before that decision.

## Files

- `results/external_validation_independent_block_feasibility_v1.json`
- `results/external_validation_independent_block_power_grid_v1.csv`
- `results/external_validation_balanced_human_block_feasibility_v1.json`
- `results/external_validation_balanced_human_block_screen_v1.csv`
- `results/external_validation_empirical_bernstein_block_feasibility_v1.json`
- `results/external_validation_empirical_bernstein_block_power_grid_v1.csv`
- `results/external_validation_confirmatory_redesign_checkpoint_v1.json`
"""

(docs / "external_validation_confirmatory_redesign_checkpoint_20260919.md").write_text(
    md, encoding="utf-8"
)

print("Evidence validation: PASS")
print(f"Nominal rho=.10 first targeted G passing lower-MC 0.80: {g_nominal_pass}")
print(f"rho=.20 largest-tested G={rho20_best_g}, lower-MC full power={rho20_best_lower:.3f}")
PY

# Validate JSON syntax and intended evidence files.
python -m json.tool results/external_validation_confirmatory_redesign_checkpoint_v1.json >/dev/null
python -m json.tool results/external_validation_independent_block_feasibility_v1.json >/dev/null
python -m json.tool results/external_validation_balanced_human_block_feasibility_v1.json >/dev/null
python -m json.tool results/external_validation_empirical_bernstein_block_feasibility_v1.json >/dev/null

git add \
  results/external_validation_independent_block_feasibility_v1.json \
  results/external_validation_independent_block_power_grid_v1.csv \
  results/external_validation_balanced_human_block_feasibility_v1.json \
  results/external_validation_balanced_human_block_screen_v1.csv \
  results/external_validation_empirical_bernstein_block_feasibility_v1.json \
  results/external_validation_empirical_bernstein_block_power_grid_v1.csv \
  results/external_validation_confirmatory_redesign_checkpoint_v1.json \
  docs/external_validation_confirmatory_redesign_checkpoint_20260919.md

git diff --cached --check

echo
echo "=== STAGED SUMMARY ==="
git diff --cached --stat

echo
echo "=== COMMITTING ONLY THE CHECKPOINT FILES ABOVE ==="
git commit -m "analysis: document confirmatory redesign feasibility checkpoint"

echo
echo "=== FINAL STATUS ==="
git status --short

echo
echo "=== HEAD ==="
git log -1 --oneline

echo
echo "IMPORTANT: nothing was pushed to GitHub."
