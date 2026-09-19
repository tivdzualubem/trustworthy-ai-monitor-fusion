# External-validation confirmatory redesign checkpoint — 2026-09-19

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

- **36 clear anti-conservative type-I rows**

Therefore this inference route is rejected as the primary confirmatory procedure.

Source:
`results/external_validation_balanced_human_block_feasibility_v1.json`

### 3. Assumption-light empirical-Bernstein author-block design

The finite-sample bounded-mean candidate produced:

- **0 clear anti-conservative type-I rows** in the targeted tested domain.
- First coarse point-power candidate: **G=2400 authors/cell, m=40 retained eligible rows/author**
- At that candidate, targeted full-preservation lower-MC power at rho=0.10 was only **0.793**, below 0.80.
- The first targeted G that cleared 0.80 lower-MC power at rho=0.10 across the tested generators was **G=2800 authors/cell**.
- Under the rho=0.20 sensitivity, even the largest tested **G=2800** reached only **0.760** lower-MC full-preservation power.

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
