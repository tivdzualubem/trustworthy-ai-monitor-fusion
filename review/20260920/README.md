# Confirmatory redesign checkpoint

## Current status

W0 has not started. No fresh monitor scoring or validation collection is authorized.

The claim parameters remain unchanged:

- FNR non-inferiority margin: 0.03
- absolute FNR ceiling: 0.10
- one-sided safety alpha: 0.025
- FPR ceiling: 0.05
- one-sided FPR alpha: 0.05
- planning power target: 0.80

## Results

### Independent singleton units

The independent-unit design was calibrated successfully over the tested boundary grid.

For the full preservation claim, the first tested size whose Monte Carlo 95% lower power bound reached 0.80 was:

- 1,400 independent eligible primary units per validation cell.

For human-source cells this means roughly the same order of distinct independent authors producing eligible retained observations. The statistical design is workable, but the recruitment requirement is high.

Files:

- `results/external_validation_independent_block_feasibility_v1.json`
- `results/external_validation_independent_block_power_grid_v1.csv`

### Balanced repeated-author design

A repeated-author design was tested to reduce the number of independent contributors.

The initial power screen found a candidate at 150 authors with 40 retained rows per author, but the targeted calibration produced 36 clear type-I failures. The failures included human-vs-human non-inferiority, absolute FNR, and FPR tests.

This procedure is not suitable as the primary confirmatory analysis.

Files:

- `results/external_validation_balanced_human_block_feasibility_v1.json`
- `results/external_validation_balanced_human_block_screen_v1.csv`

### Empirical-Bernstein author-block design

A finite-sample bounded-mean analysis was then tested on independent author-block means.

Results:

- 0 clear type-I failures in the targeted calibration grid.
- The coarse screen first crossed 0.80 point power at 2,400 authors per cell with 40 retained rows per author.
- Targeted confirmation at 2,400 authors gave a lower Monte Carlo bound of about 0.793 for full preservation, so it did not meet the 0.80 planning target.
- At 2,800 authors, the nominal-dependence scenarios cleared the target with a lower bound of about 0.849.
- Under the higher-dependence sensitivity (`rho = 0.20`), the lower bound at 2,800 authors was about 0.760–0.763, still below target.

This route is well calibrated in the tested domain, but it requires thousands of independent authors and is still underpowered in the higher-dependence sensitivity.

Files:

- `results/external_validation_empirical_bernstein_block_feasibility_v1.json`
- `results/external_validation_empirical_bernstein_block_power_grid_v1.csv`

## Where this leaves the design

No primary design tested so far is both:

1. adequately calibrated, and
2. operationally feasible for the planned prospective human-source collection.

The clean independent-unit design is statistically viable but requires a large number of independent human contributors. Repeated observations per author can reduce that requirement only by introducing an analysis problem: the Welch/t version failed type-I calibration, while the assumption-light bounded-mean version pushes the independent-author requirement into the thousands.

The earlier CR2/WCR candidates remain rejected based on the existing calibration evidence.

## Decision needed before W0

The remaining choice changes the confirmatory design rather than the implementation.

The two defensible paths are:

- narrow the confirmatory claim or scope so that the required number of independent human units is feasible; or
- use a stronger model-based primary analysis, with the model assumptions and nuisance treatment explicitly justified and validated.

Until that is decided:

- W0 remains blocked;
- validation quotas and caps remain unfrozen;
- the human sampling mechanism remains unfrozen;
- the primary FNR/FPR inference procedure remains unfrozen.

## Reproduction

The exact run wrappers used for this checkpoint are archived in:

`review/20260920/scripts/`

The result files are committed under `results/`.

The checkpoint synthesis used to create this review package is:

- `docs/external_validation_confirmatory_redesign_checkpoint_20260919.md`
- `results/external_validation_confirmatory_redesign_checkpoint_v1.json`
