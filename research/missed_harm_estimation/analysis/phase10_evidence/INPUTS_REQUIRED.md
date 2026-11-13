# Input requirements for historical reconciliation checks

The eight legacy reconciliation tests use five archived source files placed in a local `inputs/` directory. These are deliberately not bundled into the public code archive:

- `phase8b_population_robustness_20261112.json` (synthetic population aggregation)
- `phase9b_audit_20261112.json` (detailed reviewer aggregation; retain locally)
- `phase5b1_blocked_allowed_baselines_v1.zip`
- `phase5b2_stratppi_mean_development_v1.zip`
- `phase5b3_amppi_restricted_v1(1).zip`

To rerun locally, place the inputs in `analysis/phase10_evidence/inputs/`, then run `python -m unittest discover -p 'test_*.py'` in that folder. The published archive includes its aggregate result JSON but not these detailed input packages. The standard `verify_local.sh` does not invoke this input-dependent suite.
