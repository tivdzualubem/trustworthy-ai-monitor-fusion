# Results Summary

This repository contains the implementation and result artifacts for the pre-final stage of the Budget-Aware Runtime Safety Monitor Fusion project.

## Main result

The selected primary frozen policy is a cost-tuned cascade. On `final_test`, it achieved:

- Recall: `0.5833`
- False-positive rate: `0.0571`
- Average cost: `186.39 ms`

The held-out GCG split is reported as an attack-family shift diagnostic. Its false-positive rate rises to `0.5625`, so calibration behavior is reported separately from shift behavior.

## Curated outputs

Figures:

- `results/figures/monitor_strength_vs_cost.png`
- `results/figures/policy_recall_cost.png`
- `results/figures/primary_policy_shift_diagnostic.png`

Tables:

- `results/tables/monitor_diagnostics.csv`
- `results/tables/selected_policy_results.csv`
- `results/tables/primary_policy_evaluation.csv`
- `results/tables/calibration_bounds.csv`
- `results/tables/all_policy_metrics_by_split.csv`

Detailed reports:

- `reports/monitor_diagnostics/summary.md`
- `reports/policy_selection/summary.md`
- `reports/frozen_policy/summary.md`
- `reports/final_evaluation/summary.md`
