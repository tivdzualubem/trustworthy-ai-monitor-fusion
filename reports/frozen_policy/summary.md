# Frozen policies

Frozen at `2026-07-09T11:17:36.185365+00:00` from commit `ebc1a95c0873b952c848a4a4d9235e9de56106c2`.

Policy selection used only `policy_train` and `policy_selection`. The `calibration`, `final_test`, and `held_out_shift` splits remain unused for policy selection.

## Primary frozen budget-aware policy

- Family: `cost_tuned_cascade`
- Policy ID: `cascade_rl0_rh0.7498_cl0.005781_ch0.4674_j0.52`
- Freeze rule: Select the highest-recall candidate on policy_selection among required baseline families subject to policy_selection FPR <= 0.05; ties prefer lower average cost, then lower FPR.

## Frozen selected policies

| family | policy_id | role | selection_recall | selection_fpr | selection_avg_cost_ms |
| --- | --- | --- | ---: | ---: | ---: |
| cost_tuned_cascade | cascade_rl0_rh0.7498_cl0.005781_ch0.4674_j0.52 | primary_budget_aware_policy | 0.625 | 0.0372493 | 188.1 |
| learned_stacker_router | learned_router_low0.3668_high0.7676_final0.9112 | required_baseline | 0.611111 | 0.0315186 | 195.717 |
| strongest_single_monitor | judge_only_t_0.52004469 | required_baseline | 0.569444 | 0.0114613 | 207.954 |
| all_monitors_always_on | all_max_t_0.73319402 | required_baseline | 0.416667 | 0.034384 | 213.661 |
| cheapest_monitor_only | rule_only_t_0.7498301 | required_baseline | 0.138889 | 0.0286533 | 1.28534 |
| fixed_cascade | fixed_rule_gate_010_to_judge_005 | required_baseline | 0.125 | 0.00286533 | 25.3804 |

## Next stage

Use this frozen artifact for calibration, final-test, and held-out-shift evaluation. Any finite-sample bounds are valid only after this freeze step and only under the stated in-distribution exchangeability assumptions.
