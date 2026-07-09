# Baseline policy selection

Generated at `2026-07-09T11:12:23.697130+00:00`.

This stage trains or configures baseline policies using `policy_train` and compares candidate operating points on `policy_selection`. It does **not** use `calibration`, `final_test`, or `held_out_shift`.

The selected table below chooses the highest-recall candidate within each required baseline family subject to selection FPR <= `0.05`; ties prefer lower average cost.

## Selected required baselines at FPR <= 0.05

| family                   | policy_id                                       | recall | fpr      | precision | avg_cost_ms | normalized_avg_cost_vs_rule |
| ------------------------ | ----------------------------------------------- | ------ | -------- | --------- | ----------- | --------------------------- |
| cost_tuned_cascade       | cascade_rl0_rh0.7498_cl0.005781_ch0.4674_j0.52  | 0.625  | 0.03725  | 0.7759    | 188.1       | 193.7                       |
| learned_stacker_router   | learned_router_low0.3668_high0.7676_final0.9112 | 0.6111 | 0.03152  | 0.8       | 195.7       | 201.5                       |
| strongest_single_monitor | judge_only_t_0.52004469                         | 0.5694 | 0.01146  | 0.9111    | 208         | 214.1                       |
| all_monitors_always_on   | all_max_t_0.73319402                            | 0.4167 | 0.03438  | 0.7143    | 213.7       | 220                         |
| cheapest_monitor_only    | rule_only_t_0.7498301                           | 0.1389 | 0.02865  | 0.5       | 1.285       | 1.323                       |
| fixed_cascade            | fixed_rule_gate_010_to_judge_005                | 0.125  | 0.002865 | 0.9       | 25.38       | 26.13                       |

## Pareto frontier preview

| family                   | policy_id                                            | recall  | fpr      | precision | avg_cost_ms | normalized_avg_cost_vs_rule |
| ------------------------ | ---------------------------------------------------- | ------- | -------- | --------- | ----------- | --------------------------- |
| cheapest_monitor_only    | rule_only_t_0.99991411                               | 0       | 0        |           | 1.285       | 1.323                       |
| strongest_single_monitor | judge_only_t_0.87518229                              | 0.09722 | 0        | 1         | 208         | 214.1                       |
| strongest_single_monitor | judge_only_t_0.89531841                              | 0.09722 | 0        | 1         | 208         | 214.1                       |
| cheapest_monitor_only    | rule_only_t_0.98035428                               | 0.01389 | 0.002865 | 0.5       | 1.285       | 1.323                       |
| fixed_cascade            | fixed_rule_gate_010_to_judge_005                     | 0.125   | 0.002865 | 0.9       | 25.38       | 26.13                       |
| learned_stacker_router   | learned_router_low0.3723_high0.8336_final0.9876      | 0.25    | 0.002865 | 0.9474    | 120.8       | 124.4                       |
| strongest_single_monitor | judge_only_t_0.73646367                              | 0.3056  | 0.002865 | 0.9565    | 208         | 214.1                       |
| learned_stacker_router   | learned_router_low0.3682_high0.8336_final0.9876      | 0.2778  | 0.005731 | 0.9091    | 150.2       | 154.6                       |
| learned_stacker_router   | learned_router_low0.3672_high0.8336_final0.9876      | 0.3056  | 0.005731 | 0.9167    | 189.6       | 195.3                       |
| cheapest_monitor_only    | rule_only_t_0.92185796                               | 0.08333 | 0.008596 | 0.6667    | 1.285       | 1.323                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.009102_ch0.4674_j0.52  | 0.1806  | 0.008596 | 0.8125    | 21.82       | 22.46                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.009102_ch0.6219_j0.52  | 0.1806  | 0.008596 | 0.8125    | 21.82       | 22.46                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.007173_ch0.4674_j0.52  | 0.2639  | 0.008596 | 0.8636    | 35.91       | 36.97                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.007173_ch0.6219_j0.52  | 0.2639  | 0.008596 | 0.8636    | 35.91       | 36.97                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.006242_ch0.4674_j0.52  | 0.2778  | 0.008596 | 0.8696    | 42.21       | 43.46                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.006242_ch0.6219_j0.52  | 0.2778  | 0.008596 | 0.8696    | 42.21       | 43.46                       |
| learned_stacker_router   | learned_router_low0.3723_high0.8336_final0.9112      | 0.4028  | 0.008596 | 0.9062    | 120.8       | 124.4                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.009102_ch0.01579_j0.52 | 0.1944  | 0.01146  | 0.7778    | 14.34       | 14.77                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.007173_ch0.01579_j0.52 | 0.2778  | 0.01146  | 0.8333    | 28.43       | 29.27                       |
| cost_tuned_cascade       | cascade_rl0.3148_rh0.9219_cl0.006242_ch0.01579_j0.52 | 0.2917  | 0.01146  | 0.84      | 34.73       | 35.76                       |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.009102_ch0.4674_j0.52       | 0.3611  | 0.01146  | 0.8667    | 84.69       | 87.21                       |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.009102_ch0.6219_j0.52       | 0.3611  | 0.01146  | 0.8667    | 84.69       | 87.21                       |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.007173_ch0.4674_j0.52       | 0.5139  | 0.01146  | 0.9024    | 136         | 140.1                       |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.007173_ch0.6219_j0.52       | 0.5139  | 0.01146  | 0.9024    | 136         | 140.1                       |
| learned_stacker_router   | learned_router_low0.3672_high0.8336_final0.9112      | 0.5417  | 0.01146  | 0.907     | 189.6       | 195.3                       |
| learned_stacker_router   | learned_router_low0.3669_high0.8336_final0.9112      | 0.5556  | 0.01146  | 0.9091    | 203.7       | 209.8                       |
| strongest_single_monitor | judge_only_t_0.52004469                              | 0.5694  | 0.01146  | 0.9111    | 208         | 214.1                       |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.006242_ch0.4674_j0.52       | 0.5694  | 0.01433  | 0.8913    | 183.5       | 189                         |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.006242_ch0.6219_j0.52       | 0.5694  | 0.01433  | 0.8913    | 183.5       | 189                         |
| cost_tuned_cascade       | cascade_rl0_rh0.9219_cl0.005781_ch0.4674_j0.52       | 0.6111  | 0.01719  | 0.88      | 196.4       | 202.2                       |

## Required baseline coverage

- `cheapest_monitor_only`: rule filter only.
- `strongest_single_monitor`: Qwen3Guard judge only.
- `all_monitors_always_on`: rule + compact + judge always run, max-score decision.
- `fixed_cascade`: fixed rule gate into judge.
- `cost_tuned_cascade`: grid-searched rule/compact/judge cascade.
- `learned_stacker_router`: logistic cheap router plus full logistic stacker when routed to judge.

## Files

- `all_candidate_policies.csv`
- `selected_baselines_fpr05.csv`
- `selection_pareto_frontier.csv`
- `learned_router_coefficients.json`
- `policy_selection_manifest.json`
