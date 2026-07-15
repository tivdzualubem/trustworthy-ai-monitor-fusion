# Professor Confirmation Report

## Project

**Budget-Aware Runtime Safety Monitor Fusion**

Repository base commit before this final confirmation package: `4af2c2dd092dacb2e6a84eacbb755a5b24c0a7de`.

Prepared: `2026-07-15T09:52:22.601722+00:00`.

## Executive conclusion

The requested measurement, model-serialization, label-audit, robustness, risk-control, and timing requirements have been implemented.

**NO-GO for a routing-performance paper under the current audited-label evidence.**

The cost advantage is real in the measured setting, but the 5% false-positive risk constraint is not reliable under audited final and held-out shift evaluations. The project therefore pivots to a **measurement-validity paper**.

## Requirement completion

| requirement | status | evidence |
| --- | --- | --- |
| Official Qwen3Guard template and structured parsing | complete | Pinned official model revision and all three input modes preserved. |
| Controlled monitor and end-to-end timing | complete | Tesla T4, batch 1, excluded warm-up, CUDA synchronization, generation and parsing included. |
| Complete serialized learned router and fusion pipelines | complete | Scaler, calibrated estimator, feature order, and provenance serialized and reload-verified. |
| Regenerated score cache and raw outputs | complete | Rule and compact exact regeneration; Qwen raw generations and manifests preserved. |
| Author label audit | complete | 200 reviewed rows and 19 final corrections. |
| Prompt-only, response-only, prompt-response comparison | complete | Audited-label metrics regenerated in final tables. |
| Full-information versus selective fusion at common FPRs | complete | Targets 0.01, 0.025, 0.05, and 0.10 with paired bootstrap intervals. |
| All prespecified policy families | complete | Six families selected, frozen, and evaluated. |
| Replacement for influenced final split | complete | Nested leave-source-out and leave-family-out evaluation. |
| Formal risk control | complete_with_scope_limit | Per-fold fixed-sequence Learn-then-Test on untouched risk-control partitions; no shift-transfer claim. |
| Stop/go decision | complete_no_go | Cost advantage passes; risk gate fails; measurement-validity pivot confirmed. |
| Final v3 route timing and reproducibility audit | complete | Locked route timed on Tesla T4; two unique Qwen examples changed classification. |

## Data and label audit

- Total examples: `2159`.
- Audited-label negatives: `1756`.
- Audited-label positives: `403`.
- Manually reviewed audit rows: `200`.
- Final label corrections: `19`.
- A focused second review of 40 high-priority rows was also performed.

## Qwen input-mode measurement validity

The task label is response harmfulness. Prompt-only classification therefore mixes prompt harmfulness with the response-level target.

| mode | roc_auc | average_precision | recall | fpr | precision | safe_label_n | controversial_label_n | unsafe_label_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prompt_only | 0.830830 | 0.423630 | 0.890819 | 0.259681 | 0.440491 | 1224 | 120 | 815 |
| response_only | 0.905483 | 0.761312 | 0.687345 | 0.018793 | 0.893548 | 1695 | 154 | 310 |
| prompt_response | 0.925570 | 0.767658 | 0.813896 | 0.034738 | 0.843188 | 1694 | 76 | 389 |

Prompt-response is strongest overall, while response-only has a lower unsafe-threshold FPR. Prompt-only has substantially higher FPR, consistent with prompt contamination. This is evidence of measurement contamination, not causal proof.

## Six prespecified policy families

| family | policy_id | n | positive_n | negative_n | tp | fn | fp | tn | recall | fpr | precision | accuracy | intercept_rate | avg_cost_ms | median_cost_ms | p95_cost_ms | rule_call_rate | compact_call_rate | qwen_call_rate | policy_details_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cost_tuned_cascade | cascade_rl0_rh0.9219_cl0.005781_ch0.4674_q1 | 421 | 70 | 351 | 61 | 9 | 17 | 334 | 0.871429 | 0.048433 | 0.782051 | 0.938242 | 0.185273 | 1218.392660 | 1407.760426 | 1407.760426 | 1.000000 | 0.978622 | 0.862233 | {"compact_high_intercept_threshold": 0.46738710564095515, "compact_low_allow_threshold": 0.0057813970110146325, "qwen_threshold": 1.0, "route": "rule allow/intercept bands; compact allow/intercept bands; Qwen only for unresolved examples", "rule_high_intercept_threshold": 0.9218579568600582, "rule_low_allow_threshold": 0.0} |
| strongest_single_monitor | qwen_only_t_1 | 421 | 70 | 351 | 61 | 9 | 16 | 335 | 0.871429 | 0.045584 | 0.792208 | 0.940618 | 0.182898 | 1368.638805 | 1368.638805 | 1368.638805 | 0.000000 | 0.000000 | 1.000000 | {"monitor": "qwen3guard_gen_4b_prompt_response", "threshold": 1.0} |
| all_monitors_always_on | all_max_t_1 | 421 | 70 | 351 | 61 | 9 | 16 | 335 | 0.871429 | 0.045584 | 0.792208 | 0.940618 | 0.182898 | 1407.760426 | 1407.760426 | 1407.760426 | 1.000000 | 1.000000 | 1.000000 | {"aggregation": "max_score", "monitors": ["rule_filter_v1", "koala_text_moderation", "qwen3guard_gen_4b_prompt_response"], "threshold": 1.0} |
| learned_stacker_router | serialized_router_low0.1121_high0.3989_full0.8128 | 421 | 70 | 351 | 34 | 36 | 12 | 339 | 0.485714 | 0.034188 | 0.739130 | 0.885986 | 0.109264 | 601.531391 | 39.121621 | 1407.760426 | 1.000000 | 1.000000 | 0.410926 | {"cheap_features": ["rule_score", "compact_unsafe_score"], "full_features": ["rule_score", "compact_unsafe_score", "qwen_prompt_response_score"], "full_fusion_threshold": 0.8127985050108008, "route": "run rule and compact; allow/intercept when the serialized cheap router is confident; otherwise run Qwen and use the serialized full fusion model", "router_high_intercept_threshold": 0.3988659788416563, "router_low_allow_threshold": 0.11207411205638249, "serialized_bundle": "artifacts/fusion_models_v3/fusion_bundle.joblib"} |
| fixed_cascade | fixed_rule_gate_010_to_qwen_005 | 421 | 70 | 351 | 14 | 56 | 3 | 348 | 0.200000 | 0.008547 | 0.823529 | 0.859857 | 0.040380 | 114.817439 | 1.035116 | 1369.673920 | 1.000000 | 0.000000 | 0.083135 | {"qwen_threshold": 1.0, "route": "run rule; if rule score reaches the training 10% FPR gate, run Qwen; intercept at the training 5% FPR Qwen threshold", "rule_gate_threshold": 0.5934303402594009} |
| cheapest_monitor_only | rule_only_t_0.7498301 | 421 | 70 | 351 | 11 | 59 | 9 | 342 | 0.157143 | 0.025641 | 0.550000 | 0.838480 | 0.047506 | 1.035116 | 1.035116 | 1.035116 | 1.000000 | 0.000000 | 0.000000 | {"monitor": "rule_filter_v1", "threshold": 0.7498301045120662} |

All six required policy families were selected using only policy-training and policy-selection partitions, frozen, and evaluated without refitting.

## Full-information versus selective acquisition

At target FPR 0.05 on the previously used final-test partition:

- Full-information recall: `0.756757`.
- Full-information FPR: `0.028736`.
- Full-information one-sided 95% FPR upper bound: `0.048254`.
- Selective recall: `0.716216`.
- Selective FPR: `0.040230`.
- Selective one-sided 95% FPR upper bound: `0.062178`.
- Selective estimated cost reduction: `0.301800`.

The selective policy reduced estimated cost but had lower recall and higher FPR on calibration, final-test, and held-out-shift evaluations. The final selective upper bound exceeded 0.05.

| split | target_fpr | full_recall | full_fpr | full_fpr_one_sided95_upper | full_qwen_call_rate | full_avg_estimated_cost_ms | selective_recall | selective_fpr | selective_fpr_one_sided95_upper | selective_qwen_call_rate | selective_avg_estimated_cost_ms | selective_estimated_cost_reduction | recall_diff_selective_minus_full | recall_diff_ci95_low | recall_diff_ci95_high | fpr_diff_selective_minus_full | fpr_diff_ci95_low | fpr_diff_ci95_high | estimated_cost_reduction_ci95_low | estimated_cost_reduction_ci95_high | cost_advantage_observed | cost_advantage_ci_positive | selective_observed_fpr_within_target | selective_upper_bound_within_target | risk_gate_pass | split_stop_go_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 0.050000 | 0.723684 | 0.014451 | 0.030143 | 1.000000 | 1407.760426 | 0.710526 | 0.026012 | 0.044951 | 0.706161 | 1005.601156 | 0.285673 | -0.013158 | -0.065789 | 0.026316 | 0.011561 | 0.000000 | 0.026012 | 0.244204 | 0.331749 | True | True | True | True | True | True |
| final_test | 0.050000 | 0.756757 | 0.028736 | 0.048254 | 1.000000 | 1407.760426 | 0.716216 | 0.040230 | 0.062178 | 0.689573 | 982.898617 | 0.301800 | -0.040541 | -0.121622 | 0.040541 | 0.011494 | 0.002874 | 0.022989 | 0.258027 | 0.343268 | True | True | True | False | False | False |
| held_out_shift | 0.050000 | 0.921053 | 0.166667 | 0.438105 | 1.000000 | 1407.760426 | 0.894737 | 0.250000 | 0.527327 | 0.840000 | 1188.778217 | 0.155554 | -0.026316 | -0.078947 | 0.000000 | 0.083333 | 0.000000 | 0.250000 | 0.077777 | 0.252775 | True | True | False | False | False | False |

## Nested replacement for the influenced final split

The existing split assignments were ignored. Models and thresholds were relearned inside six outer folds: three leave-source-out folds and three leave-family-out folds. Every outer example was excluded from fitting and threshold selection in its fold.

### Pooled target-FPR 0.05 diagnostics

| scheme | target_fpr | policy | n | positive_n | negative_n | tp | fn | fp | tn | recall | fpr | precision | accuracy | intercept_rate | recall_ci95_low | recall_ci95_high | fpr_ci95_low | fpr_ci95_high | fpr_one_sided95_upper | qwen_call_rate | avg_estimated_cost_ms | estimated_cost_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_family_out | 0.050000 | full_information_always_on | 200 | 119 | 81 | 118 | 1 | 15 | 66 | 0.991597 | 0.185185 | 0.887218 | 0.920000 | 0.665000 | 0.954069 | 0.999787 | 0.107517 | 0.286976 | 0.270733 | 1.000000 | 1407.760426 | 0.000000 |
| leave_family_out | 0.050000 | selective_acquisition | 200 | 119 | 81 | 118 | 1 | 15 | 66 | 0.991597 | 0.185185 | 0.887218 | 0.920000 | 0.665000 | 0.954069 | 0.999787 | 0.107517 | 0.286976 | 0.270733 | 0.990000 | 1394.074037 | 0.009722 |
| leave_source_out | 0.050000 | full_information_always_on | 2159 | 403 | 1756 | 331 | 72 | 72 | 1684 | 0.821340 | 0.041002 | 0.821340 | 0.933302 | 0.186660 | 0.780380 | 0.857503 | 0.032218 | 0.051359 | 0.049662 | 1.000000 | 1407.760426 | 0.000000 |
| leave_source_out | 0.050000 | selective_acquisition | 2159 | 403 | 1756 | 336 | 67 | 88 | 1668 | 0.833747 | 0.050114 | 0.792453 | 0.928208 | 0.196387 | 0.793734 | 0.868772 | 0.040384 | 0.061380 | 0.059539 | 0.943492 | 1330.421873 | 0.054937 |

The pooled leave-source-out selective FPR was approximately 0.0501 with a one-sided upper bound approximately 0.0595. Leave-family-out FPR was much higher. These results support the measurement-validity pivot.

## Formal Learn-then-Test risk control

Each outer fold used 40% model training, 20% policy selection, and 40% untouched risk control. Candidates were ordered from conservative to aggressive and tested with an exact fixed-sequence binomial procedure for `H0: FPR >= 0.05`, at 95% confidence.

All `12` fold/method sequences returned a candidate whose risk-control one-sided upper bound was at most 0.05.

The renamed `selection_zero_intercept_fallback` means zero interceptions on the selection partition only. It does not promise zero interceptions on unseen risk-control or outer-fold examples.

### Certificate transfer diagnostic

| scheme | method | n | positive_n | negative_n | tp | fn | fp | tn | recall | fpr | precision | accuracy | intercept_rate | recall_ci95_low | recall_ci95_high | fpr_ci95_low | fpr_ci95_high | fpr_one_sided95_upper | qwen_call_rate | avg_estimated_cost_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_family_out | full_information_always_on | 200 | 119 | 81 | 99 | 20 | 10 | 71 | 0.831933 | 0.123457 | 0.908257 | 0.850000 | 0.545000 | 0.752437 | 0.894212 | 0.060820 | 0.215345 | 0.200409 | 1.000000 | 1407.760426 |
| leave_family_out | selective_acquisition | 200 | 119 | 81 | 99 | 20 | 12 | 69 | 0.831933 | 0.148148 | 0.891892 | 0.840000 | 0.555000 | 0.752437 | 0.894212 | 0.078962 | 0.244489 | 0.228955 | 0.880000 | 1243.523769 |
| leave_source_out | full_information_always_on | 2159 | 403 | 1756 | 241 | 162 | 32 | 1724 | 0.598015 | 0.018223 | 0.882784 | 0.910144 | 0.126447 | 0.548329 | 0.646250 | 0.012497 | 0.025629 | 0.024401 | 1.000000 | 1407.760426 |
| leave_source_out | selective_acquisition | 2159 | 403 | 1756 | 134 | 269 | 37 | 1719 | 0.332506 | 0.021071 | 0.783626 | 0.858268 | 0.079203 | 0.286648 | 0.380846 | 0.014878 | 0.028927 | 0.027627 | 0.182029 | 288.253187 |

The certificate scope is each untouched risk-control distribution. It does **not** certify excluded source or attack-family shift. Multiple outer groups exceeded 5% FPR, demonstrating failed transfer under shift.

## Final v3 end-to-end timing

The final locked v3 route was remeasured on a Tesla T4 using 128 deterministic audited-label calibration examples, batch size 1, eight excluded warm-up rows per policy, alternating policy order, CUDA synchronization, official Qwen generation and parsing, and no model refitting.

### Latency

| component | mode_or_policy | n | mean_ms | std_ms | min_ms | p50_ms | p95_ms | p99_ms | max_ms | expensive_call_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| end_to_end_policy | full_information_always_on | 128 | 1654.372208 | 556.587739 | 976.430359 | 1493.172941 | 2511.239174 | 3824.356324 | 4313.808649 | 1.000000 |
| end_to_end_policy | selective_acquisition | 128 | 1251.708934 | 936.998856 | 31.305919 | 1407.091752 | 2515.261509 | 3872.316007 | 4352.964455 | 0.695312 |
| rule_filter_v1 | prompt_response | 128 | 1.270986 | 0.820172 | 0.121209 | 1.109976 | 2.594630 | 4.016773 | 4.232479 |  |
| koala_text_moderation | prompt_response | 128 | 45.657291 | 18.415633 | 21.756934 | 41.631916 | 70.655028 | 76.610611 | 78.028378 |  |
| qwen3guard_official | prompt_response | 128 | 1597.560570 | 541.278173 | 942.742251 | 1441.374094 | 2431.596961 | 3744.088463 | 4234.158438 |  |

### Selective route distribution

| route | n | mean_latency_ms | p50_latency_ms | p95_latency_ms | rate |
| --- | --- | --- | --- | --- | --- |
| cheap_allow | 36 | 42.136954 | 34.226234 | 72.838538 | 0.281250 |
| cheap_intercept | 3 | 73.583905 | 73.136500 | 77.407005 | 0.023438 |
| expensive_fusion | 89 | 1780.686084 | 1663.122676 | 2685.695199 | 0.695312 |

Key timing results:

- Full mean latency: `1654.372208` ms.
- Selective mean latency: `1251.708934` ms.
- Mean latency reduction: `0.243393`.
- Selective Qwen call rate: `0.695312`.
- Full p95: `2511.239174` ms.
- Selective p95: `2515.261509` ms.
- Full p99: `3824.356324` ms.
- Selective p99: `3872.316007` ms.

Mean and median latency improved, but p95 and p99 did not improve. No tail-latency improvement is claimed.

## Regeneration reproducibility finding

- Rule maximum absolute difference: `0.0`.
- Compact maximum absolute difference: `3.748573362827301e-07`.
- Qwen maximum absolute difference: `0.5`.
- Qwen mismatch rows: `4`.
- Qwen mismatch unique examples: `2`.
- Exact regenerated-call match rate: `0.981567`.
- Qwen parse rate: `1.000000`.

The pinned Qwen model and official template produced parseable outputs, but two unique benchmark examples changed between cached and regenerated classifications. Raw outputs and the mismatch table are preserved. This is central measurement-reproducibility evidence.

## Claims supported

1. The final pipeline, labels, serialized models, routes, and timing procedure are documented and reproducible, subject to recorded Qwen classification variability.
2. Selective acquisition can reduce mean compute cost and mean latency in the measured setting.
3. A 5% FPR certificate on an in-distribution risk-control partition does not automatically transfer to excluded source or attack-family shifts.
4. Prompt-only guard evaluation can contaminate a response-harmfulness measurement.
5. The appropriate paper framing is measurement validity, not a successful routing-performance result.

## Claims not supported

- A universal or shift-robust 5% FPR guarantee.
- An untouched claim for the old final-test partition.
- Improved p95 or p99 latency.
- Exact deterministic Qwen classification parity across repeated runs.
- A superior selective-policy safety-performance frontier under all shifts.

## Primary evidence locations

- `reports/professor_confirmation/requirements_checklist.csv`
- `results/tables/final_*.csv`
- `reports/final_v3_policy_timing/`
- `artifacts/final_v3_policy_timing_results.zip`
- `reports/nested_leave_group_out_v1/`
- `reports/nested_ltt_risk_control_v1/`
- `reports/stop_go_v3/decision.md`
- `data/metadata/professor_confirmation_package_manifest.json`

## Confirmation requested

Please confirm that the implemented requirements and the measurement-validity paper pivot are acceptable before manuscript drafting continues.
