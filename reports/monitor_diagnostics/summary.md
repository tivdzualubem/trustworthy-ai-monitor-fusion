# Pre-policy monitor diagnostics

Generated at `2026-07-09T10:54:10.616864+00:00`.

These diagnostics use labels only from `policy_train`. The `calibration`, `final_test`, and `held_out_shift` splits are not used here for policy selection.

## Split label counts

| split            | y | n   |
| ---------------- | - | --- |
| calibration      | 0 | 350 |
| calibration      | 1 | 72  |
| final_test       | 0 | 350 |
| final_test       | 1 | 72  |
| held_out_shift   | 0 | 16  |
| held_out_shift   | 1 | 34  |
| policy_selection | 0 | 349 |
| policy_selection | 1 | 72  |
| policy_train     | 0 | 700 |
| policy_train     | 1 | 144 |

## Per-monitor ROC/PR summary on `policy_train`

| monitor_id            | n   | positives | negatives | roc_auc | average_precision |
| --------------------- | --- | --------- | --------- | ------- | ----------------- |
| rule_filter_v1        | 844 | 144       | 700       | 0.6479  | 0.3155            |
| koala_text_moderation | 844 | 144       | 700       | 0.6487  | 0.2744            |
| qwen3guard_gen_4b     | 844 | 144       | 700       | 0.8951  | 0.7563            |

## Latency/cost summary on `policy_train`

| monitor_id            | median_latency_ms | mean_latency_ms | p95_latency_ms | normalized_median_cost_vs_rule |
| --------------------- | ----------------- | --------------- | -------------- | ------------------------------ |
| rule_filter_v1        | 0.9712            | 1.226           | 2.89           | 1                              |
| koala_text_moderation | 3.397             | 4.209           | 6.177          | 3.497                          |
| qwen3guard_gen_4b     | 168.9             | 214.3           | 472.3          | 173.9                          |

## Recall at fixed false-positive rates on `policy_train`

| monitor_id            | max_fpr | selected_threshold | observed_recall | observed_fpr | observed_precision |
| --------------------- | ------- | ------------------ | --------------- | ------------ | ------------------ |
| rule_filter_v1        | 0.01    | 0.9336             | 0.07639         | 0.008571     | 0.6471             |
| rule_filter_v1        | 0.05    | 0.6629             | 0.2153          | 0.05         | 0.4697             |
| rule_filter_v1        | 0.1     | 0.5157             | 0.2778          | 0.1          | 0.3636             |
| koala_text_moderation | 0.01    | 0.5655             | 0.03472         | 0.01         | 0.4167             |
| koala_text_moderation | 0.05    | 0.04185            | 0.09028         | 0.04571      | 0.2889             |
| koala_text_moderation | 0.1     | 0.01409            | 0.2014          | 0.1          | 0.2929             |
| qwen3guard_gen_4b     | 0.01    | 0.6057             | 0.3819          | 0.01         | 0.8871             |
| qwen3guard_gen_4b     | 0.05    | 0.3748             | 0.6875          | 0.04714      | 0.75               |
| qwen3guard_gen_4b     | 0.1     | 0.2716             | 0.7847          | 0.09143      | 0.6384             |

## Pairwise harmful co-miss rates

Thresholds use the `policy_train` operating point with maximum FPR `0.05`.

| left_monitor          | right_monitor         | harmful_examples | both_miss_count | both_miss_rate_among_harmful |
| --------------------- | --------------------- | ---------------- | --------------- | ---------------------------- |
| rule_filter_v1        | koala_text_moderation | 144              | 100             | 0.6944                       |
| rule_filter_v1        | qwen3guard_gen_4b     | 144              | 37              | 0.2569                       |
| koala_text_moderation | qwen3guard_gen_4b     | 144              | 41              | 0.2847                       |
| ALL_THREE             | ALL_THREE             | 144              | 33              | 0.2292                       |

## Files

- `per_monitor_roc_pr_metrics.csv`
- `recall_at_fixed_fpr.csv`
- `monitor_latency_costs.csv`
- `pairwise_harmful_comiss_rates.csv`
- `roc_curve_*_policy_train.csv`
- `pr_curve_*_policy_train.csv`
- `thresholds_fixed_fpr_05.json`
- `monitor_diagnostics_manifest.json`
