# Nested Value-Estimator Training Targets

These are development-only training targets for the value estimator.
For each value-estimator outer fold, the current outer evaluation
fold is excluded before downstream target generation.

- Rows: 13496
- Unique examples: 1687
- Outer folds: 5
- Inner folds: 4
- Global outer targets used for estimator training: no
- Final test used: no
- Held-out shift used: no

## Setup summary

```text
               setup_id  rows  unique_examples  mean_value  positive_value_n  zero_value_n  negative_value_n
     compact_after_rule  6748             1687    0.002371                69          6626                53
qwen_after_rule_compact  6748             1687    0.110996               997          5503               248
```

The artifact contains targets and audit metadata only. Predictor
features are merged later under each outer fold, with embedding PCA
fit only on that outer-training partition.

This artifact does not itself establish value predictability or pass
the professor's development milestone.
