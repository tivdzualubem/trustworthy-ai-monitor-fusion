# Policy evaluation v3 methodology

This evaluation uses the final author-reviewed labels and the complete
serialized v3 fusion pipelines.

## Data-use boundaries

- `policy_train`: model fitting and train-derived candidate grids.
- `policy_selection`: threshold and policy selection.
- `calibration`, `final_test`, `held_out_shift`: evaluation only.
- No model is refit during policy selection or evaluation.

## Prespecified policy families

1. Cheapest monitor only.
2. Strongest single monitor.
3. All monitors always on with max-score aggregation.
4. Fixed rule-to-Qwen cascade.
5. Cost-tuned rule/compact/Qwen cascade.
6. Serialized learned cheap-router/full-fusion policy.

## Full-information comparison

Full-information always-on fusion and selective expensive-monitor acquisition
are compared at common selection target FPRs of 1%, 2.5%, 5%, and 10%.
Paired stratified bootstrap confidence intervals use 2,000 replicates.

## Cost

Estimated policy cost uses component p50 values from the controlled,
CUDA-synchronized T4 benchmark. These are not replacements for the separate
measured end-to-end policy timing benchmark.

## Statistical scope

The exact binomial intervals and paired bootstrap intervals are descriptive
evaluation intervals. They are not a formal Neyman-Pearson or
Learn-then-Test risk certificate. Formal risk control is a later,
separate professor requirement.
