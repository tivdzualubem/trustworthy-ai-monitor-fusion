# Serialized fusion models v3

These models were trained on `policy_train` from the final
author-reviewed cache `data/processed/monitor_score_cache_v3.parquet`.

## Complete serialized artifacts

- `artifacts/fusion_models_v3/cheap_router.joblib` uses ['rule_score', 'compact_unsafe_score']
- `artifacts/fusion_models_v3/full_information_fusion.joblib` uses ['rule_score', 'compact_unsafe_score', 'qwen_prompt_response_score']
- `artifacts/fusion_models_v3/fusion_bundle.joblib` contains both fitted pipelines, feature order, and provenance

Each pipeline contains fitted standardization, class-balanced logistic
regression, and five-fold sigmoid calibration. Reloaded probabilities matched
the in-memory fitted models exactly on every split.

No operating threshold was selected in this stage. The
`policy_selection` split remains reserved for the next professor-required
policy-selection step.
