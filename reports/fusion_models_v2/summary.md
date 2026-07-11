# Serialized fusion models v2

Two complete fitted pipelines were trained on `policy_train` using the repaired
official-Qwen score cache.

## Artifacts

- `artifacts/fusion_models_v2/cheap_router.joblib`: calibrated cheap router using ['rule_score', 'compact_unsafe_score']
- `artifacts/fusion_models_v2/full_information_fusion.joblib`: calibrated full-information fusion model using ['rule_score', 'compact_unsafe_score', 'qwen_prompt_response_score']
- `artifacts/fusion_models_v2/fusion_bundle.joblib`: complete serialized bundle used by downstream evaluation

The bundle includes fitted standardization, logistic estimators, calibration
objects, feature order, and provenance. Reloaded probabilities exactly match
the in-memory fitted models on both `policy_train` and `policy_selection`.

No operating threshold was selected in this step. The current artifact is
marked provisional because the assistant-assisted label audit still requires
author review before corrected labels are locked.
