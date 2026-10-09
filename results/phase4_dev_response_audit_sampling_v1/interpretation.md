# Development-only sampling pilot (exploratory)

No deployed native-guard result, causal inference, FNR guarantee, or new safety certificate.
All estimators use historical response-level development labels. No sealed data were accessed.
The decision column is archived cross-fitted `base_prediction`, not a single fixed deployed guard.
Uniform, proportional stratification, score-tilted stratification, and allow-only sampling use only
cost-free pre-review fields (D, base_uncertainty) for audit selection. Optional scores are unused.

## Results

### compact_after_rule

| Budget | Method | Miss RMSE | Miss bias | 95% CI coverage | FNR RMSE |
|---:|---|---:|---:|---:|---:|
| 40 | allowed_only | 0.0506 | +0.0014 | 0.960 | n/a |
| 40 | proportional_stratified | 0.0556 | -0.0052 | 0.895 | 0.1425 |
| 40 | score_tilted_stratified | 0.0545 | +0.0052 | 0.900 | 0.1224 |
| 40 | uniform | 0.0584 | +0.0040 | 0.885 | 0.1646 |
| 80 | allowed_only | 0.0357 | +0.0002 | 0.945 | n/a |
| 80 | proportional_stratified | 0.0376 | -0.0005 | 0.925 | 0.0964 |
| 80 | score_tilted_stratified | 0.0351 | +0.0002 | 0.940 | 0.1041 |
| 80 | uniform | 0.0401 | -0.0018 | 0.910 | 0.1093 |
| 160 | allowed_only | 0.0220 | -0.0010 | 0.970 | n/a |
| 160 | proportional_stratified | 0.0238 | -0.0024 | 0.950 | 0.0584 |
| 160 | score_tilted_stratified | 0.0235 | -0.0013 | 0.945 | 0.1030 |
| 160 | uniform | 0.0251 | +0.0023 | 0.975 | 0.0721 |

Development finite-pool miss prevalence: 0.13574

### qwen_after_rule_compact

| Budget | Method | Miss RMSE | Miss bias | 95% CI coverage | FNR RMSE |
|---:|---|---:|---:|---:|---:|
| 40 | allowed_only | 0.0463 | -0.0001 | 0.970 | n/a |
| 40 | proportional_stratified | 0.0570 | +0.0024 | 0.890 | 0.1279 |
| 40 | score_tilted_stratified | 0.0542 | +0.0045 | 0.930 | 0.1236 |
| 40 | uniform | 0.0590 | +0.0055 | 0.880 | 0.1689 |
| 80 | allowed_only | 0.0400 | +0.0004 | 0.935 | n/a |
| 80 | proportional_stratified | 0.0385 | +0.0001 | 0.915 | 0.0856 |
| 80 | score_tilted_stratified | 0.0349 | -0.0024 | 0.910 | 0.1118 |
| 80 | uniform | 0.0399 | -0.0017 | 0.925 | 0.1102 |
| 160 | allowed_only | 0.0255 | -0.0003 | 0.910 | n/a |
| 160 | proportional_stratified | 0.0254 | -0.0008 | 0.935 | 0.0646 |
| 160 | score_tilted_stratified | 0.0246 | -0.0014 | 0.930 | 0.1005 |
| 160 | uniform | 0.0247 | +0.0017 | 0.965 | 0.0733 |

Development finite-pool miss prevalence: 0.13515

## Interpretation warnings

- The allow-only strategy identifies miss prevalence, **not FNR** because its audit never labels blocked outcomes.
- The normal confidence intervals are an exploratory design-based approximation, not a validated rare-event bound.
- Historical development labels are already known to the researcher and may include label errors or source effects.
- Scores and cross-fitted decisions derive from historically fitted models; this is no clean prospective deployment test.
- No nested strategy tuning on full labels and no published active-testing implementation is claimed.
- Next: verify native prompt-response guard D, incorporate optional-monitor measurement cost, and then evaluate published active designs.
