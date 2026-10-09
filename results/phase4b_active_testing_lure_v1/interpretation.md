# Phase 4B: Active Testing LURE benchmark (development-only)

Source: Kossen et al., ICML 2021, Eq. 4, with authors' FancyUnbiasedRiskEstimator.
Paper: https://proceedings.mlr.press/v139/kossen21a.html
Authors' estimator: https://github.com/jlko/active-testing/blob/main/activetesting/risk_estimators.py

**Scope:** historical development Y and archived cross-fitted classifier D; NOT native-guard deployment.
Positive-probability sequential active acquisition plus LURE unbiased *linear* mean estimates.
FNR/FPR ratios are not unbiased; intervals not reported because they need separate validation.
Risk-proxy strategies are project-specific adaptations, not exact authors' experimental surrogates.
All proposals use pre-review cheap scores and/or labels already purchased during that repetition.

## Results and matched previous baselines

### compact_after_rule

| Budget | Method | Miss RMSE | FNR RMSE |
|---:|---|---:|---:|
| 40 | allowed_only | 0.0506 | n/a |
| 40 | lure_adaptive_beta | 0.0557 | 0.1961 |
| 40 | lure_fixed_cheap_rank | 0.0570 | 0.2114 |
| 40 | proportional_stratified | 0.0556 | 0.1425 |
| 40 | score_tilted_stratified | 0.0545 | 0.1224 |
| 40 | uniform | 0.0584 | 0.1646 |
| 80 | allowed_only | 0.0357 | n/a |
| 80 | lure_adaptive_beta | 0.0371 | 0.1371 |
| 80 | lure_fixed_cheap_rank | 0.0404 | 0.1567 |
| 80 | proportional_stratified | 0.0376 | 0.0964 |
| 80 | score_tilted_stratified | 0.0351 | 0.1041 |
| 80 | uniform | 0.0401 | 0.1093 |
| 160 | allowed_only | 0.0220 | n/a |
| 160 | lure_adaptive_beta | 0.0254 | 0.0972 |
| 160 | lure_fixed_cheap_rank | 0.0264 | 0.1074 |
| 160 | proportional_stratified | 0.0238 | 0.0584 |
| 160 | score_tilted_stratified | 0.0235 | 0.1030 |
| 160 | uniform | 0.0251 | 0.0721 |

### qwen_after_rule_compact

| Budget | Method | Miss RMSE | FNR RMSE |
|---:|---|---:|---:|
| 40 | allowed_only | 0.0463 | n/a |
| 40 | lure_adaptive_beta | 0.0538 | 0.2074 |
| 40 | lure_fixed_cheap_rank | 0.0552 | 0.2158 |
| 40 | proportional_stratified | 0.0570 | 0.1279 |
| 40 | score_tilted_stratified | 0.0542 | 0.1236 |
| 40 | uniform | 0.0590 | 0.1689 |
| 80 | allowed_only | 0.0400 | n/a |
| 80 | lure_adaptive_beta | 0.0362 | 0.1494 |
| 80 | lure_fixed_cheap_rank | 0.0369 | 0.1437 |
| 80 | proportional_stratified | 0.0385 | 0.0856 |
| 80 | score_tilted_stratified | 0.0349 | 0.1118 |
| 80 | uniform | 0.0399 | 0.1102 |
| 160 | allowed_only | 0.0255 | n/a |
| 160 | lure_adaptive_beta | 0.0253 | 0.0971 |
| 160 | lure_fixed_cheap_rank | 0.0236 | 0.1090 |
| 160 | proportional_stratified | 0.0254 | 0.0646 |
| 160 | score_tilted_stratified | 0.0246 | 0.1005 |
| 160 | uniform | 0.0247 | 0.0733 |

## Interpretation safeguards

- LURE is unbiased for finite-pool *means* of per-example losses under correct recorded nonzero q. Empirical Monte Carlo bias can be nonzero.
- FNR uses estimated numerator / estimated harm prevalence; biased in finite samples, and may be undefined when estimated harm=0.
- Existing Phase4A sampling controls use a different variance structure; comparison is same development pool, budgets and 200 repeats, not a randomized clinical superiority test.
- `base_score` ranked for proposal construction, never assumed a calibrated probability.
- Beta(2,8) updated exclusively from sampled audited labels. Fixed rank proposal uses no audited labels.
- Sampling spends no optional Qwen score. This is NOT yet a priced extra-monitor selection study.
- The original source labels and archived decisions can be imperfect; this is exploratory only.
- Previous pilot calculated approximate CI coverage; Phase4B does not assert any confidence bound or matched CIs.
- Original Stage-A 0/4 readout-transfer screening and all existing files remain untouched.

## Stop/go

Compare estimation error, variance and feasibility by method, setup and budget. If no consistent reduction over uniform or stratified auditing, report a negative result and prioritize valid labels/decision provenance before new methods.
