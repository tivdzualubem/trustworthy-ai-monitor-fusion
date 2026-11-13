# Phase 10 — Completed evidence reconciliation / work-completion gate

> INTERNAL WORKING DOCUMENT — not a research lead report, paper claim, safety certificate, or sign-off.

## Primary estimand and frame

The primary estimand is **A=P(Y=1,D=0)** where Y is actual policy-defined assistant-response harmfulness and D=0 is allowed by a frozen post-generation, pre-delivery monitor. The historical 1,687-case development D is a cross-fitted classifier decision, not live deployed guard behavior. FNR is secondary.

## 1. Previously completed historical baseline comparisons

| 160-label method (Qwen setup) | A RMSE | Labels spent tuning |
|---|---:|---:|
| `allowed_only_srs` | 0.02369 | 0 |
| `stratppi_proportional_fixed1` | 0.02542 | 0 |
| `stratppi_proportional_pilot_lambda` | 0.02936 | 32 |
| `stratppi_allowed70_pilot_lambda` | 0.03552 | 32 |

**Result:** Allowed-only random sampling is better than these limited StratPPI mean estimators on the archived historical-label experiment. This is **not** a claim against full StratPPI. The old labels and classifier are not externally validated.

Archived Qwen improvement (one historical setup): **1 fewer allowed harmful response(s) out of 1,687**, equivalent to 0.0593 percentage points in A; one more blocked harmful response. Not a new monitor deployment result.

### Earlier restricted AM-PPI cost-validity audit
| Method | A RMSE | Actual simulation over-budget frequency |
|---|---:|---:|
| `blocked_allowed_srs` | 0.02819 | 0% |
| `cheap_only_active_ppi` | 0.03255 | 41% |
| `restricted_amppi_2predictor` | 0.05095 | 49% |

**Not a paper-faithful AM-PPI comparison:** Costs used hypothetical 30s reviews and historical proxy means; expected-budget designs can exceed a realized hard budget. These results are negative exploratory evidence, not the final published-method comparison.

## 2. Independent-population synthetic robustness evidence

16 synthetic populations ×30 repetitions/geometry/scenario, human active-time budget 4,800 seconds, proxy acquisition cost set to zero (limitation). RMSE is against synthetic Y, not real Y.

| Geometry | Scenario | SRS RMSE | Proxy-stratified RMSE | 3-review majority RMSE | Majority–SRS 95% paired population-bootstrap CI |
|---|---|---:|---:|---:|---|
| development_footprint_1687 | perfect_binary | 0.01230 | 0.00676 | 0.02289 | [+0.00809, +0.01379] |
| development_footprint_1687 | noisy_independent | 0.03645 | 0.03685 | 0.02259 | [-0.01646, -0.01123] |
| development_footprint_1687 | noisy_correlated | 0.07088 | 0.07654 | 0.05074 | [-0.02297, -0.01688] |
| development_footprint_1687 | correlated_with_abstention | 0.09370 | 0.09418 | 0.05997 | [-0.03808, -0.02928] |
| phase6_5000 | perfect_binary | 0.01146 | 0.00684 | 0.01973 | [+0.00685, +0.00974] |
| phase6_5000 | noisy_independent | 0.03057 | 0.03340 | 0.02137 | [-0.01123, -0.00708] |
| phase6_5000 | noisy_correlated | 0.06091 | 0.06339 | 0.04436 | [-0.01860, -0.01416] |
| phase6_5000 | correlated_with_abstention | 0.08067 | 0.08100 | 0.04937 | [-0.03533, -0.02734] |

**Conditional result:** better proxy stratification with perfect truth; majority voting smaller RMSE with specified noisy raters. CIs for simulated reviewer outcomes are not valid intervals for latent Y unless the reviewer-error model is justified. Real reviewer errors are uncalibrated.

## 3. Completed real blinded review

96 historical cases, 288 primary assessments, 48 selected fourth ratings; 15844.9 active seconds and 23638.0 wall seconds. The fourth reviewer contributed 4054.5 active seconds (25.6%). These are observed browser-active timings, **not** total economic costs.

| Reviewer-defined binary Harmful endpoint | Weighted A | Approx. SE | Approx. 95% normal CI |
|---|---:|---:|---|
| R1 | 0.0554 | 0.0234 | [0.0096, 0.1012] |
| R2 | 0.3116 | 0.0473 | [0.2188, 0.4044] |
| R3 | 0.1222 | 0.0341 | [0.0554, 0.1889] |

These CIs assume the six development source×decision samples were SRS without replacement within strata and approximate normality; several strata are tiny. They target each **reviewer’s judgment**, not true harmfulness. Cross-reviewer comparisons are paired on the same cases; do not use independent-difference standard errors. Full latent-Y inference remains impossible without extra error assumptions or validated reference labels.

## 4. Mathematical limits and prior-method overlap

Without any restriction on P(R1,R2,R3,R4|Y,D), Y is not identified by review votes: any observed voting distribution can coexist with multiple latent-Y prevalences. This is a classical, **not novel**, nonidentification argument. Three conditionally independent binary reviewers may generically identify a two-class mixture up to label switching, but conditional independence is not validated by the current 96-case pilot; the three-reviewer 2×2×2 binary model is just-identified. The selected R4 vote is an additional opinion, not gold truth.

MultiPPI/AM-PPI optimize predictor/label acquisition under verified-outcome assumptions. StratPPI handles stratified PPI. This project must not rename their allocation formulas, inverse probability weights, difference estimates, or elementary bounded-label inequalities as new mathematics.

## 5. Research lead prior findings completion gate

| Requirement | Evidence state | Remaining definitive action |
|---|---|---|
| Primary estimand / monitor timing | Defined | Freeze native prospective D before making deployment claim |
| Explain Qwen allowed/blocked contrast | Historical decomposition complete | Verify on truly native frozen monitor / fresh data if claimed generally |
| Strong simple sampling | Historical Phase5B1, B2 completed | Confirm on credible independent Y, under equal full costs |
| Published StratPPI / MultiPPI / AM-PPI | Mean specialization / restricted AM only | Reproduce relevant **published** estimator, tuning, CIs, costs under same assumptions and data; no fidelity claim otherwise |
| Mathematical impossibility/necessary conditions | Classical no-identification regime and conditional-bounds mapped | Derive/verify any additional claimed theorem against prior literature; if none, choose empirical contribution |
| Real independent label collection and uncertainty | 96×3 + selected 48 completed | Qualified independent reference procedure or predeclared sensitivity bounds for latent Y; **do not** re-open ad hoc R2 QC |
| Cost: calibration, humans, adjudication, inference | Active timings measured, historic inference available | Per-case native inference and calibration/adjudication overhead; distinguish acquisition from wall-clock/cost |
| Confidence coverage / distribution shift | Synthetic robustness with 16 independent generated populations | Reference-truth coverage / fresh native population remains unavailable; cannot certify |
| Reproducible freeze | Hashes produced below | Freeze WSL execution code, native data source, predictions, groups, labels, timing logs; run tests and do privacy audit |
| GitHub and research review report | Held intentionally | Publish only after method/cost/reliability requirements are addressed or scientifically justified as scope limits |

### Decision rule

**Do not prolong the 12-case R2 diagnosis.** The next high-value step is one carefully frozen, genuinely independent native-monitor/reference-label study with measured per-component costs and a fair published-method benchmark. If qualifying new labels are infeasible, frame only an honest development-stage methodology result; do not claim full completion, true latent risk, or paper-ready novelty.

## 6. Provenance / SHA256

- `phase8b_population_robustness_20261112.json`: `981bdc062e518f4ee5efae6615d217cb2c4249c5dc844b8d3d6faf5120dd93fc`
- `phase9b_audit_20261112.json`: `0baf4ca75c22c1d3175849aa68a8e7eb449f26114eb05d961f82600821322d9c`
- `phase5b1_blocked_allowed_baselines_v1.zip`: `b3571b1d9e61ad65cb3bf4daa9ba4845206760f06e74facc1409d75cdcec0132`
- `phase5b2_stratppi_mean_development_v1.zip`: `ac57a045b12acb85a09860c9da495b12fc45a3ee4a18fed344003d05d5a9d358`
- `phase5b3_amppi_restricted_v1(1).zip`: `331441e6422ff80046ec69870f75bbefd396d952a42be4225ab97efee2183711`

All inputs are aggregated or archived historical developmental summaries. No private case records were read.
