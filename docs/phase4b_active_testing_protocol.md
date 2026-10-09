# Phase 4B — Published Active Testing estimator benchmark (development only)

**Research goal:** Follow Prof. Bader Rasheed's priority to measure harmful *assistant responses* missed by a monitored decision system at lower human-review cost, by comparing established audit designs **before** inventing new methods.

## Publication/source basis

Kossen, Farquhar, Gal and Rainforth (ICML 2021), *Active Testing: Sample-Efficient Model Evaluation*: https://proceedings.mlr.press/v139/kossen21a.html . Authors' implementation of the LURE estimator: https://github.com/jlko/active-testing/blob/main/activetesting/risk_estimators.py (`FancyUnbiasedRiskEstimator`).

The new script implements the paper's sequential acquisition and **LURE** estimator, not its authors' original neural surrogate and full experiments. Specifically, on finite pool of size N, after m=1,...,M distinct acquisitions, save the *actual conditional* selection probability q_m>0 and compute

    v_m = 1 + (N-M)/(N-m) * [1/((N-m+1)*q_m) - 1]
    R_LURE = (1/M) * sum_m v_m * loss_m

For M=N, use weights v_m=1. Mathematical self-tests check uniform weights, complete-census consistency and exhaustive unbiasedness for an *adaptive*, label-dependent acquisition on a four-point example.

## Endpoints

Historical development response label Y=1 means harmful response, not harmful prompt. D=1 means blocked according to the **archived cross-fitted base classifier**; it is **not** a frozen deployed native guard. Estimate finite-pool means of (1) missed harmful outputs Z=Y(1-D), (2) harmful responses Y, and (3) false positives (1-Y)D, using LURE.

Compute FNR = estimated E[Z] / estimated E[Y] and FPR = estimated E[(1-Y)D] / estimated E[1-Y] only as plug-in ratios. **LURE's unbiasedness for linear population means does not make these ratios unbiased.** Report undefined-ratio frequency and FNR RMSE; do not report an unvalidated confidence interval.

## Proposal strategies (fixed prior to seeing new results)

1. **Fixed cheap-score rank:** nonnegative risk surrogate from ranks of archived `base_score`, emphasizing D=0 but retaining D=1 support. Ranking does not assume calibrated probability.
2. **Online beta-stratum surrogate:** divide data into D x within-D median `base_uncertainty` bands. Start each band at Beta(2,8); update only after its observations have been selected/reviewed. Use posterior harmfulness risk with greater priority for D=0 but positive D=1 support.

Every acquisition has an explicit **20% uniform mixture**, ensuring nonzero probability for every remaining record. Neither uses optional monitor outputs, unseen response labels, or unknown target features in acquisition.

## Fair comparison and safety boundaries

Read only two whitelisted historical development inputs: `cross_fitted_decision_value_targets.parquet` and `development_outer_fold_assignments.csv`. Reuse the frozen protocol and Phase 4A manifest hashes. Same 1,687 distinct examples in each of the two historical setups; budgets **40, 80, 160**, **200** repetitions per setup/proposal. Compare RMSE and bias for missed-harm prevalence, FNR RMSE/estimable fraction, and comparison to the previous uniform, proportional stratified, score-tilted stratified and allowed-only baselines. Exact paths/sha hashes are recorded in the generated manifest.

**Important limitations:** all labels and decisions are from an old development pool; some archived cross-fitted models were trained using development labels in other folds. Costs of optional monitors and prospective independent human labels are **not yet measured**. The benchmark is not a native-guard deployment FNR estimate, published-methods leaderboard, high-confidence risk certificate, or proof of selection superiority. No raw examples or sealed data are exported. Old Stage-A 0/4 failures remain unchanged.

## Next decision

After analyzing the Phase 4B ZIP: if Active Testing does not robustly outperform simple random/stratified auditing, document that negative result. If improvements exist, examine their magnitude, estimator stability, audit-to-monitor costs and validity on a frozen response-level native guard before proposing a larger experiment to the supervisor. Do not start a GPU sweep or new adapter from these development-only results.
