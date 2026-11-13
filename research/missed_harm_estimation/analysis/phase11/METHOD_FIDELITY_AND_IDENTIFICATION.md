# Phase 11: Estimation equations and truth status

## Primary quantity

Let `D=0` mean historical allowed, `Y` latent response-level C1–C8 harmfulness, `Z` independently sampled human-review judgment under a simulated observation model, `q=P(D=0)`. We seek `A=P(Y=1,D=0)=q E[Y|D=0]`. Unless we have a separately validated Y, a PPI estimator based on reviews actually estimates `q E[Z|D=0]`, even when it uses expensive proxy signals. **Prediction-powered corrections remove predictor bias relative to the label Z, not unknown Z↔Y error.**

### Whole/allowed SRS

For sampled `i` with replacement, `\hat A_Z,whole = mean[ Z_i * 1(D_i=0) ]`, or `\hat A_Z,allowed=q mean[Z_i]`. Clopper–Pearson bins plus multiplication by q give conditional exact coverage for a stable binary reviewer endpoint under iid draws.

### PPI++ mean specialization

With `f0` observed on every allowed case, independent calibration pilot estimating `λ`, and an independent reviewed sample:

`\hat A_Z=q*(λ * mean_all_allowed(f0)+ mean_reviewed(Z−λ f0))`.

For any pilot-dependent λ chosen independently of the evaluation sample, the expectation is exactly q times the reviewer-endpoint prevalence. The residual lies in an interval of width at most `1+|λ|`; Hoeffding yields a conservative finite-sample reviewer-endpoint CI conditional on pilot and observed census proxy scores.

### StratPPI mean specialization

Define disjoint proxy-based allowed strata with known population fractions w_h. Fit λ_h in paid independent stratum pilots, then:

`\hat A_Z=q sum_h w_h [λ_h mean_stratum(f0)+mean_evaluation_h(Z−λ_h f0)]`.

Pilot-Neyman weights proportional to `w_h * SD_pilot(Z−λ_h f0)` decide evaluation allocation. The 95% CI uses a Bonferroni union bound over two stratum-specific Hoeffding intervals. It is conservative and not the paper's general asymptotic StratPPI confidence construction.

### MultiPPI subset allocation (paper-grounded component)

Let X=(Z,f0,f1). For each nonempty subset I of the three variables, independently collect n_I IID samples of X_I. For pilot covariance Σ, optimize continuous n_I under a cost constraint using the MultiPPI information matrix

`M(n) = sum_I n_I P_I^T Σ_I^{-1} P_I`

and variance objective `a^T M(n)^† a`, where a=(1,0,0). Use nonnegative SLSQP allocation, integer rounding and realized paid acquisitions. Estimate `μ` by generalized least squares: `M^† sum_I n_I P_I^T Σ_I^{-1} mean(X_I)` and report μ_0. Validate `a^T M^† M = a^T` numerically (or stop), preserving the linear unbiasedness constraint. Under independent subset observations bounded [0,1], the weights imply a conservative Hoeffding CI for the reviewer endpoint. Pilot covariance shrinkage, integral rounding, fallback gold-only acquisitions, and SLSQP have not been certified to reproduce the paper's official SOCP solver or optimality theorem. This is **not a full author-reproduced MultiPPI benchmark**.

### Active AIPW restricted

Select a cheap/expensive prediction subset using only fully charged cheap proxy scores. Fit both prediction functions on a paid pilot, buy the selected expensive scores, and use independent importance draws `i~p` with replacement where p_i is positive and proportional to estimated residual difficulty. The estimator is

`\hat A_Z = q * [ N^-1 sum_i m_i + n^-1 sum_{j=1}^n (Z_{I_j}−m_{I_j})/(N p_{I_j}) ]`.

The correction identity is exact for each fixed prediction vector and probability distribution p, without requiring the predictions to be unbiased. **It is not the AM-PPI paper's jointly optimized predictor routing, Bernoulli label propensity or KKT fixed-point algorithm.** The finite-sample reviewer-Z interval uses a conservative Hoeffding bound on the importance-weighted correction, conditional on paid pilot and score acquisitions.

### Costs, uncertainty and generalization

Methods spend one *normalized* budget on observed browser-active human-review seconds (resampled from 336 completed judgments when DB provided) plus **assumed** human-second-equivalent prices per cheap/expensive proxy. All pilot reviews and their proxy acquisitions are paid; all subset acquisitions and all subsequent reviews are paid. Real GPU runtime cannot be inferred from a synthetic score-generation function. The human timing model assumes time is independent of label and difficulty. No true independently validated latent Y, prospective monitor D, source/attack shift, adjudicated ground truth, measured dollar pricing, or deployment certificate is produced.

## Go/no-go claim

A negative result for any of these specific mean specializations/adaptations **does not refute the full published algorithms**. If conditional performance is no better than allowed-only SRS, report that narrow result and maintain the real latent-Y identification limit. Novelty remains subject to full paper theorem equivalence and qualified reference data.
