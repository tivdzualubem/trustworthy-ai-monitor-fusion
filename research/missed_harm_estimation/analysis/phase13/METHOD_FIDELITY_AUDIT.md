# Phase 13: audited method-to-paper correspondence (2026-11-12)

**Study snapshot:** published estimator identities, restricted optimization, and interval reporting. This is an internal scientific audit; no real hidden-harm prevalence is estimated here.

## Frozen estimand

Historical allowed probability `q=M/N`. Scientific target is `A_Y=P(Y=1,D=0)` for latent policy-concordant response harmfulness `Y`; historical review-defined proxy endpoint is `A_Z=P(Z=1,D=0)`. Knowing `q` means both values belong to `[0,q]` but does not equate `Y` to `Z`.

## Sources checked

1. Fisch, Maynez, Hofer, Dhingra, Globerson & Cohen, *Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models*, NeurIPS 2024, https://arxiv.org/abs/2406.04291 (sections 4.1-4.3). Author derives general weighted M-estimator; the mean specialization is a valid restricted case.
2. Cowen-Breen et al., *Multiple-Prediction-Powered Inference*, ICLR 2026, https://arxiv.org/pdf/2603.27414 (equations 5,6,7; sections 4.1-4.2). Author algorithm optimizes subset query counts and weights, via SOCP (single resource constraint), SDP (multiple); can reuse fully observed burn-in samples in final estimation.
3. Brawand et al., *Active Multiple-Prediction-Powered Inference*, arXiv 2026, https://arxiv.org/abs/2605.08429 (equations 1,2 and joint optimization in section 3). Per-instance Bernoulli sampling, selection probabilities, predictors, and weights are jointly optimized. It controls expected sampling cost as formulated, which is not automatically a deterministic realized cap.
4. Angelopoulos et al., *PPI++*, https://arxiv.org/abs/2311.01453 (power tuning for linear mean estimation).

## Exact status per arm

| Arm | Relation to paper | What was done | Not yet done |
|---|---|---|---|
| Whole/allowed SRS | Standard survey baseline | Known population masses; iid-with-replacement; exact CP interval for Bernoulli reviewer endpoint; cost-accounted | Prospective policy-certified latent Y |
| PPI++ | Linear mean-specialization | `q * [lambda mean(f) + mean(Z-lambda f)]`; independent charged pilot tunes lambda; conditional Hoeffding | Same-data power tuning and author's asymptotic interval; other non-mean estimands |
| StratPPI | Mean-specialization of NeurIPS 2024 | Proxy-quantile strata, known weights, independently pilot-tuned lambda per stratum, stratified rectifier, charged acquisition | Full weighted M-estimator, paper's asymptotic variance/CI; optimized stratum scheme independently validated |
| MultiPPI | Covariance-GLS allocation based on Eq. 7 | Seven subsets of {human,cheap,expensive}; unconstrained linear-unbiased GLS weights for given counts; SLSQP fractions, charged covariance pilot, conditional Hoeffding; **Phase 13 adds a convex global-optimality gap certificate for the *actual restricted allocation objective*** | Author SOCP/SDP code, minimum-cost ratios, globally certified **integer** solution, reuse of burn-in in inference, actual author CI and empirical predictor vectors |
| Restricted active AIPW | Not full AM-PPI | Cheap-score-driven one-time expensive-route heuristic, iid-with-replacement importance sampling correction, hard-budget ledger | Joint per-instance KKT optimum, paper's Bernoulli inclusion design and asymptotic CI; no certified author-equivalent comparison |
| **AM-PPI Eq.(1) identity test** (new, separate) | **Exact published estimator identity** for a *fixed* pilot-frozen policy | `q mean[m + xi(Z-m)/pi]` under independent per-instance Bernoulli sampling; exhaustively tested for unbiasedness and unbiased design variance estimate with fixed endpoint Z | This identity is not the routing/propensity/weight optimizer and cannot be used as a full AM-PPI benchmark |

The original Phase 11 source is copied byte-for-byte as `phase11_reference_unmodified.py`; this archive does **not** overwrite Phase 11 nor claim to replace the published author implementations.

### Convex certificate precision

Let `Q_j = P_j^T Sigma_j^-1 P_j`, `c_j>0`, `u` a fraction vector, `a=e_0`, and `M(u)=sum_j u_j Q_j/c_j + 1e-8 I`. The restricted objective is `f(u)=a^T M(u)^-1 a`, which is convex on its admissible domain. Its gradient is `g_j = -a^T M(u)^-1 (Q_j/c_j) M(u)^-1 a`.

By convexity, for any optimum `u*` on this restricted simplex `f(u)-f(u*) <= g^T u - min_{v in feasible simplex} g^T v`. The implemented `multi_optimality_certificate` computes this *rigorous convex first-order upper bound* for the **regularized continuous proxy objective** with the original Phase 11 `sum(u[:4])>=0.12` constraint. An optimality gap of 0 would certify that restricted continuous optimizer. A small nonzero gap bounds its objective suboptimality. This does **not** certify the original MultiPPI paper's finite-budget integer allocation, shrinkage choices, or statistical coverage.

### Why method effects remain conditional

* In all Phase 11/12B simulations, `Y` and proxy scores were synthetically generated, not observed native monitor outputs.
* The 336 observations from the local DB are the duration distribution only; actual human judgments cannot be converted into latent `Y`.
* Original MultiPPI's burn-in reuse may improve economic/statistical efficiency relative to the conservative Phase 11 independent-pilot design. Thus a worse adaptation **must not** be cited as a failure of official MultiPPI.
* In Phase 11/12B the cheap and expensive models were attributed measured historical *latencies* only. No GPU price-to-wage ratio was established and the synthetic predictors were not actual Koala/Qwen scores.
* AM-PPI Eq.(1) with independent Bernoulli labels does not meet a deterministic hard action budget: realized label count is random. Introducing a post-hoc cap without inclusion-probability correction can bias the estimator.

## Identifiability / frozen review reference

The real study contains 96 stratified historical-development cases and 336 non-gold review actions. The original C1-C8 reviewer ontology fixes the *definition* of true response harm, not a verified rater sensitivity/specificity. Arbitrarily correlated reviewer errors permit multiple latent `Y` distributions matching the same observed votes. Thus the historical latent target `A_Y` is not point-identified without additional assumptions or an externally anchored validation design. This is classical measurement-error/latent-class nonidentification, **not novel theory**.

A valid project endpoint now is `A_Z` for an **explicitly named, fixed, reproducible reviewer protocol**, with uncertainty conditional on the historical sampling design. Claiming true `A_Y` needs (a) a frozen target response policy, (b) independent qualified reference adjudication with a documented error protocol, (c) probability-based inclusion with positive known chances and adequate representation, and (d) response/monitor provenance and temporal or source transport scope. The original historical 96 and nonrandom R4 follow-up do not provide (b)-(d) automatically.

**Do not revive the R2-only 12-case Phase 9D study:** it was not run on real cases and is not necessary to audit estimator mathematical fidelity. The qualified-reference task, if ultimately required, targets the population estimand and must not be chosen around a single disagreeing rater.
