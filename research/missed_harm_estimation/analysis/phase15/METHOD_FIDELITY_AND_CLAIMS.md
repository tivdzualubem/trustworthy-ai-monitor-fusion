# Method fidelity and evidence boundary — 2026-11-13

## Proven from the uploaded Phase14 aggregate JSONs

- Both geometries (`N=1687, allowed M=1557`; `N=5000, allowed M=4000`), each 8 synthetic populations ×25 repetitions across 3 simulated reviewer regimes.
- Restricted MultiPPI continuous-subset GLS allocator: repaired solver certified 1,200/1,200 run-level relative first-order gap upper bounds below 0.5%; no recorded budget overspends.
- In the 1,687 geometry, repaired and old **realized estimators are numerically identical at aggregate precision**: perfect RMSE=0.01385255; independent errors=0.04094852; correlated errors=0.07709185.
- In the 5,000 geometry, repaired RMSE=0.01317481 perfect; 0.03822958 independent errors versus old 0.03822346; 0.06458694 correlated versus old 0.06458358. These differences are negligible and in two noisy regimes slightly worse, not improvement.
- Convex optimization certificate is only on **Phase11's restricted continuous objective**. Integer rounding and sample scheduling may erase objective-level gains.

## Method comparison context (source-authored distinctions)

- **StratPPI:** Fisch et al., NeurIPS 2024, mean specializations and strata weighting. The published procedure provides stratified prediction-powered estimation/inference; Phase11 is not a complete reproduction of all paper M-estimation/inference options. https://arxiv.org/abs/2406.04291
- **MultiPPI:** Cowen-Breen et al., ICLR 2026, optimally allocates diverse predictor measurement subsets, uses covariance estimation and allocation; Phase14's certified GLS component does not reproduce full author cost-reuse/integer and CI procedure. https://arxiv.org/abs/2603.27414
- **AM-PPI:** Brawand et al., 2026, jointly optimizes instance-specific predictor subset, Bernoulli inclusion probabilities and correction weights under a deployment-time budget. Phase13 verified an estimator identity, not the coupled KKT solver nor full published CIs. https://arxiv.org/abs/2605.08429
- A third-party package advertising a `MultiPPIMeanEstimator` is not automatically an implementation of **Cowen-Breen et al.** Verify its cited reference and algorithm before calling it official.

## Research decision

Numerical repair and repeated synthetic repricing are **closed**. Primary uncertainty is not a numerical optimizer error but **unvalidated response-level outcomes** and truly cost-comparable predictor acquisitions. Work from a new frozen response-level frame, or explicitly restrict claims to the old development-frame reviewer endpoint; the latter is not a deployment risk certificate. If published-method superiority is central, first verify author-equivalent code or faithfully reproduce its complete allocation and inferential objective using actual native proxy measurements, not simulated scores.

No elementary interval clipping, Bernoulli weighting identity, exact hypergeometric inversion, or first-order convex certificate is claimed as a novel statistical result.
