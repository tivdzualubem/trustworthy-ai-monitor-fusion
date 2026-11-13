# Phase 13 uncertainty audit

## Parameter support clipping

The scientific quantity `A=P(Y=1,D=0)` (or its fixed reviewer-protocol counterpart using `Z`) has an exactly known range `[0,q]` where `q=P(D=0)` in the frozen population. If `I(X)` is a valid confidence set and `A` lies in `[0,q]`, then

`P(A in I(X) intersection [0,q]) = P(A in I(X))`.

Therefore clipping **cannot reduce frequentist coverage** while shortening intervals. The original interval endpoints are retained, and the clipped endpoints are tracked separately. If the old interval misses `[0,q]`, intersection is empty and is reported as such; it is not silently replaced with a fake point interval.

## Asymptotic normal alternative — explicitly exploratory

For non-SRS methods, an additional *diagnostic* interval is computed as `[mu +- 1.96 * plug-in-se] intersection [0,q]`. In the Phase 11 method code, the PPI and stratified standard errors use sample residual variances; MultiPPI's current `se` uses the pilot covariance under the chosen subset counts; restricted active AIPW uses sample importance-weighted residual variance. These **do not** reproduce all paper CI algorithms or prove finite-sample coverage. The plug-in normal interval can severely under-cover when harm is rare, propensity weights are heavy, or variance estimates from tiny pilots are unstable. Report widths and simulated reviewer endpoint coverage side-by-side; do not prefer the narrowest interval without coverage.

## Conservative interval interpretation

The prior Hoeffding intervals rely on: independent calibration pilot / frozen tuning for the estimation sample; bounded proxy scores; independent iid-with-replacement sampling from the fixed synthetic population (or within predeclared strata); a cost stopping mechanism independent of the outcome conditional on sample count; and fixed/known Q and stratum weights. Their nominal coverage is for the **reviewer-defined binary outcome process**, not latent harmfulness under unrestricted reviewer error. MultiPPI covariance is derived from a separate pilot and its custom conditional Hoeffding bound is not the paper's CI.

## Targeted recheck, not another general sensitivity sweep

The Phase 11 per-trial samples were not exported, so confidence interval endpoints cannot be reconstructed from only the uploaded aggregate means. The Phase 13 script regenerates the same synthetic geometry and estimator algorithms **once** only to audit corrected intervals and the restricted optimization gap. It does not add a model, outcome experiment, budget sweep, or new paper claim.
