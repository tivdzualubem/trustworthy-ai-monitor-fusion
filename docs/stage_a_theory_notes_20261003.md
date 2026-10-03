# Stage-A Theory Notes

## 1. Marginal FNR does not identify monitor degradation under changing composition

Let (e) index an environment or formulation condition and let (Z) denote case characteristics such as harm category, severity, and base intent. For a fixed guard,

\[
\mathrm{FNR}_e
=
P(D=0\mid Y=1,e)
=
\sum_z
P(D=0\mid Y=1,Z=z,e)
P(Z=z\mid Y=1,e).
\]

A marginal FNR difference across environments can therefore arise from at least two distinct mechanisms:

1. the conditional behaviour of the guard changes,
   (P(D=0\mid Y=1,Z=z,e)); or
2. the distribution of harmful cases changes,
   (P(Z=z\mid Y=1,e)).

In the original shifted-cell design there is an additional selection problem: changing a prompt can change the target-model response, and the analysis then conditions on the subset for which (Y=1). Consequently, equal category allocation alone does not make the harmful-response populations comparable.

Without fixing, identifying, or standardizing the harmful-case composition, a marginal FNR difference does not by itself identify degradation of the guard.

The Stage-A paired design reduces this ambiguity by evaluating the same underlying base intent across controlled formulation changes. Strict source comparisons additionally require semantic retention and severity alignment. These paired comparisons target operating-point transport rather than a marginal deployment-risk contrast.

## 2. Independent versus paired information requirements

Consider two conditions, left (L) and right (R). Let (D=1) denote intercept/block and (D=0) denote non-intercept/miss.

For the paired design define

\[
q_{10}=P(D_L=1,D_R=0),
\qquad
q_{01}=P(D_L=0,D_R=1),
\]

and

\[
q=q_{10}+q_{01}.
\]

The right-minus-left change in miss probability is

\[
\Delta
=
\mathrm{FNR}_R-\mathrm{FNR}_L
=
q_{10}-q_{01}.
\]

Define the paired random variable

\[
X_i
=
(1-D_{i,R})-(1-D_{i,L})
=
D_{i,L}-D_{i,R}.
\]

Then

\[
X_i\in\{-1,0,1\},
\qquad
E[X_i]=\Delta,
\]

and

\[
\operatorname{Var}(X_i)
=
q-\Delta^2.
\]

Therefore, for (n) paired base cases,

\[
\operatorname{Var}(\widehat{\Delta}_{\mathrm{pair}})
=
\frac{q-\Delta^2}{n}.
\]

Only discordant pairs contribute nonzero information to the paired difference. This is why a small discordance probability can make a paired design efficient.

For comparison, suppose the two FNRs are estimated from independent samples of size (n) per condition. Let

\[
p_L=\mathrm{FNR}_L,
\qquad
p_R=\mathrm{FNR}_R.
\]

Then

\[
\operatorname{Var}(\widehat{\Delta}_{\mathrm{ind}})
=
\frac{p_L(1-p_L)+p_R(1-p_R)}{n}.
\]

Under the same normal-approximation confidence or power criterion, the leading sample-size terms are therefore

\[
n_{\mathrm{pair}}
\propto
\frac{q-\Delta^2}{\Delta^2},
\]

and

\[
n_{\mathrm{ind}}
\propto
\frac{p_L(1-p_L)+p_R(1-p_R)}{\Delta^2}.
\]

The corresponding first-order efficiency ratio is

\[
\frac{n_{\mathrm{pair}}}{n_{\mathrm{ind}}}
\approx
\frac{q-\Delta^2}
{p_L(1-p_L)+p_R(1-p_R)}.
\]

This is a design approximation rather than a replacement for the final exact power calculation. It makes the key point explicit: pairing reduces the information requirement only when the within-case decisions are sufficiently concordant. If (q) is small, the numerator can be much smaller than the independent Bernoulli variance. If (q) approaches the scale of the marginal variance terms, the advantage becomes limited.

Stage-A therefore uses (q) as an empirical feasibility diagnostic rather than as a safety metric. The observed values indicate potentially useful pairing for some direct-source comparisons, particularly Granite Guardian and Qwen3Guard, but not uniformly across guards and not after representation shift.

## 3. Common-mode and defense-in-depth reliability

Let

\[
M_g=1
\]

denote a miss by guard (g). A multi-guard defense is characterized by the joint distribution

\[
P(M_1,\ldots,M_G),
\]

not only by the individual marginal miss probabilities.

For a panel that blocks whenever at least one guard blocks, complete panel failure occurs when every guard misses:

\[
P_{\mathrm{panel\ fail}}
=
P(M_1=1,\ldots,M_G=1).
\]

If guard misses were independent on the evaluated case population, this probability would factor as

\[
P_{\mathrm{panel\ fail}}^{\mathrm{ind}}
=
\prod_{g=1}^{G}P(M_g=1).
\]

That factorization is not valid when guards share failure modes or when difficult cases induce correlated misses. Positive dependence can therefore make a defense-in-depth panel substantially less redundant than marginal FNRs suggest.

A useful development-stage diagnostic is to compare observed joint miss patterns with the independence picture, while avoiding a parametric common-cause model before there is enough evidence to support one. In particular, high rates of two-of-four or three-of-four misses reveal dependence structure even when no all-four failure is observed in a limited sample.

Stage-A shows precisely this pattern under obfuscation: simultaneous three-of-four misses become frequent, with ShieldGemma, Granite Guardian, and Qwen3Guard often failing on the same cases while Llama Guard remains the sole blocker. This is evidence of near-common-mode vulnerability under the tested representation shifts and evidence that Llama Guard contributes failure diversity in this sample.

The result should not be interpreted as an identified architectural cause, a deployment-wide common-cause probability, or proof that the four-guard panel is safe. No all-four miss was observed in Stage-A, but zero observed events only yields an upper confidence bound; it does not establish a zero population failure probability.

The present evidence is sufficient to motivate a reliability/common-cause direction and to preserve joint-failure quantities in future study design. It is not sufficient to fit a beta-factor, latent common-cause model, or other parametric reliability model as a primary claim.
