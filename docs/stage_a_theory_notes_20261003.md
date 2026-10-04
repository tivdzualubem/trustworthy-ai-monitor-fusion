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

This is a design approximation rather than a replacement for the final exact power calculation. The key point is that `q` cannot be interpreted in isolation. Pairing is efficient only when the paired variance component `q-Delta^2` is small relative to the independent Bernoulli variance `p_L(1-p_L)+p_R(1-p_R)`.

In the strict direct-source Stage-A comparison, the plug-in paired/independent variance ratios are approximately 0.296 for ShieldGemma, 0.534 for Llama Guard, 1.000 for Qwen3Guard, and 1.020 for Granite Guardian. Thus ShieldGemma shows the largest paired variance reduction despite having a larger `q` than Granite or Qwen. Granite and Qwen have very small marginal non-intercept probabilities, so their independent-sample variances are already small and their low discordance does not produce a pairing advantage.

Stage-A therefore retains `q` as a descriptive discordance statistic, while paired-sampling efficiency is assessed with the paired-versus-independent variance ratio.

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

Stage-A shows that the apparent near-common-mode pattern is not uniform across the tested transformations. It is concentrated primarily in the O3 Base64 condition: all 38 human O3 cases have exactly three native-policy non-intercepts with Llama Guard as the sole blocker, and 14 of 19 eligible model O3 cases have the same three-of-four pattern. O1 produces very little panel concurrence and O2 produces a weaker intermediate signal. The current evidence therefore supports a Base64-specific panel phenomenon, not a general common-mode failure claim across obfuscations.

The result should not be interpreted as an identified architectural cause, a deployment-wide common-cause probability, or proof that the four-guard panel is safe. No all-four miss was observed in Stage-A, but zero observed events only yields an upper confidence bound; it does not establish a zero population failure probability.

The present evidence is sufficient to motivate a reliability/common-cause direction and to preserve joint-failure quantities in future study design. It is not sufficient to fit a beta-factor, latent common-cause model, or other parametric reliability model as a primary claim.


## 4. Representation invariance and metamorphic testing

Let \(x\) denote a semantic request and \(T(x)\) a representation transformation intended to preserve the request's semantics. For a guard decision rule \(D(\cdot)\), a basic representation-invariance property is

\[
D(T(x)) = D(x)
\]

for transformations that should not alter the safety-relevant meaning.

The follow-up does not assume that all representation changes must leave every finite-model decision identical. Instead, it treats systematic decision changes under semantics-preserving transformations as metamorphic test failures that require explanation.

The raw Base64 condition provides the clearest example. On the same 38 harmful intents, Granite Guardian and Qwen3Guard move from 38/38 intercepts on decoded text to 0/38 on raw Base64, while Llama Guard moves in the opposite direction and intercepts all 38. On matched benign controls, Llama also intercepts all 38 raw Base64 prompts, whereas the other three intercept none.

Appending the decoded text to the same raw Base64 representation largely restores the direct operating point. This makes representation accessibility, rather than the underlying semantic request alone, a central explanatory variable.

A useful diversity claim therefore requires more than disagreement among monitors. For monitor \(g\), additional harmful-case interception under a representation should be considered useful only together with evidence that the same behavior does not arise indiscriminately on matched benign controls. The present raw-Base64 Llama pattern fails that stronger criterion.

The correct primary framing is therefore measurement invariance / metamorphic testing, with defense-in-depth reliability treated as a consequence of representation-dependent guard behavior rather than as an independently established common-cause mechanism.
