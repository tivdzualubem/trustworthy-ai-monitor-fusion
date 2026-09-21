# Stage A redesign: deployment risk, standardized risk, and paired operating-point transport

## Scope

The previous design used marginal false-negative-rate (FNR) comparisons across source, representation, joint, and temporal shift cells. Those comparisons are operationally useful, but they do not isolate degradation of the guard itself. A marginal FNR change can arise because the guard changes, because the harmful-case spectrum changes, or because a changed prompt produces a different target-model response and the analysis then conditions on the subset with (Y=1).

The redesign separates three quantities that answer different questions and should not be interpreted interchangeably.

## 1. Deployment FNR

For guard (g) in environment (e), let (D_g=1) denote an intercept/block decision and (D_g=0) a non-blocking decision. Let (Y=1) denote a harmful target-model response under the study policy.

[
\mathrm{FNR}^{dep}_{g,e}
=
P(D_g=0 \mid Y=1,E=e).
]

Deployment FNR answers:

> In the environment as it actually occurs, how often does the guard fail to intercept a harmful response?

This quantity intentionally includes the observed environment-specific mixture of category, severity, formulation, base intent, and target-model behavior. It is therefore the correct operational quantity for describing deployed risk.

A difference such as

[
\mathrm{FNR}^{dep}_{g,e_1}
-
\mathrm{FNR}^{dep}_{g,e_0}
]

does not by itself identify degradation of the guard. The two environments may contain different spectra of harmful cases, and changing the prompt can change which target responses satisfy (Y=1). Deployment FNR will therefore remain a descriptive safety measure, not a stand-alone causal or transport claim about the guard.

## 2. Standardized FNR

Let

[
Z=(C,V,B)
]

denote the case characteristics used for standardization:

- (C): harm category;
- (V): severity;
- (B): base intent or paired base-case identity.

For a common reference distribution (w(z)),

[
\mathrm{FNR}^{std}_{g,e}
=
\sum_z
P(D_g=0 \mid Y=1,E=e,Z=z)\,w(z).
]

The same reference weights are used for all environments being compared.

Standardized FNR answers:

> If the environments were evaluated over the same spectrum of category, severity, and underlying base intent, would the guard miss rates still differ?

The main diagnostic comparison is between the marginal deployment effect and the standardized effect. If a source or obfuscation difference is large marginally but becomes small after standardization, the earlier result is mainly explained by spectrum/composition change. If the difference remains, there is stronger evidence of a guard-specific transport problem.

Standardization addresses observed case-mix differences but does not automatically remove selection created by generating different target responses and then conditioning on (Y=1). The paired quantity below is therefore required.

## 3. Paired operating-point transport

Stage A will construct genuinely paired base cases with source and representation variants. Where feasible, each base case will contain:

- human/direct;
- model/direct;
- human/obfuscated;
- model/obfuscated.

Pair membership requires preservation of the same underlying harmful intent. Semantic equivalence and severity are checked separately. A variant that fails the semantic-equivalence requirement is not used as evidence of paired operating-point transport for that comparison.

For paired variants (a) and (b),

[
D_{g,i}^{(a)},D_{g,i}^{(b)}\in\{0,1\}.
]

The threshold-decision discordance is

[
q_g^{a,b}
=
P\left(D_g^{(a)}\neq D_g^{(b)}\right).
]

Directional discordance is retained separately:

[
q_{10}
=
P(D_g^{(a)}=1,D_g^{(b)}=0),
]

[
q_{01}
=
P(D_g^{(a)}=0,D_g^{(b)}=1).
]

When (a) is the reference representation and (b) is the shifted representation, (q_{10}) identifies cases that were intercepted before the transformation but missed afterward. The paired change in non-blocking probability is

[
\Delta_g^{pair}=q_{10}-q_{01}.
]

Paired operating-point transport answers:

> Holding the underlying harmful case fixed, does changing source or representation move the guard across its operating threshold?

This quantity is the main Stage-A identification diagnostic because it avoids comparing unrelated harmful-case populations.

## Stage-A experiment

Stage A uses only public/development data and approximately 100–200 genuinely paired base cases. No W0 or prospective confirmation-domain data are used.

For each base case:

1. construct the required source/representation variants;
2. evaluate semantic equivalence;
3. evaluate severity separately;
4. retain explicit category and base-intent identifiers;
5. run the same retained pairs through the complete candidate guard panel;
6. record the frozen operating-point decision for every guard and variant.

The first feasibility diagnostic is paired threshold-decision discordance (q). A value around (0.04) would make a paired prospective design potentially useful for reducing the required independent sample size. Values around (0.08)–(0.10) or higher would indicate that pairing is unlikely to solve the feasibility problem.

The same Stage-A cases are also used for two additional checks.

First, source and obfuscation effects are recalculated after matching or standardizing for category, severity, and base intent. This distinguishes spectrum change from a persistent guard-transport effect.

Second, failures are compared across the full guard panel. The analysis checks whether different guards begin missing the same cases under shift. A strong increase in shared misses would support a separate common-mode/defense-in-depth result.

## Decision boundary

Stage A is exploratory and identification-oriented. It does not freeze a new prospective primary claim, authorize W0, or authorize fresh confirmatory scoring.

Temporal shift remains secondary during this stage.

The next prospective study is chosen only after the Stage-A evidence shows which phenomenon is present:

- persistent operating-point transport after controlling for the same underlying harm;
- common-mode failure across guards under shift;
- sufficiently low paired discordance to support a sample-efficient certification design.

Local threshold fragility and score geometry may be examined later as explanatory mechanisms, but they are not the primary Stage-A objective.
