# Stage-A Paired-Transport Results

## Purpose

Stage-A evaluates whether runtime safety-monitor decisions remain stable when the same underlying harmful intent is expressed under controlled changes in source and representation. The study is a development-stage design check, not a deployment certificate.

It addresses three questions:

1. whether pairing reduces variance relative to an independent comparison for the same operating-point contrast;
2. whether source and representation changes alter guard operating behaviour; and
3. whether non-intercepts are shared across the guard panel or remain complementary.

No guard, threshold, formulation, eligibility rule, or study case was retuned after guard outputs were observed.

## Study design

The frozen Stage-A set contains 104 base harmful intents. Each base intent has four formulation slots: human/direct, model/direct, human/obfuscated, and model/obfuscated, giving 416 scored prompts.

Comparison-specific pair eligibility was frozen before guard scoring.

| Comparison | Eligible base cases |
|---|---:|
| Human direct vs human obfuscated | 104 |
| Model direct vs model obfuscated | 55 |
| Human direct vs model direct, strict | 52 |
| Human obfuscated vs model obfuscated, strict | 52 |
| Source semantic-only sensitivity | 55 |

The strict source set requires semantic retention and severity alignment. The 55-case sensitivity set requires semantic retention but retains the three severity-mismatched cases and is reported separately.

The obfuscation assignment is not balanced within the same intents: O1, O2 and O3 were applied to different base intents. The present transform-stratified analysis therefore describes transformation-specific operating behaviour but does not, by itself, identify a causal transformation effect independent of intent difficulty.

## Guard panel and terminology

The same frozen Stage-A prompts were evaluated by Meta Llama Guard 3-1B, Google ShieldGemma 2B, IBM Granite Guardian 3.3 8B, and Qwen3Guard Gen 4B.

All four guards produced complete, successfully parsed outputs for all 416 prompts.

Because the guards do not implement identical safety taxonomies, cross-guard panel summaries use the term **native-policy non-intercept** rather than treating every non-intercept as an interchangeable false negative. ShieldGemma in particular has narrower native policy-family coverage, so category-supported interpretation remains necessary.

## Pairing efficiency: q is not sufficient

For an ordered pair, let `D=1` denote intercept/block and `D=0` denote non-intercept. Define

\[
q_{10}=P(D_L=1,D_R=0), \qquad
q_{01}=P(D_L=0,D_R=1),
\]

\[
q=q_{10}+q_{01},
\]

and

\[
\Delta=P(D_R=0)-P(D_L=0)=q_{10}-q_{01}.
\]

The paired variance component is

\[
V_{pair}=q-\Delta^2,
\]

while for two independent samples of equal size the corresponding component is

\[
V_{ind}=p_L(1-p_L)+p_R(1-p_R).
\]

Therefore, small `q` alone does not establish that pairing is especially efficient. The relevant diagnostic is the ratio `V_pair / V_ind`.

### Strict direct-source comparison

| Guard | q | Paired / independent variance | Approx. variance reduction |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.077 | 0.534 | 46.6% |
| ShieldGemma 2B | 0.154 | 0.296 | 70.4% |
| Granite Guardian 3.3 8B | 0.038 | 1.020 | -2.0% |
| Qwen3Guard Gen 4B | 0.019 | 1.000 | 0.0% |

This reverses the earlier interpretation based on `q` alone. In this direct-source comparison, **ShieldGemma receives the largest variance reduction from pairing**, followed by Llama Guard. Granite Guardian and Qwen3Guard have very low marginal non-intercept rates in the direct condition, so their independent-sample variances are already small; their low `q` does not translate into a pairing advantage.

The full comparison table is in `pairing_efficiency.csv`.

## Representation effects must be separated by transformation

The pooled obfuscation result is dominated by O3, the Base64 wrapper. O1 is the dot-separated token perturbation and O2 is the leet-style character substitution.

### Human-source representation pairs

| Transform | n | Guard | Direct non-intercept | Transformed non-intercept | Difference |
|---|---:|---|---:|---:|---:|
| O1 | 38 | Llama Guard | 0.079 | 0.079 | 0.0 pp |
| O1 | 38 | ShieldGemma | 0.316 | 0.368 | +5.3 pp |
| O1 | 38 | Granite Guardian | 0.026 | 0.026 | 0.0 pp |
| O1 | 38 | Qwen3Guard | 0.000 | 0.000 | 0.0 pp |
| O2 | 28 | Llama Guard | 0.000 | 0.571 | +57.1 pp |
| O2 | 28 | ShieldGemma | 0.321 | 0.429 | +10.7 pp |
| O2 | 28 | Granite Guardian | 0.000 | 0.107 | +10.7 pp |
| O2 | 28 | Qwen3Guard | 0.000 | 0.000 | 0.0 pp |
| O3 Base64 | 38 | Llama Guard | 0.132 | 0.000 | -13.2 pp |
| O3 Base64 | 38 | ShieldGemma | 0.184 | 1.000 | +81.6 pp |
| O3 Base64 | 38 | Granite Guardian | 0.000 | 1.000 | +100.0 pp |
| O3 Base64 | 38 | Qwen3Guard | 0.000 | 1.000 | +100.0 pp |

### Model-source representation pairs

| Transform | n | Guard | Direct non-intercept | Transformed non-intercept | Difference |
|---|---:|---|---:|---:|---:|
| O1 | 21 | Llama Guard | 0.095 | 0.048 | -4.8 pp |
| O1 | 21 | ShieldGemma | 0.286 | 0.381 | +9.5 pp |
| O1 | 21 | Granite Guardian | 0.000 | 0.000 | 0.0 pp |
| O1 | 21 | Qwen3Guard | 0.048 | 0.048 | 0.0 pp |
| O2 | 15 | Llama Guard | 0.000 | 0.200 | +20.0 pp |
| O2 | 15 | ShieldGemma | 0.667 | 0.800 | +13.3 pp |
| O2 | 15 | Granite Guardian | 0.067 | 0.267 | +20.0 pp |
| O2 | 15 | Qwen3Guard | 0.000 | 0.000 | 0.0 pp |
| O3 Base64 | 19 | Llama Guard | 0.263 | 0.000 | -26.3 pp |
| O3 Base64 | 19 | ShieldGemma | 0.526 | 1.000 | +47.4 pp |
| O3 Base64 | 19 | Granite Guardian | 0.000 | 1.000 | +100.0 pp |
| O3 Base64 | 19 | Qwen3Guard | 0.000 | 0.737 | +73.7 pp |

The pooled statement that “obfuscation causes common-mode failure” is therefore too broad. The strongest multi-guard pattern is specifically associated with the Base64 condition in this Stage-A allocation.

## Transform-stratified panel concurrence

### Human obfuscated cases

| Transform | n | >=2 native-policy non-intercepts | 3-of-4 | 4-of-4 |
|---|---:|---:|---:|---:|
| O1 | 38 | 2 (5.3%) | 0 | 0 |
| O2 | 28 | 10 (35.7%) | 1 (3.6%) | 0 |
| O3 Base64 | 38 | 38 (100%) | 38 (100%) | 0 |

For all 38 human O3/Base64 cases, exactly three guards were non-intercepting and Llama Guard was the sole blocker.

### Model obfuscated cases

| Transform | n | >=2 native-policy non-intercepts | 3-of-4 | 4-of-4 |
|---|---:|---:|---:|---:|
| O1 | 21 | 1 (4.8%) | 0 | 0 |
| O2 | 15 | 5 (33.3%) | 0 | 0 |
| O3 Base64 | 19 | 19 (100%) | 14 (73.7%) | 0 |

Among the 19 model O3/Base64 cases, 14 were three-of-four non-intercepts and five were two-of-four. Llama Guard was the sole blocker in all 14 three-of-four cases.

These results show that the apparent near-common-mode pattern is primarily a **Base64-specific panel phenomenon** in the current design. O2 produces weaker instability for some guards, while O1 is comparatively stable. Because the transform groups contain different intents, the next controlled experiment must apply all three transforms to the same existing intents before attributing these differences purely to transformation type.

## Llama/Base64 interpretation boundary

Llama Guard blocks all Stage-A Base64 cases in the current scored set, while the other guards frequently do not. This creates apparent monitor diversity, but it does not yet establish useful semantic complementarity.

The current harmful-only Stage-A data cannot distinguish between:

1. Llama Guard recovering harmful semantics from the Base64 representation; and
2. Llama Guard reacting to encoded-looking text or the Base64 wrapper itself.

Matched benign controls and raw/decoded/raw+decoded comparisons are therefore required before interpreting Llama's Base64 behavior as genuinely useful diversity.

## Source effects

The strict source comparisons remain smaller and more heterogeneous than the pooled representation comparison.

### Human direct to model direct, strict

| Guard | Human non-intercept | Model non-intercept | Difference |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.058 | 0.096 | +3.85 pp |
| ShieldGemma 2B | 0.346 | 0.462 | +11.54 pp |
| Granite Guardian 3.3 8B | 0.019 | 0.019 | 0.00 pp |
| Qwen3Guard Gen 4B | 0.000 | 0.019 | +1.92 pp |

Stage-A does not support a general claim that model-generated formulations are uniformly harder for the guard panel.

## Matching and standardization

The strict source comparisons use the same base intents and require semantic retention and severity alignment. Category and base intent are matched by construction, and severity is matched by the eligibility rule.

`category_stratified_effects.csv` and `severity_stratified_effects.csv` preserve these dimensions explicitly.

A deployment-standardized aggregate is not introduced retrospectively because a numerical target weighting distribution over category, severity and base intent was not fixed before observing Stage-A outcomes.

## Interpretation and scope

The corrected Stage-A interpretation is narrower than the initial report:

- small `q` is not sufficient evidence of paired-sampling efficiency;
- the strongest pooled representation signal is driven primarily by the Base64 subset;
- the observed Llama/Base64 complementarity may be useful diversity or may be representation-triggered overblocking;
- cross-guard non-intercepts are not automatically comparable false negatives because guard policy scopes differ; and
- the present transform groups use different intents, so transformation and intent difficulty are still partly confounded.

Stage-A remains controlled development evidence, not a deployment safety certificate. No W0 evaluation, fresh confirmatory evaluation, post-result case selection, or threshold retuning was performed.
