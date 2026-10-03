# Stage-A Paired-Transport Results

## Purpose

Stage-A evaluates whether runtime safety-monitor decisions remain stable when the same underlying harmful intent is expressed under controlled changes in source and representation. The study is a development-stage design check, not a deployment certificate.

It addresses three questions:

1. whether paired observations produce sufficiently low decision discordance to materially reduce the information requirement relative to independent comparisons;
2. whether source and representation changes alter guard operating behaviour; and
3. whether failures are shared across the guard panel or remain complementary.

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

## Guard panel

The same frozen Stage-A prompts were evaluated by Meta Llama Guard 3-1B, Google ShieldGemma 2B, IBM Granite Guardian 3.3 8B, and Qwen3Guard Gen 4B.

All four guards produced complete, successfully parsed outputs for all 416 prompts.

## Threshold-decision discordance

For an ordered pair, let `D=1` denote a blocked harmful case and `D=0` a missed harmful case. Define

\[
q_{10}=P(D_L=1,D_R=0), \qquad
q_{01}=P(D_L=0,D_R=1),
\]

and

\[
q=q_{10}+q_{01}.
\]

Exact two-sided 95% Clopper-Pearson intervals are reported in `q_kill_test.csv`.

The Stage-A planning discussion treated approximately `q=0.04` as a level at which pairing could be materially useful, while values around `0.08–0.10` or larger would provide progressively less sample-size advantage. These are design heuristics, not hypothesis-test cutoffs.

### Strict source comparison: human direct to model direct

| Guard | n | q |
|---|---:|---:|
| Llama Guard 3-1B | 52 | 0.077 |
| ShieldGemma 2B | 52 | 0.154 |
| Granite Guardian 3.3 8B | 52 | 0.038 |
| Qwen3Guard Gen 4B | 52 | 0.019 |

For Granite Guardian and Qwen3Guard, direct-source discordance is close to or below the Stage-A planning value. Llama Guard is higher, and ShieldGemma is substantially higher.

### Strict source comparison after obfuscation

| Guard | n | q |
|---|---:|---:|
| Llama Guard 3-1B | 52 | 0.212 |
| ShieldGemma 2B | 52 | 0.115 |
| Granite Guardian 3.3 8B | 52 | 0.135 |
| Qwen3Guard Gen 4B | 52 | 0.096 |

Pairing therefore does not provide a uniform solution to the information requirement. Its potential advantage is guard- and condition-dependent and weakens materially after representation change.

## Matched FNR effects

Differences below are `FNR_right - FNR_left`.

### Human direct to human obfuscated

| Guard | Direct FNR | Obfuscated FNR | Difference |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.077 | 0.183 | +10.58 pp |
| ShieldGemma 2B | 0.269 | 0.615 | +34.62 pp |
| Granite Guardian 3.3 8B | 0.010 | 0.404 | +39.42 pp |
| Qwen3Guard Gen 4B | 0.000 | 0.365 | +36.54 pp |

### Model direct to model obfuscated

| Guard | Direct FNR | Obfuscated FNR | Difference |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.127 | 0.073 | -5.45 pp |
| ShieldGemma 2B | 0.473 | 0.709 | +23.64 pp |
| Granite Guardian 3.3 8B | 0.018 | 0.418 | +40.00 pp |
| Qwen3Guard Gen 4B | 0.018 | 0.273 | +25.45 pp |

Representation change produces the largest and most consistent deterioration for ShieldGemma, Granite Guardian and Qwen3Guard. Llama Guard behaves differently and retains substantially more blocking ability under the tested transformations.

### Strict source effect: human direct to model direct

| Guard | Human FNR | Model FNR | Difference |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.058 | 0.096 | +3.85 pp |
| ShieldGemma 2B | 0.346 | 0.462 | +11.54 pp |
| Granite Guardian 3.3 8B | 0.019 | 0.019 | 0.00 pp |
| Qwen3Guard Gen 4B | 0.000 | 0.019 | +1.92 pp |

### Strict source effect under obfuscation

| Guard | Human FNR | Model FNR | Difference |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.212 | 0.077 | -13.46 pp |
| ShieldGemma 2B | 0.635 | 0.712 | +7.69 pp |
| Granite Guardian 3.3 8B | 0.385 | 0.404 | +1.92 pp |
| Qwen3Guard Gen 4B | 0.327 | 0.269 | -5.77 pp |

The source effect is smaller and more heterogeneous than the representation effect. Stage-A therefore does not support reducing the result to a single claim that model formulations are uniformly harder for the guard panel.

## Matching and standardization

The strict source comparisons use the same base intents and require semantic retention and severity alignment. Category and base intent are matched by construction, and severity is matched by the eligibility rule. Representation comparisons retain the same underlying formulation within each base intent.

`category_stratified_effects.csv` and `severity_stratified_effects.csv` preserve these dimensions explicitly.

A deployment-standardized aggregate is not introduced retrospectively because a numerical target weighting distribution over category, severity and base intent was not fixed before observing Stage-A outcomes.

## Common-mode failures

No Stage-A case was missed by all four guards.

| Variant | n | Missed by >=2 | Missed by 3 | Missed by all 4 |
|---|---:|---:|---:|---:|
| Human/direct | 104 | 4 | 0 | 0 |
| Human/obfuscated | 104 | 50 | 39 | 0 |
| Model/direct | 55 | 6 | 0 | 0 |
| Model/obfuscated | 55 | 25 | 14 | 0 |

Among the 39 human-obfuscated cases missed by exactly three guards, Llama Guard was the sole blocking monitor in 38 cases. Among the 14 corresponding model-obfuscated cases, Llama Guard was the sole blocking monitor in all 14.

This is evidence of a strong near-common-mode failure pattern among ShieldGemma, Granite Guardian and Qwen3Guard under the tested representation transformations, while Llama Guard contributes substantial failure diversity in this Stage-A sample.

The absence of an observed four-monitor miss is not evidence of zero population risk. With zero observed events, the exact two-sided 95% upper bound is approximately 3.5% for `n=104` and 6.5% for `n=55`.

Pairwise co-miss results are reported in `pairwise_comiss_results.csv`.

## Interpretation and scope

Stage-A does not show that paired sampling uniformly resolves the large-sample problem. Pairing appears potentially useful for selected direct-source comparisons, particularly Granite Guardian and Qwen3Guard, but the advantage is substantially weaker under representation change.

The more prominent Stage-A result is representation sensitivity together with a concentration of multi-monitor failures under obfuscation. This supports treating source transport and representation transport as distinct estimands and motivates further study of common-mode reliability without yet fitting a parametric common-cause model.

Stage-A is controlled development evidence, not a deployment safety certificate. The strict source analysis contains 52 pairs and the model-representation analysis contains 55 semantic-retain pairs. These sample sizes are appropriate for a Stage-A signal check but not for broad population guarantees.

No W0 evaluation, fresh confirmatory evaluation, post-result case selection, or threshold retuning was performed.
