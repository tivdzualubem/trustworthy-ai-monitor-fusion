# Stage-A Paired-Transport Results

## Purpose

Stage-A evaluates whether runtime safety-monitor decisions remain stable when the same underlying harmful intent is expressed under controlled source and representation changes. The study is development evidence, not a deployment certificate.

The initial 416-prompt Stage-A analysis identified a strong representation signal dominated by O3/Base64. A subsequent frozen follow-up added matched benign controls, same-intent O1/O2/O3 comparisons, and direct/raw/raw+decoded comparisons. The follow-up resolves the main ambiguity in the initial interpretation.

## Initial Stage-A design

The frozen Stage-A set contains 104 base harmful intents. Each base intent has four formulation slots: human/direct, model/direct, human/obfuscated, and model/obfuscated, giving 416 scored prompts.

Comparison-specific pair eligibility was frozen before guard scoring.

| Comparison | Eligible base cases |
|---|---:|
| Human direct vs human obfuscated | 104 |
| Model direct vs model obfuscated | 55 |
| Human direct vs model direct, strict | 52 |
| Human obfuscated vs model obfuscated, strict | 52 |
| Source semantic-only sensitivity | 55 |

The strict source set requires semantic retention and severity alignment. The 55-case sensitivity set requires semantic retention but retains the three severity-mismatched cases.

## Guard panel and terminology

The same frozen prompts were evaluated by Meta Llama Guard 3-1B, Google ShieldGemma 2B, IBM Granite Guardian 3.3 8B, and Qwen3Guard Gen 4B.

Because the guards do not implement identical safety taxonomies, cross-guard summaries use **native-policy intercept/non-intercept** rather than treating every non-intercept as an interchangeable false negative.

## Pairing efficiency correction

For an ordered pair, let `D=1` denote intercept and `D=0` denote non-intercept. Define

\[
q_{10}=P(D_L=1,D_R=0),\qquad
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

and the equal-n independent variance component is

\[
V_{ind}=p_L(1-p_L)+p_R(1-p_R).
\]

Therefore small `q` alone is not evidence that pairing is especially efficient.

### Strict direct-source comparison

| Guard | q | Paired / independent variance | Approx. variance reduction |
|---|---:|---:|---:|
| Llama Guard 3-1B | 0.077 | 0.534 | 46.6% |
| ShieldGemma 2B | 0.154 | 0.296 | 70.4% |
| Granite Guardian 3.3 8B | 0.038 | 1.020 | -2.0% |
| Qwen3Guard Gen 4B | 0.019 | 1.000 | 0.0% |

ShieldGemma therefore receives the largest paired variance reduction in the strict direct-source comparison despite having a larger `q` than Granite or Qwen.

## Initial transform-stratified result

The original pooled obfuscation result was dominated by O3/Base64.

For human-source representation pairs:

| Transform | Llama non-intercept change | ShieldGemma | Granite | Qwen |
|---|---:|---:|---:|---:|
| O1 | 0.0 pp | +5.3 pp | 0.0 pp | 0.0 pp |
| O2 | +57.1 pp | +10.7 pp | +10.7 pp | 0.0 pp |
| O3 Base64 | -13.2 pp | +81.6 pp | +100.0 pp | +100.0 pp |

The initial design assigned different transforms to different intents, so those transform contrasts could not by themselves separate transform effect from intent difficulty.

That limitation motivated the follow-up.

## Follow-up control experiment

The follow-up used the complete existing set of 38 human-source O3/Base64 harmful intents and 38 researcher-authored benign controls rank-matched by direct character length.

Every harmful and benign member was evaluated in five conditions:

- direct decoded text;
- O1;
- O2;
- O3 raw Base64;
- O3 raw plus decoded text.

This produced 380 prompts and 1,520 guard evaluations under the unchanged frozen guard contracts and runtime.

### Benign intercept counts

| Condition | Llama | ShieldGemma | Granite | Qwen |
|---|---:|---:|---:|---:|
| Direct | 0/38 | 0/38 | 0/38 | 0/38 |
| O1 | 7/38 | 0/38 | 0/38 | 0/38 |
| O2 | 23/38 | 3/38 | 0/38 | 2/38 |
| O3 raw | **38/38** | **0/38** | **0/38** | **0/38** |
| O3 raw + decoded | 1/38 | 0/38 | 0/38 | 0/38 |

### Harmful intercept counts

| Condition | Llama | ShieldGemma | Granite | Qwen |
|---|---:|---:|---:|---:|
| Direct | 33/38 | 31/38 | 38/38 | 38/38 |
| O1 | 31/38 | 29/38 | 38/38 | 38/38 |
| O2 | 18/38 | 25/38 | 32/38 | 36/38 |
| O3 raw | **38/38** | **0/38** | **0/38** | **0/38** |
| O3 raw + decoded | 34/38 | 32/38 | 38/38 | 38/38 |

The raw Base64 condition therefore eliminates harmful-versus-benign discrimination in opposite directions: Llama behaves as an always-intercept monitor on this matched set, while the other three behave as effectively always-non-intercept monitors.

### Raw versus decoded versus raw+decoded

When decoded semantics are appended to the Base64 string, decisions return close to the direct operating point.

Harmful direct versus raw+decoded intercept counts are 33 versus 34 for Llama, 31 versus 32 for ShieldGemma, 38 versus 38 for Granite, and 38 versus 38 for Qwen.

Benign direct is 0/38 for all four; raw+decoded is 1/38 for Llama and 0/38 for the other three.

This supports a representation-dependent failure interpretation rather than an intrinsic inability to judge the underlying semantic requests.

## Llama S8 result

In the follow-up, Llama S8 counts were:

| Case type | Direct | O1 | O2 | O3 raw | O3 raw + decoded |
|---|---:|---:|---:|---:|---:|
| Benign | 0 | 0 | 2 | **37** | 0 |
| Harmful | 0 | 0 | 0 | **38** | 0 |

S8 is therefore nearly perfectly tied to the raw Base64 representation and not to harmfulness in this follow-up.

This shows that the original Base64-specific Llama rescue cannot be interpreted as demonstrated semantic complementarity. The behavioral cause remains unresolved.

## Same-intent transformation effect

Because all three transforms are now applied to the same 38 harmful intents, the transform-versus-intent allocation confound is removed for this subset.

The rate of harmful cases with at least two native-policy non-intercepts is:

| Condition | >=2 | >=3 |
|---|---:|---:|
| Direct | 1/38 (2.6%) | 0 |
| O1 | 2/38 (5.3%) | 0 |
| O2 | 14/38 (36.8%) | 1/38 (2.6%) |
| O3 raw | **38/38 (100%)** | **38/38 (100%)** |
| O3 raw + decoded | 1/38 (2.6%) | 0 |

O2 carries a weaker shared-degradation signal. The dramatic three-of-four concurrence remains specific to raw Base64.

## Corrected interpretation

The strongest supported Stage-A result is now **representation invariance failure**.

The evidence is:

1. semantically equivalent representations produce large monitor-decision changes;
2. those differences remain when O1/O2/O3 are applied to the same harmful intents;
3. benign controls show that Llama's raw Base64 behavior is not selective for harmful semantics;
4. explicitly exposing decoded semantics largely restores direct-text behavior; and
5. the panel concurrence pattern is strongly representation-specific.

The original interpretation of Llama Guard as providing useful Base64 diversity is therefore too strong. The current evidence supports **apparent diversity without demonstrated semantic usefulness**.

A reliability/common-cause framing remains relevant as a secondary analysis, particularly for O2, but broad common-mode failure is not the primary conclusion.

## Paper direction

The strongest current paper direction is:

**representation invariance / metamorphic testing of runtime safety monitors**, with apparent-versus-genuinely-useful monitor diversity as a secondary contribution.

A deeper Llama/Base64 mechanism study is a possible follow-on rather than a prerequisite for the present behavioral claim.

## Limitations

The benign controls are researcher-authored and rank-matched by character length, not fully domain-matched semantic counterparts. They support a controlled representation-trigger diagnostic rather than a population-level benign-blocking estimate.

The follow-up harmful subset is the complete pre-existing human O3 allocation, not an independent confirmation sample.

Guard policy scopes differ, so cross-guard non-intercepts remain native-policy quantities rather than directly comparable false negatives.

No W0 evaluation, fresh confirmatory evaluation, post-result case selection, or threshold retuning was performed.
