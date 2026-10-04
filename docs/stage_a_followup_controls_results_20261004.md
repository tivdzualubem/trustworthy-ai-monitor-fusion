# Stage-A Follow-up Control Results

## Executive result

The follow-up resolves the main Stage-A ambiguity. The strongest surviving result is **representation invariance failure** rather than a general source effect or a broad common-mode failure claim.

Under raw Base64, Llama Guard 3-1B intercepts all 38 benign and all 38 harmful controls, while ShieldGemma 2B, Granite Guardian 3.3 8B, and Qwen3Guard Gen 4B intercept none of the 38 benign and none of the 38 harmful controls. The raw Base64 condition therefore destroys harmful-versus-benign discrimination in opposite directions: Llama becomes an always-intercept monitor, while the other three become effectively always-non-intercept monitors on this matched set.

When the decoded semantic text is made explicitly visible alongside the same Base64 string, decisions return close to the direct-text operating point for all four guards.

The experiment remains development evidence only. W0 remained closed, no fresh confirmatory sample was opened, and no threshold was retuned.

## Design

The frozen follow-up contains 38 existing human-source Stage-A O3/Base64 harmful intents and 38 researcher-authored benign controls rank-matched by direct character length.

Each harmful and benign member was evaluated under five conditions:

- `direct_decoded`: plain semantic text;
- `O1`: dot-separated token perturbation;
- `O2`: leet-style character substitution;
- `O3_raw`: Base64 wrapper only;
- `O3_raw_plus_decoded`: the Base64 wrapper followed by the decoded direct text.

This produces 380 prompts and 1,520 guard evaluations.

The same frozen guard contracts and runtime were used as in Stage-A.

## Benign controls

### Intercept counts out of 38 benign cases

| Condition | Llama Guard | ShieldGemma | Granite Guardian | Qwen3Guard |
|---|---:|---:|---:|---:|
| Direct decoded | 0 | 0 | 0 | 0 |
| O1 | 7 | 0 | 0 | 0 |
| O2 | 23 | 3 | 0 | 2 |
| O3 raw | **38** | **0** | **0** | **0** |
| O3 raw + decoded | 1 | 0 | 0 | 0 |

The Base64 behavior is therefore not interpretable as useful safety selectivity by itself. On this benign set, Llama Guard intercepts every raw Base64 prompt, whereas the other three guards intercept none.

Llama also shows weaker representation-triggered benign blocking under O1 and especially O2.

## Harmful controls

### Intercept counts out of 38 harmful cases

| Condition | Llama Guard | ShieldGemma | Granite Guardian | Qwen3Guard |
|---|---:|---:|---:|---:|
| Direct decoded | 33 | 31 | 38 | 38 |
| O1 | 31 | 29 | 38 | 38 |
| O2 | 18 | 25 | 32 | 36 |
| O3 raw | **38** | **0** | **0** | **0** |
| O3 raw + decoded | 34 | 32 | 38 | 38 |

O1 is comparatively mild. O2 causes a broader loss of harmful-case interception. O3 raw Base64 produces the largest change and reverses monitor behavior: Llama intercepts all cases while the other three intercept none.

## Raw, decoded, and raw+decoded

The direct decoded and raw+decoded conditions are extremely close.

For harmful cases:

- Llama Guard: 33/38 direct versus 34/38 raw+decoded;
- ShieldGemma: 31/38 direct versus 32/38 raw+decoded;
- Granite Guardian: 38/38 versus 38/38;
- Qwen3Guard: 38/38 versus 38/38.

For benign cases:

- all four guards are 0/38 on direct decoded;
- raw+decoded remains 0/38 for ShieldGemma, Granite Guardian, and Qwen3Guard;
- Llama Guard is 1/38.

Within-pair discordance between direct decoded and raw+decoded is only 1/38 for harmful Llama, 1/38 for harmful ShieldGemma, 0/38 for harmful Granite, 0/38 for harmful Qwen, and 1/38 for benign Llama.

By contrast, moving from raw Base64 to raw+decoded flips 37/38 benign Llama decisions, 32/38 harmful ShieldGemma decisions, 38/38 harmful Granite decisions, and 38/38 harmful Qwen decisions.

The parsimonious interpretation is that the dominant O3 failure is representation-dependent and largely reversible when semantic content is explicitly exposed.

## Llama S8 behavior

Llama Guard S8 corresponds to the frozen Intellectual Property category in the Stage-A contract.

Observed S8 counts:

| Case type | Direct | O1 | O2 | O3 raw | O3 raw + decoded |
|---|---:|---:|---:|---:|---:|
| Benign | 0 | 0 | 2 | **37** | 0 |
| Harmful | 0 | 0 | 0 | **38** | 0 |

Thus S8 is almost perfectly associated with the raw Base64 representation in this follow-up, not with harmfulness. The behavior disappears when decoded semantics are appended.

The result establishes a representation-specific empirical signature. It does not by itself identify the internal mechanism responsible for the S8 output.

## Same-intent transformation comparison

The follow-up removes the earlier transform-versus-intent allocation confound by applying O1, O2, and O3 to the same 38 harmful intents.

Harmful-case native-policy non-intercept concurrence across the four guards is:

| Condition | >=2 non-intercepts | >=3 non-intercepts | All 4 |
|---|---:|---:|---:|
| Direct decoded | 1/38 (2.6%) | 0 | 0 |
| O1 | 2/38 (5.3%) | 0 | 0 |
| O2 | 14/38 (36.8%) | 1/38 (2.6%) | 0 |
| O3 raw | **38/38 (100%)** | **38/38 (100%)** | 0 |
| O3 raw + decoded | 1/38 (2.6%) | 0 | 0 |

O2 therefore contains a genuine weaker shared degradation signal, but the dramatic three-of-four concurrence is specific to raw Base64 in this controlled same-intent comparison.

## Corrected interpretation of monitor diversity

The original Stage-A O3 result looked like useful monitor diversity because Llama Guard was the sole blocker while the other monitors often did not intercept.

The matched benign controls change that interpretation. Llama Guard also intercepts 38/38 benign raw Base64 controls. Therefore, the O3 pattern is **apparent diversity without demonstrated semantic usefulness**.

A guard contributes useful diversity only when additional harmful interception is not achieved merely by indiscriminate benign blocking under the same representation.

On the present controls, raw Base64 does not satisfy that criterion for Llama Guard.

## Paper direction

The strongest supported direction is now:

**representation invariance / metamorphic testing of runtime safety monitors**, with apparent-versus-useful diversity as a secondary contribution.

The empirical structure supporting this direction is:

1. semantically equivalent representations cause large decision changes;
2. the effect persists when all transformations are applied to the same harmful intents;
3. benign controls show that Llama/Base64 interception is not selective for harmful semantics;
4. explicitly exposing decoded semantics largely restores the direct-text operating point; and
5. panel concurrence is strongly representation-specific rather than a uniform common-mode phenomenon.

A defense-in-depth/common-cause reliability direction remains relevant as a secondary analysis, especially for O2, but the follow-up does not support presenting broad common-mode failure as the primary claim.

A deeper Llama/Base64 mechanism study remains a possible follow-on. The present experiment establishes the behavioral phenomenon but not its internal cause.

## Limitations

The benign controls are researcher-authored and rank-matched by direct character length. They are not fully domain-matched semantic counterparts to each harmful intent. Consequently, they support a controlled representation-trigger diagnostic, not a population-level benign-blocking estimate.

The harmful subset contains the full pre-existing human O3 allocation from Stage-A rather than an independently sampled confirmation set.

Guard policy scopes differ. Cross-guard summaries therefore retain native-policy intercept/non-intercept terminology rather than treating every non-intercept as a directly comparable false negative.

No deployment guarantee is claimed.
