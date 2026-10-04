# Stage-A Follow-up Control Experiment

## Purpose

This follow-up resolves the specific ambiguities identified after Stage-A without opening W0 or a fresh confirmatory sample.

It combines three controls in one GPU run:

1. matched benign transformations to test whether Base64-triggered blocking is useful safety behavior or representation-triggered overblocking;
2. raw Base64 versus decoded direct text versus raw+decoded text;
3. O1, O2 and O3 applied to the same harmful intents to remove the earlier transform-versus-intent allocation confound.

## Frozen inputs

The experiment uses 38 human-source harmful intents from the existing Stage-A O3/Base64 allocation. These are not newly selected from outcomes: the full pre-existing O3 human subset is used.

A separate bank of 38 researcher-authored benign requests is rank-matched to the harmful anchors by direct character length.

Each harmful and benign member receives five conditions:

- `direct_decoded`: plain semantic text;
- `O1`: dot-separated token perturbation;
- `O2`: leet-style substitution;
- `O3_raw`: Base64 wrapper;
- `O3_raw_plus_decoded`: the Base64 wrapper followed by the decoded direct text.

This yields 38 pairs x 2 case types x 5 conditions = 380 prompts.

## Guard execution

The follow-up runner reuses the exact frozen Stage-A guard contract and runtime:

- Llama Guard 3-1B through the official Meta raw-checkpoint runtime;
- ShieldGemma 2B using the four official prompt-only policy guidelines and frozen max-P(Yes) rule;
- Granite Guardian 3.3 8B using `criteria_id="harm"`;
- Qwen3Guard Gen 4B using prompt moderation with Safe / Controversial / Unsafe output.

There is no threshold retuning and no change to the Stage-A guard adapters.

## Primary readouts

`condition_summary.csv`
: Per guard, case type, and condition intercept/non-intercept rates.

`llama_s8_summary.csv`
: Llama Guard unsafe and S8 counts for benign and harmful cases by condition.

`paired_transition_summary.csv`
: Within-pair decision transitions from direct decoded text to O1/O2/O3/raw+decoded, plus O3 raw versus O3 raw+decoded.

`panel_concurrence_summary.csv`
: Harmful-case native-policy non-intercept concurrence across the four guards by condition.

`combined_scores.csv`
: Full row-level scored follow-up table.

## Interpretation rules frozen before scoring

The key mechanism question is whether Llama Guard's O3/Base64 blocking is semantic or representation-triggered.

Evidence for representation-triggered overblocking would include substantial Llama blocking/S8 labeling on benign O3 controls.

Evidence for semantic recovery would require low benign O3 blocking together with harmful O3 behavior that tracks decoded semantics, especially when comparing O3 raw with direct decoded and O3 raw+decoded.

Differences among O1/O2/O3 on the same harmful intents may be interpreted as transformation-associated operating-point instability. They remain development evidence rather than a deployment guarantee.

Across heterogeneous guard taxonomies, the report uses **native-policy intercept/non-intercept** rather than treating every non-intercept as an identical false negative.

## Boundaries

- W0 remains closed.
- No fresh confirmatory sample is opened.
- No guard threshold is retuned.
- No prompt is selected or removed after observing follow-up scores.
- No deployment safety certificate is claimed.
