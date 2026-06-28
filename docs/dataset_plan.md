# Final Dataset and Split Plan

This document freezes the dataset choices for the current project scope.

## Prediction unit

Each prediction unit is one prompt-response pair:

X = (prompt, response)

The runtime monitors assess whether the response should be intercepted in the
context of the prompt.

## Primary target

Y = 1 when the response materially provides, facilitates, endorses, or
contains disallowed harmful content and should therefore be intercepted.

Y = 0 for benign responses, safe refusals, safe redirections, and harmless
responses to harmful prompts.

The following concepts remain separate from Y:

- prompt harmfulness;
- response refusal;
- over-refusal;
- jailbreak success;
- toxicity;
- attack family.

## Final dataset selection

### 1. JailbreakBench judge comparison — adversarial harmful set

Configuration:

- repository: JailbreakBench/JBB-Behaviors;
- configuration: judge_comparison;
- adversarial rows only: indices 0 through 199;
- response label: human_majority.

Attack-family mapping documented by JailbreakBench:

- indices 0--99: PAIR;
- indices 100--149: GCG;
- indices 150--199: adversarial random search.

The 100 benign XSTest-derived rows at indices 200--299 are excluded to avoid
overlap with the separate XSTest source.

The complete GCG family is reserved as the held-out attack-family shift.

### 2. XSTest — benign and over-refusal set

The project uses:

- xstest_prompts.csv;
- model_completions/xstest_v2_completions_gpt4.csv;
- only the 250 prompts whose source label is safe.

All selected XSTest examples have Y = 0.

The existing manual response annotation is retained separately:

- 1_full_compliance;
- 2_full_refusal;
- 3_partial_refusal.

A full or partial refusal to a safe prompt is recorded as over_refusal = 1,
but it does not change Y.

### 3. WildGuardTest — mixed safety set

The project uses the WildGuardTest configuration from allenai/wildguardmix.

Rows without a response-harm label are excluded.

Mapping:

- response_harm_label = harmful gives Y = 1;
- response_harm_label = unharmful gives Y = 0;
- prompt_harm_label is stored separately;
- response_refusal_label is stored separately.

## Five disjoint splits

Each prompt group belongs to exactly one split:

1. policy_train
2. policy_selection
3. calibration
4. final_test
5. held_out_shift

The complete JailbreakBench GCG family is assigned to held_out_shift.

All remaining examples are assigned deterministically using approximate
proportions:

- policy_train: 40%;
- policy_selection: 20%;
- calibration: 20%;
- final_test: 20%.

Identical normalized prompts are kept in the same split.

The calibration split must not be used to train models, tune thresholds,
select cascades, or choose among candidate policies.
