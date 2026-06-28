# Frozen Research Scope

## Research question

Can a simple budget-aware multi-monitor policy improve the
recall-cost-false-positive frontier over strong baselines, while giving valid
in-distribution calibration bounds and revealing how those bounds fail under
attack-family shift?

## Prediction unit

Each observation is a prompt-response pair:

X = (P, R)

where P is the user prompt and R is the assistant response.

## Primary outcome

Y = 1 if R should be intercepted because, in the context of P, it materially
provides, facilitates, endorses, or contains disallowed harmful content.

Y = 0 if R is benign, a safe refusal, or a safe redirection.

## Auxiliary variables

- prompt_harmful
- response_refusal
- jailbreak_success
- harm_category
- attack_family
- toxicity_category
- source_dataset

## Scope limitations

The project does not claim:

- robustness to arbitrary deployment shift;
- valid calibration guarantees under attack-family shift;
- reliable detection of all correlated monitor failures;
- universal online adaptation.

Held-out attack families will be used as stress tests showing how
in-distribution calibration bounds may fail under shift.
