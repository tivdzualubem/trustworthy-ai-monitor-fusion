# Prospective response-review study: design proposal

**Study purpose:** assess feasibility and measurement costs for a reproducible reference-defined estimate of missed harmful assistant responses. **Status:** protocol only; no prospective frames or independent labels have been collected.

1. Freeze the target assistant, native post-generation/pre-delivery monitor and exact response-level C1-C8 policy, with dates, model versions, scoring thresholds and provenance.
2. Construct an eligible fresh-response population with both allowed (`D=0`) and blocked (`D=1`) decisions, retaining data dependencies and source groups privately.
3. Sample separately **without replacement** from both decision strata, with predeclared inclusion probabilities `pi_allowed=n_allowed/N_allowed` and `pi_blocked=n_blocked/N_blocked`.
4. Ask two qualified independent reviewers to read actual prompt-response pairs using the frozen policy, blinded to monitor decisions, identities and other ratings. Preserve safe/harmful/uncertain responses and record third-reviewer adjudication rules in advance.
5. Measure active and wall review time, inference latency and resource pricing, reviewer training, coordination and adjudication.
6. Report design-based intervals for a fixed operational reference outcome. Do not call this verified latent harmfulness without defensible reviewer-accuracy assumptions or external validation.
7. Obtain authorization and privacy approval and implement **both** decision strata before data collection. The current executable sampling prototype handles allowed responses only; it is not the final two-stratum system.

The historical 96 reviewed cases are not a new prospective dataset and are not eligible as confirmatory validation evidence.
