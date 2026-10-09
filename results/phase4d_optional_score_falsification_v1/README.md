# Phase 4D: Does the purchased optional-monitor SCORE add information?

Post-hoc exploratory, development-only. D is an archived cross-fitted classifier, not a deployed guard.
Scores are used only on records where simulated extra-monitor acquisition occurs.
Matched human review budgets across arms. At fixed monitor count, acquisition and randomization are shared.
The no-monitor control incurs zero optional calls; the other three arms are charged identically.

## FNR RMSE (smaller is better)

| Setup | Monitor calls | Human reviews | Cheap only (0 calls) | Acquisition only | Shuffled scores | Real scores |
|---|---:|---:|---:|---:|---:|---:|
| compact_after_rule | 80 | 80 | 0.0809 | 0.0819 | 0.0819 | 0.0845 |
| compact_after_rule | 80 | 160 | 0.0608 | 0.0598 | 0.0582 | 0.0593 |
| compact_after_rule | 320 | 80 | 0.0850 | 0.0962 | 0.0896 | 0.0858 |
| compact_after_rule | 320 | 160 | 0.0612 | 0.0576 | 0.0598 | 0.0540 |
| qwen_after_rule_compact | 80 | 80 | 0.0843 | 0.0856 | 0.0863 | 0.0743 |
| qwen_after_rule_compact | 80 | 160 | 0.0593 | 0.0632 | 0.0606 | 0.0549 |
| qwen_after_rule_compact | 320 | 80 | 0.0779 | 0.0940 | 0.0870 | 0.0596 |
| qwen_after_rule_compact | 320 | 160 | 0.0627 | 0.0621 | 0.0605 | 0.0421 |

## Interpretation and restrictions

- Negative real-minus-control RMSE means better real-score performance in this Monte Carlo simulation.
- Paired bootstrap intervals quantify Monte Carlo uncertainty only; they are not validation or deployment intervals.
- Shuffling nulls the per-case link between paid scores and outcomes, retaining decision-group score distribution.
- The acquisition-only arm retains the allocation effects of prioritizing cheap-uncertain queries.
- Review labels are historical, simulated as purchased; no new independently labeled cases or fresh confirmatory data.
- FNR and FPR are ratios and do not inherit the conditional unbiasedness of the linear stratum mean estimators.
- Optional monitor latency uses historical batch-mean per-call cost; no new wall-clock timing or human wage cost.
- This specific four-arm ablation was devised *after* reviewing Phase 4C. It must not be sold as pre-registered confirmation.
- No algorithm selection, new model training, or evidence of deployment superiority follows from this pilot.
