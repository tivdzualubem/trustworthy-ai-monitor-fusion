# Phase 4C: Cost-aware optional-monitor auditing — development only

No native guard deployment risk claim. Archived optional scores are revealed only on query-selected rows, and review selection uses no unsampled Y.

| Setup | Human reviews | Optional checks | Method | Miss RMSE | FNR RMSE | Legacy optional ms |
|---|---:|---:|---|---:|---:|---:|
| compact_after_rule | 40 | 80 | priority_proportional | 0.0612 | 0.1083 | 3652.6 |
| compact_after_rule | 40 | 80 | priority_tilted | 0.0590 | 0.1441 | 3652.6 |
| compact_after_rule | 40 | 80 | random_proportional | 0.0609 | 0.1605 | 3652.6 |
| compact_after_rule | 80 | 80 | priority_proportional | 0.0394 | 0.0959 | 3652.6 |
| compact_after_rule | 80 | 80 | priority_tilted | 0.0376 | 0.0872 | 3652.6 |
| compact_after_rule | 80 | 80 | random_proportional | 0.0376 | 0.0978 | 3652.6 |
| compact_after_rule | 160 | 80 | priority_proportional | 0.0261 | 0.0577 | 3652.6 |
| compact_after_rule | 160 | 80 | priority_tilted | 0.0260 | 0.0562 | 3652.6 |
| compact_after_rule | 160 | 80 | random_proportional | 0.0245 | 0.0569 | 3652.6 |
| compact_after_rule | 40 | 320 | priority_proportional | 0.0582 | 0.1412 | 14610.3 |
| compact_after_rule | 40 | 320 | priority_tilted | 0.0520 | 0.1176 | 14610.3 |
| compact_after_rule | 40 | 320 | random_proportional | 0.0546 | 0.1197 | 14610.3 |
| compact_after_rule | 80 | 320 | priority_proportional | 0.0380 | 0.0931 | 14610.3 |
| compact_after_rule | 80 | 320 | priority_tilted | 0.0363 | 0.0904 | 14610.3 |
| compact_after_rule | 80 | 320 | random_proportional | 0.0355 | 0.0999 | 14610.3 |
| compact_after_rule | 160 | 320 | priority_proportional | 0.0239 | 0.0569 | 14610.3 |
| compact_after_rule | 160 | 320 | priority_tilted | 0.0243 | 0.0565 | 14610.3 |
| compact_after_rule | 160 | 320 | random_proportional | 0.0254 | 0.0551 | 14610.3 |
| qwen_after_rule_compact | 40 | 80 | priority_proportional | 0.0671 | 0.1789 | 127804.8 |
| qwen_after_rule_compact | 40 | 80 | priority_tilted | 0.0582 | 0.1198 | 127804.8 |
| qwen_after_rule_compact | 40 | 80 | random_proportional | 0.0590 | 0.1426 | 127804.8 |
| qwen_after_rule_compact | 80 | 80 | priority_proportional | 0.0387 | 0.0774 | 127804.8 |
| qwen_after_rule_compact | 80 | 80 | priority_tilted | 0.0369 | 0.0705 | 127804.8 |
| qwen_after_rule_compact | 80 | 80 | random_proportional | 0.0365 | 0.0981 | 127804.8 |
| qwen_after_rule_compact | 160 | 80 | priority_proportional | 0.0251 | 0.0533 | 127804.8 |
| qwen_after_rule_compact | 160 | 80 | priority_tilted | 0.0238 | 0.0517 | 127804.8 |
| qwen_after_rule_compact | 160 | 80 | random_proportional | 0.0272 | 0.0619 | 127804.8 |
| qwen_after_rule_compact | 40 | 320 | priority_proportional | 0.0474 | 0.1109 | 511219.4 |
| qwen_after_rule_compact | 40 | 320 | priority_tilted | 0.0493 | 0.0965 | 511219.4 |
| qwen_after_rule_compact | 40 | 320 | random_proportional | 0.0584 | 0.1312 | 511219.4 |
| qwen_after_rule_compact | 80 | 320 | priority_proportional | 0.0370 | 0.0640 | 511219.4 |
| qwen_after_rule_compact | 80 | 320 | priority_tilted | 0.0390 | 0.0663 | 511219.4 |
| qwen_after_rule_compact | 80 | 320 | random_proportional | 0.0393 | 0.0962 | 511219.4 |
| qwen_after_rule_compact | 160 | 320 | priority_proportional | 0.0259 | 0.0438 | 511219.4 |
| qwen_after_rule_compact | 160 | 320 | priority_tilted | 0.0239 | 0.0372 | 511219.4 |
| qwen_after_rule_compact | 160 | 320 | random_proportional | 0.0248 | 0.0602 | 511219.4 |

## Guardrails

- The additional monitor is a simulated, *charged* selective acquisition of an already-cached score; it does not change the historical D.
- The human auditor is simulated from historical development labels, not an actual human annotator.
- The miss/harm prevalence HT stratum means are conditionally design-unbiased; FNR and FPR ratios generally are not.
- Nominal confidence intervals are exploratory, not deployment certificates.
- Human review counts and historic monitor inference time are separate budget coordinates; comparisons to no-monitor baselines cannot be called cheaper without an explicit utility/cost model.
- Selected hypotheses and new methods need a fresh independent validation sample before publication claims.
