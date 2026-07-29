# Cross-Fitted Decision-Value Targets

This artifact uses only the frozen development pool. `final_test` and `held_out_shift` were not used.

For each optional monitor, the realized target is:

`base zero-one loss - augmented zero-one loss`.

Positive values indicate that acquiring the optional monitor corrected the downstream decision. Negative values indicate that it worsened the decision.

## Development-only pooled diagnostics

### compact_after_rule

- Rows: 1687
- Base recall: 0.213058
- Base FPR: 0.050860
- Base one-sided 95% FPR upper bound: 0.061617
- Base decision loss: 0.177830
- Augmented recall: 0.213058
- Augmented FPR: 0.046562
- Augmented one-sided 95% FPR upper bound: 0.056932
- Augmented decision loss: 0.174274
- Decision-loss reduction: 0.003557
- Positive-value examples: 20 (0.011855)
- Negative-value examples: 14 (0.008299)
- Decision-change rate: 0.020154

### qwen_after_rule_compact

- Rows: 1687
- Base recall: 0.216495
- Base FPR: 0.047994
- Base one-sided 95% FPR upper bound: 0.058496
- Base decision loss: 0.174867
- Augmented recall: 0.865979
- Augmented FPR: 0.053725
- Augmented one-sided 95% FPR upper bound: 0.064730
- Augmented decision loss: 0.067576
- Decision-loss reduction: 0.107291
- Positive-value examples: 249 (0.147599)
- Negative-value examples: 68 (0.040308)
- Decision-change rate: 0.187908

These are target-construction diagnostics only. They do not yet show that decision value is predictable from legitimate pre-acquisition features, and they do not pass the professor's milestone by themselves.
