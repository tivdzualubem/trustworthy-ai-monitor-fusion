# Cross-Fitted Value-Predictability Diagnostic

This report is development-only. It does not use `final_test` or `held_out_shift`, and it does not change the overall project `no-go` status.

## Primary prespecified comparison

- Setup: `qwen_after_rule_compact`
- Feature family: `all_features`
- Integrated learned-minus-uncertainty advantage: 0.00481624
- Paired bootstrap 95% CI: [-0.00170458, 0.01148525]
- Value-predictability criterion: NO-GO

## All prespecified comparisons

```text
               setup_id       feature_family  primary_comparison      mse      mae  spearman  positive_value_average_precision  integrated_advantage  paired_bootstrap_lower95  paired_bootstrap_upper95  predictability_criterion_pass
     compact_after_rule         all_features               False 0.019460 0.041389  0.083805                          0.247952              0.000119                 -0.002090                  0.002134                          False
     compact_after_rule       cheap_features               False 0.018622 0.033620  0.066284                          0.363167             -0.000563                 -0.002802                  0.001526                          False
     compact_after_rule cheap_plus_embedding               False 0.019712 0.041877  0.103466                          0.230680              0.000963                 -0.001319                  0.003364                          False
     compact_after_rule  cheap_plus_metadata               False 0.019084 0.037785  0.074846                          0.264923              0.000059                 -0.002223                  0.002371                          False
     compact_after_rule     frozen_embedding               False 0.021261 0.044484  0.019224                          0.019623             -0.002312                 -0.005498                  0.000934                          False
     compact_after_rule     runtime_metadata               False 0.021084 0.041335  0.032315                          0.049165             -0.001986                 -0.004876                  0.000935                          False
qwen_after_rule_compact         all_features                True 0.166194 0.255197  0.252910                          0.361909              0.004816                 -0.001705                  0.011485                          False
qwen_after_rule_compact       cheap_features               False 0.169309 0.243155  0.210860                          0.328050              0.000622                 -0.005587                  0.006402                          False
qwen_after_rule_compact cheap_plus_embedding               False 0.166968 0.252529  0.245486                          0.343944              0.005246                 -0.001008                  0.011411                          False
qwen_after_rule_compact  cheap_plus_metadata               False 0.166330 0.241273  0.244256                          0.364625              0.004698                 -0.001720                  0.010967                          False
qwen_after_rule_compact     frozen_embedding               False 0.179283 0.265804  0.121661                          0.243676             -0.009425                 -0.017888                 -0.001138                          False
qwen_after_rule_compact     runtime_metadata               False 0.171962 0.251243  0.209425                          0.301684             -0.000370                 -0.007514                  0.006358                          False
```

The total safety-cost frontier and common-risk selective point remain required before the professor's overall milestone can pass.
