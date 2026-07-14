# Nested leave-group-out evaluation v1

This is the professor-required replacement for relying on the previously
influenced final-test split.

## Design

- Outer leave-source-out folds: ['jailbreakbench_judge_comparison', 'wildguardtest', 'xstest_safe_gpt4']
- Outer leave-family-out folds: ['GCG', 'PAIR', 'random_search']
- Each outer fold is excluded from both model fitting and threshold selection.
- Remaining rows are deterministically split into inner training and inner
  policy-selection sets with stratification.
- The existing `split` column is ignored.
- Complete calibrated cheap-router and full-fusion pipelines are fitted,
  serialized, reloaded, and verified independently inside each outer fold.
- All six prespecified policy families are selected within each fold.
- Full-information and selective acquisition are compared at common selection
  target FPRs of [0.01, 0.025, 0.05, 0.1].

## Interpretation limit

This is nested cross-fitted robustness evaluation, not a genuinely new
untouched external test set. The XSTest source fold contains only negative
examples, so it supports FPR evaluation but not source-specific recall.

## Target FPR 0.05 fold results

          scheme                      outer_fold  target_fpr                     policy    n  positive_n  negative_n  tp  fn  fp   tn   recall      fpr  precision  accuracy  intercept_rate  recall_ci95_low  recall_ci95_high  fpr_ci95_low  fpr_ci95_high  fpr_one_sided95_upper  qwen_call_rate  avg_estimated_cost_ms  estimated_cost_reduction
leave_source_out jailbreakbench_judge_comparison        0.05 full_information_always_on  200         119          81 118   1  16   65 0.991597 0.197531   0.880597  0.915000        0.670000         0.954069          0.999787      0.117331       0.300863               0.284426        1.000000            1407.760426                  0.000000
leave_source_out jailbreakbench_judge_comparison        0.05      selective_acquisition  200         119          81 115   4  15   66 0.966387 0.185185   0.884615  0.905000        0.650000         0.916171          0.990767      0.107517       0.286976               0.270733        0.965000            1359.858067                  0.034027
leave_source_out                   wildguardtest        0.05 full_information_always_on 1709         284        1425 213  71  46 1379 0.750000 0.032281   0.822394  0.931539        0.151551         0.695423          0.799277      0.023728       0.042825               0.041088        1.000000            1407.760426                  0.000000
leave_source_out                   wildguardtest        0.05      selective_acquisition 1709         284        1425 221  63  62 1363 0.778169 0.043509   0.780919  0.926858        0.165594         0.725308          0.825123      0.033518       0.055432               0.053476        0.946167            1334.082971                  0.052337
leave_source_out                xstest_safe_gpt4        0.05 full_information_always_on  250           0         250   0   0  10  240      NaN 0.040000   0.000000  0.960000        0.040000              NaN               NaN      0.019345       0.072329               0.066904        1.000000            1407.760426                  0.000000
leave_source_out                xstest_safe_gpt4        0.05      selective_acquisition  250           0         250   0   0  11  239      NaN 0.044000   0.000000  0.956000        0.044000              NaN               NaN      0.022166       0.077363               0.071780        0.908000            1281.845655                  0.089443
leave_family_out                             GCG        0.05 full_information_always_on   50          38          12  38   0   2   10 1.000000 0.166667   0.950000  0.960000        0.800000         0.907487          1.000000      0.020863       0.484138               0.438105        1.000000            1407.760426                  0.000000
leave_family_out                             GCG        0.05      selective_acquisition   50          38          12  38   0   2   10 1.000000 0.166667   0.950000  0.960000        0.800000         0.907487          1.000000      0.020863       0.484138               0.438105        1.000000            1407.760426                  0.000000
leave_family_out                            PAIR        0.05 full_information_always_on  100          43          57  42   1  10   47 0.976744 0.175439   0.807692  0.890000        0.520000         0.877110          0.999411      0.087473       0.299058               0.279394        1.000000            1407.760425                  0.000000
leave_family_out                            PAIR        0.05      selective_acquisition  100          43          57  42   1  10   47 0.976744 0.175439   0.807692  0.890000        0.520000         0.877110          0.999411      0.087473       0.299058               0.279394        0.980000            1380.387649                  0.019444
leave_family_out                   random_search        0.05 full_information_always_on   50          38          12  38   0   3    9 1.000000 0.250000   0.926829  0.940000        0.820000         0.907487          1.000000      0.054861       0.571858               0.527327        1.000000            1407.760426                  0.000000
leave_family_out                   random_search        0.05      selective_acquisition   50          38          12  38   0   3    9 1.000000 0.250000   0.926829  0.940000        0.820000         0.907487          1.000000      0.054861       0.571858               0.527327        1.000000            1407.760426                  0.000000

## Target FPR 0.05 pooled out-of-fold results

          scheme  target_fpr                     policy    n  positive_n  negative_n  tp  fn  fp   tn   recall      fpr  precision  accuracy  intercept_rate  recall_ci95_low  recall_ci95_high  fpr_ci95_low  fpr_ci95_high  fpr_one_sided95_upper  qwen_call_rate  avg_estimated_cost_ms  estimated_cost_reduction
leave_family_out        0.05 full_information_always_on  200         119          81 118   1  15   66 0.991597 0.185185   0.887218  0.920000        0.665000         0.954069          0.999787      0.107517       0.286976               0.270733        1.000000            1407.760426                  0.000000
leave_family_out        0.05      selective_acquisition  200         119          81 118   1  15   66 0.991597 0.185185   0.887218  0.920000        0.665000         0.954069          0.999787      0.107517       0.286976               0.270733        0.990000            1394.074037                  0.009722
leave_source_out        0.05 full_information_always_on 2159         403        1756 331  72  72 1684 0.821340 0.041002   0.821340  0.933302        0.186660         0.780380          0.857503      0.032218       0.051359               0.049662        1.000000            1407.760426                  0.000000
leave_source_out        0.05      selective_acquisition 2159         403        1756 336  67  88 1668 0.833747 0.050114   0.792453  0.928208        0.196387         0.793734          0.868772      0.040384       0.061380               0.059539        0.943492            1330.421873                  0.054937

## Validity statement

No outer-fold label or feature row is used for fitting or threshold selection
within that fold. Formal Neyman-Pearson or Learn-then-Test risk certification
is not performed here and remains the next professor-required stage.
