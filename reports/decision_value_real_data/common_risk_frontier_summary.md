# Common-Risk Safety-Cost Frontier

This is a development-only diagnostic.

## Cost accounting

- Optional Qwen mean cost: 1597.560570 ms/acquired example
- Frozen embedding mean runtime: 159.468164 ms/example
- PCA plus value-estimator inference: 0.029189 ms/example
- Learned fixed overhead: 159.497354 ms/example

## Selective-point candidates

```text
 learned_budget  learned_acquisition_rate  learned_incremental_cost_ms_per_example  learned_recall  learned_fpr  risk_pass  best_uncertainty_budget_under_cost  best_uncertainty_cost_ms_per_example  best_uncertainty_recall_under_cost  recall_margin_over_uncertainty  uncertainty_dominance_pass  best_random_budget_under_cost  best_random_cost_ms_per_example  best_random_recall_upper95_under_cost  recall_margin_over_random_upper95  random_dominance_pass  selective_point_pass
           0.00                  0.000000                               159.497354        0.216495     0.047994       True                                0.10                            156.252219                            0.233677                       -0.017182                       False                            0.1                       156.252219                               0.304210                          -0.087715                  False                 False
           0.05                  0.047421                               235.256006        0.278351     0.037249       True                                0.10                            156.252219                            0.233677                        0.044674                        True                            0.1                       156.252219                               0.304210                          -0.025859                  False                 False
           0.10                  0.097807                               315.749573        0.336770     0.028653       True                                0.10                            156.252219                            0.233677                        0.103093                        True                            0.1                       156.252219                               0.304210                           0.032560                   True                  True
           0.20                  0.198577                               476.736708        0.487973     0.025072       True                                0.20                            317.239354                            0.388316                        0.099656                        True                            0.2                       317.239354                               0.379811                           0.108162                   True                  True
           0.30                  0.299348                               637.723843        0.587629     0.030086       True                                0.40                            636.372675                            0.549828                        0.037801                        True                            0.2                       317.239354                               0.379811                           0.207818                   True                  True
           0.40                  0.398340                               795.870029        0.649485     0.034384       True                                0.40                            636.372675                            0.549828                        0.099656                        True                            0.2                       317.239354                               0.379811                           0.269674                   True                  True
           0.50                  0.499111                               956.857164        0.721649     0.039398       True                                0.60                            956.452979                            0.718213                        0.003436                        True                            0.2                       317.239354                               0.379811                           0.341838                   True                  True
           0.60                  0.598696                              1115.950333        0.745704     0.042264       True                                0.60                            956.452979                            0.718213                        0.027491                        True                            0.2                       317.239354                               0.379811                           0.365893                   True                  True
           0.75                  0.748074                              1354.590086        0.797251     0.045845       True                                0.75                           1195.092732                            0.790378                        0.006873                        True                            0.2                       317.239354                               0.379811                           0.417440                   True                  True
```

- Frontier condition: PASS
- Predictability condition: NO-GO
- Overall milestone: NO-GO

The overall result remains no-go regardless of the frontier condition because the prespecified value-predictability confidence interval included zero.
