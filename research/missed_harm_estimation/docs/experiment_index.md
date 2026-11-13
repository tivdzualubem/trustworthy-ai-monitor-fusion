# Experimental index

This index links each research question to its supporting code and aggregate results. It is not a task-status or correspondence record.

| Analysis | Measured or tested quantity | Artifacts |
|---|---|---|
| Historical monitor decomposition | Harm-labeled allowed cases 229 to 228; blocked cases 62 to 63 (N=1,687) | Historical monitor records and report Section 3 |
| Human-label reliability | 96 cases, 336 reviews, Fleiss kappa 0.3082 | Public aggregate summary; report Section 4 |
| Historical estimator baselines | 160-label comparison with simple sampling | Report Section 6; archived development tables |
| Repeated-review simulation | Noisy reviewer model and sampling/review-time tradeoffs | `analysis/phase8b_wrapper/`; `results/phase8b_population_robustness_20261112.json` |
| Equal-budget estimators | SRS, PPI++ mean, StratPPI mean, restricted MultiPPI and active AIPW | `analysis/phase11/`; `results/phase11_*` |
| Latency sensitivity | Historical Koala and Qwen component latency and assumed conversion rate | `analysis/phase12b/`; `results/phase12b_*` |
| Confidence interval analysis | Feasible-range clipping and empirical simulated coverage | `analysis/phase13/`; `results/phase13_*` |
| Allocation optimizer | Restricted convex optimization and primal-dual gap certificate | `analysis/phase14/`; `results/phase14_*` |
| Prospective reference study | Design and software tests only | `docs/prospective_study_design.md`; `analysis/phase15/` |

**Scientific boundaries:** historical benchmark labels and selected rater votes are not verified true harmfulness. The current active and multiple-prediction methods are restricted adaptations, and latency-to-human economic conversion is assumed. No new safety certificate or confirmed model superiority is established.
