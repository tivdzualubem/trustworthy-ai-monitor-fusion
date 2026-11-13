# Statistical methods and limitations

The primary parameter is A = P(Y=1, D=0), with the monitor's decision frozen after assistant generation and before delivery. Sources of uncertainty include finite sampling, reviewer misclassification, abstentions, heterogeneous response categories, covariate or source dependence, and estimated cost ratios.

**Estimator comparison:** Whole-population and allowed-only simple random sampling are retained as controls. PPI++ and StratPPI use mean-estimation specializations with charged pilot labels. Restricted MultiPPI minimizes a covariance-information criterion over seven nonempty observation subsets, using a checked first-order gap certificate. Restricted active AIPW uses with-replacement sampling and is not the full per-instance Bernoulli-and-KKT AM-PPI method. These are controlled software experiments, not benchmark replications of every author method.

**Reference outcomes:** Three primary reviewers each labeled 96 cases; a fourth reviewed 48 selected cases. Reviewer disagreement and uncertain outcomes prevent treating a single vote, majority or fourth opinion as verified Y. The selected R4 cases do not form an independent probability sample for reference truth. Bounds for Y would require externally justified rater-accuracy assumptions or additional reference validation.

**Costs:** The human-review distribution comes from 336 active time observations. Historical Koala and Qwen latencies are separately measured. Economic conversion between GPU and human seconds is a sensitivity parameter, and recruitment, adjudication and organizer overhead are not independently priced.

**Inference:** Support-clipping is applied only to intervals for a fixed, identified endpoint; it cannot correct unknown label bias. Narrow exploratory Wald intervals undercovered in several synthetic settings. No fresh native-monitor deployment population has been evaluated.

## Related work

- Fisch et al., *Stratified Prediction-Powered Inference*, NeurIPS 2024. https://arxiv.org/abs/2406.04291
- Cowen-Breen et al., *Multiple-Prediction-Powered Inference*, ICLR 2026. https://arxiv.org/abs/2603.27414
- Brawand et al., *Active Multiple-Prediction-Powered Inference*, 2026. https://arxiv.org/abs/2605.08429
- Feng et al., *Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges*, ICLR 2026. https://arxiv.org/abs/2601.20913
- Schirmer et al., *Online Safety Monitoring for LLMs*, 2026. https://arxiv.org/abs/2607.02510
