# Reliable and Cost-Aware Evaluation of Runtime Safety Monitors for Large Language Models

**Research progress update | 19 October 2026**

## 1. Question and scope

We propose to measure the frequency of harmful generated assistant responses that a fixed runtime monitor would allow through: **miss prevalence**, `A = P(Y=1, D=0)`, where `Y=1` denotes a harmful *assistant response* under a fixed response-level policy, and `D=1` denotes a block. The monitor is proposed to act after generation and before delivery. The secondary estimand is `FNR = P(D=0 | Y=1) = A/(A+B)`, where `B = P(Y=1,D=1)` denotes harmful responses already blocked. These quantities answer different questions.

All results below concern **1,687 historical development prompt-response exchanges** (291 historically labeled harmful) under two archived cross-fitted classifier setups. Here `D` is **not a native deployed guard decision**, and the data do not measure real-world user exposure. No protected old final-test or shift split was reopened. The proposed monitor timing and primary estimand need agreement before a prospective pilot.

## 2. Results from the existing data

**Explaining Qwen.** Phase 5A replayed the Phase 4D sampling procedure and reproduced its recorded errors. At 320 simulated Qwen score acquisitions and 160 human-label purchases, the paired comparison of real versus within-decision-group shuffled scores gave:

| Estimand | Shuffled-score RMSE | Real-score RMSE | Reduction |
|---|---:|---:|---:|
| Allowed harmful response prevalence, A | 0.02574 | 0.02517 | 2.2% |
| Blocked harmful response prevalence, B | 0.01019 | 0.00535 | 47.5% |
| FNR = A/(A+B) | 0.06054 | 0.04210 | 30.5% |

Thus the FNR gain predominantly reflects more accurate estimation of **harmful blocked responses**, which affect its denominator. It does not establish a comparable improvement in missed-harm prevalence, our proposed primary outcome. The available intervals describe variation across Monte Carlo sampling repetitions, not generalization to new deployments.

**Blocked/allowed allocation.** With 160 simulated reviews in the Qwen historical setup, ordinary proportional sampling obtained A RMSE **0.0262**, versus **0.0291** with a fixed 70%-allowed/30%-blocked allocation. For FNR, the same two methods obtained RMSE **0.0631** and **0.0472**, respectively. Review allocation therefore changes the relative precision of these two estimands. An allowed-only audit remains essential when only A is targeted.

**Published-method comparisons.** Phase 5B2 implemented a **one-dimensional mean-estimation specialization** of StratPPI using already available cheap scores, charging calibration labels to the human budget. At 160 reviews on the Qwen setup, fixed-weight StratPPI with a 70%-allowed allocation yielded FNR RMSE **0.0470**, compared with **0.0472** for simple sampling. Allowed-only random auditing obtained A RMSE **0.0237**; its FNR is not estimable without blocked-group information. Some low-budget PPI ratio estimates fell outside [0,1] before clipping, warning against assuming stable finite-sample FNR inference.

Phase 5B3 tested a **restricted two-predictor adaptation**, not the full published Active MultiPPI optimization or confidence-interval method. Across its **16** cost/human-budget configurations, simple blocked/allowed sampling had lower FNR RMSE in all comparisons. The restricted adaptation exceeded its specified expected total budget in approximately **49.6%** of repetitions, and optional-call settings were **hints rather than hard caps**. It used hypothetical 30- and 120-second human review costs and historical average inference times; the results are **not** a definitive comparison with the published AM-PPI method.

The older readout-transfer screen still passed **0/4 guards** and the failed routing results remain negative findings. We have not developed or validated a new router or demonstrated a reduction in harmful content reaching users.

## 3. Reproducibility and validity

The development source parquet, fold assignments and two frozen protocol files all matched their recorded SHA-256 hashes; local Git HEAD was `fa6762f2beb6ff04eb9afd6f5ef1bd56777ea13a` at the time of the checks. The four reviewed protocol/synthetic test modules passed **38/38 tests**. Safe checks also verified the development group/fold and cross-fitted decision-value identities. Three historical test modules remain **excluded**, not passed, because they invoke readers of protected mixed-split datasets or restricted historical artifacts. Replacing the relevant assertions with development-only checks does not retroactively validate those historical tests.

Original prompt-response text is present for all 1,687 authorized development records: WildGuardTest (1,367), XSTest safe GPT-4 (200) and JailbreakBench judge comparison (120). Their recorded license fields do **not** establish permission to distribute response text to outside reviewers. Historical Y labels have **not** been independently checked. The existing local blinded-review software (three reviewers plus fourth-rater adjudication/audit, uncertain-case tracking and time measurement) has passed synthetic tests only; it is not a completed human-review study. Institutional/ethics determination, data rights, consent, secure processing and independent reviewers remain necessary before use.

Full costs have **not** been established. Earlier calculations charge the historical mean optional-Qwen time of about 1.598 seconds per call (~511 seconds for 320 calls), but have not measured selected-case wall-clock inference, independent review, reviewer disagreements/adjudication or nonrating overhead. Calibration labels were counted in Phase 5B2/5B3; real human-review time was not.

## 4. Direction and next decisions

The current evidence supports a **cost-aware empirical study of response-level safety auditing, including negative findings when straightforward methods match or outperform more elaborate methods**. We do not currently claim a new monitoring algorithm, a statistical safety certificate, or superiority over fully implemented StratPPI/AM-PPI. A rare-event inference contribution or label-validity study could become more appropriate if properly measured independent labels and strict matched-budget comparisons reveal that problem.

The immediate work is to obtain guidance on independent review and data use; then independently review a justified, small development sample and measure review/adjudication time. In parallel, determine whether a faithful, strictly cost-matched AM-PPI benchmark is feasible and useful, including training/calibration and purchased model scores. Only if a significant question remains should a separate **prospective feasibility pilot** be designed with one frozen native pre-delivery monitor, fresh allowed and blocked responses, blinded reviewers, actual costs and a predeclared sample-size calculation. It would **not** certify safety.

**Decisions requested:** agreement on (i) miss prevalence as the primary outcome and the post-generation/pre-delivery monitor definition; (ii) the proposed empirical/negative-results direction; and (iii) the independent-review/ethics/data-access procedure and whether a more faithful AM-PPI implementation must precede the prospective pilot.

## References and evidence

- Kossen et al., *Active Testing: Sample-Efficient Model Evaluation*, ICML 2021: https://proceedings.mlr.press/v139/kossen21a.html
- Fisch et al., *Stratified Prediction-Powered Inference*, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract.html
- Brawand et al., *Active Multiple-Prediction-Powered Inference*, 2026: https://arxiv.org/abs/2605.08429
- Frozen Stage-A / Phase 3–4 release: https://github.com/tivdzualubem/trustworthy-ai-monitor-fusion/tree/stage-a-paired-transport-20260921
- Phase 5A/5B1/5B2/5B3 exported aggregates, paired errors, manuscripts and local provenance/test archives in the accompanying release package. No sealed example-level records are included.