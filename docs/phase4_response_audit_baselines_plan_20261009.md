# Main research direction — label-efficient measurement of missed harmful responses

**Project:** Trustworthy AI / Monitor Fusion · **Checkpoint:** 9 October 2026  
**Scientific priority:** Prof. Bader Rasheed's latest feedback, specifically: “The main path is to test whether smarter selection of human reviews and extra monitor checks can reliably measure missed harmful outputs at lower cost. First compare against strong existing methods.”

## Scientific scope and constraints

The unit is **a prompt–target-assistant-response exchange**, not an isolated unsafe-looking prompt. The study outcome is **response-level harm** under the frozen C1–C8 ontology in `configs/external_validation_common_ontology_v1.json` and target policy `configs/external_validation_target_policy_v1.json`. A refusal to an unsafe prompt is *Y=0*. Native prompt-moderation decisions from October Stage-A are not response-level labels.

Define `Y_i∈{0,1}` as harmful **response** outcome, `D_i∈{0,1}` as the decision of one **specified fixed deployed monitor policy** (`D=1` blocks), and `Z_i=Y_i(1−D_i)` as a missed harmful response. Choose the deployed policy and evaluation population explicitly **before** any comparison. On a finite population of `N` eligible exchanges:

- Primary recommended estimand: `p_miss = N⁻¹ Σ Z_i`, missed harmful responses per exchange.
- Secondary: `FNR = (Σ Z_i)/(Σ Y_i)` where the denominator is positive, and false-alarm rate `FPR = Σ (1−Y_i)D_i / Σ (1−Y_i)`.
- Distinguish this fixed finite development-pool quantity from a genuine prospective deployment or shift estimate. The existing July development set is not fresh confirmatory data.

**Data boundary:** The existing `configs/decision_value_real_data_protocol_v1.json` describes `N=1,687` *development-only* examples (`policy_train`, `policy_selection`, `calibration`), historically 291 response positives and 1,396 negatives. These should be treated as old, imperfect development labels, not new verified truth. Explicitly avoid inspecting any protected `final_test` or `held_out_shift` rows. The v2 contract permits only four previously produced development artifacts; the safe next script checks existence/metadata of those four paths, not the sealed mixed-split containers. It does not inspect records or export prompt texts.

## Strong comparator ladder — no novel method yet

| Comparator | Sampling/estimation mechanism | Role and caveat |
|---|---|---|
| A. Uniform probability audit | Simple random sampling of review units; estimator is the sample mean of `Z`. | Mandatory neutral baseline; design-based confidence intervals, account for grouping and finite population. |
| B. Stratified probability audit | Define strata from *already available* cheap monitor score and/or `D`, possibly family; random sample within each with explicit `N_h, n_h`; weighted stratum estimator. | Simple strong comparator; allocation via proportional/minimum-per-stratum versus train-only Neyman allocation, without oracle use of unreviewed labels. |
| C. Importance-weighted / Poisson active audit | Bias selection toward likely misses with strictly positive known inclusion probabilities `π_i`; use Horvitz–Thompson for totals or justified ratio/Hájek estimator. | Avoid naive unweighted mean of actively reviewed cases; record exact inclusion probabilities and sequential selection design. |
| D. Published **active-testing** strategies | Implement or reproduce a recognized sequential sample-efficient *evaluation*, not active *training*, strategy with honest weighting. | Existing method must be tried before claiming novelty; include selection-model training cost and uncertainty. |
| E. Rare-class active-importance sampling | Follow published low-shot validation estimator design for rare failures; compare its label economy to A–D. | Relevance to a small missed-harm stratum; assess conditions for estimator/variance applicability rather than transferring results blindly. |
| F. Selective additional-monitor acquisition | Optional expensive-monitor score acquired for selected cases only; include random sentinels if learning about common miss patterns requires all scores; evaluate *full cost* of monitor + human labeling. | Cost and missing-by-design auxiliary scores matter; **do not** select using optional scores that were never bought. |

**Prespecified lightweight feasibility pilot, only after inspecting permitted development schemas:** use repeated simulated audits of an *already labeled development-only* population, with budgets such as `40, 80, 160` human judgments (plus a clear extra-monitor cost schedule). Compare finite-population error, bias, empirical CI coverage and mean/tail cost across multiple deterministic seeds. Assess prevalence and FNR separately; FNR is a **ratio**, not an unbiased Horvitz–Thompson estimate. A tiny rare-event denominator can make FNR noisy or undefined. Group/cluster dependencies require group-aware sampling/uncertainty, not iid binomial assumptions. Do not use the fully known development labels to design sampling strata or tune strategies subsequently judged on that same development data; use nested partitions / fixed inexpensive scores and out-of-fold auxiliaries as appropriate.

## Literature to implement/compare first

1. **Kossen, Farquhar, Gal & Rainforth (ICML 2021)**, *Active Testing: Sample-Efficient Model Evaluation*. Sample-efficient test labeling with corrected acquisition bias; official [PMLR paper](https://proceedings.mlr.press/v139/kossen21a.html), [code](https://github.com/jlko/active-testing).
2. **Poms et al. (ICCV 2021)**, *Low-Shot Validation: Active Importance Sampling for Estimating Classifier Performance on Rare Categories*. Calibration plus importance sampling for rare-category performance; [open-access ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/html/Poms_Low-Shot_Validation_Active_Importance_Sampling_for_Estimating_Classifier_Performance_on_ICCV_2021_paper.html).
3. **Katariya, Iyer & Sarawagi (ICDM 2012)**, *Active Evaluation of Classifiers on Large Datasets*. Iterative stratified sampling for label-efficient accuracy estimates; [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/active-evaluation-classifiers-large-datasets/).
4. **Yilmaz et al. (2021 preprint)**, *Sample Efficient Model Evaluation*. Importance versus Poisson sampling and confidence intervals; [arXiv 2109.12043](https://arxiv.org/abs/2109.12043). Verify publication history before calling it peer-reviewed.

These methods **do not establish superiority in our safety application**. Their objectives, accessible covariates, rare-event assumptions, estimator bias and costs need to be mapped to response-level `Y, D, Z` before comparison.

## Locked stop/go ordering

1. **Completed:** October 7 original four-guard exploratory kill test, artifact restoration (all evidence), October 8 Phase 2, Phase 3A/B cheap controls, Phase 3C new text/lexical pilot, researcher signoff on 38 prompts plus 304 guard-specific and 38 O2-equivalence AI-assisted draft judgments. Original readout-transfer screen **0/4 failed**, unchanged. Phase 3C labels retain **18 uncertain**; no independent adjudicators.
2. **Immediate next action:** run `phase4_response_audit_readiness.py` in the existing WSL branch. It checks schema/availability **only for the permitted four development-only artifacts** and automatically copies a small JSON to Windows Downloads. No Kaggle or data collection.
3. **Next analysis (not yet run):** on an available, explicitly defined response-level development population, compare A–E with matched human-label budgets and design-corrected estimates. Choose `D` before evaluation and ensure that it was actually computed on response/prompt-response inputs, not prompt-only Stage-A scores.
4. **Optional backup after primary pilot:** cheap native O2 inverse/known-pair normalization and strict same-case benign/harmful rates. A replay of cached native outputs for the exact decoded source **is not** a general operational decoder. Only if residual native-policy failures survive simple controls and valid labels should we run small new GPU scoring of Phase 3C prompts; no new adapter/theory now.
5. **Supervisor report:** summarize *completed* correction evidence, negative screen and remaining uncertainty, results of strong literature-backed audit baselines, any meaningful unresolved problem and exact proposed next pilot; solicit approval before expanded prospective data collection and confirmatory claims.

**Not authorized yet:** fresh W0 confirmatory sample; prospectively changed attack/source collection; post-hoc modification of old screening rules; deployment safety guarantee or 5% FNR/FPR certificate.
