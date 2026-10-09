# Trustworthy AI — Phase 4D Evidence and Supervisor Alignment Checkpoint

**Date:** 9 October 2026  
**Purpose:** Research evidence checkpoint, **not** a final paper, formal guarantee, approved protocol amendment, or a ready-to-send claim of deployment improvement.  
**Main research direction (Prof. Bader Rasheed):** Determine whether better selection of human reviews and additional monitor checks can measure missed *harmful assistant responses* at lower cost; start with strong existing methods.

## 1. What was checked

- Primary Phase 4D archive: `phase4d_optional_score_falsification_v1.zip` (five files). The CSV contains 32 aggregate rows, 48 paired RMSE contrasts and 12,800 per-repetition scalar-error records (400 repetitions in each of 32 arms/conditions).
- Recalculated all aggregate bias and RMSE fields directly from scalar errors and all 48 RMSE contrasts: **matched**. Reproduced selected Monte Carlo bootstrap intervals using the script's documented random seeds: **matched**.
- Earlier uploaded Phase 4A, 4B, 4C and Phase 3C archives were inspected for consistency in task definitions and reported scope. This is a check of delivered results and code, **not** an independent end-to-end rerun on the user's WSL data or an audit of original response labels.
- Phase 4D input/protocol hashes are recorded in its manifest. The actual source parquet and fold CSV were not uploaded here and therefore their bytes could not be independently rehashed in this workspace.

## 2. Precisely defined target

For historical response-level labels `Y=1` (harmful *assistant response*) and historical cross-fitted classifier output `D=1` (blocked/flagged):

- **FNR** = `P(D=0 | Y=1)` — proportion of harmful responses missed.
- **Missed-harm prevalence** = `P(Y=1, D=0)` — missed harmful responses per traffic example. **Not the same denominator.**

Across **1,687** historical development examples per setup, the archived `base_prediction` is a cross-fitted classifier decision, **not the native deployed guard decision**. The full development reference values are:

| Historical setup | FNR | Missed-harm prevalence |
|---|---:|---:|
| Compact-after-rule | 0.78694 | 0.13574 |
| Qwen-after-rule/compact | 0.78351 | 0.13515 |

High values here **are not measured live-deployment failure rates**, safety certificates, or prompt-classification scores.

## 3. Phase 4D design

Four arms were compared with matched human-review budgets (80/160), optional-monitor budgets (80/320), and 400 Monte Carlo repetitions per design cell:

1. **Cheap only** — no optional monitor calls, cheap-score/decision stratification.
2. **Acquisition indicator only** — same 80% cheap-uncertainty-priority + 20% random optional-call allocation as the purchased-score arms, but stratify without the scores.
3. **Permuted purchased scores** — shuffle the scores among acquired examples **within decision group**. Preserves that group's purchased-score distribution, destroys original case-score matching; does **not** control for every cheap-score correlation.
4. **Real purchased scores** — use only the optional scores whose acquisition cost was simulated as paid.

All FNR estimates are ratios of stratum-weighted totals. Their ratios are **not exactly unbiased**, even if the component totals have design-unbiased stratified estimators. The reported bootstrap intervals describe Monte Carlo variation in simulated RMSE differences, **not uncertainty under deployment or distribution shift**.

## 4. Main finding: Qwen helps FNR, especially at higher check budget

**FNR RMSE** (lower is better):

| Setup | Optional calls | Human reviews | Cheap only (0 calls) | Acquisition only | Permuted score | Real score |
|---|---:|---:|---:|---:|---:|---:|
| Compact | 80 | 160 | 0.0608 | 0.0598 | 0.0582 | 0.0593 |
| Compact | 320 | 160 | 0.0612 | 0.0576 | 0.0598 | 0.0540 |
| Qwen | 80 | 80 | 0.0843 | 0.0856 | 0.0863 | **0.0743** |
| Qwen | 80 | 160 | 0.0593 | 0.0632 | 0.0606 | **0.0549** |
| Qwen | 320 | 80 | 0.0779 | 0.0940 | 0.0870 | **0.0596** |
| **Qwen** | **320** | **160** | **0.0627** | **0.0621** | **0.0605** | **0.0421** |

At **320 Qwen calls / 160 human reviews**, real scores outperform shuffled scores by **0.01844 RMSE** (30.5% lower relative RMSE). The paired 95% **Monte Carlo** bootstrap interval on **real minus shuffled** is **[-0.02402, -0.01310]**. Real Qwen scores also outperform acquisition-only (paired difference -0.02003; interval [-0.02471, -0.01532]) and cheap-only (-0.02059; [-0.02591, -0.01538]).

All four Qwen configurations have negative real-versus-permuted FNR RMSE differences with Monte Carlo intervals below zero. The compact monitor shows a smaller/inconsistent effect (only its 320-call/160-review real-versus-permuted interval excludes zero: approximately [-0.01058, -0.00092]).

**The Qwen result is for FNR, not both response-level estimands.** For missed-harm prevalence at 320 Qwen calls/160 reviews, real vs cheap-only RMSE difference is -0.00223 with MC interval **[-0.00461, +0.00022]**, which includes zero. Avoid proclaiming a general risk-estimation win.

## 5. Costs and human-review trade-off remain unsettled

| Purchased checks | Compact historical aggregate monitor time | Qwen historical aggregate monitor time |
|---|---:|---:|
| 80 | 3.65 s | **127.80 s** |
| 320 | 14.61 s | **511.22 s (8.52 min)** |

Phase 4D uses frozen *historical mean latency per optional call*, **not newly measured concurrent wall-clock time**, and does not price human annotators, reviewer disagreements, or processing overhead.

An exploratory cross-budget comparison at 320 Qwen calls gives FNR RMSE **0.05964** with **80 human reviews** versus **0.06269** for no extra monitor with **160 human reviews**; the paired Monte Carlo interval on the RMSE difference is approximately **[-0.0114, +0.0056]**, so parity/improvement is uncertain. Ignoring all overhead, 511 seconds / 80 potentially avoided human reviews = **6.39 seconds per saved review** as a mathematical break-even threshold; **not an actual demonstrated cost saving**. Real human annotation generally has other variable costs, but none are measured here.

## 6. Phase 3 / supervisor issue checklist

| Supervisor item | Status and boundary |
|---|---|
| 1. Easy topic/lexical controls | **Addressed diagnostically:** word/char within-representation baseline accuracy 100% on original easy controls; with 38 researcher-approved similar-topic benign controls drops to approximately **67.1%/72.4%**. **New native guard/hidden-probe evaluation on hard controls not run.** |
| 2. Native decision vs trained probe vs cause | **Distinguished explicitly.** No causal explanation or proof of native understanding. |
| 3. Source-probe cutoff / mean shift | **CPU diagnostic completed.** Held-out O2 balanced accuracy after scalar cutoff: Llama 64.47%, Shield 78.95%, Granite 94.74%, Qwen 93.42%. Mean shift on a linear probe is just an offset. Original failed screen unmodified. |
| 4. Harmful recall *and* benign alarms; decoding/normalization | The original O3 both-class interception failure is recorded. **Actual new native-guard decoding/normalization forward-pass check remains unperformed**; this belongs to the backup path and should not automatically prompt a GPU rerun. |
| 5. Response/prompt task, policy, semantic validity | **Distinguished.** Hard benign text and AI-assisted policy/O2 judgments approved by **one researcher**; 18 policy uncertainties retained. **No independent multiple-rater agreement** and no independent response-level adjudication for the historical Phase 4 Y. |
| 6. Missing representations, scores, examples | Original frozen artifacts restored to repo on Oct 8; prespecified Stage-A **0/4 screen remains failed**. New Phase 3/4 outputs are separate exploratory packages; do not say these are pushed without verifying. |

## 7. Decision and remaining work

**Proceed no further with repeated post-hoc acquisition-method modifications on these same development labels.** The current Qwen-specific result warrants discussion, but not a new method, formal guarantee, or deployment claim.

To close the professor's **main** research request rigorously, seek his agreement on a small prospective response-level study, including:

- Frozen **native prompt–response monitor decision D**, target-assistant response policy and outcome Y, reference population/traffic sources, disjoint selection/evaluation/confirmation data, and independent annotation/adjudication plan.
- Strong simple sampling comparators (uniform/stratified/allowed-only where its estimand permits) and a published active-testing method; a very small optional-Qwen acquisition arm if cost and reviewer budget justify it.
- Predeclared *primary estimand* (prefer explicit FNR and/or per-traffic miss prevalence), inferential intervals and failure/stop gates, positive sampling probabilities, compute + human cost (separately and jointly), meaningful paired budgets, and holdout uncertainty adequate for a generalization claim.
- For the backup representation story, consider simple deterministic decoding/normalization and native-score threshold controls **only after** a targeted question and agreed budget; the strong primary research direction is measurement, not probe repair.

**Suggested supervisor message (for later review, not sent):**

> We completed the requested lexical-shortcut and source-probe cutoff diagnostics, retained the original 0/4 failed screen, and conducted development-only comparisons of random/stratified auditing, Active Testing, and optional-monitor-assisted audits. Hard same-topic benign controls substantially weaken lexical separation. Active Testing did not consistently improve FNR estimation over proportional stratification. In a paired score-shuffling ablation, purchased Qwen scores improved FNR estimation beyond acquisition-only and score-shuffling controls on the historical development data, but the cost-effectiveness, independent response labels, and native guard/generalization questions remain unresolved. We propose a small frozen response-level pilot with independently reviewed outcomes and matched human/compute budgets before making a paper claim. Could we agree on that prospective design?

### Evidence references

- Uploaded: `phase4d_optional_score_falsification_v1.zip`, especially `paired_scalar_errors.csv`, `ablation_summary.csv`, `paired_rmse_contrasts.csv`, `manifest.json`.
- Uploaded: `phase4_dev_response_audit_sampling_v1.zip`, `phase4b_active_testing_lure_v1.zip`, `phase4c_monitor_assisted_audit_v1.zip`.
- Uploaded: `stage_a_phase3_results.zip`, `stage_a_phase3c_researcher_approved_20261009.zip`, `stage_a_phase3c_guard_policy_researcher_approved_20261009.zip`.
- Frozen GitHub branch: <https://github.com/tivdzualubem/trustworthy-ai-monitor-fusion/tree/stage-a-paired-transport-20260921>.
- Original research supervisor message: attached `EDITS(5).txt` (9 October conversation), six issues and preferred primary/backup paths.
