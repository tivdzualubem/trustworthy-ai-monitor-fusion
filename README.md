# Trustworthy AI — Runtime Safety-Monitor Cascades

This repository studies **runtime safety-monitor cascades for prompt–response systems**. The original project investigated whether inexpensive safety monitors could selectively invoke a stronger monitor while maintaining harmful-response recall, controlling false positives, and reducing runtime cost.

The current evidence is treated as a **development-only evaluation/measurement pilot**. The stronger next research question is whether finite-sample safety/risk certificates transport across source or attack-family shift, especially with dependent or template-generated examples. A narrow evaluation/measurement paper remains a fallback only if the main pilot phenomena replicate on genuinely different monitor families and genuinely fresh data.

## Project status

The earlier v2 router study is a **no-go on the existing development data**. No signed-value router family had a jointly feasible development operating point, and no router was selected.

The historical v2 results remain in the repository for auditability, with the following interpretation:

- reported v2 costs are **summed measured component latencies**, not direct wall-clock end-to-end (E2E) policy latency;
- the historical development split had **1,687 effective groups for 1,687 examples**, which was effectively one group per example and did not protect against template or near-duplicate dependence;
- the 35 s quantity used in the bounded-cost analysis was **not a demonstrated mechanically enforced runtime bound**;
- the historical `max(1 ms, 1% of budget)` equivalence margin was a prespecified engineering choice and was **not externally justified** as a confirmatory equivalence margin;
- the historical final v2 CSV snapshot is **not fully regenerable** from the committed repository because complete downstream timing provenance was not preserved; the available historical evidence and its audit entry point are retained;
- fresh calibration, joint FPR+cost certification, fresh source/time-separated confirmation, and **multi-rater labels** were not available;
- `final_test` and `held_out_shift` were **historically evaluated and included in the label audit**. They were not used in the development-only pilot, but neither split is eligible as fresh confirmatory data.

These limitations define the claim boundary for the historical router analysis.

## Monitor system

The project uses three monitor levels:

- a deterministic rule-based filter;
- `KoalaAI/Text-Moderation` as the compact moderation model;
- `Qwen/Qwen3Guard-Gen-4B` as the stronger optional monitor.

The prediction unit is a **prompt–response pair**. A positive example is a response that should be intercepted because, in the context of the prompt, it materially provides, facilitates, endorses, or contains disallowed harmful content. Safe refusals, benign answers, and safe redirections are negative examples.

## Evaluation questions

The evaluation/measurement study focuses on six issues that can affect conclusions about selective safety routing:

1. **Component-sum cost vs direct E2E cost**
2. **Offline ranked routing vs deployable threshold routing**
3. **Singleton grouping vs dependency-aware grouping**
4. **Empirical FPR screening vs finite-sample risk control**
5. **Original labels vs audited harmful-response labels**
6. **Iso-cost screening vs Pareto-style comparison**

The final item is important methodologically. An iso-cost question requires a justified equivalence margin. A **Pareto comparison** instead asks whether a policy has higher recall, non-higher cost, and acceptable FPR risk; a policy should not be rejected simply because two-sided cost equivalence fails when it is also cheaper.

## Small multi-stack / multi-dataset pilot

A frozen development-only pilot was run across three monitor stacks:

1. `rule -> compact`
2. `compact -> Qwen3Guard`
3. `rule + compact -> Qwen3Guard`

and three data sources:

- JailbreakBench judge-comparison
- WildGuardTest
- XSTest-safe

The authorized development view contains **1,687 prompt–response examples**.

The label-blind, dependency-closed T4 timing sample contains **363 examples**:

- 120 JailbreakBench
- 123 WildGuardTest
- 120 XSTest-safe

### Main findings

| Evaluation comparison | Changed conclusions |
|---|---:|
| Empirical FPR vs finite-sample risk gate | 36 / 75 |
| Grouping full-protocol contrast (confounded) | 19 / 75 |
| Label full-protocol contrast (confounded) | 14 / 75 |
| Ranked vs deployable routing | 4 / 45 |
| Component-sum vs direct E2E mean-cost ordering | 0 / 105 |

### FPR treatment

Forty of 75 deployable policy evaluations have empirical FPR at or below 0.05, but only four pass the frozen finite-sample row-and-dependency risk gate. This changes **36 of 75** policy conclusions.

### Grouping and label contrasts

The earlier **19/75 grouping** and **14/75 label** numbers remain full-protocol contrasts, not isolated grouping or label effects. The completed factorial decomposition separates measurement-only changes from retraining/reselection changes on a common outer holdout; the results are summarized below.

### Ranked vs deployable routing

Offline ranked acquisition and a reusable threshold are not operationally equivalent. Risk pass/fail changes in **4 of 45** matched comparisons, and the realized acquisition rate differs in **45 of 45** comparisons.

### Direct E2E vs component-sum cost

Direct wall-clock E2E latency is slightly higher than the same-run component sum for every one of the 15 timed policies.

However, none of the **105 pairwise mean-cost orderings** reverses.

This is a useful negative result: direct E2E remains the correct cost estimand, but correcting the measurement does not change policy ordering in this pilot.


## Factorial measurement decomposition

The 2×2 decomposition varies audited versus original labels and dependency-aware versus singleton grouping at three levels:

| Decomposition layer | Grouping contrast | Label contrast |
|---|---:|---:|
| Fixed-policy measurement | 0/75 | 1/75 |
| Retraining/reselection with fixed outer holdout | 6/75 | 4/75 |
| Full protocol | 19/75 | 14/75 |

The **fixed-policy measurement** layer freezes the evaluation rows, fitted models, thresholds, routes, and predictions and changes only the evaluation label or risk grouping. The **retraining/reselection** layer freezes a dependency-closed outer holdout and always measures it with audited labels and dependency-aware risk grouping while allowing the 2×2 conditions to change training and selection. The **full-protocol** layer retains the original condition-specific folds, training data, models, thresholds, labels, and holdouts.

The 75-policy aggregate counts are descriptive across the five prespecified seeds; they are **not pooled inferential observations**. Seed-specific contrasts and factorial interaction diagnostics are retained under `reports/factorial_measurement_decomposition_v1/`.

## Numerical route stability

The numerical route-stability study is complete on the development-only evidence. The archived taxonomy is **two route-threshold mismatches and five prediction mismatches**: two prediction changes are downstream of the route flips and three are pure decision-threshold mismatches. The original mismatches sit at frozen boundaries: the reference route margins for the two routing cases are exactly zero, while the three pure decision cases are at or effectively at their frozen decision thresholds.

Across the full archived 5,445 policy-call population, float64 Python/NumPy/PyTorch CPU arithmetic is effectively invariant. Float32 arithmetic is materially more sensitive near frozen boundaries and produces route/prediction flips in the diagnostic recomputation, so exact bit-level parity is not an appropriate invariance criterion.

In the controlled hardware study, the same float32 PyTorch policy arithmetic produced **0/5 route flips** and **0/5 prediction flips** between CPU and Tesla T4. The compact monitor was deterministic within each repeated hardware/precision cell, with maximum CPU-versus-T4 float32 score difference **7.363e-09**. T4 float16 was exploratory and is not a supported path: it changed **2/5 routes** and **5/5 predictions** relative to T4 float32.

The dead-band analysis uses the prespecified epsilon grid rather than choosing a tolerance to erase mismatches. The first prespecified grid point covering the conservative non-exploratory perturbation envelope is **1e-06**. At that reporting point, **6.12%** of archived selective-route evaluations and **0.28%** of archived decision evaluations are numerically ambiguous. This is an invariance/measurement result, not a deployment rule: no accept/defer/reject action is assigned here.

The threshold-tie diagnostic resolves the previously reported route dead-band on the archived development-only population. At the frozen `1e-6` reporting epsilon, 200/3267 route evaluations (6.12%) are ambiguous, and all 200 already have exactly zero canonical reference route margin; the epsilon adds 0 non-tied route rows. Among the exact-tie rows, 197 share an exactly repeated raw cheap-monitor input state and 197 lie on an exactly repeated routing-distance mass point. At runtime, 198 remain exact ties and 2 reference ties move off exact equality. Both 2 archived route flips occur at exact reference ties (2/2), with 0 observed non-tied route crossings. This is an archived boundary/mass-point diagnostic, not evidence of a general floating-point safety failure. The separate controlled CPU–T4 float32 study remains 0/5 route flips and 0/5 prediction flips.

`final_test` and `held_out_shift` were not used, and no fresh-confirmatory, router-superiority, Pareto, universal-hardware-invariance, or production claim is made.

## Safety–availability control-plane kill study

A small development-only novelty-kill study tests both **routing/bypass pressure** and **load/escalation pressure** for the existing selective cascades. It uses the prespecified 0.25 selective policy from each stack, with `rule_compact_to_qwen` primary and the other two stacks as sensitivities. No model is retrained and no router is rescued.

The routing study is a synthetic control-plane probability stress test, not a realizable adversarial-attack claim. It incorporates the completed numerical-stability dead-band (`1e-6`) as fail-closed routing ambiguity: ambiguous requests escalate rather than silently bypass.

The load study compares three behaviors:

- **fail-open budget cap** — bounds expensive work but can bypass the stronger monitor under overflow;
- **fail-closed without admission control** — preserves the expensive-check requirement but accumulates backlog when escalated arrival exceeds expensive-stage capacity;
- **fail-closed with defer/reject** — preserves the expensive-check requirement for ordinary responses while bounding expensive-stage load by making overflow explicit.

The formal result is a necessary-condition result rather than a novel security theorem: if all accepted requests can be forced onto an expensive stage with cost `c_e`, serving all `N` requests requires `B >= N*c_e`; and if escalated arrival exceeds expensive service capacity, accept-all fail-closed service has positive backlog drift. Unless the system is provisioned for the all-expensive worst case, an explicit admission/defer/reject action is therefore necessary to preserve a hard resource bound without silently failing open.

The internal kill criterion did not produce the prespecified nontrivial dead-band-scale bypass signal across enough stacks. The overload result reduces to the generic finite-capacity/admission-control condition, so this study does not support a standalone security-paper direction on its own.

`final_test` and `held_out_shift` are not used. The study makes no fresh-confirmatory, router-superiority, Pareto, production, or external literature-novelty claim.

## Direct E2E timing

Timing was performed on a single NVIDIA Tesla T4 with:

- batch size 1;
- 20 untimed warm-up requests;
- CUDA synchronization;
- model loading excluded from timing;
- no runtime cap;
- no post-hoc clipping;
- 15 deployable policies;
- 363 timing examples;
- **5,445 raw policy calls**.

Two runtime route mismatches were observed. The frozen protocol required exact route parity before joining GPU cost evidence with the frozen CPU recall/risk evidence, so:

```text
cpu_cost_join_valid=false
pareto_claim_available=false
```

No Pareto-dominance conclusion is reported from this pilot.

## Current claim boundary

The repository currently supports an **evaluation/measurement case study**, not a validated router-superiority claim.

The current evidence does **not** establish:

- router superiority;
- Pareto dominance for the pilot;
- fresh joint FPR+cost certification;
- fresh source/time-separated confirmation;
- multi-rater confirmation;
- arbitrary distribution-shift robustness;
- correlated-failure detection;
- production readiness.

The finite-sample risk bounds used in the pilot are internal development safeguards, not external deployment certificates.

The current grouping/label and numerical phenomena have not yet been replicated on genuinely different monitor families and genuinely fresh data. The present evaluation/measurement evidence is therefore a pilot, not a completed general publication result.

## Reproducibility

The corrected evaluation/measurement workflow has committed entry points and hash-recorded artifacts.

Key commands are:

```bash
bash scripts/reproduce_historical_v2_evidence.sh
python scripts/rebuild_evaluation_measurement_dependency_groups.py --check
python scripts/run_evaluation_measurement_pilot_v1_cpu.py
python scripts/analyze_evaluation_measurement_pilot_v1.py
python scripts/run_factorial_measurement_decomposition.py
python scripts/verify_reproducibility.py --strict-hashes
```

The T4 direct-E2E benchmark and its package builder are also committed and hash-recorded. Generated Kaggle upload bundles are kept outside Git and identified by SHA-256.

To verify the repository:

```bash
source .venv/bin/activate
python scripts/verify_reproducibility.py --strict-hashes
pytest -q
```

## Repository structure

- `configs/` — frozen experiment protocols.
- `data/metadata/` — provenance, grouping, model, timing, and reproducibility manifests.
- `data/processed/` — authorized development views and monitor-score caches.
- `src/monitor_fusion/` — reusable monitor and evaluation implementation.
- `scripts/` — canonical reproduction, timing, analysis, and verification entry points.
- `reports/evaluation_measurement_pilot_v1/` — CPU results, T4 timing evidence, and combined pilot analysis.
- `reports/exact_cost_risk_cascade_v2/` — historical v2 evidence retained for auditability.
- `paper/` — current report and historical report artifacts.
- `tests/` — methodological and reproducibility checks.

## Current report

Current PDF: `paper/Evaluation_Measurement_Runtime_Safety_Monitor_Cascades.pdf`

Current source: `paper/evaluation_measurement_current/main.tex`

Build command:

```bash
bash paper/evaluation_measurement_current/build.sh
```

The current report integrates the corrected legacy-split provenance, the factorial grouping/label decomposition, the numerical route-stability analysis, and the small safety-availability/control-plane kill study. It explicitly treats the existing evidence as a development-only pilot and does not claim router superiority, fresh confirmation, a general numerical safety failure, or a standalone security-paper result.

The completed threshold-tie diagnostic shows that all 200 rows in the 6.12% route dead-band at epsilon=1e-6 are exact canonical threshold ties; 197/200 occur on repeated/discrete cheap-state mass points, and the two archived route mismatches also occur at exact ties.

## Current work

The threshold-tie diagnostic is complete. **Stop using the existing data for discovery**: the completed project data remain closed to further hypothesis selection, source/attack selection, monitor selection, threshold selection, or router rescue. They remain available only for reproducibility, audit, overlap exclusion, and reporting of completed pilot analyses.

The current prospective study is **external validation of frozen safety-monitor operating points under source and attack-family shift**. **False-negative rate (FNR)** is the primary safety outcome because it measures missed harmful responses. FPR is treated separately as an **overblocking** outcome rather than as a safety certificate.

The approved design uses a contemporaneous **2 x 2** source-by-attack factorial in **W1**, including a human/direct **temporal anchor**, plus independent human/direct W0 optimization and baseline-validation sets. It specifies independent multi-rater labels, common-ontology mapping, provenance-aware clustering, deterministic threshold selection, explicit sampling/stopping rules, FNR power planning, the 5% individual-cell FPR operating constraint, and multiplicity control.

**W0 collection has not started.** Before W0 begins, the exact **precollection registries** and analysis implementation must be frozen: source-generation rules, attack-family transformations, collection windows, target-assistant contract, common ontology, annotation instructions, dependency/provenance schema, monitor revisions and inference contracts, sampling quotas, and analysis code. **Confirmation-domain retuning is not permitted.**

Next action: freeze the precollection registries and analysis implementation.
