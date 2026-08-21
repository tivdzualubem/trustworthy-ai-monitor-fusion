# Trustworthy AI — Runtime Safety-Monitor Cascades

This repository studies **runtime safety-monitor cascades for prompt–response systems**. The original project investigated whether inexpensive safety monitors could selectively invoke a stronger monitor while maintaining harmful-response recall, controlling false positives, and reducing runtime cost.

The current work uses that system as a case study in **evaluation and measurement**. The main question is whether common evaluation choices can change the apparent conclusion of a selective safety-monitor experiment.

## Project status

The earlier v2 router study is a **no-go on the existing development data**. No signed-value router family had a jointly feasible development operating point, and no router was selected.

The historical v2 results remain in the repository for auditability, with the following interpretation:

- reported v2 costs are **summed measured component latencies**, not direct wall-clock end-to-end (E2E) policy latency;
- the historical development split had **1,687 effective groups for 1,687 examples**, which was effectively one group per example and did not protect against template or near-duplicate dependence;
- the 35 s quantity used in the bounded-cost analysis was **not a demonstrated mechanically enforced runtime bound**;
- the historical `max(1 ms, 1% of budget)` equivalence margin was a prespecified engineering choice and was **not externally justified** as a confirmatory equivalence margin;
- the historical final v2 CSV snapshot is **not fully regenerable** from the committed repository because complete downstream timing provenance was not preserved; the available historical evidence and its audit entry point are retained;
- fresh calibration, joint FPR+cost certification, fresh source/time-separated confirmation, and **multi-rater labels** were not available;
- the **protected legacy `final_test`** and `held_out_shift` partitions remain sealed.

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
| Singleton vs dependency-aware grouping | 19 / 75 |
| Original vs audited labels | 14 / 75 |
| Ranked vs deployable routing | 4 / 45 |
| Component-sum vs direct E2E mean-cost ordering | 0 / 105 |

### FPR treatment

Forty of 75 deployable policy evaluations have empirical FPR at or below 0.05, but only four pass the frozen finite-sample row-and-dependency risk gate. This changes **36 of 75** policy conclusions.

### Grouping

Replacing singleton grouping with dependency-aware grouping changes **19 of 75** conclusions.

### Label quality

Replacing the original labels with the audited harmful-response labels changes **14 of 75** conclusions.

### Ranked vs deployable routing

Offline ranked acquisition and a reusable threshold are not operationally equivalent. Risk pass/fail changes in **4 of 45** matched comparisons, and the realized acquisition rate differs in **45 of 45** comparisons.

### Direct E2E vs component-sum cost

Direct wall-clock E2E latency is slightly higher than the same-run component sum for every one of the 15 timed policies.

However, none of the **105 pairwise mean-cost orderings** reverses.

This is a useful negative result: direct E2E remains the correct cost estimand, but correcting the measurement does not change policy ordering in this pilot.

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

## Reproducibility

The corrected evaluation/measurement workflow has committed entry points and hash-recorded artifacts.

Key commands are:

```bash
bash scripts/reproduce_historical_v2_evidence.sh
python scripts/rebuild_evaluation_measurement_dependency_groups.py --check
python scripts/run_evaluation_measurement_pilot_v1_cpu.py
python scripts/analyze_evaluation_measurement_pilot_v1.py
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

The current report presents:

- the corrected interpretation of the historical v2 study;
- direct E2E timing;
- dependency-aware grouping;
- ranked vs deployable routing;
- empirical vs finite-sample FPR treatment;
- label-audit sensitivity;
- the completed multi-stack / multi-dataset pilot;
- source-specific and tail-latency results;
- the current claim boundary.

Earlier reports are kept for traceability. The current project status is defined by the corrected v2 interpretation and the completed evaluation/measurement pilot.

## Current work

The project now examines how cost measurement, routing procedure, grouping, FPR treatment, and label quality affect conclusions about selective safety-monitor cascades. The existing router system is the development case study for this evaluation.
