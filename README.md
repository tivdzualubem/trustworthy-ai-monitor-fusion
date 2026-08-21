# Evaluation and Measurement of Runtime Safety-Monitor Cascades
<!-- V2_STATUS_START -->

## Current research direction: evaluation and measurement

Following the August 17 review, the router is a **no-go on the existing data**.
The project is now focused on an evaluation/measurement study using this
system as the first case study.

The historical v2 development results must be interpreted with these
limitations:

- the reported development costs are **summed measured component latencies**,
  not the frozen direct wall-clock end-to-end policy-latency estimand;
- the 1,687 development examples resolve to **1,687 effective groups**, so the
  historical grouping does not protect against template or near-duplicate
  dependence;
- the 35 s value used by bounded-cost code was a post-hoc recording cap, not a
  demonstrated mechanically enforced runtime bound;
- the historical `max(1 ms, 1% of budget)` cost-equivalence margin was
  prespecified but **not externally justified**;
- the historical v2 CSV snapshot is not fully regenerable from committed raw
  timing artifacts because downstream raw timing provenance is incomplete;
- fresh calibration, joint FPR+cost certification, fresh source/time-separated
  confirmation, and multi-rater labels were not obtained.

No signed-value family had a jointly feasible historical development operating
point, and no router was selected. The protected legacy `final_test` and
`held_out_shift` partitions remain sealed.

The historical component-cost tables are retained unchanged for auditability.
Corrected experiments must use a new measurement namespace and direct E2E cost
when making E2E or exact-cost claims.

Future evaluation distinguishes two questions:

1. **Iso-cost comparison:** use an externally justified equivalence margin.
2. **Pareto comparison:** test higher recall, non-higher cost, and controlled
   FPR without requiring two-sided cost equivalence first.

- [Current v2 report](paper/Exact_Cost_Development_Screening_and_Risk_Constrained_Evaluation.pdf)
- [Publication source](paper/v2_publication/main.tex)
- [Historical v2 provenance status](reports/exact_cost_risk_cascade_v2/historical_v2_provenance_status.json)
- [Frozen v2 protocol](configs/exact_cost_risk_cascade_protocol_v2.json)
- [v2 evidence snapshot](reports/exact_cost_risk_cascade_v2/)


The historical bounded-cost analysis normalized per-example cost with a 35 s constant. That constant was not a demonstrated mechanically enforced runtime bound; it was an analysis normalization bound, so the historical bounded-mean cost calculation is not a deployment runtime-bound certificate.

<!-- V2_STATUS_END -->

This repository contains an audited runtime safety-monitoring study with three
monitor levels: a deterministic rule filter, a compact moderation classifier,
and Qwen3Guard-Gen-4B. The completed work includes the original budget-aware
fusion pipeline and a professor-directed pivot from uncertainty routing to
**monitor-specific decision-value acquisition**.

The prediction unit is a prompt-response pair. The positive label means that
the response should be intercepted because, in the context of the prompt, it
materially provides, facilitates, endorses, or contains disallowed harmful
content. Benign answers, safe refusals, and safe redirections are negative.

## Earlier decision-value result

The result is mixed and remains **no-go** under the prespecified development
milestone:

- **Complementary monitor value: pass.** Adding Qwen after rule and compact
  monitoring reduced cross-fitted decision loss from 0.17487 to 0.06758.
- **Value predictability versus uncertainty: no-go.** The primary all-features
  learned policy had integrated matched-budget advantage 0.004816, but the
  paired 95% interval was `[-0.001705, 0.011485]`.
  acquired Qwen for 9.78% of development examples, reached recall 0.33677,
  and had pooled outer-OOF FPR 0.02865.
- **Scope control: pass.** The decision-value analysis used 1,687 development
  rows only. `final_test` and `held_out_shift` were not used.

Because every milestone condition was required, the project does not claim a
new validated routing method, robustness to changing attack distributions,
detection of correlated monitor failures, or deployment readiness.

## Final report

- [Compiled technical report](paper/Exact_Cost_Development_Screening_and_Risk_Constrained_Evaluation.pdf)
- [Current publication source](paper/v2_publication/main.tex)
- [Bibliography](paper/references.bib)
- [Report figures](paper/figures/)

Build the report with:

```bash
bash paper/build_report.sh
```

## Main experiment stages

1. **Audited monitor-fusion baseline**
   - 2,159 prompt-response examples;
   - deterministic rule filter, KoalaAI/Text-Moderation, and Qwen3Guard;
   - pinned model revisions, serialized policies, nested group holdouts,
     Learn-then-Test diagnostics, and Tesla-T4 timing.

2. **Decision-value theory and counterexample**
   - conditional value
     `Delta(h) = r(h) - E[r(h, Z) | h]`;
   - acquire exactly when expected decision improvement exceeds acquisition
     cost;
   - identical uncertainty can coexist with different optional-monitor value.

3. **Two-context Gaussian simulation**
   - 30 independent seeds;
   - random, uncertainty, oracle-value, and learned-value acquisition compared
     at identical acquisition counts;
   - at 25% acquisition, learned value reduced decision loss by 0.027839
     relative to uncertainty, with paired 95% interval
     `[0.027368, 0.028310]`.

4. **Leakage-controlled real-data diagnostic**
   - five outer and four inner stratified group folds;
   - 13,496 nested value-estimator training targets;
   - complete-text 384-dimensional frozen embeddings with 100% token coverage
     and zero truncation;
   - six prespecified feature families;
   - exact matched budgets and 100 random repetitions per budget.

5. **Common-risk cost accounting**
   - optional Qwen mean runtime: 1,597.56 ms;
   - complete-text embedding mean runtime: 159.47 ms/example;
   - PCA plus value inference: 0.029 ms/example;
   - risk feasibility defined by pooled development outer-OOF FPR <= 0.05.

## Repository layout

- `data/processed/`: audited dataset and final score cache.
- `data/metadata/`: source, monitor, model, timing, policy, and reproducibility
  manifests.
- `src/monitor_fusion/`: reusable monitor implementation.
- `scripts/`: data building, scoring, cross-fitting, simulation, diagnostics,
  timing, and verification entry points.
- `artifacts/`: serialized fusion and risk-control objects.
- `reports/decision_value_theory/`: counterexample outputs.
- `reports/decision_value_gaussian_simulation/`: controlled simulation.
- `reports/decision_value_real_data/`: cross-fitted targets, embeddings,
  value-estimator results, matched-budget curves, and safety-cost frontier.
- `results/tables/`: canonical tables from the original fusion evaluation.
- `tests/`: reproducibility and methodological checks.
- `paper/`: final LaTeX report, bibliography, figures, and PDF.

## Environment setup

The CPU analysis environment was validated with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The complete-text embedding environment is pinned separately:

```bash
python -m pip install --no-cache-dir -r requirements-embedding-cpu.txt
```

## Verification

Verify committed hashes and the full test suite:

```bash
python scripts/verify_reproducibility.py --strict-hashes
pytest -q
```

Equivalent Make targets:

```bash
make verify
make test
```

## Reproducing the decision-value analyses

The principal scripts are:

```bash
python scripts/run_decision_value_counterexample.py
python scripts/run_two_context_gaussian_decision_value.py
python scripts/build_cross_fitted_decision_value_targets.py
python scripts/build_full_text_prompt_response_embeddings.py
python scripts/build_nested_value_training_targets.py
python scripts/run_cross_fitted_value_predictability.py
python scripts/run_common_risk_safety_cost_frontier.py
```

The committed artifacts are the source of truth for the final report. The
excluded `final_test` and `held_out_shift` partitions must remain unused unless
a new protocol is approved and frozen before evaluation.

## Earlier runtime result

The original selective threshold router reduced mean Tesla-T4 latency from
1,654.37 ms to 1,251.71 ms, a 24.34% reduction, while calling Qwen on 69.53%
of the timing examples. The upper-tail latency did not improve, and the
prespecified risk condition did not transfer reliably across audited shift
diagnostics. This result is retained as measurement-validity evidence, not as
a deployment claim.

## Limitations

- Development-only evidence for the decision-value pivot.
- Single-author label audit.
- Sparse positive and negative realized-value events.
- General-purpose embeddings rather than a representation trained for monitor
  complementarity.
- Mixed CPU/GPU timing components in incremental cost accounting.
- No claim of arbitrary distribution-shift robustness, correlated-failure
  detection, online adaptation, or production readiness.
