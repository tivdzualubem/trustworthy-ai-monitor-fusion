# Risk-Controlled Decision-Value Acquisition for Runtime Safety Monitor Fusion
<!-- V2_STATUS_START -->

## Exact-cost risk-cascade v2

The current v2 study evaluates selective safety-monitor acquisition under a frozen exact-cost and risk-constrained protocol with repeated grouped splits, multiple model families, deployable threshold semantics, stronger baselines, heterogeneous latency measurement, and protected-data boundaries.

**Development outcome:** no signed-value model family had a jointly feasible development operating point. Ridge, HistGradientBoostingRegressor, and RandomForestRegressor each passed **1 of 16** primary pairwise cost-equivalence comparisons. No router family was therefore selected.

This repository does **not** claim that the router improves recall under controlled FPR and cost. Independent fresh calibration and fresh source- and time-separated multi-rater confirmatory data were unavailable, so formal joint-risk certification and the confirmatory superiority gate were not executed. The protected legacy `final_test` and `held_out_shift` splits remained sealed.

The supported contribution is an **evaluation and measurement methodology for exact-cost, risk-constrained safety-monitor cascades**.

- [Research report](paper/Exact_Cost_Development_Screening_and_Risk_Constrained_Evaluation.pdf)
- [Publication source](paper/v2_publication/main.tex)
- [Frozen v2 protocol](configs/exact_cost_risk_cascade_protocol_v2.json)
- [Protocol amendment](configs/exact_cost_risk_cascade_protocol_v2_amendment_001.json)
- [v2 evaluation evidence](reports/exact_cost_risk_cascade_v2/)

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

## Final scientific result

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
- [LaTeX source](paper/Risk_Controlled_Decision_Value_Acquisition_Report.tex)
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
