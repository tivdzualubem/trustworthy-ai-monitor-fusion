# Phase 4C — Selective additional-monitor checks for measuring harmful-response misses

**Date:** 2026-10-09. **Status:** prespecified exploratory CPU-only simulation, **not** a fresh deployment evaluation, independent validation, new method, or risk certificate.

## Why this test

Prof. Bader's latest preferred path asks whether selectively acquiring extra monitor checks **and** human reviews can measure missed harmful **outputs** at lower cost, compared with strong existing methods. Phase 4A evaluated four simple probability-sampling methods; Phase 4B evaluated two adapted Active Testing/LURE acquisition proposals. Phase 4B did **not** consistently beat simple stratified audits, especially for FNR. Therefore the next small question is whether *paying for some optional-monitor scores* produces enough useful audit allocation to justify their additional cost. Do not tune a novel strategy until this simple test leaves an important gap.

## Input identity and scope

- Original historical cross-fitted, development-only response-level data: `reports/decision_value_real_data/cross_fitted_decision_value_targets.parquet`, 1,687 distinct examples × 2 optional-monitor setups. Frozen fold assignments and exact input/protocol SHA-256 checks must pass.
- The binary harm label `Y` belongs to archived response-level development data, *not* Stage-A prompt proxy labels. The frozen archival `base_prediction` is used as `D`: a cross-fitted classifier, *not* one deployed native guard.
- `optional_monitor_score` is a cached value for **simulation**. A row's value is made available to acquisition stratification **only when an optional monitor call for that row has been counted**. No extra GPU inference happens here.
- Optional monitor setups and historical mean per-call costs from `configs/decision_value_real_data_protocol_v1.json`: `compact_after_rule`: 45.657291 ms per Koala compact call; `qwen_after_rule_compact`: 1597.56057 ms per Qwen prompt-response call. Hardware/runtime conditions may differ now; **not fresh timing measurements**.
- Strict whitelist: only pre-existing development data and protocol/aggregate files; final-test and held-out-shift data remain sealed. Export aggregate metrics only.

## Estimands and correct limitations

On the finite development population, estimate two linear prevalence quantities and two ratios:

- Harm prevalence `P(Y=1)`.
- Miss prevalence `P(Y=1,D=0)` — missed harmful **responses per traffic example**.
- `FNR=P(D=0 | Y=1)` — ratio of the first two means; biased/unstable in small samples even if numerator/denominator means are individually unbiased.
- `FPR=P(D=1 | Y=0)` — similarly a ratio.

The primary outcomes are the repeated-sampling **RMSE** for missed-harm prevalence and FNR, with both false alarms and harmful misses retained. Precision/CI coverage for prevalence is descriptive and **not a validated adaptive guarantee**.

## Fixed protocol (no optimization after seeing results)

- Human reviews `B ∈ {40,80,160}`; optional-monitor calls `M ∈ {80,320}`.
- 200 independent randomized simulations per method, monitor count and setup.
- **Three simple comparisons**, not new learned estimators:
  1. `random_proportional`: sample optional-monitor calls uniformly; stratify the ensuing human-review sample by observable response decision `D`, monitor-acquired flag, and low/high cheap or optional score; allocate review proportionally.
  2. `priority_proportional`: acquire 80% of optional scores on highest *cheap* uncertainty examples, plus 20% randomized sentinel calls; proportional human allocation.
  3. `priority_tilted`: use the same priority/sentinel monitor acquisition but double the sampling allocation weight for the high optional-score stratum.
- Each population response belongs to one known post-acquisition stratum. Within every nonempty stratum, use simple random without replacement, with at least two human reviews per stratum if feasible. The stratum **population size** and sampling count are recorded, so the estimator is the design-based weighted sum `(1/N)Σ_h N_h mean(sampled Y(1−D) | h)` and analogues for harm and false alarms. This is unbiased **conditional on the monitor-acquired partition** for the *linear means*; the FNR/FPR ratios are not unbiased.
- Review acquisition and optional-monitor selection never inspect `Y`; no unacquired optional score enters the strata or allocation. Cached labels are visible only at simulated review, with full historical labels used separately as *evaluation truth*.
- A baseline with `M=0` has already been evaluated in Phases 4A/4B; its prior results are not rewritten. Do **not** call an increased-monitor strategy cheaper merely because its review count is the same.

## Cost reporting and validity

- Report human review count and historical optional-monitor latency separately, without assuming a financial cost per human label or extrapolating to deployed throughput. Example: 320 Qwen optional calls cost ~511.2 seconds of *historical mean monitor inference time* versus ~14.6 seconds for 320 compact calls. Not wall-clock predictions of the new experiment.
- Avoid comparing `augmented_prediction` directly to `base_prediction` as proof of guard-level transport; neither is a single frozen deployed guard policy.
- Use any apparent improvements as **hypothesis-generating**; do not choose the best setting post hoc then present its development RMSE as fresh confirmatory evidence.

## Decision rule / supervisor reporting

- If optional-monitor-informed audits do not improve **both** the relevant estimation accuracy and defensible price-of-information tradeoff over simple probability-sampling baselines, stop; do not build a more complex acquisition model merely to chase gains.
- If a clear cost-relevant signal remains, specify a new pre-frozen response-level protocol with reliable annotation/decision provenance, independent labels and a confirmatory sample for supervisor approval; do not use Stage-A prompt-only features/labels as substitutes.
- On reporting back, explicitly include negative Stage-A 0/4 screen, lexical hard-negative findings, source-probe cutoff control, the exact Phase 4A–C limits, and no deployment safety guarantee.
