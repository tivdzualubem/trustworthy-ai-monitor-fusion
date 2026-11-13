# Trustworthy AI — Phase 8B population-variability validity audit

**Why this is necessary:** The completed Phase 8A run simulated 300 reviewer realizations on **one fixed synthetic population** per geometry. In the 1,687-response development geometry, an independent reconstruction of the fixed-seed proxy showed that all **34** harmful allowed responses landed in the **65** proxy-positive allowed responses; none were proxy-negative. Balanced proxy-stratified sampling therefore almost exhausts a favorable group, explaining its unusually low RMSE (0.0009). The paired Monte Carlo confidence intervals from Phase 8A are conditional on that population and do not measure population-generator variability.

**Research lead's approved objective:** `P(Y=1,D=0)` (harmful assistant responses allowed by the monitor) is primary, FNR secondary, and the monitor acts after generation but before delivery. He asks whether uncertainty in *human labels* matters more than estimator choice, under actual review costs, supported by sound mathematics and comparisons with published methods.

## What the audit does

- **Reuses the existing `phase8_cost_matched.py`, unchanged.** This is a wrapper, not a replacement or new estimator.
- Runs independent synthetic populations (`--populations 16`, default), each with new true harmfulness assignment and cheap-proxy realization; within each population, runs all the same sampling/reviewer-noise method arms (`--repetitions-per-population 30`). Both population geometries can be examined in one run at a 4,800-second active-review budget.
- Reuses empirical active-time distributions from the existing study through the Phase 8A code. The SQLite file is opened read-only and only `rater`/`active_seconds` are selected; no case IDs, text, reviewer votes or credentials are exported.
- Aggregates squared errors over *independent populations* and paired method differences; reports bootstrap intervals with the **population** as the resampling unit, not repeated vote simulations treated as independent populations. Also reports proxy-positive stratum counts and failures of proxy sensitivity to identify the one-seed artifact.
- Checks that every realized and reserved cost remains within the same hard active-time cap.

**Scientific limitations:** This checks robustness to synthetic population generation, not new theory; proxy quality and review error models are still assumed. Proxy acquisition, model inference, training and administrative costs are not measured or charged. R4's reserved-time strategy can use substantially less actual time than competitors. These are not fully cost-equivalent in all resources, and these are not complete MultiPPI, AM-PPI, or tuned StratPPI implementations.

## Run once in the established VS Code Ubuntu project

1. Download `Trustworthy_AI_Phase8B_Population_Robustness_20261112.zip` to Windows Downloads.
2. From the existing Phase 8A folder, locate the ZIP, extract **next to the existing script**, and run. All commands below are safe to paste in Ubuntu WSL:

```bash
cd ~/trustworthy-ai-monitor-fusion/reviewer_analysis/Phase8_Cost_Matched_20261112

ZIP="$(find /mnt/c/Users -maxdepth 4 -type f -iname 'Trustworthy_AI_Phase8B_Population_Robustness_20261112*.zip' -print -quit 2>/dev/null)"
if [ -z "$ZIP" ]; then
    echo "ZIP not found in Windows Downloads. Download the file first."
else
    python3 -m zipfile -e "$ZIP" . && \
    cp -n phase8b_robustness/phase8b_population_audit.py ./phase8b_population_audit.py && \
    ~/trustworthy-ai-monitor-fusion/.venv/bin/python \
      phase8b_population_audit.py \
      --db ~/tai_review_real_01/cases_private.sqlite \
      --populations 16 --repetitions-per-population 30 \
      --budget-seconds 4800 \
      --out ~/tai_review_real_01/phase8b_population_robustness_20261112.json && \
    cp -n ~/tai_review_real_01/phase8b_population_robustness_20261112.json "$(dirname "$ZIP")/"
fi
```

This uses **one new aggregate output filename** and will refuse to overwrite it. It does not rerun, reset or alter your 96-case human review or the original Phase 8A outputs. The Phase 8A script remains unmodified.

**Check:** After completion, the terminal shows each geometry's method RMSE, CI width, review time, and frequency of the artificially perfect H-positive separation. The aggregate JSON is automatically copied into the same Windows directory as the downloaded ZIP (normally Downloads) and is safe to share; the private SQLite DB is not. Optionally, use `python3 -m unittest -v phase8b_robustness/test_phase8b_population_audit.py` from the same directory; this tests only the aggregator with synthetic fixtures.

## Then proceed without another round of small tasks

1. Read the population-aggregated results. If stratification still benefits **perfect labels** but loses its advantage under noise across independently generated populations, treat that as a **robust simulation finding**, not a theorem.
2. Complete one faithful published-method comparison using exact assumptions, algorithms, all calibration labels, score acquisition cost and actual inference+review/adjudication time. Important primary sources: [MultiPPI (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d79ead018586772763ee1e486bd12b3a-Abstract-Conference.html), [AM-PPI (2026)](https://arxiv.org/abs/2605.08429), and [StratPPI (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract.html). The existing Phase8A proxy and PPI arms are **not** full published replicas.
3. In parallel, audit theorem-level prior art for partial identification and two-phase validation under correlated/uncertain human labels. Do not claim a novel result solely from well-known impossibility of identifying true prevalence with completely unconstrained misclassification.
4. **Only when fully verified**, prepare a reproducible research report and explicitly stage safe code and aggregate statistics for GitHub, then report to Prof. Bader with chosen direction and honest limitations. Do not stage private DB, raw response content, tokens or protected legacy splits.
