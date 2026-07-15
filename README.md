# Budget-Aware Runtime Safety Monitor Fusion

This repository contains the data, monitor outputs, serialized policies,
evaluation code, robustness analyses, and fixed-hardware timing evidence for a
budget-aware runtime safety-monitor fusion study.

The prediction unit is a prompt-response pair. The target is whether the
response should be intercepted because it materially contains or provides
disallowed harmful content. Benign responses, refusals, and safe redirections
are negative examples.

## Main finding

Selective acquisition reduced mean compute cost and mean latency in the
measured setting, but it did not satisfy the prespecified false-positive risk
criterion under all audited and held-out shift evaluations. The resulting
paper direction is therefore **measurement validity**, not a claim of a
successful deployment-ready routing policy.

The final report is:

[`paper/Budget_Aware_Runtime_Safety_Monitor_Fusion_Report.pdf`](paper/Budget_Aware_Runtime_Safety_Monitor_Fusion_Report.pdf)

## Repository contents

- `data/processed/`: audited prompt-response dataset and final monitor-score
  cache used by the analyses.
- `data/metadata/`: source, label, monitor, model, policy, timing, and
  evaluation manifests.
- `src/monitor_fusion/`: reusable monitor implementation.
- `scripts/`: data processing, monitor scoring, model training, policy
  evaluation, robustness analysis, timing, and verification entry points.
- `artifacts/`: serialized fusion models, frozen policies, risk-control
  artifacts, and the final timing-result archive.
- `exports/final_v3_policy_timing_package.zip`: portable Tesla-T4 timing
  package.
- `reports/`: detailed scientific outputs and raw generations.
- `results/tables/`: canonical final result tables.
- `tests/`: reproducibility smoke tests.
- `paper/`: final technical report.

Historical v2 files are retained where they provide the direct lineage for
the audited v3 cache or are still referenced by final v3 scripts. They are not
the recommended final entry points.

## Environment setup

The CPU analysis environment was validated with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The final GPU timing package has its own requirements and was executed on a
Tesla T4. A compatible CUDA-enabled PyTorch installation is required for GPU
regeneration; PyTorch is supplied by the Kaggle accelerator image in the
documented timing workflow.

## Fast verification

Verify the committed artifacts, hashes, serialized model, data counts, final
tables, and timing manifest:

```bash
python scripts/verify_reproducibility.py --strict-hashes
pytest -q
```

Equivalent Make targets are available:

```bash
make verify
make test
```

## Reproduce the CPU analyses

The committed audited score cache allows the final statistical analyses to be
rerun without downloading or regenerating the large monitor models:

```bash
bash scripts/reproduce_cpu_results.sh
```

This executes, in order:

1. serialized v3 fusion-model training;
2. all prespecified policy evaluation;
3. nested leave-source-out and leave-family-out evaluation;
4. nested fixed-sequence Learn-then-Test risk control;
5. the final stop/go decision.

The scripts write their detailed outputs under `artifacts/`, `reports/`,
`results/tables/`, and `data/metadata/`.

## Regenerate monitor outputs and fixed-hardware timing

The repository preserves raw Qwen generations, compact-monitor outputs,
manifests, pinned model revisions, and the rule-filter implementation.

For the exact final timing workflow:

1. Upload `exports/final_v3_policy_timing_package.zip` to a Kaggle notebook.
2. Select a Tesla T4 accelerator and enable internet access for the initial
   model download.
3. Provide a Kaggle secret named `HF_TOKEN`.
4. Extract the package and run:

```bash
bash final_v3_policy_timing_package/run_gpu.sh \
  final_v3_policy_timing_package \
  final_v3_policy_timing_results
```

The committed measured result archive is
`artifacts/final_v3_policy_timing_results.zip`. Its manifest records the GPU,
batch size, warm-up policy, CUDA synchronization, model revisions, parsing
rate, score comparisons, and policy latencies.

## Data and label audit

The final audited dataset contains 2,159 examples:

- 403 positive response-harmfulness labels;
- 1,756 negative labels;
- 200 manually reviewed audit rows;
- 19 final label corrections;
- a focused second review of 40 high-priority rows.

The main files are:

- `data/processed/unified_dataset_label_audited_v1.parquet`
- `data/processed/monitor_score_cache_v3.parquet`
- `docs/label_audit_protocol.md`
- `reports/label_audit/`

## Key measured results

On the 128-example Tesla-T4 timing benchmark:

- full-information mean latency: 1,654.37 ms;
- selective mean latency: 1,251.71 ms;
- measured mean-latency reduction: 24.34%;
- selective expensive-monitor call rate: 69.53%.

Median latency improved, but p95 and p99 did not improve. No tail-latency
improvement is claimed.

Rule scores regenerated exactly and compact-monitor scores agreed within
`3.75e-7`. Qwen outputs remained parseable, but two unique examples changed
classification between the cached and regenerated runs. The raw generations
and mismatch records are retained as measurement-reproducibility evidence.

## Risk-control interpretation

The fixed-sequence Learn-then-Test procedure produced valid certificates on
the untouched in-fold risk-control partitions. Those certificates do not imply
that the same 5% false-positive bound transfers to an excluded dataset source
or attack family. Several outer-shift evaluations exceeded the target.

The old final-test partition had influenced development and is retained only
as descriptive evidence. Nested leave-source-out and leave-family-out
evaluation replaces any claim that it remained untouched.

## Limitations

- The manual label audit was performed by one author.
- The fixed-hardware benchmark used one accelerator type and 128 examples.
- Qwen classification outputs were not exactly deterministic across the two
  recorded generations.
- The risk-control guarantees are scoped to their in-fold risk-control
  distributions and do not establish universal shift robustness.
- The repository supports research reproduction; it does not establish
  deployment readiness.
