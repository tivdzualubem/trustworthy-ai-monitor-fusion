# Stage-A readout-transfer experiment: frozen scoring artifacts

**Run:** Stage-A readout-transfer kill test (2026-10-07)
**Status:** exploratory; original pre-specified screening outcome unchanged

This directory contains the frozen inputs and the exact GPU-scoring and analysis artifacts from the four-guard experiment.

## Contents

- `readout_probe_prompts.csv`, `o3_ablation_prompts.csv`, `input_audit.json`: input files committed before GPU scoring (not overwritten by this restore).
- `features/*_probe_features.npz`: the 152 frozen feature rows per guard, along with the corresponding native scores and decisions.
- `features/*_checkpoint.meta.json`: input, runner, guard-contract, runtime-lock, and registry hashes, matched to each guard.
- `o3_scores/*_o3_ablation_scores.csv`: 304 per-guard ablation scores and native outputs.
- `analysis/oof_predictions.csv`: 2,736 out-of-fold individual predictions.
- `analysis/native_thresholds.csv`: 20 fold-specific threshold rows.
- `analysis/probe_metrics.csv`, `analysis/o3_ablation_summary.csv`, `analysis/screening_decision.csv`: aggregate outputs.
- `gpu_manifest.json` and `*_manifest.json`: hardware/runtime, guard, and provenance records.
- `artifact_integrity.json`: SHA-256 checksums for the restored artifacts.

## Interpretive restrictions

The dataset labels *prompts*, not harmful model responses. The guards use different native safety policies. Researcher-authored benign controls were length matched, not hard topic/wording matched. Simple lexical-classifier baselines, source-probe threshold-only calibration, policy-aligned label auditing, and normalization/decoding controls are not provided in this frozen experiment and remain follow-up work.

The prespecified `readout_transfer_signal` in `analysis/screening_decision.csv` is **false for all four guards**. Do not change that rule or file retrospectively. Success of a newly trained probe cannot be interpreted as a native guard's mechanism or proof of semantic understanding. Paired mean-shift plus a fixed linear probe is an intercept/threshold-like adjustment, not evidence of nonlinear representation repair.

## Frozen versions and exact reconstruction

Code and inputs: `stage-a-paired-transport-20260921` branch at frozen input commit
`82275073e44109cf16f42b8a4506e5f019946816`.

The extraction protocol, model revisions and paths, feature-position choice, score definitions, five pair-ID folds, and logistic-regression parameters are recorded in
`configs/stage_a_readout_transfer_kill_contract_v1.json`,
`docs/stage_a_readout_transfer_kill_protocol_20261007.md`,
`scripts/run_stage_a_readout_transfer_kill_kaggle.py`, and
`scripts/analyze_stage_a_readout_transfer_kill.py`.

Run environment: Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, Transformers 5.16.1, two Tesla T4 GPUs. Analysis used NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1 (the analysis versions are recorded in `gpu_manifest.json`).

This is an artifact restoration only, not a revised experiment or renewed claim.
