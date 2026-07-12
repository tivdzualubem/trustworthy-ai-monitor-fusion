# Controlled timing benchmark v2

Repository commit: `479ab50c4f769984038d91f68f6fc4c6363342c7`

Hardware: `Tesla T4` on `cuda:0`

Rows: 128 deterministic calibration examples

Batch size: 1

Warm-up: 8 rows per monitor mode and
8 rows per complete policy, excluded from results.

Qwen uses the official chat template, deterministic generation, and structured
output parsing. GPU timings use synchronization before and after measured work.

The policy benchmark loads the exact serialized fusion bundle and locked
target-FPR 0.05 operating point. It does not refit any model.

See `latency_summary.csv`, `qwen_stage_latency_summary.csv`, the per-example
Parquet files, and `run_manifest.json`.
