# Official Qwen3Guard scoring repair

This replaces the earlier Qwen scoring path.

Use this in Colab after uploading the repo and generated dataset files:

1. Install requirements:

    python -m pip install -r requirements/monitor_scoring.txt
    python -m pip install -e .

2. Run official Qwen3Guard scoring:

    python scripts/score_qwen3guard_official_colab.py --dataset data/processed/unified_dataset.parquet --output-dir data/interim/qwen3guard_official --modes prompt_only response_only prompt_response --batch-size 1 --max-new-tokens 128 --checkpoint-every 16 --warmup-examples 4

3. Copy back the entire folder:

    data/interim/qwen3guard_official

The output folder contains per-mode Parquet scores, a combined cache, raw generated outputs, and a run manifest.
