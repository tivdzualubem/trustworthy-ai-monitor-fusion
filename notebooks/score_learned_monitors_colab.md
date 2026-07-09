# Colab Workflow: Learned Monitor Scoring

This workflow scores the two learned monitors and creates the final monitor
score cache. Run it in Google Colab with a GPU runtime.

## Outputs

The workflow creates:

- data/interim/compact_scores.parquet
- data/interim/judge_scores.parquet
- data/processed/monitor_score_cache.parquet
- data/metadata/monitor_scoring_run.json

The script is resumable. If Colab disconnects, rerun it and already-scored
example_id values will be skipped.

## Colab commands

### 1. Clone repository

Use HTTPS unless SSH is configured in Colab:

    !git clone https://github.com/tivdzualubem/trustworthy-ai-monitor-fusion.git
    %cd trustworthy-ai-monitor-fusion

### 2. Install dependencies

    !python -m pip install -r requirements/data.txt
    !python -m pip install -r requirements/monitor_scoring.txt
    !python -m pip install -e .

### 3. Authenticate to Hugging Face

Use a read token. Do not print or commit the token.

    from huggingface_hub import login
    login()

### 4. Mount Google Drive

    from google.colab import drive
    drive.mount("/content/drive")

### 5. Copy local generated files into the repo

Before running Colab, upload these two local files to Google Drive:

- data/processed/unified_dataset.parquet
- data/interim/rule_scores.parquet

Recommended Drive location:

- MyDrive/trustworthy-ai-monitor-fusion/unified_dataset.parquet
- MyDrive/trustworthy-ai-monitor-fusion/rule_scores.parquet

Then in Colab run:

    !mkdir -p data/processed data/interim
    !cp /content/drive/MyDrive/trustworthy-ai-monitor-fusion/unified_dataset.parquet data/processed/
    !cp /content/drive/MyDrive/trustworthy-ai-monitor-fusion/rule_scores.parquet data/interim/

### 6. Run learned monitor scoring

    !python scripts/score_learned_monitors_colab.py

### 7. Save outputs back to Drive

    !mkdir -p /content/drive/MyDrive/trustworthy-ai-monitor-fusion/monitor_scores
    !cp data/interim/compact_scores.parquet /content/drive/MyDrive/trustworthy-ai-monitor-fusion/monitor_scores/
    !cp data/interim/judge_scores.parquet /content/drive/MyDrive/trustworthy-ai-monitor-fusion/monitor_scores/
    !cp data/processed/monitor_score_cache.parquet /content/drive/MyDrive/trustworthy-ai-monitor-fusion/monitor_scores/
    !cp data/metadata/monitor_scoring_run.json /content/drive/MyDrive/trustworthy-ai-monitor-fusion/monitor_scores/

## Methodology rules

During inference, learned monitors receive only:

- example_id
- prompt
- response

Labels and split metadata are reattached only after scoring through example_id.

No thresholds are selected during score generation.
