from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from monitor_fusion.monitors.rule_filter import RULES, score_prompt_response


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/processed/unified_dataset.parquet"
OUT_PARQUET = ROOT / "data/interim/rule_scores.parquet"
OUT_CSV = ROOT / "data/interim/rule_scores.csv"
RUN_METADATA = ROOT / "data/interim/rule_filter_v1_run.json"

INFERENCE_FIELDS = ["example_id", "prompt", "response"]
METADATA_FIELDS = [
    "example_id",
    "split",
    "y",
    "source_dataset",
    "attack_family",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    df = pd.read_parquet(DATASET_PATH)

    missing = [
        field
        for field in INFERENCE_FIELDS + METADATA_FIELDS
        if field not in df.columns
    ]
    if missing:
        raise ValueError(f"Dataset is missing required fields: {missing}")

    inference_df = df[INFERENCE_FIELDS].copy()

    records = []
    for row in inference_df.itertuples(index=False):
        result = score_prompt_response(
            prompt=row.prompt,
            response=row.response,
        )
        records.append(
            {
                "example_id": row.example_id,
                "rule_score": result["rule_score"],
                "rule_weighted_sum": result["rule_weighted_sum"],
                "rule_match_count": result["rule_match_count"],
                "rule_matches": json.dumps(
                    result["rule_matches"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "rule_latency_ms": result["rule_latency_ms"],
            }
        )

    scores = pd.DataFrame.from_records(records)

    if len(scores) != len(df):
        raise RuntimeError("Rule-score row count does not match dataset row count.")
    if scores["example_id"].nunique() != len(scores):
        raise RuntimeError("Rule-score example_id values are not unique.")

    merged = df[METADATA_FIELDS].merge(
        scores,
        on="example_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(df):
        raise RuntimeError("Merged rule-score cache row count mismatch.")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PARQUET, index=False)
    merged.to_csv(OUT_CSV, index=False)

    metadata = {
        "scoring_run_id": "rule_filter_v1_local_cpu",
        "created_at": utc_now(),
        "monitor_id": "rule_filter_v1",
        "input_dataset": str(DATASET_PATH.relative_to(ROOT)),
        "output_parquet": str(OUT_PARQUET.relative_to(ROOT)),
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
        "num_examples": int(len(merged)),
        "monitor_input_fields": INFERENCE_FIELDS,
        "reattached_metadata_fields": METADATA_FIELDS,
        "threshold_selected": False,
        "rule_count": len(RULES),
        "rules": [
            {
                "rule_id": rule.rule_id,
                "pattern": rule.pattern,
                "weight": rule.weight,
                "description": rule.description,
            }
            for rule in RULES
        ],
        "hardware_metadata": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        },
    }

    RUN_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Rows scored:", len(merged))
    print("Output parquet:", OUT_PARQUET)
    print("Output csv:", OUT_CSV)
    print("Run metadata:", RUN_METADATA)
    print()
    print("Rule score summary:")
    print(merged["rule_score"].describe())
    print()
    print("Median latency ms:", merged["rule_latency_ms"].median())
    print("95th percentile latency ms:", merged["rule_latency_ms"].quantile(0.95))
    print()
    print("Split counts:")
    print(merged["split"].value_counts().sort_index())


if __name__ == "__main__":
    main()
