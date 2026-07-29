#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/decision_value_real_data_protocol_v1.json"
DATASET = ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
CACHE = ROOT / "data/processed/monitor_score_cache_v3.parquet"


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def effective_groups(frame: pd.DataFrame) -> pd.Series:
    def clean(column: str) -> pd.Series:
        values = frame[column].fillna("").astype(str).str.strip()
        return values.mask(values.isin({"", "nan", "None", "<NA>"}))

    group_id = clean("group_id")
    pair_id = clean("pair_id")
    example_id = clean("example_id")

    result = group_id.map(
        lambda value: f"group:{value}" if pd.notna(value) else pd.NA
    )
    result = result.fillna(
        pair_id.map(
            lambda value: f"pair:{value}" if pd.notna(value) else pd.NA
        )
    )
    return result.fillna(
        example_id.map(
            lambda value: f"example:{value}" if pd.notna(value) else pd.NA
        )
    )


def validate() -> dict:
    protocol = load_protocol()
    dataset = pd.read_parquet(DATASET)
    cache = pd.read_parquet(CACHE)

    assert protocol["status"] == "frozen_before_real_data_modeling"
    assert dataset["example_id"].is_unique
    assert cache["example_id"].is_unique
    assert set(dataset["example_id"]) == set(cache["example_id"])

    scope = protocol["scope"]
    development_splits = set(scope["development_splits"])
    excluded_splits = set(scope["excluded_splits"])
    assert development_splits.isdisjoint(excluded_splits)

    development = dataset[dataset["split"].isin(development_splits)].copy()
    excluded = dataset[dataset["split"].isin(excluded_splits)].copy()

    assert len(development) == scope["expected_development_rows"]
    counts = development["y"].astype(int).value_counts().to_dict()
    assert counts == {
        0: scope["expected_negative_n"],
        1: scope["expected_positive_n"],
    }
    assert len(excluded) == 472

    development["effective_group"] = effective_groups(development)
    assert development["effective_group"].notna().all()
    group_n = development["effective_group"].nunique()
    assert group_n >= protocol["cross_fitting"]["outer_folds"]

    cache_columns = set(cache.columns)
    for setup in protocol["optional_monitor_setups"]:
        assert set(setup["base_features"]) < set(setup["augmented_features"])
        assert set(setup["augmented_features"]).issubset(cache_columns)

    predictors = protocol["predictor_families"]
    compact_pre = (
        predictors["cheap_features"]["compact_after_rule"]
        + predictors["runtime_metadata"]["compact_after_rule"]
    )
    qwen_pre = (
        predictors["cheap_features"]["qwen_after_rule_compact"]
        + predictors["runtime_metadata"]["qwen_after_rule_compact"]
    )
    assert not any(name.startswith("compact_") for name in compact_pre)
    assert not any(name.startswith("qwen_") for name in qwen_pre)

    embedding = predictors["frozen_embedding"]
    assert embedding["resolved_full_revision_required"] is True
    assert embedding["measure_runtime"] is True
    assert embedding["dimension"] == 384

    assert protocol["operating_risk"]["target"] == 0.05
    assert protocol["cross_fitting"]["outer_folds"] == 5
    assert protocol["cross_fitting"]["inner_folds"] == 4
    assert protocol["random_repetitions"] == 100

    result = {
        "status": "PASS",
        "development_rows": len(development),
        "negative_n": counts[0],
        "positive_n": counts[1],
        "excluded_rows": len(excluded),
        "effective_group_n": int(group_n),
        "optional_setups": [
            item["setup_id"] for item in protocol["optional_monitor_setups"]
        ],
        "predictor_comparisons": predictors["comparisons"],
        "budgets": protocol["matched_budgets"],
        "fpr_target": protocol["operating_risk"]["target"],
        "embedding_model": embedding["model_id"],
        "embedding_requested_revision": embedding["requested_revision"]
    }
    return result


def main() -> None:
    output = (
        ROOT
        / "reports/decision_value_real_data/protocol_validation.json"
    )
    result = validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("decision-value real-data protocol validation passed")
    print(json.dumps(result, indent=2))
    print("output:", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
