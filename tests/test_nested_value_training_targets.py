from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/decision_value_real_data"
TARGET_PATH = (
    REPORT_DIR / "nested_value_training_targets.parquet"
)
METRICS_PATH = (
    REPORT_DIR / "nested_value_training_target_fold_metrics.csv"
)
MANIFEST_PATH = (
    REPORT_DIR / "nested_value_training_target_manifest.json"
)
PROTOCOL_PATH = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)


def test_protocol_forbids_global_outer_targets_for_training() -> None:
    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    value_estimator = protocol["value_estimator"]

    assert value_estimator[
        "current_outer_evaluation_fold_excluded_from_target_generation"
    ] is True
    assert value_estimator[
        "global_outer_target_rows_allowed_for_estimator_training"
    ] is False
    assert value_estimator["training_target_artifact"] == (
        "nested_value_training_targets.parquet"
    )

    matching = [
        item
        for item in protocol["protocol_amendments"]
        if item.get("amendment_id")
        == "nested_inner_oof_value_training_targets_v1"
    ]
    assert len(matching) == 1
    assert matching[0]["value_estimator_training_started"] is False
    assert matching[0]["final_test_or_shift_used"] is False


def test_nested_target_shape_and_repetition() -> None:
    targets = pd.read_parquet(TARGET_PATH)

    assert len(targets) == 13496
    assert targets["example_id"].nunique() == 1687
    assert targets["setup_id"].nunique() == 2
    assert targets["value_outer_fold"].nunique() == 5

    assert not targets.duplicated(
        ["setup_id", "value_outer_fold", "example_id"]
    ).any()

    counts = targets.groupby(
        ["setup_id", "example_id"]
    ).size()
    assert (counts == 4).all()

    assert set(
        targets["realized_decision_value"].unique()
    ).issubset({-1, 0, 1})
    assert set(
        targets["downstream_inner_fold"].unique()
    ) == {0, 1, 2, 3}


def test_current_outer_evaluation_fold_is_excluded() -> None:
    targets = pd.read_parquet(TARGET_PATH)
    assignments = pd.read_csv(
        REPORT_DIR / "development_outer_fold_assignments.csv"
    )

    merged = targets.merge(
        assignments.rename(
            columns={"outer_fold": "assigned_outer_fold"}
        ),
        on="example_id",
        how="left",
        validate="many_to_one",
    )

    assert merged["assigned_outer_fold"].notna().all()
    assert (
        merged["assigned_outer_fold"]
        != merged["value_outer_fold"]
    ).all()


def test_no_optional_output_is_available_as_predictor() -> None:
    targets = pd.read_parquet(TARGET_PATH)

    forbidden = {
        "optional_monitor_score",
        "augmented_score",
        "compact_unsafe_score",
        "qwen_prompt_response_score",
        "source_dataset",
        "attack_family",
        "prompt",
        "response",
    }
    assert forbidden.isdisjoint(targets.columns)

    assert (
        targets["target_generation_scope"]
        == "inner_oof_within_current_outer_train"
    ).all()


def test_thresholds_match_frozen_downstream_builder() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    frozen = pd.read_csv(
        REPORT_DIR / "cross_fitted_target_fold_metrics.csv"
    )

    merged = metrics.merge(
        frozen[
            [
                "setup_id",
                "outer_fold",
                "base_threshold",
                "augmented_threshold",
                "outer_train_n",
                "outer_test_n",
            ]
        ],
        left_on=["setup_id", "value_outer_fold"],
        right_on=["setup_id", "outer_fold"],
        suffixes=("_nested", "_frozen"),
        validate="one_to_one",
    )

    assert len(merged) == 10
    assert (
        merged["base_threshold_nested"]
        == merged["base_threshold_frozen"]
    ).all()
    assert (
        merged["augmented_threshold_nested"]
        == merged["augmented_threshold_frozen"]
    ).all()
    assert (
        merged["outer_train_n_nested"]
        == merged["outer_train_n_frozen"]
    ).all()
    assert (
        merged["outer_eval_n"]
        == merged["outer_test_n"]
    ).all()


def test_nested_target_manifest_scope() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    assert manifest["rows"] == 13496
    assert manifest["expected_rows"] == 13496
    assert manifest["unique_examples"] == 1687
    assert manifest["rows_per_example_per_setup"] == 4
    assert (
        manifest["global_outer_target_rows_used_for_training"]
        is False
    )
    assert (
        manifest["current_outer_evaluation_fold_excluded"]
        is True
    )
    assert manifest["final_test_used"] is False
    assert manifest["held_out_shift_used"] is False
    assert (
        manifest["optional_monitor_outputs_written_as_predictors"]
        is False
    )
