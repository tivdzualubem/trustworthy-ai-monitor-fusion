from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/build_cross_fitted_decision_value_targets.py"
)
OUTPUT_DIR = ROOT / "reports/decision_value_real_data"

spec = importlib.util.spec_from_file_location(
    "build_cross_fitted_decision_value_targets",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {SCRIPT}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_threshold_selection_respects_fpr_target() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.9])

    selected = module.select_threshold_at_fpr(
        y=y,
        scores=scores,
        target_fpr=0.25,
    )

    prediction = (scores >= selected.threshold).astype(int)
    metrics = module.binary_metrics(y, prediction)

    assert metrics["fpr"] <= 0.25
    assert selected.recall == metrics["recall"]


def test_cross_fitted_target_artifact_is_development_only() -> None:
    path = (
        OUTPUT_DIR
        / "cross_fitted_decision_value_targets.parquet"
    )
    assert path.exists()

    targets = pd.read_parquet(path)
    dataset = pd.read_parquet(
        ROOT
        / "data/processed/unified_dataset_label_audited_v1.parquet"
    )

    development_ids = set(
        dataset.loc[
            dataset["split"].isin(
                [
                    "policy_train",
                    "policy_selection",
                    "calibration",
                ]
            ),
            "example_id",
        ]
    )
    excluded_ids = set(
        dataset.loc[
            dataset["split"].isin(
                ["final_test", "held_out_shift"]
            ),
            "example_id",
        ]
    )

    assert len(targets) == 1687 * 2
    assert not targets.duplicated(
        ["example_id", "setup_id"]
    ).any()
    assert set(targets["example_id"]) == development_ids
    assert not set(targets["example_id"]).intersection(
        excluded_ids
    )
    assert set(targets["setup_id"]) == {
        "compact_after_rule",
        "qwen_after_rule_compact",
    }


def test_realized_value_identity_and_cross_fitting_manifest() -> None:
    targets = pd.read_parquet(
        OUTPUT_DIR
        / "cross_fitted_decision_value_targets.parquet"
    )
    manifest = json.loads(
        (
            OUTPUT_DIR
            / "cross_fitted_target_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert np.array_equal(
        targets["realized_decision_value"].to_numpy(),
        (
            targets["base_loss"]
            - targets["augmented_loss"]
        ).to_numpy(),
    )
    assert set(
        targets["realized_decision_value"].unique()
    ).issubset({-1, 0, 1})
    assert targets["outer_fold"].nunique() == 5
    assert manifest["excluded_rows_used"] is False
    assert manifest["final_test_used"] is False
    assert manifest["held_out_shift_used"] is False
    assert manifest["outer_folds"] == 5
    assert manifest["inner_folds"] == 4


def test_each_development_example_has_one_outer_fold() -> None:
    assignments = pd.read_csv(
        OUTPUT_DIR
        / "development_outer_fold_assignments.csv"
    )

    assert len(assignments) == 1687
    assert assignments["example_id"].is_unique
    assert set(assignments["outer_fold"]) == {0, 1, 2, 3, 4}
