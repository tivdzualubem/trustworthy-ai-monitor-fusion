from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/decision_value_real_data"

OOF_PATH = (
    REPORT_DIR
    / "value_estimator_oof_predictions.parquet"
)
FOLD_METRIC_PATH = (
    REPORT_DIR
    / "value_estimator_fold_metrics.csv"
)
CURVE_PATH = (
    REPORT_DIR
    / "matched_budget_value_curves.csv"
)
RANDOM_PATH = (
    REPORT_DIR
    / "matched_budget_random_repetitions.csv"
)
SUMMARY_PATH = (
    REPORT_DIR
    / "value_predictability_summary.csv"
)
MANIFEST_PATH = (
    REPORT_DIR
    / "value_predictability_manifest.json"
)


def test_oof_prediction_shape_and_uniqueness() -> None:
    frame = pd.read_parquet(OOF_PATH)

    assert len(frame) == 1687 * 2 * 6
    assert frame["example_id"].nunique() == 1687
    assert frame["setup_id"].nunique() == 2
    assert frame["feature_family"].nunique() == 6
    assert not frame.duplicated(
        [
            "example_id",
            "setup_id",
            "feature_family",
        ]
    ).any()
    assert np.isfinite(
        frame["predicted_decision_value"]
    ).all()


def test_oof_rows_are_development_only() -> None:
    frame = pd.read_parquet(OOF_PATH)
    dataset = pd.read_parquet(
        ROOT
        / "data/processed/"
        "unified_dataset_label_audited_v1.parquet"
    )
    excluded_ids = set(
        dataset.loc[
            dataset["split"].isin(
                ["final_test", "held_out_shift"]
            ),
            "example_id",
        ]
    )
    assert not set(frame["example_id"]).intersection(
        excluded_ids
    )


def test_all_folds_and_families_are_present() -> None:
    metrics = pd.read_csv(FOLD_METRIC_PATH)

    assert len(metrics) == 2 * 5 * 6
    assert set(metrics["outer_fold"]) == set(range(5))
    assert metrics["setup_id"].nunique() == 2
    assert metrics["feature_family"].nunique() == 6
    assert (
        metrics["selected_candidate_id"]
        .isin([0, 1, 2])
        .all()
    )

    embedding_families = {
        "frozen_embedding",
        "cheap_plus_embedding",
        "all_features",
    }
    embedded = metrics[
        "feature_family"
    ].isin(embedding_families)

    assert metrics.loc[
        embedded,
        "embedding_used",
    ].all()
    assert (
        metrics.loc[
            embedded,
            "pca_components",
        ]
        == 32
    ).all()
    assert not metrics.loc[
        ~embedded,
        "embedding_used",
    ].any()


def test_exact_matched_budget_curves() -> None:
    curves = pd.read_csv(CURVE_PATH)

    expected_policies = {
        "ordinary_uncertainty",
        "learned_decision_value",
        "oracle_realized_value_diagnostic",
    }
    expected_budgets = {
        0.0,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.75,
        1.0,
    }

    assert set(curves["policy"]) == expected_policies
    assert set(curves["budget"]) == expected_budgets
    assert len(curves) == 2 * 6 * 10 * 3

    pivot = curves.pivot_table(
        index=[
            "setup_id",
            "feature_family",
            "budget",
        ],
        columns="policy",
        values="acquired_n",
    )
    assert (
        pivot.nunique(axis=1) == 1
    ).all()


def test_random_repetitions_are_complete() -> None:
    random = pd.read_csv(RANDOM_PATH)

    assert len(random) == 2 * 6 * 10 * 100
    assert random["repetition"].nunique() == 100
    assert set(random["policy"]) == {"random"}


def test_primary_comparison_is_prespecified() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    primary = summary.loc[
        summary["primary_comparison"]
    ]

    assert len(summary) == 12
    assert len(primary) == 1
    assert primary.iloc[0]["setup_id"] == (
        "qwen_after_rule_compact"
    )
    assert primary.iloc[0]["feature_family"] == (
        "all_features"
    )

    for column in [
        "integrated_advantage",
        "paired_bootstrap_lower95",
        "paired_bootstrap_upper95",
    ]:
        assert np.isfinite(summary[column]).all()


def test_manifest_keeps_overall_no_go() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    assert manifest["status"] == (
        "development_only_value_predictability_completed"
    )
    assert manifest["overall_project_status"] == "no-go"
    assert (
        manifest["overall_milestone_claim_made"]
        is False
    )
    assert (
        manifest["scope"]["final_test_used"]
        is False
    )
    assert (
        manifest["scope"]["held_out_shift_used"]
        is False
    )
    assert (
        manifest["scope"][
            "global_outer_target_rows_used_for_training"
        ]
        is False
    )
    assert (
        manifest["scope"][
            "current_outer_evaluation_fold_excluded"
        ]
        is True
    )
