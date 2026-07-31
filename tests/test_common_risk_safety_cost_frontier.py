from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/decision_value_real_data"

RUNTIME_PATH = (
    REPORT_DIR / "value_policy_inference_runtime.csv"
)
RANDOM_SUMMARY_PATH = (
    REPORT_DIR / "common_risk_random_summary.csv"
)
FRONTIER_PATH = (
    REPORT_DIR / "common_risk_safety_cost_frontier.csv"
)
CANDIDATE_PATH = (
    REPORT_DIR / "common_risk_selective_candidates.csv"
)
SUMMARY_PATH = (
    REPORT_DIR / "common_risk_frontier_summary.json"
)


def test_runtime_benchmark_covers_all_outer_folds() -> None:
    frame = pd.read_csv(RUNTIME_PATH)

    assert len(frame) == 5
    assert set(frame["outer_fold"]) == set(range(5))
    assert (frame["eval_n"] > 0).all()
    assert (
        frame["timed_repetitions"] == 200
    ).all()
    assert (
        frame["mean_per_example_ms"] > 0.0
    ).all()
    assert np.isfinite(
        frame["mean_per_example_ms"]
    ).all()


def test_random_summary_is_complete() -> None:
    frame = pd.read_csv(RANDOM_SUMMARY_PATH)

    assert len(frame) == 10
    assert (frame["repetitions"] == 100).all()
    assert set(frame["budget"]) == {
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
    assert (
        frame["recall_lower95"]
        <= frame["mean_recall"]
    ).all()
    assert (
        frame["mean_recall"]
        <= frame["recall_upper95"] + 1e-12
    ).all()


def test_frontier_has_exact_deterministic_points() -> None:
    frame = pd.read_csv(FRONTIER_PATH)

    assert len(frame) == 20
    assert set(frame["policy"]) == {
        "learned_decision_value",
        "ordinary_uncertainty",
    }
    assert frame.groupby("policy").size().to_dict() == {
        "learned_decision_value": 10,
        "ordinary_uncertainty": 10,
    }
    assert (
        frame["incremental_cost_ms_per_example"]
        >= 0.0
    ).all()


def test_selective_candidates_are_below_full_acquisition() -> None:
    frame = pd.read_csv(CANDIDATE_PATH)

    assert len(frame) == 9
    assert (
        frame["learned_acquisition_rate"] < 1.0
    ).all()
    assert (
        frame["learned_incremental_cost_ms_per_example"]
        > 0.0
    ).all()


def test_summary_preserves_overall_no_go() -> None:
    summary = json.loads(
        SUMMARY_PATH.read_text(encoding="utf-8")
    )

    assert summary["status"] == (
        "development_only_frontier_completed"
    )
    assert (
        summary["predictability_condition_pass"]
        is False
    )
    assert summary["overall_milestone_pass"] is False
    assert summary["overall_project_status"] == "no-go"
    assert (
        summary["scope"]["final_test_used"]
        is False
    )
    assert (
        summary["scope"]["held_out_shift_used"]
        is False
    )
