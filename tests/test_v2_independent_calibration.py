from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from monitor_fusion.evaluation.independent_calibration import (
    FRESH_CALIBRATION_OPTIMIZATION_SPLIT,
    FRESH_CALIBRATION_RISK_SPLIT,
    calibrate_selected_policy_independently,
    partition_fresh_calibration_roles,
)


def fresh_calibration_rows():
    splits = np.array(
        [
            "fresh_calibration_optimization",
            "fresh_calibration_optimization",
            "fresh_calibration_optimization",
            "fresh_calibration_optimization",
            "fresh_calibration_risk",
            "fresh_calibration_risk",
            "fresh_calibration_risk",
            "fresh_calibration_risk",
        ],
        dtype=object,
    )

    example_ids = np.array(
        [
            f"fresh-example-{index}"
            for index in range(len(splits))
        ],
        dtype=object,
    )

    effective_groups = np.array(
        [
            f"fresh-group-{index}"
            for index in range(len(splits))
        ],
        dtype=object,
    )

    return splits, example_ids, effective_groups


def test_split_ids_match_frozen_protocol() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "configs/exact_cost_risk_cascade_protocol_v2.json"
    )

    protocol = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )

    assert set(protocol["scope"]["fresh_split_ids"]) == {
        FRESH_CALIBRATION_OPTIMIZATION_SPLIT,
        FRESH_CALIBRATION_RISK_SPLIT,
        "fresh_confirmatory",
    }

    assert protocol["independent_calibration"]["source"] == (
        "new_data_not_used_for_training_model_selection_"
        "or_legacy_evaluation"
    )


def test_fresh_roles_are_disjoint_and_complete() -> None:
    splits, example_ids, groups = fresh_calibration_rows()

    roles = partition_fresh_calibration_roles(
        splits,
        example_ids,
        groups,
    )

    all_indices = np.concatenate(
        [
            roles.optimization_indices,
            roles.risk_testing_indices,
        ]
    )

    assert len(np.unique(all_indices)) == len(splits)
    assert set(all_indices.tolist()) == set(
        range(len(splits))
    )


def test_optimizer_uses_only_fresh_optimization_rows() -> None:
    splits, example_ids, groups = fresh_calibration_rows()

    scores = np.array(
        [0.9, 0.8, 0.2, 0.1, 100, 90, 80, 70],
        dtype=float,
    )
    no_cost = np.full(len(scores), 10.0)
    acquisition_cost = np.full(len(scores), 110.0)

    first = calibrate_selected_policy_independently(
        scores,
        no_cost,
        acquisition_cost,
        split=splits,
        example_id=example_ids,
        effective_group=groups,
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=35.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    changed_scores = scores.copy()
    changed_scores[4:] = [-1000, -900, -800, -700]

    changed_no_cost = no_cost.copy()
    changed_no_cost[4:] = [1000, 1100, 1200, 1300]

    changed_acquisition_cost = acquisition_cost.copy()
    changed_acquisition_cost[4:] = [
        2000,
        2100,
        2200,
        2300,
    ]

    second = calibrate_selected_policy_independently(
        changed_scores,
        changed_no_cost,
        changed_acquisition_cost,
        split=splits,
        example_id=example_ids,
        effective_group=groups,
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=35.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    assert first.policy == second.policy
    assert first.optimization_split == (
        "fresh_calibration_optimization"
    )
    assert first.risk_testing_split == (
        "fresh_calibration_risk"
    )
    assert first.optimization_example_count == 4
    assert first.risk_testing_example_count == 4


def test_optimization_changes_only_from_optimization_rows() -> None:
    splits, example_ids, groups = fresh_calibration_rows()

    scores = np.array(
        [0.9, 0.8, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1],
        dtype=float,
    )
    no_cost = np.full(len(scores), 10.0)
    acquisition_cost = np.full(len(scores), 110.0)

    first = calibrate_selected_policy_independently(
        scores,
        no_cost,
        acquisition_cost,
        split=splits,
        example_id=example_ids,
        effective_group=groups,
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=35.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    changed = scores.copy()
    changed[:4] = [0.05, 0.04, 0.03, 0.02]

    second = calibrate_selected_policy_independently(
        changed,
        no_cost,
        acquisition_cost,
        split=splits,
        example_id=example_ids,
        effective_group=groups,
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=35.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    assert first.policy != second.policy


@pytest.mark.parametrize(
    "splits",
    [
        [
            "policy_train",
            "fresh_calibration_risk",
        ],
        [
            "calibration",
            "fresh_calibration_risk",
        ],
        [
            "final_test",
            "fresh_calibration_risk",
        ],
        [
            "fresh_confirmatory",
            "fresh_calibration_risk",
        ],
        [
            "fresh_calibration_optimization",
        ],
        [
            "fresh_calibration_optimization",
            "unexpected",
        ],
    ],
)
def test_non_fresh_calibration_split_sets_fail_closed(
    splits: list[str],
) -> None:
    ids = [
        f"id-{index}"
        for index in range(len(splits))
    ]
    groups = [
        f"group-{index}"
        for index in range(len(splits))
    ]

    with pytest.raises(ValueError):
        partition_fresh_calibration_roles(
            splits,
            ids,
            groups,
        )


def test_effective_group_cannot_cross_role_boundary() -> None:
    with pytest.raises(
        ValueError,
        match="effective_group",
    ):
        partition_fresh_calibration_roles(
            [
                "fresh_calibration_optimization",
                "fresh_calibration_risk",
            ],
            ["optimization-example", "risk-example"],
            ["shared-group", "shared-group"],
        )


def test_duplicate_example_ids_fail_closed() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        partition_fresh_calibration_roles(
            [
                "fresh_calibration_optimization",
                "fresh_calibration_risk",
            ],
            ["same", "same"],
            ["optimization-group", "risk-group"],
        )


def test_blank_selected_candidate_fails_closed() -> None:
    splits, example_ids, groups = fresh_calibration_rows()
    scores = np.arange(len(splits), dtype=float)
    no_cost = np.full(len(splits), 10.0)
    acquisition_cost = np.full(len(splits), 110.0)

    with pytest.raises(
        ValueError,
        match="selected_candidate_id",
    ):
        calibrate_selected_policy_independently(
            scores,
            no_cost,
            acquisition_cost,
            split=splits,
            example_id=example_ids,
            effective_group=groups,
            selected_candidate_id=" ",
            absolute_cost_budget_ms=35.0,
            policy_id="signed_value_router",
            boundary_hash_seed=1729,
        )
