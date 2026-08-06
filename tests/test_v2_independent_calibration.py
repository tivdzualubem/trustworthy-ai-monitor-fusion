from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.independent_calibration import (
    calibrate_selected_policy_independently,
    partition_independent_development_roles,
)


def development_rows():
    splits = np.array(
        [
            "policy_train",
            "policy_train",
            "policy_selection",
            "policy_selection",
            "calibration",
            "calibration",
            "calibration",
            "calibration",
        ],
        dtype=object,
    )

    example_ids = np.array(
        [f"example-{index}" for index in range(len(splits))],
        dtype=object,
    )

    return splits, example_ids


def test_frozen_roles_are_disjoint_and_complete() -> None:
    splits, example_ids = development_rows()

    roles = partition_independent_development_roles(
        splits,
        example_ids,
    )

    all_indices = np.concatenate(
        [
            roles.policy_train_indices,
            roles.policy_selection_indices,
            roles.calibration_indices,
        ]
    )

    assert len(np.unique(all_indices)) == len(splits)
    assert set(all_indices.tolist()) == set(range(len(splits)))


def test_threshold_uses_only_calibration_rows() -> None:
    splits, example_ids = development_rows()

    scores = np.array(
        [100, 90, 80, 70, 0.9, 0.8, 0.2, 0.1],
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
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=60.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    changed_non_calibration = scores.copy()
    changed_non_calibration[:4] = [-1000, -900, -800, -700]

    second = calibrate_selected_policy_independently(
        changed_non_calibration,
        no_cost,
        acquisition_cost,
        split=splits,
        example_id=example_ids,
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=60.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    assert first.policy == second.policy
    assert first.calibration_split == "calibration"
    assert first.calibration_example_count == 4


def test_calibration_changes_when_calibration_scores_change() -> None:
    splits, example_ids = development_rows()

    scores = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.2, 0.1]
    )

    no_cost = np.full(len(scores), 10.0)
    acquisition_cost = np.full(len(scores), 110.0)

    first = calibrate_selected_policy_independently(
        scores,
        no_cost,
        acquisition_cost,
        split=splits,
        example_id=example_ids,
        selected_candidate_id="Ridge:000",
        absolute_cost_budget_ms=35.0,
        policy_id="signed_value_router",
        boundary_hash_seed=1729,
    )

    changed = scores.copy()
    changed[4:] = [0.05, 0.04, 0.03, 0.02]

    second = calibrate_selected_policy_independently(
        changed,
        no_cost,
        acquisition_cost,
        split=splits,
        example_id=example_ids,
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
            "policy_selection",
            "final_test",
        ],
        [
            "policy_train",
            "policy_selection",
        ],
        [
            "policy_train",
            "policy_selection",
            "calibration",
            "unexpected",
        ],
    ],
)
def test_invalid_split_sets_fail_closed(
    splits: list[str],
) -> None:
    ids = [f"id-{index}" for index in range(len(splits))]

    with pytest.raises(ValueError):
        partition_independent_development_roles(
            splits,
            ids,
        )


def test_duplicate_example_ids_fail_closed() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        partition_independent_development_roles(
            [
                "policy_train",
                "policy_selection",
                "calibration",
            ],
            ["same", "same", "other"],
        )
