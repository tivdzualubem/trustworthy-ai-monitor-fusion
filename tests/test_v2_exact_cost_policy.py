from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.policies.exact_cost import (
    apply_threshold_mixture,
    calibrate_exact_cost_threshold_mixture,
    sha256_uniform,
    total_cost_per_example,
)


def synthetic_cost_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    no_acquisition = np.array([10.0, 10.0, 10.0, 10.0])
    acquisition = np.array([110.0, 210.0, 310.0, 410.0])
    return scores, no_acquisition, acquisition


def test_calibration_matches_absolute_expected_cost_with_heterogeneous_costs() -> None:
    scores, no_acquisition, acquisition = synthetic_cost_inputs()

    policy = calibrate_exact_cost_threshold_mixture(
        scores,
        no_acquisition,
        acquisition,
        absolute_cost_budget_ms=60.0,
        policy_id="signed_value_cost_aware",
        hash_seed=1729,
    )

    assert policy.calibration_upper_threshold_cost_ms == 35.0
    assert policy.calibration_lower_threshold_cost_ms == 85.0
    assert policy.lower_threshold_probability == pytest.approx(0.5)
    assert policy.calibration_expected_total_cost_ms == pytest.approx(60.0)
    assert policy.lower_acquisition_threshold < policy.upper_acquisition_threshold


@pytest.mark.parametrize(
    ("target", "expected_threshold_cost"),
    [(10.0, 10.0), (260.0, 260.0)],
)
def test_exact_endpoint_needs_no_randomized_threshold(
    target: float,
    expected_threshold_cost: float,
) -> None:
    scores, no_acquisition, acquisition = synthetic_cost_inputs()

    policy = calibrate_exact_cost_threshold_mixture(
        scores,
        no_acquisition,
        acquisition,
        absolute_cost_budget_ms=target,
        policy_id="threshold_distance",
        hash_seed=2718,
    )

    assert (
        policy.lower_acquisition_threshold
        == policy.upper_acquisition_threshold
    )
    assert policy.calibration_expected_total_cost_ms == expected_threshold_cost


def test_frozen_policy_is_deterministic_and_order_invariant() -> None:
    scores, no_acquisition, acquisition = synthetic_cost_inputs()
    policy = calibrate_exact_cost_threshold_mixture(
        scores,
        no_acquisition,
        acquisition,
        absolute_cost_budget_ms=60.0,
        policy_id="signed_value_cost_aware",
        hash_seed=3141,
    )
    example_ids = np.array(["a", "b", "c", "d"])

    forward = apply_threshold_mixture(scores, example_ids, policy)
    repeated = apply_threshold_mixture(scores, example_ids, policy)
    reverse = apply_threshold_mixture(scores[::-1], example_ids[::-1], policy)

    np.testing.assert_array_equal(forward, repeated)
    np.testing.assert_array_equal(forward, reverse[::-1])


def test_hash_uniform_is_stable_and_policy_specific() -> None:
    first = sha256_uniform(
        "example-7",
        policy_id="current_error_prediction",
        hash_seed=104729,
    )
    repeated = sha256_uniform(
        "example-7",
        policy_id="current_error_prediction",
        hash_seed=104729,
    )
    different_policy = sha256_uniform(
        "example-7",
        policy_id="random_acquisition",
        hash_seed=104729,
    )

    assert 0.0 <= first < 1.0
    assert first == repeated
    assert first != different_policy


def test_realized_total_cost_uses_the_matching_per_example_path() -> None:
    result = total_cost_per_example(
        [False, True, False, True],
        [10.0, 11.0, 12.0, 13.0],
        [110.0, 211.0, 312.0, 413.0],
    )

    np.testing.assert_allclose(result, [10.0, 211.0, 12.0, 413.0])


def test_calibration_rejects_cost_ceiling_substitution() -> None:
    scores, no_acquisition, acquisition = synthetic_cost_inputs()

    with pytest.raises(ValueError, match="must lie between"):
        calibrate_exact_cost_threshold_mixture(
            scores,
            no_acquisition,
            acquisition,
            absolute_cost_budget_ms=5.0,
            policy_id="signed_value_cost_aware",
            hash_seed=1729,
        )
    with pytest.raises(ValueError, match="must lie between"):
        calibrate_exact_cost_threshold_mixture(
            scores,
            no_acquisition,
            acquisition,
            absolute_cost_budget_ms=300.0,
            policy_id="signed_value_cost_aware",
            hash_seed=1729,
        )


def test_calibration_keeps_equal_scores_as_one_boundary_group() -> None:
    policy = calibrate_exact_cost_threshold_mixture(
        [0.9, 0.9, 0.1],
        [10.0, 10.0, 10.0],
        [110.0, 210.0, 310.0],
        absolute_cost_budget_ms=60.0,
        policy_id="threshold_distance",
        hash_seed=5772,
    )

    assert policy.calibration_upper_threshold_cost_ms == 10.0
    assert policy.calibration_lower_threshold_cost_ms == 110.0
    assert policy.lower_threshold_probability == pytest.approx(0.5)
