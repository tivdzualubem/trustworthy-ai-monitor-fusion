from __future__ import annotations

import inspect

import numpy as np
import pytest

from monitor_fusion.policies.baselines import (
    FROZEN_RANDOM_POLICY_SEEDS,
    SELECTIVE_BASELINE_IDS,
    always_acquire_endpoint,
    apply_selective_baseline,
    calibrate_selective_baseline_exact_cost,
    current_error_prediction_score,
    direct_fusion_endpoint,
    never_acquire_endpoint,
    random_acquisition_score,
    threshold_distance_score,
)


def synthetic_costs() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    return (
        np.full(6, 10.0),
        np.array(
            [
                110.0,
                160.0,
                210.0,
                260.0,
                310.0,
                360.0,
            ]
        ),
    )


def test_required_baseline_registry() -> None:
    assert set(SELECTIVE_BASELINE_IDS) == {
        "threshold_distance",
        "current_error_prediction",
        "random_acquisition",
    }

    assert FROZEN_RANDOM_POLICY_SEEDS == (
        104729,
        130363,
        155921,
        181081,
        205759,
    )


def test_threshold_distance_matches_protocol() -> None:
    score = threshold_distance_score(
        [0.1, 0.4, 0.5, 0.6, 0.9],
        frozen_base_decision_threshold=0.5,
    )

    np.testing.assert_allclose(
        score,
        [-0.4, -0.1, 0.0, -0.1, -0.4],
    )


def test_threshold_distance_is_symmetric() -> None:
    score = threshold_distance_score(
        [0.3, 0.7],
        frozen_base_decision_threshold=0.5,
    )

    assert score[0] == pytest.approx(score[1])


def test_current_error_prediction_is_identity_score() -> None:
    probability = np.array([0.05, 0.2, 0.8])

    score = current_error_prediction_score(
        probability
    )

    np.testing.assert_allclose(
        score,
        probability,
    )

    assert score is not probability


def test_random_score_is_deterministic() -> None:
    identifiers = np.array(
        ["a", "b", "c", "d"]
    )

    first_seed = FROZEN_RANDOM_POLICY_SEEDS[0]
    second_seed = FROZEN_RANDOM_POLICY_SEEDS[1]

    forward = random_acquisition_score(
        identifiers,
        policy_seed=first_seed,
    )

    repeated = random_acquisition_score(
        identifiers,
        policy_seed=first_seed,
    )

    reverse = random_acquisition_score(
        identifiers[::-1],
        policy_seed=first_seed,
    )

    other_seed = random_acquisition_score(
        identifiers,
        policy_seed=second_seed,
    )

    np.testing.assert_array_equal(
        forward,
        repeated,
    )

    np.testing.assert_array_equal(
        forward,
        reverse[::-1],
    )

    assert not np.array_equal(
        forward,
        other_seed,
    )

    assert np.all(
        (forward >= 0.0)
        & (forward < 1.0)
    )


def test_random_score_rejects_unfrozen_seed() -> None:
    with pytest.raises(
        ValueError,
        match="five frozen",
    ):
        random_acquisition_score(
            ["a", "b"],
            policy_seed=1,
        )


@pytest.mark.parametrize(
    "policy_id",
    SELECTIVE_BASELINE_IDS,
)
def test_selective_baselines_use_same_exact_cost(
    policy_id: str,
) -> None:
    no_acquisition, acquisition = synthetic_costs()

    identifiers = np.array(
        ["a", "b", "c", "d", "e", "f"]
    )

    if policy_id == "threshold_distance":
        scores = threshold_distance_score(
            [
                0.05,
                0.2,
                0.45,
                0.55,
                0.8,
                0.95,
            ],
            frozen_base_decision_threshold=0.5,
        )
    elif policy_id == "current_error_prediction":
        scores = current_error_prediction_score(
            [
                0.1,
                0.9,
                0.3,
                0.8,
                0.2,
                0.7,
            ]
        )
    else:
        scores = random_acquisition_score(
            identifiers,
            policy_seed=(
                FROZEN_RANDOM_POLICY_SEEDS[0]
            ),
        )

    policy = calibrate_selective_baseline_exact_cost(
        scores,
        no_acquisition,
        acquisition,
        absolute_cost_budget_ms=60.0,
        policy_id=policy_id,
        boundary_hash_seed=1729,
    )

    assert policy.policy_id == policy_id

    assert (
        policy.calibration_expected_total_cost_ms
        == pytest.approx(60.0)
    )

    forward = apply_selective_baseline(
        scores,
        identifiers,
        policy,
    )

    reverse = apply_selective_baseline(
        scores[::-1],
        identifiers[::-1],
        policy,
    )

    np.testing.assert_array_equal(
        forward,
        reverse[::-1],
    )


def test_direct_fusion_is_full_cost_endpoint() -> None:
    result = direct_fusion_endpoint(
        [0, 1, 1, 0]
    )

    assert result.policy_id == "direct_fusion"

    np.testing.assert_array_equal(
        result.acquisition,
        [True, True, True, True],
    )

    np.testing.assert_array_equal(
        result.decision,
        [0, 1, 1, 0],
    )


def test_direct_fusion_is_not_selective_cost_matched() -> None:
    with pytest.raises(
        ValueError,
        match="selective baseline",
    ):
        calibrate_selective_baseline_exact_cost(
            [0.1, 0.9],
            [10.0, 10.0],
            [100.0, 100.0],
            absolute_cost_budget_ms=55.0,
            policy_id="direct_fusion",
            boundary_hash_seed=1729,
        )


def test_never_and_always_acquire_endpoints() -> None:
    never = never_acquire_endpoint(
        [0, 1, 0]
    )

    always = always_acquire_endpoint(
        [1, 1, 0]
    )

    assert never.policy_id == "never_acquire"
    assert always.policy_id == "always_acquire"

    np.testing.assert_array_equal(
        never.acquisition,
        [False, False, False],
    )

    np.testing.assert_array_equal(
        always.acquisition,
        [True, True, True],
    )

    np.testing.assert_array_equal(
        never.decision,
        [0, 1, 0],
    )

    np.testing.assert_array_equal(
        always.decision,
        [1, 1, 0],
    )


def test_pre_acquisition_interfaces_exclude_forbidden_inputs() -> None:
    functions = (
        threshold_distance_score,
        current_error_prediction_score,
        random_acquisition_score,
    )

    forbidden = {
        "y_true",
        "labels",
        "optional_monitor_output",
        "optional_probability",
        "post_acquisition_latency",
    }

    for function in functions:
        parameters = set(
            inspect.signature(function).parameters
        )

        assert parameters.isdisjoint(
            forbidden
        )


@pytest.mark.parametrize(
    ("function", "args", "kwargs"),
    [
        (
            threshold_distance_score,
            [[-0.1, 0.5]],
            {
                "frozen_base_decision_threshold": 0.5
            },
        ),
        (
            threshold_distance_score,
            [[0.1, 0.5]],
            {
                "frozen_base_decision_threshold": 1.1
            },
        ),
        (
            current_error_prediction_score,
            [[0.2, 1.2]],
            {},
        ),
        (
            direct_fusion_endpoint,
            [[0, 2]],
            {},
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    function: object,
    args: list[object],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        function(
            *args,
            **kwargs,
        )
