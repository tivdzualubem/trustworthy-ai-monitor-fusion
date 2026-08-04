from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.signed_value import (
    acquire_by_net_value,
    cost_aware_signed_value_score,
    realized_policy_specific_value,
    zero_one_loss,
)


def test_realized_value_is_signed_and_policy_specific() -> None:
    labels = np.array([1, 0, 1, 0])
    base = np.array([0, 0, 1, 1])
    augmented_a = np.array([1, 1, 0, 1])
    augmented_b = np.array([0, 0, 1, 1])

    value_a = realized_policy_specific_value(labels, base, augmented_a)
    value_b = realized_policy_specific_value(labels, base, augmented_b)

    np.testing.assert_array_equal(value_a, np.array([1, -1, -1, 0]))
    np.testing.assert_array_equal(value_b, np.zeros(4, dtype=int))
    assert set(value_a) == {-1, 0, 1}


def test_realized_value_equals_difference_in_zero_one_loss() -> None:
    labels = np.array([0, 1, 1, 0, 1])
    base = np.array([0, 0, 1, 1, 0])
    augmented = np.array([1, 1, 1, 0, 0])

    expected = zero_one_loss(labels, base) - zero_one_loss(labels, augmented)
    actual = realized_policy_specific_value(labels, base, augmented)

    np.testing.assert_array_equal(actual, expected)


def test_cost_aware_score_uses_prespecified_positive_floor() -> None:
    score = cost_aware_signed_value_score(
        [0.5, -0.5, 1.0],
        [0.0, 0.5, 4.0],
        cost_floor_ms=1.0,
    )

    np.testing.assert_allclose(score, [0.5, -0.5, 0.25])


def test_net_value_rule_is_strict_at_the_boundary() -> None:
    acquisition = acquire_by_net_value(
        [0.2, 0.2001, -0.1],
        [2.0, 2.0, 1.0],
        cost_weight_per_ms=0.1,
    )

    np.testing.assert_array_equal(acquisition, [False, True, False])


@pytest.mark.parametrize(
    ("labels", "base", "augmented"),
    [
        ([0, 2], [0, 1], [0, 1]),
        ([0, 1], [0], [0, 1]),
        ([0, 1], [0, 1], [[0, 1]]),
    ],
)
def test_realized_value_rejects_invalid_inputs(
    labels: object,
    base: object,
    augmented: object,
) -> None:
    with pytest.raises(ValueError):
        realized_policy_specific_value(labels, base, augmented)


def test_cost_functions_reject_negative_or_nonfinite_costs() -> None:
    with pytest.raises(ValueError):
        cost_aware_signed_value_score([0.1], [-1.0])
    with pytest.raises(ValueError):
        acquire_by_net_value(
            [0.1],
            [np.inf],
            cost_weight_per_ms=0.1,
        )
