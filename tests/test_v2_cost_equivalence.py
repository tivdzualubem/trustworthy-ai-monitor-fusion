from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.cost_equivalence import (
    paired_group_bootstrap_cost_equivalence,
    primary_equivalence_margin_ms,
)


def test_primary_margin_matches_frozen_rule() -> None:
    assert primary_equivalence_margin_ms(50.0) == 1.0
    assert primary_equivalence_margin_ms(1000.0) == 10.0


def test_identical_costs_are_equivalent() -> None:
    comparator = np.linspace(100.0, 140.0, 40)

    result = paired_group_bootstrap_cost_equivalence(
        comparator.copy(),
        comparator,
        np.repeat(np.arange(10), 4),
        equivalence_margin_ms=1.0,
        bootstrap_repetitions=500,
        random_seed=1729,
    )

    assert result.observed_mean_difference_ms == 0.0
    assert result.lower_one_sided_bound_ms == 0.0
    assert result.upper_one_sided_bound_ms == 0.0
    assert result.group_count == 10
    assert result.equivalent is True


def test_material_cost_difference_fails() -> None:
    comparator = np.full(60, 100.0)
    policy = comparator + 5.0

    result = paired_group_bootstrap_cost_equivalence(
        policy,
        comparator,
        np.repeat(np.arange(15), 4),
        equivalence_margin_ms=2.0,
        bootstrap_repetitions=500,
        random_seed=2718,
    )

    assert result.observed_mean_difference_ms == 5.0
    assert result.equivalent is False


def test_group_bootstrap_is_deterministic() -> None:
    comparator = np.arange(48, dtype=float) + 100.0

    policy = comparator + np.tile(
        [-1.0, 1.0, 2.0, -2.0],
        12,
    )

    groups = np.repeat(np.arange(12), 4)

    first = paired_group_bootstrap_cost_equivalence(
        policy,
        comparator,
        groups,
        equivalence_margin_ms=2.0,
        bootstrap_repetitions=500,
        random_seed=3141,
    )

    second = paired_group_bootstrap_cost_equivalence(
        policy,
        comparator,
        groups,
        equivalence_margin_ms=2.0,
        bootstrap_repetitions=500,
        random_seed=3141,
    )

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"equivalence_margin_ms": 0.0},
        {"equivalence_margin_ms": 1.0, "alpha": 0.5},
        {
            "equivalence_margin_ms": 1.0,
            "bootstrap_repetitions": 99,
        },
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        paired_group_bootstrap_cost_equivalence(
            [1.0, 2.0],
            [1.0, 2.0],
            ["a", "b"],
            **kwargs,
        )
