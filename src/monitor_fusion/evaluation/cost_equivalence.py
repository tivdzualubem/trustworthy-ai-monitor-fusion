"""Paired grouped exact-cost equivalence testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PairedCostEquivalenceResult:
    observed_mean_difference_ms: float
    equivalence_margin_ms: float
    lower_one_sided_bound_ms: float
    upper_one_sided_bound_ms: float
    alpha: float
    bootstrap_repetitions: int
    group_count: int
    equivalent: bool


def _numeric_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    try:
        result = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values"
        ) from exc

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")

    return result


def primary_equivalence_margin_ms(
    absolute_cost_budget_ms: float,
) -> float:
    """Historical frozen v2 engineering margin.

    This preserves max(1 ms, 1% of budget) for historical
    reproducibility. It is not an externally justified universal
    equivalence margin and must not be promoted to that role.
    """

    if (
        not np.isfinite(absolute_cost_budget_ms)
        or absolute_cost_budget_ms <= 0
    ):
        raise ValueError(
            "absolute_cost_budget_ms must be finite and positive"
        )

    return max(
        1.0,
        0.01 * float(absolute_cost_budget_ms),
    )


def paired_group_bootstrap_cost_equivalence(
    policy_total_cost_ms: ArrayLike,
    comparator_total_cost_ms: ArrayLike,
    groups: Iterable[object],
    *,
    equivalence_margin_ms: float,
    alpha: float = 0.05,
    bootstrap_repetitions: int = 2000,
    random_seed: int = 1729,
) -> PairedCostEquivalenceResult:
    """Run a paired group-bootstrap two-one-sided equivalence test."""

    policy = _numeric_vector(
        policy_total_cost_ms,
        name="policy_total_cost_ms",
    )
    comparator = _numeric_vector(
        comparator_total_cost_ms,
        name="comparator_total_cost_ms",
    )

    if len(policy) != len(comparator):
        raise ValueError("paired cost arrays must have equal length")

    if np.any(policy < 0) or np.any(comparator < 0):
        raise ValueError("total costs must be nonnegative")

    group_labels = np.asarray(
        [str(value) for value in groups],
        dtype=str,
    )

    if len(group_labels) != len(policy):
        raise ValueError(
            "groups must have the same length as the paired costs"
        )

    if np.any(group_labels == ""):
        raise ValueError("group identifiers must not be empty")

    if (
        not np.isfinite(equivalence_margin_ms)
        or equivalence_margin_ms <= 0
    ):
        raise ValueError(
            "equivalence_margin_ms must be finite and positive"
        )

    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError(
            "alpha must be strictly between zero and one half"
        )

    if (
        isinstance(bootstrap_repetitions, bool)
        or not isinstance(bootstrap_repetitions, int)
        or bootstrap_repetitions < 100
    ):
        raise ValueError(
            "bootstrap_repetitions must be at least 100"
        )

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError(
            "random_seed must be a nonnegative integer"
        )

    difference = policy - comparator

    unique_groups, inverse = np.unique(
        group_labels,
        return_inverse=True,
    )

    group_count = len(unique_groups)

    if group_count < 2:
        raise ValueError(
            "at least two effective groups are required"
        )

    group_sums = np.bincount(
        inverse,
        weights=difference,
        minlength=group_count,
    ).astype(float)

    group_sizes = np.bincount(
        inverse,
        minlength=group_count,
    ).astype(float)

    generator = np.random.default_rng(random_seed)

    bootstrap_means = np.empty(
        bootstrap_repetitions,
        dtype=float,
    )

    for repetition in range(bootstrap_repetitions):
        sampled_groups = generator.integers(
            0,
            group_count,
            size=group_count,
        )

        bootstrap_means[repetition] = (
            np.sum(group_sums[sampled_groups])
            / np.sum(group_sizes[sampled_groups])
        )

    lower_bound = float(
        np.quantile(
            bootstrap_means,
            alpha,
            method="linear",
        )
    )

    upper_bound = float(
        np.quantile(
            bootstrap_means,
            1 - alpha,
            method="linear",
        )
    )

    margin = float(equivalence_margin_ms)

    return PairedCostEquivalenceResult(
        observed_mean_difference_ms=float(
            np.mean(difference)
        ),
        equivalence_margin_ms=margin,
        lower_one_sided_bound_ms=lower_bound,
        upper_one_sided_bound_ms=upper_bound,
        alpha=float(alpha),
        bootstrap_repetitions=bootstrap_repetitions,
        group_count=group_count,
        equivalent=bool(
            lower_bound > -margin
            and upper_bound < margin
        ),
    )
