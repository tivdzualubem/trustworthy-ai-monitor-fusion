"""Fresh independent calibration enforcement for protocol-v2 policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from monitor_fusion.policies.exact_cost import (
    ExactCostThresholdMixture,
    calibrate_exact_cost_threshold_mixture,
)


IntArray = NDArray[np.int64]

FRESH_CALIBRATION_OPTIMIZATION_SPLIT = (
    "fresh_calibration_optimization"
)
FRESH_CALIBRATION_RISK_SPLIT = "fresh_calibration_risk"
FRESH_CONFIRMATORY_SPLIT = "fresh_confirmatory"

FRESH_CALIBRATION_SPLITS = (
    FRESH_CALIBRATION_OPTIMIZATION_SPLIT,
    FRESH_CALIBRATION_RISK_SPLIT,
)

LEGACY_SPLITS = frozenset(
    {
        "policy_train",
        "policy_selection",
        "calibration",
        "final_test",
        "held_out_shift",
    }
)

FORBIDDEN_CALIBRATION_SPLITS = (
    LEGACY_SPLITS | {FRESH_CONFIRMATORY_SPLIT}
)


@dataclass(frozen=True)
class FreshCalibrationRoles:
    """Disjoint optimization and risk-testing row indices."""

    optimization_indices: IntArray
    risk_testing_indices: IntArray


@dataclass(frozen=True)
class IndependentlyCalibratedPolicy:
    """Policy optimized without using independent risk-testing rows."""

    selected_candidate_id: str
    optimization_split: str
    risk_testing_split: str
    optimization_example_count: int
    risk_testing_example_count: int
    policy: ExactCostThresholdMixture


def _string_vector(
    values: Iterable[object],
    *,
    name: str,
) -> NDArray[np.object_]:
    array = np.asarray(
        [str(value) for value in values],
        dtype=object,
    )

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    if any(not str(value).strip() for value in array):
        raise ValueError(f"{name} contains an empty value")

    return array


def partition_fresh_calibration_roles(
    split: Iterable[object],
    example_id: Iterable[object],
    effective_group: Iterable[object],
) -> FreshCalibrationRoles:
    """Validate the frozen fresh-calibration data boundary."""

    splits = _string_vector(split, name="split")
    example_ids = _string_vector(
        example_id,
        name="example_id",
    )
    effective_groups = _string_vector(
        effective_group,
        name="effective_group",
    )

    row_count = len(splits)

    if len(example_ids) != row_count:
        raise ValueError(
            "split and example_id lengths differ"
        )

    if len(effective_groups) != row_count:
        raise ValueError(
            "split and effective_group lengths differ"
        )

    if len(set(example_ids.tolist())) != row_count:
        raise ValueError(
            "example_id values must be unique"
        )

    observed = set(splits.tolist())

    forbidden = sorted(
        observed.intersection(
            FORBIDDEN_CALIBRATION_SPLITS
        )
    )

    if forbidden:
        raise ValueError(
            "Legacy, protected, and confirmatory splits "
            "are forbidden during fresh calibration: "
            + ", ".join(forbidden)
        )

    expected = set(FRESH_CALIBRATION_SPLITS)

    if observed != expected:
        raise ValueError(
            "Fresh calibration rows must contain exactly "
            "fresh_calibration_optimization and "
            "fresh_calibration_risk"
        )

    group_roles: dict[str, str] = {}

    for group, split_name in zip(
        effective_groups,
        splits,
        strict=True,
    ):
        group_name = str(group)
        role_name = str(split_name)
        previous = group_roles.setdefault(
            group_name,
            role_name,
        )

        if previous != role_name:
            raise ValueError(
                "An effective_group may not cross the "
                "fresh calibration optimization/risk boundary"
            )

    roles = FreshCalibrationRoles(
        optimization_indices=np.flatnonzero(
            splits
            == FRESH_CALIBRATION_OPTIMIZATION_SPLIT
        ).astype(np.int64),
        risk_testing_indices=np.flatnonzero(
            splits == FRESH_CALIBRATION_RISK_SPLIT
        ).astype(np.int64),
    )

    if roles.optimization_indices.size == 0:
        raise ValueError(
            "Fresh calibration optimization rows are required"
        )

    if roles.risk_testing_indices.size == 0:
        raise ValueError(
            "Fresh calibration risk-testing rows are required"
        )

    return roles


def calibrate_selected_policy_independently(
    scores: ArrayLike,
    no_acquisition_total_cost_ms: ArrayLike,
    acquisition_total_cost_ms: ArrayLike,
    *,
    split: Iterable[object],
    example_id: Iterable[object],
    effective_group: Iterable[object],
    selected_candidate_id: str,
    absolute_cost_budget_ms: float,
    policy_id: str,
    boundary_hash_seed: int,
) -> IndependentlyCalibratedPolicy:
    """Optimize thresholds using fresh optimization rows only.

    The fresh risk-testing subset is validated and retained for
    later joint FPR and mean-cost certification. It is never passed
    to the threshold optimizer.
    """

    roles = partition_fresh_calibration_roles(
        split,
        example_id,
        effective_group,
    )

    score_array = np.asarray(scores, dtype=np.float64)
    no_acquisition = np.asarray(
        no_acquisition_total_cost_ms,
        dtype=np.float64,
    )
    acquisition = np.asarray(
        acquisition_total_cost_ms,
        dtype=np.float64,
    )

    row_count = (
        len(roles.optimization_indices)
        + len(roles.risk_testing_indices)
    )

    for name, array in (
        ("scores", score_array),
        (
            "no_acquisition_total_cost_ms",
            no_acquisition,
        ),
        (
            "acquisition_total_cost_ms",
            acquisition,
        ),
    ):
        if array.ndim != 1 or len(array) != row_count:
            raise ValueError(
                f"{name} must match the fresh calibration "
                "row count"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains non-finite values"
            )

    if not selected_candidate_id.strip():
        raise ValueError(
            "selected_candidate_id must not be empty"
        )

    optimization = roles.optimization_indices

    policy = calibrate_exact_cost_threshold_mixture(
        score_array[optimization],
        no_acquisition[optimization],
        acquisition[optimization],
        absolute_cost_budget_ms=absolute_cost_budget_ms,
        policy_id=policy_id,
        hash_seed=boundary_hash_seed,
    )

    return IndependentlyCalibratedPolicy(
        selected_candidate_id=selected_candidate_id,
        optimization_split=(
            FRESH_CALIBRATION_OPTIMIZATION_SPLIT
        ),
        risk_testing_split=FRESH_CALIBRATION_RISK_SPLIT,
        optimization_example_count=len(optimization),
        risk_testing_example_count=len(
            roles.risk_testing_indices
        ),
        policy=policy,
    )
