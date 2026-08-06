"""Independent calibration enforcement for protocol-v2 policies."""

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

POLICY_TRAIN_SPLIT = "policy_train"
POLICY_SELECTION_SPLIT = "policy_selection"
CALIBRATION_SPLIT = "calibration"

DEVELOPMENT_SPLITS = (
    POLICY_TRAIN_SPLIT,
    POLICY_SELECTION_SPLIT,
    CALIBRATION_SPLIT,
)

PROTECTED_SPLITS = frozenset(
    {
        "final_test",
        "held_out_shift",
    }
)


@dataclass(frozen=True)
class IndependentDevelopmentRoles:
    """Non-overlapping row indices for the frozen development roles."""

    policy_train_indices: IntArray
    policy_selection_indices: IntArray
    calibration_indices: IntArray


@dataclass(frozen=True)
class IndependentlyCalibratedPolicy:
    """A policy threshold calibrated only on the frozen calibration split."""

    selected_candidate_id: str
    calibration_split: str
    calibration_example_count: int
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
        raise ValueError(
            f"{name} contains an empty value"
        )

    return array


def partition_independent_development_roles(
    split: Iterable[object],
    example_id: Iterable[object],
) -> IndependentDevelopmentRoles:
    """Validate and partition the three frozen development roles."""

    splits = _string_vector(
        split,
        name="split",
    )
    example_ids = _string_vector(
        example_id,
        name="example_id",
    )

    if len(splits) != len(example_ids):
        raise ValueError(
            "split and example_id lengths differ"
        )

    if len(set(example_ids.tolist())) != len(example_ids):
        raise ValueError(
            "example_id values must be unique"
        )

    observed = set(splits.tolist())

    protected = sorted(
        observed.intersection(PROTECTED_SPLITS)
    )

    if protected:
        raise ValueError(
            "Protected splits are forbidden: "
            + ", ".join(protected)
        )

    expected = set(DEVELOPMENT_SPLITS)

    if observed != expected:
        raise ValueError(
            "Development rows must contain exactly "
            "policy_train, policy_selection, and calibration"
        )

    roles = IndependentDevelopmentRoles(
        policy_train_indices=np.flatnonzero(
            splits == POLICY_TRAIN_SPLIT
        ).astype(np.int64),
        policy_selection_indices=np.flatnonzero(
            splits == POLICY_SELECTION_SPLIT
        ).astype(np.int64),
        calibration_indices=np.flatnonzero(
            splits == CALIBRATION_SPLIT
        ).astype(np.int64),
    )

    for indices in (
        roles.policy_train_indices,
        roles.policy_selection_indices,
        roles.calibration_indices,
    ):
        if indices.size == 0:
            raise ValueError(
                "Every development role must contain rows"
            )

    return roles


def calibrate_selected_policy_independently(
    scores: ArrayLike,
    no_acquisition_total_cost_ms: ArrayLike,
    acquisition_total_cost_ms: ArrayLike,
    *,
    split: Iterable[object],
    example_id: Iterable[object],
    selected_candidate_id: str,
    absolute_cost_budget_ms: float,
    policy_id: str,
    boundary_hash_seed: int,
) -> IndependentlyCalibratedPolicy:
    """Calibrate an online threshold using calibration rows only."""

    roles = partition_independent_development_roles(
        split,
        example_id,
    )

    score_array = np.asarray(
        scores,
        dtype=np.float64,
    )
    no_acquisition = np.asarray(
        no_acquisition_total_cost_ms,
        dtype=np.float64,
    )
    acquisition = np.asarray(
        acquisition_total_cost_ms,
        dtype=np.float64,
    )

    expected_length = (
        len(roles.policy_train_indices)
        + len(roles.policy_selection_indices)
        + len(roles.calibration_indices)
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
        if array.ndim != 1 or len(array) != expected_length:
            raise ValueError(
                f"{name} must match the development row count"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains non-finite values"
            )

    if not selected_candidate_id.strip():
        raise ValueError(
            "selected_candidate_id must not be empty"
        )

    calibration = roles.calibration_indices

    policy = calibrate_exact_cost_threshold_mixture(
        score_array[calibration],
        no_acquisition[calibration],
        acquisition[calibration],
        absolute_cost_budget_ms=absolute_cost_budget_ms,
        policy_id=policy_id,
        hash_seed=boundary_hash_seed,
    )

    return IndependentlyCalibratedPolicy(
        selected_candidate_id=selected_candidate_id,
        calibration_split=CALIBRATION_SPLIT,
        calibration_example_count=len(calibration),
        policy=policy,
    )
