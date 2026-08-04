"""Required baseline policies for exact-cost cascade evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from monitor_fusion.policies.exact_cost import (
    ExactCostThresholdMixture,
    apply_threshold_mixture,
    calibrate_exact_cost_threshold_mixture,
    sha256_uniform,
)


SELECTIVE_BASELINE_IDS = (
    "threshold_distance",
    "current_error_prediction",
    "random_acquisition",
)

FROZEN_RANDOM_POLICY_SEEDS = (
    104729,
    130363,
    155921,
    181081,
    205759,
)


@dataclass(frozen=True)
class EndpointPolicyResult:
    """Acquisition and final decisions for a fixed endpoint policy."""

    policy_id: str
    acquisition: NDArray[np.bool_]
    decision: NDArray[np.int64]


def _finite_vector(
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
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return result


def _probability_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    result = _finite_vector(values, name=name)

    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(
            f"{name} must lie in [0, 1]"
        )

    return result


def _binary_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.int64]:
    array = np.asarray(values)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain binary numeric values"
        ) from exc

    if not np.all(finite) or not np.all(
        np.isin(array, (0, 1))
    ):
        raise ValueError(
            f"{name} must contain only zero and one"
        )

    return array.astype(np.int64, copy=False)


def _example_ids(
    values: Iterable[object],
) -> list[str]:
    identifiers = [str(value) for value in values]

    if not identifiers:
        raise ValueError(
            "example_ids must not be empty"
        )

    if any(not identifier for identifier in identifiers):
        raise ValueError(
            "example_ids must not contain empty identifiers"
        )

    return identifiers


def threshold_distance_score(
    base_probability: ArrayLike,
    *,
    frozen_base_decision_threshold: float,
) -> NDArray[np.float64]:
    """Return negative distance from the frozen base threshold."""

    probability = _probability_vector(
        base_probability,
        name="base_probability",
    )

    if (
        not np.isfinite(frozen_base_decision_threshold)
        or not 0.0
        <= frozen_base_decision_threshold
        <= 1.0
    ):
        raise ValueError(
            "frozen_base_decision_threshold must lie in [0, 1]"
        )

    return -np.abs(
        probability
        - float(frozen_base_decision_threshold)
    )


def current_error_prediction_score(
    cross_fitted_error_probability: ArrayLike,
) -> NDArray[np.float64]:
    """Return probability that the frozen base policy is wrong."""

    return _probability_vector(
        cross_fitted_error_probability,
        name="cross_fitted_error_probability",
    ).copy()


def random_acquisition_score(
    example_ids: Iterable[object],
    *,
    policy_seed: int,
) -> NDArray[np.float64]:
    """Return deterministic hash-uniform random acquisition scores."""

    if policy_seed not in FROZEN_RANDOM_POLICY_SEEDS:
        raise ValueError(
            "policy_seed is not one of the five frozen seeds"
        )

    identifiers = _example_ids(example_ids)

    return np.fromiter(
        (
            sha256_uniform(
                identifier,
                policy_id="random_acquisition",
                hash_seed=policy_seed,
            )
            for identifier in identifiers
        ),
        dtype=np.float64,
        count=len(identifiers),
    )


def calibrate_selective_baseline_exact_cost(
    scores: ArrayLike,
    no_acquisition_total_cost_ms: ArrayLike,
    acquisition_total_cost_ms: ArrayLike,
    *,
    absolute_cost_budget_ms: float,
    policy_id: str,
    boundary_hash_seed: int,
) -> ExactCostThresholdMixture:
    """Calibrate a required selective baseline at exact expected cost."""

    if policy_id not in SELECTIVE_BASELINE_IDS:
        raise ValueError(
            "policy_id must name a required selective baseline"
        )

    return calibrate_exact_cost_threshold_mixture(
        scores,
        no_acquisition_total_cost_ms,
        acquisition_total_cost_ms,
        absolute_cost_budget_ms=absolute_cost_budget_ms,
        policy_id=policy_id,
        hash_seed=boundary_hash_seed,
    )


def apply_selective_baseline(
    scores: ArrayLike,
    example_ids: Iterable[object],
    policy: ExactCostThresholdMixture,
) -> NDArray[np.bool_]:
    """Apply a frozen baseline without evaluation-set ranking."""

    if policy.policy_id not in SELECTIVE_BASELINE_IDS:
        raise ValueError(
            "policy artifact is not a selective baseline"
        )

    return apply_threshold_mixture(
        scores,
        example_ids,
        policy,
    )


def never_acquire_endpoint(
    base_decision: ArrayLike,
) -> EndpointPolicyResult:
    """Return the frozen base-policy endpoint."""

    decision = _binary_vector(
        base_decision,
        name="base_decision",
    )

    return EndpointPolicyResult(
        policy_id="never_acquire",
        acquisition=np.zeros(
            len(decision),
            dtype=np.bool_,
        ),
        decision=decision.copy(),
    )


def always_acquire_endpoint(
    augmented_decision: ArrayLike,
) -> EndpointPolicyResult:
    """Return the always-acquire augmented endpoint."""

    decision = _binary_vector(
        augmented_decision,
        name="augmented_decision",
    )

    return EndpointPolicyResult(
        policy_id="always_acquire",
        acquisition=np.ones(
            len(decision),
            dtype=np.bool_,
        ),
        decision=decision.copy(),
    )


def direct_fusion_endpoint(
    full_information_fusion_decision: ArrayLike,
) -> EndpointPolicyResult:
    """Return the full-cost directly trained fusion baseline."""

    decision = _binary_vector(
        full_information_fusion_decision,
        name="full_information_fusion_decision",
    )

    return EndpointPolicyResult(
        policy_id="direct_fusion",
        acquisition=np.ones(
            len(decision),
            dtype=np.bool_,
        ),
        decision=decision.copy(),
    )
