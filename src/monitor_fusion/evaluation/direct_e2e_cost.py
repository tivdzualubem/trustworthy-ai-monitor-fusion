"""Fail-closed distinction between direct E2E cost and component diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from monitor_fusion.policies.exact_cost import (
    ExactCostThresholdMixture,
    calibrate_exact_cost_threshold_mixture,
    total_cost_per_example,
)


FloatArray = NDArray[np.float64]

DIRECT_E2E_ESTIMAND = (
    "direct_wall_clock_end_to_end_policy_latency_ms"
)

COMPONENT_SUM_DIAGNOSTIC = (
    "summed_component_latency_ms_diagnostic_only"
)


def _cost_vector(
    values: ArrayLike,
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    if np.any(array < 0.0):
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return array


@dataclass(frozen=True)
class PolicyCostEvidence:
    """Measured policy costs with an explicit estimand identity."""

    no_acquisition_ms: FloatArray
    acquisition_ms: FloatArray
    estimand: str

    @classmethod
    def direct_e2e(
        cls,
        no_acquisition_ms: ArrayLike,
        acquisition_ms: ArrayLike,
    ) -> "PolicyCostEvidence":
        return cls._build(
            no_acquisition_ms,
            acquisition_ms,
            estimand=DIRECT_E2E_ESTIMAND,
        )

    @classmethod
    def component_sum_diagnostic(
        cls,
        no_acquisition_ms: ArrayLike,
        acquisition_ms: ArrayLike,
    ) -> "PolicyCostEvidence":
        return cls._build(
            no_acquisition_ms,
            acquisition_ms,
            estimand=COMPONENT_SUM_DIAGNOSTIC,
        )

    @classmethod
    def _build(
        cls,
        no_acquisition_ms: ArrayLike,
        acquisition_ms: ArrayLike,
        *,
        estimand: str,
    ) -> "PolicyCostEvidence":
        without_optional = _cost_vector(
            no_acquisition_ms,
            name="no_acquisition_ms",
        )

        with_optional = _cost_vector(
            acquisition_ms,
            name="acquisition_ms",
        )

        if len(without_optional) != len(with_optional):
            raise ValueError(
                "cost arrays must have equal length"
            )

        if np.any(with_optional < without_optional):
            raise ValueError(
                "acquisition cost must not be lower than "
                "no-acquisition cost"
            )

        return cls(
            no_acquisition_ms=without_optional.copy(),
            acquisition_ms=with_optional.copy(),
            estimand=estimand,
        )

    def require_direct_e2e(self) -> None:
        if self.estimand != DIRECT_E2E_ESTIMAND:
            raise ValueError(
                "exact-cost calibration requires directly "
                "measured wall-clock end-to-end policy latency; "
                "summed component latency is diagnostic only"
            )


def calibrate_direct_e2e_threshold_mixture(
    scores: ArrayLike,
    costs: PolicyCostEvidence,
    *,
    absolute_cost_budget_ms: float,
    policy_id: str,
    hash_seed: int,
) -> ExactCostThresholdMixture:
    """Calibrate only from direct wall-clock E2E policy costs."""

    costs.require_direct_e2e()

    return calibrate_exact_cost_threshold_mixture(
        scores,
        costs.no_acquisition_ms,
        costs.acquisition_ms,
        absolute_cost_budget_ms=absolute_cost_budget_ms,
        policy_id=policy_id,
        hash_seed=hash_seed,
    )


def realized_direct_e2e_cost(
    acquisition: ArrayLike,
    costs: PolicyCostEvidence,
) -> FloatArray:
    """Return realized total cost only for direct E2E evidence."""

    costs.require_direct_e2e()

    return total_cost_per_example(
        acquisition,
        costs.no_acquisition_ms,
        costs.acquisition_ms,
    )
