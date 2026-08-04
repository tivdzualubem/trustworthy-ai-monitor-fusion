"""Policy-specific signed decision-value estimands.

The realized target compares two fixed downstream policies on the same
example.  Positive values mean that the augmented policy corrects a base
policy error; negative values mean that augmentation introduces an error.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _one_dimensional_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.generic]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _binary_array(values: ArrayLike, *, name: str) -> IntArray:
    array = _one_dimensional_array(values, name=name)
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric binary values") from exc
    if not bool(np.all(finite)):
        raise ValueError(f"{name} contains non-finite values")
    if not bool(np.all(np.isin(array, (0, 1)))):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.int64, copy=False)


def _finite_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = _one_dimensional_array(values, name=name)
    try:
        result = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _require_equal_length(
    reference: NDArray[np.generic],
    **arrays: NDArray[np.generic],
) -> None:
    for name, array in arrays.items():
        if len(array) != len(reference):
            raise ValueError(
                f"{name} has length {len(array)}; expected {len(reference)}"
            )


def zero_one_loss(y_true: ArrayLike, prediction: ArrayLike) -> IntArray:
    """Return per-example zero-one loss for a fixed binary policy."""

    labels = _binary_array(y_true, name="y_true")
    decisions = _binary_array(prediction, name="prediction")
    _require_equal_length(labels, prediction=decisions)
    return (labels != decisions).astype(np.int64)


def realized_policy_specific_value(
    y_true: ArrayLike,
    base_prediction: ArrayLike,
    augmented_prediction: ArrayLike,
) -> IntArray:
    """Compute ``L(y, base) - L(y, augmented)`` per example.

    Under zero-one loss, every target is one of ``-1``, ``0``, or ``1``.
    The target is tied to the supplied fixed policies; it is not an intrinsic
    nonnegative value of the optional monitor.
    """

    labels = _binary_array(y_true, name="y_true")
    base = _binary_array(base_prediction, name="base_prediction")
    augmented = _binary_array(
        augmented_prediction,
        name="augmented_prediction",
    )
    _require_equal_length(
        labels,
        base_prediction=base,
        augmented_prediction=augmented,
    )
    base_loss = (labels != base).astype(np.int64)
    augmented_loss = (labels != augmented).astype(np.int64)
    return base_loss - augmented_loss


def cost_aware_signed_value_score(
    estimated_signed_value: ArrayLike,
    estimated_incremental_cost_ms: ArrayLike,
    *,
    cost_floor_ms: float = 1.0,
) -> FloatArray:
    """Return the online signed-value-per-cost score.

    The positive cost floor prevents unstable ratios and is fixed before
    policy calibration.  Both inputs must be available before acquisition.
    """

    if not np.isfinite(cost_floor_ms) or cost_floor_ms <= 0.0:
        raise ValueError("cost_floor_ms must be finite and positive")
    value = _finite_float_array(
        estimated_signed_value,
        name="estimated_signed_value",
    )
    cost = _finite_float_array(
        estimated_incremental_cost_ms,
        name="estimated_incremental_cost_ms",
    )
    _require_equal_length(value, estimated_incremental_cost_ms=cost)
    if bool(np.any(cost < 0.0)):
        raise ValueError("estimated_incremental_cost_ms must be nonnegative")
    return value / np.maximum(cost, float(cost_floor_ms))


def acquire_by_net_value(
    estimated_signed_value: ArrayLike,
    estimated_incremental_cost_ms: ArrayLike,
    *,
    cost_weight_per_ms: float,
) -> BoolArray:
    """Apply the strict rule ``value - weight * cost > 0``."""

    if (
        not np.isfinite(cost_weight_per_ms)
        or cost_weight_per_ms < 0.0
    ):
        raise ValueError(
            "cost_weight_per_ms must be finite and nonnegative"
        )
    value = _finite_float_array(
        estimated_signed_value,
        name="estimated_signed_value",
    )
    cost = _finite_float_array(
        estimated_incremental_cost_ms,
        name="estimated_incremental_cost_ms",
    )
    _require_equal_length(value, estimated_incremental_cost_ms=cost)
    if bool(np.any(cost < 0.0)):
        raise ValueError("estimated_incremental_cost_ms must be nonnegative")
    return value - float(cost_weight_per_ms) * cost > 0.0
