"""Frozen heterogeneous optional-monitor cost-prediction primitives."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


COST_TARGET_TRANSFORM = "log1p"
COST_TARGET_INVERSE = "max_0_expm1"
COST_PREDICTION_FLOOR_MS = 1.0


def transform_optional_monitor_latency_ms(
    latency_ms: Iterable[object],
) -> np.ndarray:
    """Apply the frozen log1p training-target transform."""

    latency = np.asarray(
        list(latency_ms),
        dtype=np.float64,
    )

    if (
        latency.ndim != 1
        or latency.size == 0
    ):
        raise ValueError(
            "latency_ms must be a "
            "nonempty one-dimensional vector"
        )

    if not np.all(
        np.isfinite(latency)
    ):
        raise ValueError(
            "latency_ms contains "
            "non-finite values"
        )

    if np.any(
        latency < 0.0
    ):
        raise ValueError(
            "latency_ms must be nonnegative"
        )

    return np.log1p(
        latency
    )


def inverse_optional_monitor_latency_prediction(
    predicted_log_latency: Iterable[object],
    *,
    apply_online_floor: bool = False,
) -> np.ndarray:
    """Invert log1p predictions under the frozen cost contract."""

    predicted = np.asarray(
        list(predicted_log_latency),
        dtype=np.float64,
    )

    if (
        predicted.ndim != 1
        or predicted.size == 0
    ):
        raise ValueError(
            "predicted_log_latency must be "
            "a nonempty one-dimensional vector"
        )

    if not np.all(
        np.isfinite(predicted)
    ):
        raise ValueError(
            "predicted_log_latency "
            "contains non-finite values"
        )

    milliseconds = np.maximum(
        0.0,
        np.expm1(predicted),
    )

    if apply_online_floor:
        milliseconds = np.maximum(
            COST_PREDICTION_FLOOR_MS,
            milliseconds,
        )

    return milliseconds


def pooled_log_latency_mse(
    true_log_latency: Iterable[object],
    predicted_log_latency: Iterable[object],
) -> float:
    """Frozen inner-selection metric for a cost-predictor candidate."""

    truth = np.asarray(
        list(true_log_latency),
        dtype=np.float64,
    )

    prediction = np.asarray(
        list(predicted_log_latency),
        dtype=np.float64,
    )

    if (
        truth.ndim != 1
        or prediction.ndim != 1
        or truth.size == 0
    ):
        raise ValueError(
            "cost-prediction vectors must be "
            "nonempty and one-dimensional"
        )

    if truth.shape != prediction.shape:
        raise ValueError(
            "truth and prediction lengths differ"
        )

    if not (
        np.all(np.isfinite(truth))
        and np.all(
            np.isfinite(prediction)
        )
    ):
        raise ValueError(
            "cost-prediction vectors "
            "contain non-finite values"
        )

    residual = (
        prediction
        - truth
    )

    return float(
        np.mean(
            residual ** 2
        )
    )
