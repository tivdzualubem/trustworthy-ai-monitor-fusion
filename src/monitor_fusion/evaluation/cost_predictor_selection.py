"""Frozen v2 heterogeneous-cost predictor candidate selection.

This module implements only the prospectively frozen candidate families,
hyperparameter grids, pooled log-latency MSE scoring, and deterministic
inner-selection tie breaking.

Grouped fold construction and repeated-development family selection are
handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from monitor_fusion.evaluation.heterogeneous_cost_prediction import (
    pooled_log_latency_mse,
)


RIDGE_FAMILY = "Ridge_on_log_latency"
HGBR_FAMILY = (
    "HistGradientBoostingRegressor_on_log_latency"
)

LINEAR_FAMILY_ORDER = {
    RIDGE_FAMILY: 0,
    HGBR_FAMILY: 1,
}


RIDGE_GRID = (
    {
        "alpha": 0.1,
    },
    {
        "alpha": 1.0,
    },
    {
        "alpha": 10.0,
    },
)


HGBR_GRID = (
    {
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 30,
    },
    {
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
    },
    {
        "l2_regularization": 5.0,
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
    },
)


@dataclass(frozen=True)
class FrozenCostPredictorCandidate:
    """One prospectively frozen cost-predictor configuration."""

    candidate_id: str
    family: str
    grid_index: int
    parameters: dict[str, Any]


@dataclass(frozen=True)
class InnerCostPredictorEvaluation:
    """Pooled inner-validation evidence for one candidate."""

    candidate: FrozenCostPredictorCandidate
    pooled_log_latency_mse: float
    online_inference_latency_ms: float


def frozen_cost_predictor_candidates(
    *,
    fold_seed: int,
) -> tuple[FrozenCostPredictorCandidate, ...]:
    """Return exactly the six frozen v2 cost-predictor candidates."""

    seed = int(fold_seed)

    candidates: list[
        FrozenCostPredictorCandidate
    ] = []

    for index, grid in enumerate(
        RIDGE_GRID
    ):
        parameters = {
            "alpha": float(
                grid["alpha"]
            ),
            "fit_intercept": True,
            "solver": "auto",
            "tol": 0.0001,
        }

        candidates.append(
            FrozenCostPredictorCandidate(
                candidate_id=(
                    f"{RIDGE_FAMILY}:"
                    f"{index}"
                ),
                family=RIDGE_FAMILY,
                grid_index=index,
                parameters=parameters,
            )
        )

    for index, grid in enumerate(
        HGBR_GRID
    ):
        parameters = {
            **grid,
            "loss": "squared_error",
            "early_stopping": False,
            "random_state": seed,
        }

        candidates.append(
            FrozenCostPredictorCandidate(
                candidate_id=(
                    f"{HGBR_FAMILY}:"
                    f"{index}"
                ),
                family=HGBR_FAMILY,
                grid_index=index,
                parameters=parameters,
            )
        )

    if len(candidates) != 6:
        raise RuntimeError(
            "Frozen v2 cost-predictor grid "
            "must contain exactly six candidates"
        )

    return tuple(
        candidates
    )


def build_cost_predictor(
    candidate: FrozenCostPredictorCandidate,
) -> RegressorMixin:
    """Instantiate one frozen cost-predictor candidate."""

    if candidate.family == RIDGE_FAMILY:
        return Ridge(
            **candidate.parameters
        )

    if candidate.family == HGBR_FAMILY:
        return HistGradientBoostingRegressor(
            **candidate.parameters
        )

    raise ValueError(
        "Unknown frozen cost-predictor family: "
        f"{candidate.family}"
    )


def validate_regression_matrix(
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fail closed before fitting a cost predictor."""

    x = np.asarray(
        features,
        dtype=np.float64,
    )

    y = np.asarray(
        target,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional matrix"
        )

    if y.ndim != 1:
        raise ValueError(
            "target must be a one-dimensional vector"
        )

    if len(x) != len(y):
        raise ValueError(
            "features and target row counts differ"
        )

    if len(y) == 0:
        raise ValueError(
            "cost-predictor training set is empty"
        )

    if not np.all(
        np.isfinite(x)
    ):
        raise ValueError(
            "features contain non-finite values"
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            "target contains non-finite values"
        )

    return x, y


def fit_cost_predictor_candidate(
    candidate: FrozenCostPredictorCandidate,
    *,
    training_features: np.ndarray,
    training_log_latency: np.ndarray,
) -> RegressorMixin:
    """Fit one frozen candidate on the current inner-training rows."""

    x, y = validate_regression_matrix(
        training_features,
        training_log_latency,
    )

    estimator = build_cost_predictor(
        candidate
    )

    estimator.fit(
        x,
        y,
    )

    return estimator


def predict_log_latency(
    estimator: RegressorMixin,
    *,
    features: np.ndarray,
) -> np.ndarray:
    """Generate finite one-dimensional log-latency predictions."""

    x = np.asarray(
        features,
        dtype=np.float64,
    )

    if x.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional matrix"
        )

    if not np.all(
        np.isfinite(x)
    ):
        raise ValueError(
            "features contain non-finite values"
        )

    prediction = np.asarray(
        estimator.predict(x),
        dtype=np.float64,
    )

    if prediction.shape != (
        len(x),
    ):
        raise RuntimeError(
            "cost predictor returned "
            "an unexpected prediction shape"
        )

    if not np.all(
        np.isfinite(prediction)
    ):
        raise ValueError(
            "cost predictor returned "
            "non-finite predictions"
        )

    return prediction


def evaluate_inner_candidate_predictions(
    candidate: FrozenCostPredictorCandidate,
    *,
    true_log_latency: np.ndarray,
    predicted_log_latency: np.ndarray,
    online_inference_latency_ms: float,
) -> InnerCostPredictorEvaluation:
    """Score pooled grouped inner-validation predictions."""

    latency = float(
        online_inference_latency_ms
    )

    if (
        not np.isfinite(latency)
        or latency < 0.0
    ):
        raise ValueError(
            "online inference latency must be "
            "finite and nonnegative"
        )

    mse = pooled_log_latency_mse(
        true_log_latency,
        predicted_log_latency,
    )

    return InnerCostPredictorEvaluation(
        candidate=candidate,
        pooled_log_latency_mse=mse,
        online_inference_latency_ms=latency,
    )


def select_inner_cost_predictor_candidate(
    evaluations: tuple[
        InnerCostPredictorEvaluation,
        ...,
    ],
) -> InnerCostPredictorEvaluation:
    """Apply the frozen inner candidate-selection rule.

    Primary criterion:
        minimum pooled MSE on log1p optional-monitor latency.

    Exact-score ties:
        lower online inference latency, then the linear Ridge family,
        then frozen grid order for deterministic reproducibility.

    The repeated-development one-standard-error family-selection rule is
    intentionally not applied here; it belongs to the later repeated
    grouped development aggregation stage.
    """

    if not evaluations:
        raise ValueError(
            "No candidate evaluations supplied"
        )

    candidate_ids = [
        item.candidate.candidate_id
        for item in evaluations
    ]

    if len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError(
            "Duplicate candidate evaluation"
        )

    for item in evaluations:
        if not np.isfinite(
            item.pooled_log_latency_mse
        ):
            raise ValueError(
                "Candidate MSE must be finite"
            )

        if (
            item.pooled_log_latency_mse
            < 0.0
        ):
            raise ValueError(
                "Candidate MSE cannot be negative"
            )

        if not np.isfinite(
            item.online_inference_latency_ms
        ):
            raise ValueError(
                "Candidate inference latency "
                "must be finite"
            )

        if (
            item.online_inference_latency_ms
            < 0.0
        ):
            raise ValueError(
                "Candidate inference latency "
                "cannot be negative"
            )

        if (
            item.candidate.family
            not in LINEAR_FAMILY_ORDER
        ):
            raise ValueError(
                "Unknown candidate family"
            )

    return min(
        evaluations,
        key=lambda item: (
            item.pooled_log_latency_mse,
            item.online_inference_latency_ms,
            LINEAR_FAMILY_ORDER[
                item.candidate.family
            ],
            item.candidate.grid_index,
        ),
    )
