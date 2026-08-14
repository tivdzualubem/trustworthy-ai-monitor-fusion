"""Frozen inner hyperparameter selection for v2 signed-value regressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from monitor_fusion.evaluation.signed_value_models import (
    SignedValueModelCandidate,
)


@dataclass(frozen=True)
class SignedValueCandidateMetric:
    candidate_identifier: str
    pooled_inner_validation_mean_squared_error: float


@dataclass(frozen=True)
class SignedValueSelectionResult:
    selected_candidate: SignedValueModelCandidate
    candidate_metrics: tuple[SignedValueCandidateMetric, ...]


def signed_value_candidate_identifier(
    candidate: SignedValueModelCandidate,
) -> str:
    """Return the deterministic identifier for a frozen v2 candidate."""
    return (
        f"{candidate.family}:"
        f"{int(candidate.candidate_index):03d}"
    )


def _finite_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return array


def _require_single_family(
    candidates: Sequence[SignedValueModelCandidate],
) -> str:
    if not candidates:
        raise ValueError(
            "candidate list must not be empty"
        )

    families = {
        candidate.family
        for candidate in candidates
    }

    if len(families) != 1:
        raise ValueError(
            "inner hyperparameter selection must operate "
            "within exactly one signed-value model family"
        )

    return next(iter(families))


def select_inner_signed_value_candidate(
    candidates: Sequence[SignedValueModelCandidate],
    *,
    realized_signed_value: ArrayLike,
    candidate_predictions: Mapping[str, ArrayLike],
) -> SignedValueSelectionResult:
    """Select one hyperparameter candidate using pooled validation MSE.

    This is the frozen historical value-estimator selection metric carried
    forward into v2.  Candidate comparison occurs only inside one model
    family.  Model-family selection is deliberately not performed here.
    """

    _require_single_family(candidates)

    target = _finite_vector(
        realized_signed_value,
        name="realized_signed_value",
    )

    if not bool(
        np.all(np.isin(target, (-1.0, 0.0, 1.0)))
    ):
        raise ValueError(
            "realized_signed_value must contain only -1, 0, or 1"
        )

    metrics: list[SignedValueCandidateMetric] = []

    for candidate in candidates:
        identifier = signed_value_candidate_identifier(candidate)

        if identifier not in candidate_predictions:
            raise KeyError(
                f"Missing predictions for {identifier}"
            )

        prediction = _finite_vector(
            candidate_predictions[identifier],
            name=f"predictions[{identifier}]",
        )

        if len(prediction) != len(target):
            raise ValueError(
                f"predictions for {identifier} do not match target length"
            )

        mse = float(
            np.mean(
                np.square(
                    prediction - target
                )
            )
        )

        metrics.append(
            SignedValueCandidateMetric(
                candidate_identifier=identifier,
                pooled_inner_validation_mean_squared_error=mse,
            )
        )

    best_mse = min(
        metric.pooled_inner_validation_mean_squared_error
        for metric in metrics
    )

    winners = [
        metric
        for metric in metrics
        if np.isclose(
            metric.pooled_inner_validation_mean_squared_error,
            best_mse,
            rtol=0.0,
            atol=1e-15,
        )
    ]

    # The frozen hyperparameter rule specifies the metric but no additional
    # within-family exact-tie rule.  Do not invent one after results are seen.
    if len(winners) != 1:
        raise RuntimeError(
            "Frozen signed-value hyperparameter rule does not resolve "
            "an exact pooled-MSE tie within one family"
        )

    selected_identifier = winners[0].candidate_identifier

    selected = next(
        candidate
        for candidate in candidates
        if signed_value_candidate_identifier(candidate)
        == selected_identifier
    )

    return SignedValueSelectionResult(
        selected_candidate=selected,
        candidate_metrics=tuple(metrics),
    )
