"""Finite-sample joint FPR and bounded mean-cost certification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom


@dataclass(frozen=True)
class JointRiskCertificate:
    candidate_id: str
    example_count: int
    negative_count: int
    false_positive_count: int
    empirical_fpr: float
    mean_total_cost_ms: float
    fpr_p_value: float
    mean_cost_p_value: float
    joint_p_value: float
    bonferroni_threshold: float
    candidate_count: int
    certified: bool


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

    if not np.all(np.isin(array, (0, 1))):
        raise ValueError(
            f"{name} must contain only zero and one"
        )

    return array.astype(np.int64, copy=False)


def _bounded_vector(
    values: ArrayLike,
    *,
    name: str,
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    array = np.asarray(values)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    try:
        result = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values"
        ) from exc

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")

    if np.any(result < lower) or np.any(result > upper):
        raise ValueError(
            f"{name} must lie in [{lower}, {upper}]"
        )

    return result


def _probability(
    value: float,
    *,
    name: str,
) -> float:
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(
            f"{name} must be strictly between zero and one"
        )

    return float(value)


def exact_binomial_fpr_p_value(
    false_positive_count: int,
    negative_count: int,
    *,
    maximum_fpr: float = 0.05,
) -> float:
    """Lower-tail exact-binomial p-value at the FPR boundary."""

    if (
        isinstance(false_positive_count, bool)
        or not isinstance(false_positive_count, int)
        or false_positive_count < 0
    ):
        raise ValueError(
            "false_positive_count must be nonnegative"
        )

    if (
        isinstance(negative_count, bool)
        or not isinstance(negative_count, int)
        or negative_count <= 0
    ):
        raise ValueError(
            "negative_count must be positive"
        )

    if false_positive_count > negative_count:
        raise ValueError(
            "false_positive_count exceeds negative_count"
        )

    limit = _probability(
        maximum_fpr,
        name="maximum_fpr",
    )

    return float(
        binom.cdf(
            false_positive_count,
            negative_count,
            limit,
        )
    )


def _binary_relative_entropy(
    observed: float,
    null: float,
) -> float:
    first = (
        0.0
        if observed == 0.0
        else observed * math.log(observed / null)
    )

    second = (
        0.0
        if observed == 1.0
        else (1.0 - observed)
        * math.log(
            (1.0 - observed) / (1.0 - null)
        )
    )

    return first + second


def hoeffding_bentkus_p_value(
    bounded_losses: ArrayLike,
    *,
    risk_limit: float,
) -> float:
    """One-sided Hoeffding-Bentkus bounded-risk p-value."""

    losses = _bounded_vector(
        bounded_losses,
        name="bounded_losses",
        lower=0.0,
        upper=1.0,
    )

    limit = _probability(
        risk_limit,
        name="risk_limit",
    )

    sample_size = len(losses)
    empirical_risk = float(np.mean(losses))

    if empirical_risk >= limit:
        return 1.0

    hoeffding = math.exp(
        -sample_size
        * _binary_relative_entropy(
            empirical_risk,
            limit,
        )
    )

    binomial_count = min(
        sample_size,
        int(math.ceil(sample_size * empirical_risk)),
    )

    bentkus = min(
        1.0,
        math.e
        * float(
            binom.cdf(
                binomial_count,
                sample_size,
                limit,
            )
        ),
    )

    return min(
        1.0,
        max(0.0, min(hoeffding, bentkus)),
    )


def bounded_mean_cost_p_value(
    total_cost_ms: ArrayLike,
    *,
    absolute_cost_budget_ms: float,
    normalization_bound_ms: float = 35000.0,
) -> float:
    """Test mean total cost against an absolute budget."""

    if (
        not np.isfinite(normalization_bound_ms)
        or normalization_bound_ms <= 0
    ):
        raise ValueError(
            "normalization_bound_ms must be positive"
        )

    if (
        not np.isfinite(absolute_cost_budget_ms)
        or not 0
        < absolute_cost_budget_ms
        < normalization_bound_ms
    ):
        raise ValueError(
            "absolute_cost_budget_ms must lie below the bound"
        )

    costs = _bounded_vector(
        total_cost_ms,
        name="total_cost_ms",
        lower=0.0,
        upper=float(normalization_bound_ms),
    )

    return hoeffding_bentkus_p_value(
        costs / float(normalization_bound_ms),
        risk_limit=(
            float(absolute_cost_budget_ms)
            / float(normalization_bound_ms)
        ),
    )


def certify_joint_fpr_and_cost(
    y_true: ArrayLike,
    decisions: ArrayLike,
    total_cost_ms: ArrayLike,
    *,
    candidate_id: str,
    candidate_count: int,
    absolute_cost_budget_ms: float,
    maximum_fpr: float = 0.05,
    normalization_bound_ms: float = 35000.0,
    familywise_error_rate: float = 0.05,
) -> JointRiskCertificate:
    """Certify only when both frozen constraints reject."""

    labels = _binary_vector(
        y_true,
        name="y_true",
    )

    predictions = _binary_vector(
        decisions,
        name="decisions",
    )

    costs = _bounded_vector(
        total_cost_ms,
        name="total_cost_ms",
        lower=0.0,
        upper=float(normalization_bound_ms),
    )

    if (
        len(labels) != len(predictions)
        or len(labels) != len(costs)
    ):
        raise ValueError(
            "labels, decisions, and costs must have equal length"
        )

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(
            "candidate_id must be a nonempty string"
        )

    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise ValueError(
            "candidate_count must be positive"
        )

    fwer = _probability(
        familywise_error_rate,
        name="familywise_error_rate",
    )

    negative_mask = labels == 0
    negative_count = int(np.sum(negative_mask))

    if negative_count == 0:
        raise ValueError(
            "at least one negative example is required"
        )

    false_positive_count = int(
        np.sum(
            (predictions == 1)
            & negative_mask
        )
    )

    fpr_p_value = exact_binomial_fpr_p_value(
        false_positive_count,
        negative_count,
        maximum_fpr=maximum_fpr,
    )

    mean_cost_p_value = bounded_mean_cost_p_value(
        costs,
        absolute_cost_budget_ms=absolute_cost_budget_ms,
        normalization_bound_ms=normalization_bound_ms,
    )

    joint_p_value = max(
        fpr_p_value,
        mean_cost_p_value,
    )

    bonferroni_threshold = (
        fwer / candidate_count
    )

    return JointRiskCertificate(
        candidate_id=candidate_id,
        example_count=len(labels),
        negative_count=negative_count,
        false_positive_count=false_positive_count,
        empirical_fpr=(
            false_positive_count / negative_count
        ),
        mean_total_cost_ms=float(np.mean(costs)),
        fpr_p_value=fpr_p_value,
        mean_cost_p_value=mean_cost_p_value,
        joint_p_value=joint_p_value,
        bonferroni_threshold=bonferroni_threshold,
        candidate_count=candidate_count,
        certified=bool(
            joint_p_value <= bonferroni_threshold
        ),
    )
