"""Frozen classifier candidates and inner-selection rules for v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CURRENT_ERROR_FAMILY_ORDER = (
    "LogisticRegression",
    "HistGradientBoostingClassifier",
    "RandomForestClassifier",
)

DIRECT_FUSION_FAMILY_ORDER = CURRENT_ERROR_FAMILY_ORDER


@dataclass(frozen=True)
class ClassifierCandidate:
    role: str
    family: str
    candidate_index: int
    params: Mapping[str, Any]
    preprocessing: str

    @property
    def identifier(self) -> str:
        return f"{self.family}:{self.candidate_index:03d}"


@dataclass(frozen=True)
class CurrentErrorCandidateMetric:
    candidate_identifier: str
    pooled_binary_log_loss: float
    online_inference_latency_ms_per_example: float


@dataclass(frozen=True)
class CurrentErrorSelectionResult:
    selected_candidate: ClassifierCandidate
    candidate_metrics: tuple[CurrentErrorCandidateMetric, ...]


@dataclass(frozen=True)
class FprThresholdResult:
    threshold: float
    recall: float
    false_positive_rate: float
    predicted_positive_n: int


@dataclass(frozen=True)
class DirectFusionCandidateMetric:
    candidate_identifier: str
    recall: float
    false_positive_rate: float
    threshold: float


@dataclass(frozen=True)
class DirectFusionSelectionResult:
    selected_candidate: ClassifierCandidate
    selected_threshold: float
    candidate_metrics: tuple[DirectFusionCandidateMetric, ...]


def _binary_labels(values: ArrayLike) -> NDArray[np.int64]:
    y = np.asarray(values)

    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")

    if not bool(np.all(np.isin(y, (0, 1)))):
        raise ValueError("labels must be binary")

    return y.astype(np.int64, copy=False)


def _probabilities(
    values: ArrayLike,
    *,
    expected_length: int,
) -> NDArray[np.float64]:
    p = np.asarray(values, dtype=np.float64)

    if p.ndim != 1 or len(p) != expected_length:
        raise ValueError(
            "probabilities must be one-dimensional and match labels"
        )

    if not np.isfinite(p).all():
        raise ValueError("probabilities contain non-finite values")

    if bool(np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("probabilities must lie in [0, 1]")

    return p


def _candidates_from_protocol(
    protocol: Mapping[str, Any],
    *,
    protocol_key: str,
    role: str,
    expected_family_order: Sequence[str],
) -> tuple[ClassifierCandidate, ...]:
    try:
        entries = protocol["model_families"][protocol_key]
    except KeyError as exc:
        raise KeyError(
            f"Missing model_families.{protocol_key}"
        ) from exc

    observed_order = tuple(
        str(entry["family"])
        for entry in entries
    )

    if observed_order != tuple(expected_family_order):
        raise ValueError(
            f"Unexpected {protocol_key} family order: "
            f"{observed_order!r}"
        )

    candidates: list[ClassifierCandidate] = []

    for entry in entries:
        family = str(entry["family"])
        grid = entry.get("candidate_grid")

        if not isinstance(grid, list) or not grid:
            raise ValueError(
                f"{family} must have a nonempty candidate_grid"
            )

        preprocessing = str(
            entry.get("preprocessing", "none")
        )

        for index, params in enumerate(grid):
            if not isinstance(params, dict):
                raise TypeError(
                    f"{family} candidate {index} must be a mapping"
                )

            candidates.append(
                ClassifierCandidate(
                    role=role,
                    family=family,
                    candidate_index=index,
                    params=dict(params),
                    preprocessing=preprocessing,
                )
            )

    identifiers = [candidate.identifier for candidate in candidates]

    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Classifier candidate identifiers are not unique")

    return tuple(candidates)


def current_error_candidates_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[ClassifierCandidate, ...]:
    return _candidates_from_protocol(
        protocol,
        protocol_key="current_error_classifiers",
        role="current_error_classifier",
        expected_family_order=CURRENT_ERROR_FAMILY_ORDER,
    )


def direct_fusion_candidates_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[ClassifierCandidate, ...]:
    return _candidates_from_protocol(
        protocol,
        protocol_key="downstream_fusion_classifiers",
        role="downstream_fusion_classifier",
        expected_family_order=DIRECT_FUSION_FAMILY_ORDER,
    )


def candidates_by_family(
    candidates: Sequence[ClassifierCandidate],
) -> dict[str, tuple[ClassifierCandidate, ...]]:
    grouped: dict[str, list[ClassifierCandidate]] = {}

    for candidate in candidates:
        grouped.setdefault(candidate.family, []).append(candidate)

    return {
        family: tuple(values)
        for family, values in grouped.items()
    }


def build_classifier(
    candidate: ClassifierCandidate,
    *,
    random_state: int,
) -> Any:
    params = dict(candidate.params)

    if candidate.family == "LogisticRegression":
        if candidate.preprocessing != "StandardScaler":
            raise ValueError(
                "Frozen LogisticRegression requires StandardScaler"
            )

        estimator = LogisticRegression(
            random_state=int(random_state),
            **params,
        )

        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", estimator),
            ]
        )

    if candidate.family == "HistGradientBoostingClassifier":
        if candidate.preprocessing != "none":
            raise ValueError(
                "Frozen HistGradientBoostingClassifier uses no preprocessing"
            )

        return HistGradientBoostingClassifier(
            loss="log_loss",
            class_weight="balanced",
            early_stopping=False,
            random_state=int(random_state),
            **params,
        )

    if candidate.family == "RandomForestClassifier":
        if candidate.preprocessing != "none":
            raise ValueError(
                "Frozen RandomForestClassifier uses no preprocessing"
            )

        return RandomForestClassifier(
            criterion="gini",
            class_weight="balanced",
            bootstrap=True,
            n_jobs=1,
            random_state=int(random_state),
            **params,
        )

    raise KeyError(
        f"Unknown frozen classifier family: {candidate.family}"
    )


def positive_class_probability(
    model: Any,
    features: ArrayLike,
) -> NDArray[np.float64]:
    probability = np.asarray(
        model.predict_proba(features),
        dtype=np.float64,
    )

    if probability.ndim != 2:
        raise ValueError("predict_proba must return a matrix")

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("fitted classifier has no positive class")

    result = probability[:, classes.index(1)]

    if not np.isfinite(result).all():
        raise ValueError(
            "classifier produced non-finite probabilities"
        )

    if bool(np.any((result < 0.0) | (result > 1.0))):
        raise ValueError(
            "classifier probabilities must lie in [0, 1]"
        )

    return result.astype(np.float64, copy=False)


def _require_one_family(
    candidates: Sequence[ClassifierCandidate],
) -> str:
    if not candidates:
        raise ValueError("candidate list must not be empty")

    families = {candidate.family for candidate in candidates}

    if len(families) != 1:
        raise ValueError(
            "inner hyperparameter selection must operate within "
            "exactly one model family"
        )

    return next(iter(families))


def select_inner_current_error_candidate(
    candidates: Sequence[ClassifierCandidate],
    *,
    labels: ArrayLike,
    candidate_probabilities: Mapping[str, ArrayLike],
    candidate_inference_latency_ms_per_example: Mapping[str, float],
) -> CurrentErrorSelectionResult:
    """Select one hyperparameter candidate within one current-error family."""

    _require_one_family(candidates)
    y = _binary_labels(labels)

    metrics: list[CurrentErrorCandidateMetric] = []

    for candidate in candidates:
        identifier = candidate.identifier

        if identifier not in candidate_probabilities:
            raise KeyError(
                f"Missing probabilities for {identifier}"
            )

        if identifier not in candidate_inference_latency_ms_per_example:
            raise KeyError(
                f"Missing inference latency for {identifier}"
            )

        probability = _probabilities(
            candidate_probabilities[identifier],
            expected_length=len(y),
        )

        latency = float(
            candidate_inference_latency_ms_per_example[identifier]
        )

        if not np.isfinite(latency) or latency < 0.0:
            raise ValueError(
                f"Invalid inference latency for {identifier}"
            )

        metric = float(
            log_loss(
                y,
                probability,
                labels=[0, 1],
            )
        )

        metrics.append(
            CurrentErrorCandidateMetric(
                candidate_identifier=identifier,
                pooled_binary_log_loss=metric,
                online_inference_latency_ms_per_example=latency,
            )
        )

    best_loss = min(
        item.pooled_binary_log_loss
        for item in metrics
    )

    loss_ties = [
        item
        for item in metrics
        if np.isclose(
            item.pooled_binary_log_loss,
            best_loss,
            rtol=0.0,
            atol=1e-15,
        )
    ]

    best_latency = min(
        item.online_inference_latency_ms_per_example
        for item in loss_ties
    )

    latency_ties = [
        item
        for item in loss_ties
        if np.isclose(
            item.online_inference_latency_ms_per_example,
            best_latency,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if len(latency_ties) != 1:
        raise RuntimeError(
            "Frozen current-error selection rule does not resolve "
            "an exact metric-and-latency tie within one family"
        )

    selected_id = latency_ties[0].candidate_identifier
    selected = next(
        candidate
        for candidate in candidates
        if candidate.identifier == selected_id
    )

    return CurrentErrorSelectionResult(
        selected_candidate=selected,
        candidate_metrics=tuple(metrics),
    )


def select_threshold_at_empirical_fpr(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    maximum_fpr: float = 0.05,
) -> FprThresholdResult:
    """Apply the frozen direct-fusion threshold rule."""

    y = _binary_labels(labels)
    p = _probabilities(
        probabilities,
        expected_length=len(y),
    )

    if (
        not np.isfinite(maximum_fpr)
        or maximum_fpr < 0.0
        or maximum_fpr > 1.0
    ):
        raise ValueError("maximum_fpr must lie in [0, 1]")

    unique = np.unique(p)
    above_max = np.nextafter(
        float(unique.max()),
        np.inf,
    )
    thresholds = np.concatenate(
        ([above_max], unique[::-1])
    )

    positive = y == 1
    negative = y == 0

    positive_n = int(positive.sum())
    negative_n = int(negative.sum())

    if positive_n == 0 or negative_n == 0:
        raise ValueError(
            "threshold selection requires both classes"
        )

    feasible: list[FprThresholdResult] = []

    for threshold in thresholds:
        prediction = p >= threshold

        tp = int(np.sum(prediction & positive))
        fp = int(np.sum(prediction & negative))

        recall = tp / positive_n
        fpr = fp / negative_n

        if fpr <= maximum_fpr + 1e-15:
            feasible.append(
                FprThresholdResult(
                    threshold=float(threshold),
                    recall=float(recall),
                    false_positive_rate=float(fpr),
                    predicted_positive_n=int(
                        prediction.sum()
                    ),
                )
            )

    if not feasible:
        raise RuntimeError(
            "No threshold satisfies the frozen FPR constraint"
        )

    # Frozen rule:
    # 1. maximize recall;
    # 2. ties prefer lower FPR;
    # 3. then higher threshold.
    return max(
        feasible,
        key=lambda item: (
            item.recall,
            -item.false_positive_rate,
            item.threshold,
        ),
    )


def select_inner_direct_fusion_candidate(
    candidates: Sequence[ClassifierCandidate],
    *,
    labels: ArrayLike,
    candidate_probabilities: Mapping[str, ArrayLike],
    maximum_fpr: float = 0.05,
) -> DirectFusionSelectionResult:
    """Select one direct-fusion hyperparameter candidate within one family."""

    _require_one_family(candidates)
    y = _binary_labels(labels)

    metrics: list[DirectFusionCandidateMetric] = []

    for candidate in candidates:
        identifier = candidate.identifier

        if identifier not in candidate_probabilities:
            raise KeyError(
                f"Missing probabilities for {identifier}"
            )

        result = select_threshold_at_empirical_fpr(
            y,
            candidate_probabilities[identifier],
            maximum_fpr=maximum_fpr,
        )

        metrics.append(
            DirectFusionCandidateMetric(
                candidate_identifier=identifier,
                recall=result.recall,
                false_positive_rate=result.false_positive_rate,
                threshold=result.threshold,
            )
        )

    # The candidate is compared at the operating point produced by
    # the same frozen threshold rule: recall, then lower FPR, then
    # higher threshold. If two different candidates remain identical
    # on all three quantities, the protocol has not specified a safe
    # further tuning rule, so fail closed.
    best_key = max(
        (
            (
                item.recall,
                -item.false_positive_rate,
                item.threshold,
            )
            for item in metrics
        )
    )

    winners = [
        item
        for item in metrics
        if (
            item.recall,
            -item.false_positive_rate,
            item.threshold,
        )
        == best_key
    ]

    if len(winners) != 1:
        raise RuntimeError(
            "Frozen direct-fusion selection rule does not resolve "
            "an exact operating-point tie within one family"
        )

    selected_metric = winners[0]
    selected = next(
        candidate
        for candidate in candidates
        if candidate.identifier
        == selected_metric.candidate_identifier
    )

    return DirectFusionSelectionResult(
        selected_candidate=selected,
        selected_threshold=selected_metric.threshold,
        candidate_metrics=tuple(metrics),
    )
