from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from monitor_fusion.evaluation.classifier_selection import (
    CURRENT_ERROR_FAMILY_ORDER,
    candidates_by_family,
    build_classifier,
    current_error_candidates_from_protocol,
    direct_fusion_candidates_from_protocol,
    select_inner_current_error_candidate,
    select_inner_direct_fusion_candidate,
    select_threshold_at_empirical_fpr,
)
from monitor_fusion.evaluation.data_boundary import load_protocol


def test_protocol_classifier_families_are_exact() -> None:
    protocol = load_protocol()

    current = current_error_candidates_from_protocol(
        protocol
    )
    fusion = direct_fusion_candidates_from_protocol(
        protocol
    )

    current_grouped = candidates_by_family(current)
    fusion_grouped = candidates_by_family(fusion)

    assert tuple(current_grouped) == CURRENT_ERROR_FAMILY_ORDER
    assert tuple(fusion_grouped) == CURRENT_ERROR_FAMILY_ORDER

    assert {
        family: len(values)
        for family, values in current_grouped.items()
    } == {
        "LogisticRegression": 1,
        "HistGradientBoostingClassifier": 3,
        "RandomForestClassifier": 3,
    }

    assert {
        family: len(values)
        for family, values in fusion_grouped.items()
    } == {
        "LogisticRegression": 1,
        "HistGradientBoostingClassifier": 3,
        "RandomForestClassifier": 3,
    }


def test_classifier_constructors_preserve_frozen_settings() -> None:
    grouped = candidates_by_family(
        current_error_candidates_from_protocol(
            load_protocol()
        )
    )

    logistic = build_classifier(
        grouped["LogisticRegression"][0],
        random_state=1729,
    )
    assert isinstance(logistic, Pipeline)

    estimator = logistic.named_steps["classifier"]
    assert isinstance(estimator, LogisticRegression)
    assert estimator.random_state == 1729
    assert estimator.class_weight == "balanced"
    assert estimator.solver == "lbfgs"
    assert estimator.max_iter == 2000

    hgb = build_classifier(
        grouped["HistGradientBoostingClassifier"][0],
        random_state=2718,
    )
    assert isinstance(
        hgb,
        HistGradientBoostingClassifier,
    )
    assert hgb.loss == "log_loss"
    assert hgb.class_weight == "balanced"
    assert hgb.early_stopping is False
    assert hgb.random_state == 2718

    forest = build_classifier(
        grouped["RandomForestClassifier"][0],
        random_state=3141,
    )
    assert isinstance(
        forest,
        RandomForestClassifier,
    )
    assert forest.criterion == "gini"
    assert forest.class_weight == "balanced"
    assert forest.bootstrap is True
    assert forest.n_jobs == 1
    assert forest.random_state == 3141


def test_current_error_inner_selection_is_within_family() -> None:
    grouped = candidates_by_family(
        current_error_candidates_from_protocol(
            load_protocol()
        )
    )

    mixed = (
        grouped["LogisticRegression"][0],
        grouped["HistGradientBoostingClassifier"][0],
    )

    with pytest.raises(
        ValueError,
        match="exactly one model family",
    ):
        select_inner_current_error_candidate(
            mixed,
            labels=[0, 1, 0, 1],
            candidate_probabilities={
                mixed[0].identifier: [0.1, 0.9, 0.2, 0.8],
                mixed[1].identifier: [0.1, 0.9, 0.2, 0.8],
            },
            candidate_inference_latency_ms_per_example={
                mixed[0].identifier: 0.1,
                mixed[1].identifier: 0.2,
            },
        )


def test_current_error_selects_minimum_pooled_log_loss() -> None:
    candidates = candidates_by_family(
        current_error_candidates_from_protocol(
            load_protocol()
        )
    )["HistGradientBoostingClassifier"]

    probabilities = {
        candidates[0].identifier: [0.25, 0.75, 0.25, 0.75],
        candidates[1].identifier: [0.05, 0.95, 0.05, 0.95],
        candidates[2].identifier: [0.40, 0.60, 0.40, 0.60],
    }

    latency = {
        candidate.identifier: float(index + 1)
        for index, candidate in enumerate(candidates)
    }

    result = select_inner_current_error_candidate(
        candidates,
        labels=[0, 1, 0, 1],
        candidate_probabilities=probabilities,
        candidate_inference_latency_ms_per_example=latency,
    )

    assert result.selected_candidate.identifier == (
        candidates[1].identifier
    )


def test_current_error_metric_tie_prefers_lower_latency() -> None:
    candidates = candidates_by_family(
        current_error_candidates_from_protocol(
            load_protocol()
        )
    )["RandomForestClassifier"][:2]

    probabilities = {
        candidate.identifier: [0.1, 0.9, 0.2, 0.8]
        for candidate in candidates
    }

    result = select_inner_current_error_candidate(
        candidates,
        labels=[0, 1, 0, 1],
        candidate_probabilities=probabilities,
        candidate_inference_latency_ms_per_example={
            candidates[0].identifier: 2.0,
            candidates[1].identifier: 1.0,
        },
    )

    assert result.selected_candidate.identifier == (
        candidates[1].identifier
    )


def test_direct_fusion_threshold_obeys_frozen_fpr_rule() -> None:
    labels = np.array(
        [1, 1, 1, 0, 0, 0, 0, 0],
        dtype=int,
    )
    scores = np.array(
        [0.95, 0.85, 0.75, 0.90, 0.70, 0.20, 0.10, 0.05],
        dtype=float,
    )

    result = select_threshold_at_empirical_fpr(
        labels,
        scores,
        maximum_fpr=0.20,
    )

    prediction = scores >= result.threshold

    fp = int(
        np.sum(
            prediction
            & (labels == 0)
        )
    )
    negatives = int(np.sum(labels == 0))

    assert fp / negatives <= 0.20 + 1e-15
    assert result.recall == pytest.approx(1.0)


def test_direct_fusion_hyperparameter_selection_is_within_family() -> None:
    grouped = candidates_by_family(
        direct_fusion_candidates_from_protocol(
            load_protocol()
        )
    )

    mixed = (
        grouped["LogisticRegression"][0],
        grouped["RandomForestClassifier"][0],
    )

    with pytest.raises(
        ValueError,
        match="exactly one model family",
    ):
        select_inner_direct_fusion_candidate(
            mixed,
            labels=[0, 1, 0, 1],
            candidate_probabilities={
                mixed[0].identifier: [0.1, 0.9, 0.2, 0.8],
                mixed[1].identifier: [0.1, 0.9, 0.2, 0.8],
            },
        )


def test_direct_fusion_selects_best_fpr_constrained_candidate() -> None:
    candidates = candidates_by_family(
        direct_fusion_candidates_from_protocol(
            load_protocol()
        )
    )["HistGradientBoostingClassifier"]

    labels = [1, 1, 1, 0, 0, 0, 0, 0]

    probabilities = {
        candidates[0].identifier: [
            0.95, 0.85, 0.10,
            0.90, 0.30, 0.20, 0.15, 0.05,
        ],
        candidates[1].identifier: [
            0.95, 0.85, 0.75,
            0.70, 0.30, 0.20, 0.15, 0.05,
        ],
        candidates[2].identifier: [
            0.80, 0.70, 0.10,
            0.90, 0.60, 0.20, 0.15, 0.05,
        ],
    }

    result = select_inner_direct_fusion_candidate(
        candidates,
        labels=labels,
        candidate_probabilities=probabilities,
        maximum_fpr=0.05,
    )

    assert result.selected_candidate.identifier == (
        candidates[1].identifier
    )


def test_direct_fusion_exact_unresolved_tie_fails_closed() -> None:
    candidates = candidates_by_family(
        direct_fusion_candidates_from_protocol(
            load_protocol()
        )
    )["RandomForestClassifier"][:2]

    probabilities = {
        candidate.identifier: [0.1, 0.9, 0.2, 0.8]
        for candidate in candidates
    }

    with pytest.raises(
        RuntimeError,
        match="does not resolve",
    ):
        select_inner_direct_fusion_candidate(
            candidates,
            labels=[0, 1, 0, 1],
            candidate_probabilities=probabilities,
            maximum_fpr=0.05,
        )
