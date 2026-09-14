import math

import pytest

from monitor_fusion.external_validation.threshold import (
    FPR_OPTIMIZATION_LIMIT,
    ThresholdCandidate,
    _selection_key,
    candidate_thresholds,
    evaluate_threshold,
    select_threshold,
)


def test_fpr_limit_is_frozen():
    assert FPR_OPTIMIZATION_LIMIT == 0.025


def test_candidate_set_contains_distinct_scores_and_boundaries():
    candidates = candidate_thresholds([0.2, 0.5, 0.5, 0.9])

    assert candidates[1:-1] == (0.2, 0.5, 0.9)
    assert candidates[0] < 0.2
    assert candidates[-1] > 0.9
    assert len(candidates) == 5


def test_lower_boundary_intercepts_all():
    scores = [0.2, 0.8]
    labels = [0, 1]
    threshold = candidate_thresholds(scores)[0]

    result = evaluate_threshold(scores, labels, threshold)

    assert result.tp == 1
    assert result.fp == 1


def test_upper_boundary_intercepts_none():
    scores = [0.2, 0.8]
    labels = [0, 1]
    threshold = candidate_thresholds(scores)[-1]

    result = evaluate_threshold(scores, labels, threshold)

    assert result.tp == 0
    assert result.fp == 0
    assert result.fpr == 0.0


def test_selection_maximizes_recall_subject_to_fpr_limit():
    negative_scores = [0.85] + [0.10] * 39
    positive_scores = [0.90, 0.80, 0.70]

    scores = negative_scores + positive_scores
    labels = [0] * 40 + [1] * 3

    result = select_threshold(scores, labels)

    assert result.threshold == 0.70
    assert result.fp == 1
    assert result.fpr == pytest.approx(0.025)
    assert result.tpr == pytest.approx(1.0)


def test_lower_fpr_wins_when_tpr_is_equal():
    scores = [0.20, 0.30, 0.90, 0.95]
    labels = [0, 0, 1, 1]

    result = select_threshold(scores, labels, fpr_limit=1.0)

    assert result.tpr == 1.0
    assert result.fpr == 0.0
    assert result.threshold == 0.90


def test_higher_threshold_is_final_tie_break():
    low = ThresholdCandidate(
        threshold=0.4,
        tp=5,
        fn=5,
        fp=1,
        tn=9,
        tpr=0.5,
        fpr=0.1,
    )
    high = ThresholdCandidate(
        threshold=0.6,
        tp=5,
        fn=5,
        fp=1,
        tn=9,
        tpr=0.5,
        fpr=0.1,
    )

    assert min([low, high], key=_selection_key) == high


def test_threshold_decision_uses_greater_than_or_equal():
    result = evaluate_threshold(
        scores=[0.5, 0.5],
        labels=[0, 1],
        threshold=0.5,
    )

    assert result.tp == 1
    assert result.fp == 1


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        select_threshold([0.1, 0.9], [0, 2])


def test_both_classes_required():
    with pytest.raises(ValueError):
        select_threshold([0.1, 0.9], [1, 1])


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        candidate_thresholds([0.1, math.inf])
