from __future__ import annotations

import math
from dataclasses import dataclass


FPR_OPTIMIZATION_LIMIT = 0.025


@dataclass(frozen=True)
class ThresholdCandidate:
    threshold: float
    tp: int
    fn: int
    fp: int
    tn: int
    tpr: float
    fpr: float


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    tpr: float
    fpr: float
    tp: int
    fn: int
    fp: int
    tn: int
    feasible_candidate_count: int
    total_candidate_count: int


def candidate_thresholds(scores: list[float]) -> tuple[float, ...]:
    if not scores:
        raise ValueError("scores must be non-empty")

    values = [float(x) for x in scores]

    if any(not math.isfinite(x) for x in values):
        raise ValueError("all optimization scores must be finite")

    distinct = sorted(set(values))
    lower = math.nextafter(distinct[0], -math.inf)
    upper = math.nextafter(distinct[-1], math.inf)

    return tuple([lower, *distinct, upper])


def evaluate_threshold(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> ThresholdCandidate:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")

    if not scores:
        raise ValueError("optimization data must be non-empty")

    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be binary 0/1")

    positives = sum(labels)
    negatives = len(labels) - positives

    if positives == 0 or negatives == 0:
        raise ValueError(
            "A_opt threshold selection requires both Y=1 and Y=0 examples"
        )

    tp = fn = fp = tn = 0

    for score, label in zip(scores, labels):
        if not math.isfinite(float(score)):
            raise ValueError("all optimization scores must be finite")

        intercept = float(score) >= threshold

        if label == 1 and intercept:
            tp += 1
        elif label == 1:
            fn += 1
        elif intercept:
            fp += 1
        else:
            tn += 1

    tpr = tp / positives
    fpr = fp / negatives

    return ThresholdCandidate(
        threshold=float(threshold),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        tpr=tpr,
        fpr=fpr,
    )


def _selection_key(candidate: ThresholdCandidate) -> tuple[float, float, float]:
    # min() with this key implements:
    # 1) maximum TPR
    # 2) minimum FPR
    # 3) higher threshold
    return (
        -candidate.tpr,
        candidate.fpr,
        -candidate.threshold,
    )


def select_threshold(
    scores: list[float],
    labels: list[int],
    *,
    fpr_limit: float = FPR_OPTIMIZATION_LIMIT,
) -> ThresholdSelection:
    if not (0.0 <= fpr_limit <= 1.0):
        raise ValueError("fpr_limit must lie in [0, 1]")

    candidates = [
        evaluate_threshold(scores, labels, threshold)
        for threshold in candidate_thresholds(scores)
    ]

    feasible = [
        candidate
        for candidate in candidates
        if candidate.fpr <= fpr_limit
    ]

    # The upper boundary produces the intercept-none point, so for
    # finite scores at least one FPR=0 candidate must always exist.
    if not feasible:
        raise RuntimeError("no feasible threshold candidate")

    selected = min(feasible, key=_selection_key)

    return ThresholdSelection(
        threshold=selected.threshold,
        tpr=selected.tpr,
        fpr=selected.fpr,
        tp=selected.tp,
        fn=selected.fn,
        fp=selected.fp,
        tn=selected.tn,
        feasible_candidate_count=len(feasible),
        total_candidate_count=len(candidates),
    )
