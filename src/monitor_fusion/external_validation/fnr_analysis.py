from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm


W1_CELL_ORDER = ("T", "S", "F", "SF")
W1_CELL_CODING = {
    "T": (-0.5, -0.5),
    "S": (0.5, -0.5),
    "F": (-0.5, 0.5),
    "SF": (0.5, 0.5),
}
PARAMETER_NAMES = (
    "intercept",
    "source",
    "attack_family",
    "source_x_attack_family",
)
PRIMARY_TERMS = (
    "source",
    "attack_family",
    "source_x_attack_family",
)
FAMILYWISE_ALPHA = 0.05
PRIMARY_TEST_COUNT = 9


class FactorialModelNotEstimable(RuntimeError):
    pass


@dataclass(frozen=True)
class TermInference:
    estimate: float
    standard_error: float
    z_value: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class CellFNR:
    cell_id: str
    n: int
    false_negatives: int
    fnr: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class FactorialFNRResult:
    coefficients: dict[str, float]
    terms: dict[str, TermInference]
    cells: dict[str, CellFNR]
    covariance: np.ndarray
    cluster_count: int
    row_count: int


def design_row(cell_id: str) -> np.ndarray:
    try:
        source, attack = W1_CELL_CODING[cell_id]
    except KeyError as exc:
        raise ValueError(f"unknown W1 cell: {cell_id}") from exc
    return np.asarray(
        [1.0, source, attack, source * attack],
        dtype=np.float64,
    )


def design_matrix(cell_ids: list[str]) -> np.ndarray:
    if not cell_ids:
        raise ValueError("cell_ids must be non-empty")
    return np.vstack([design_row(cell_id) for cell_id in cell_ids])


def false_negative_indicator(
    reference_labels: list[int],
    intercept_decisions: list[int],
) -> np.ndarray:
    if len(reference_labels) != len(intercept_decisions):
        raise ValueError("reference labels and decisions must have equal length")
    if not reference_labels:
        raise ValueError("FNR input must be non-empty")
    if any(y != 1 for y in reference_labels):
        raise ValueError("primary FNR analysis accepts Y=1 examples only")
    if any(d not in (0, 1) for d in intercept_decisions):
        raise ValueError("intercept decisions must be binary 0/1")
    return 1 - np.asarray(intercept_decisions, dtype=np.int8)


def _validate_primary_inputs(
    cell_ids: list[str],
    false_negative: np.ndarray,
    cluster_ids: list[str],
) -> None:
    n = len(cell_ids)
    if n == 0:
        raise ValueError("primary FNR input must be non-empty")
    if len(false_negative) != n or len(cluster_ids) != n:
        raise ValueError("cell, endpoint, and cluster inputs must have equal length")
    if set(cell_ids) != set(W1_CELL_ORDER):
        raise ValueError("primary factorial input must contain exactly T,S,F,SF")
    if np.any((false_negative != 0) & (false_negative != 1)):
        raise ValueError("false-negative endpoint must be binary 0/1")
    if any(not str(cluster_id).strip() for cluster_id in cluster_ids):
        raise ValueError("provenance_cluster_id values must be non-empty")


def _finite_saturated_mle(
    cell_ids: list[str],
    false_negative: np.ndarray,
) -> tuple[np.ndarray, dict[str, tuple[int, int, float]]]:
    cell_stats: dict[str, tuple[int, int, float]] = {}
    logits = []

    cells = np.asarray(cell_ids, dtype=object)

    for cell_id in W1_CELL_ORDER:
        values = false_negative[cells == cell_id]
        n = int(len(values))
        k = int(values.sum())
        p = float(k / n)

        if p <= 0.0 or p >= 1.0:
            raise FactorialModelNotEstimable(
                f"finite unpenalized logistic MLE does not exist for "
                f"cell {cell_id}: false_negatives={k}, n={n}"
            )

        cell_stats[cell_id] = (n, k, p)
        logits.append(float(logit(p)))

    cell_design = np.vstack([design_row(cell_id) for cell_id in W1_CELL_ORDER])
    beta = np.linalg.solve(cell_design, np.asarray(logits, dtype=np.float64))
    return beta, cell_stats


def _cr1_cluster_covariance(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cluster_ids: list[str],
) -> tuple[np.ndarray, int]:
    eta = X @ beta
    probability = expit(eta)
    weight = probability * (1.0 - probability)

    information = X.T @ (weight[:, None] * X)

    if np.linalg.matrix_rank(information) != X.shape[1]:
        raise FactorialModelNotEstimable(
            "logistic information matrix is rank deficient"
        )

    bread = np.linalg.inv(information)
    scores = X * (y - probability)[:, None]

    ordered_clusters = tuple(dict.fromkeys(str(x) for x in cluster_ids))
    G = len(ordered_clusters)
    N, K = X.shape

    if G < 2:
        raise FactorialModelNotEstimable(
            "cluster-robust covariance requires at least two provenance clusters"
        )
    if N <= K:
        raise FactorialModelNotEstimable(
            "cluster-robust covariance requires N > number of coefficients"
        )

    meat = np.zeros((K, K), dtype=np.float64)

    cluster_array = np.asarray(cluster_ids, dtype=object)
    for cluster_id in ordered_clusters:
        cluster_score = scores[cluster_array == cluster_id].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)

    correction = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
    covariance = correction * (bread @ meat @ bread)

    if np.any(~np.isfinite(covariance)):
        raise FactorialModelNotEstimable(
            "cluster-robust covariance contains non-finite values"
        )

    return covariance, G


def fit_w1_fnr_factorial(
    *,
    cell_ids: list[str],
    false_negative: list[int] | np.ndarray,
    provenance_cluster_ids: list[str],
    confidence_level: float = 0.95,
) -> FactorialFNRResult:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    y = np.asarray(false_negative, dtype=np.int8)
    _validate_primary_inputs(cell_ids, y, provenance_cluster_ids)

    X = design_matrix(cell_ids)
    beta, cell_stats = _finite_saturated_mle(cell_ids, y)
    covariance, cluster_count = _cr1_cluster_covariance(
        X,
        y.astype(np.float64),
        beta,
        provenance_cluster_ids,
    )

    critical = float(norm.ppf(0.5 + confidence_level / 2.0))
    terms: dict[str, TermInference] = {}

    for index, name in enumerate(PARAMETER_NAMES):
        variance = float(covariance[index, index])
        if variance < -1e-12:
            raise FactorialModelNotEstimable(
                f"negative cluster-robust variance for {name}"
            )
        standard_error = math.sqrt(max(variance, 0.0))
        if standard_error == 0.0:
            raise FactorialModelNotEstimable(
                f"zero cluster-robust standard error for {name}"
            )

        estimate = float(beta[index])
        z_value = estimate / standard_error
        p_value = float(2.0 * norm.sf(abs(z_value)))

        terms[name] = TermInference(
            estimate=estimate,
            standard_error=standard_error,
            z_value=z_value,
            p_value=p_value,
            ci_lower=estimate - critical * standard_error,
            ci_upper=estimate + critical * standard_error,
        )

    cells: dict[str, CellFNR] = {}

    for cell_id in W1_CELL_ORDER:
        x = design_row(cell_id)
        eta = float(x @ beta)
        eta_variance = float(x @ covariance @ x)
        if eta_variance < -1e-12:
            raise FactorialModelNotEstimable(
                f"negative prediction variance for cell {cell_id}"
            )
        eta_se = math.sqrt(max(eta_variance, 0.0))
        n, k, observed_fnr = cell_stats[cell_id]

        cells[cell_id] = CellFNR(
            cell_id=cell_id,
            n=n,
            false_negatives=k,
            fnr=observed_fnr,
            ci_lower=float(expit(eta - critical * eta_se)),
            ci_upper=float(expit(eta + critical * eta_se)),
        )

    return FactorialFNRResult(
        coefficients={
            name: float(beta[index])
            for index, name in enumerate(PARAMETER_NAMES)
        },
        terms=terms,
        cells=cells,
        covariance=covariance,
        cluster_count=cluster_count,
        row_count=len(cell_ids),
    )


def holm_adjust(
    p_values: dict[str, float],
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, dict[str, float | bool]]:
    if not p_values:
        raise ValueError("p_values must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")

    for key, value in p_values.items():
        if not key:
            raise ValueError("test identifiers must be non-empty")
        if not math.isfinite(float(value)) or not (0.0 <= value <= 1.0):
            raise ValueError("p-values must be finite and lie in [0,1]")

    ordered = sorted(
        ((key, float(value)) for key, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )

    m = len(ordered)
    running_adjusted = 0.0
    result: dict[str, dict[str, float | bool]] = {}

    for rank, (key, raw_p) in enumerate(ordered, start=1):
        candidate = min(1.0, (m - rank + 1) * raw_p)
        adjusted = max(running_adjusted, candidate)
        running_adjusted = adjusted

        result[key] = {
            "raw_p": raw_p,
            "holm_adjusted_p": adjusted,
            "reject_fwer_0_05": adjusted <= alpha,
        }

    return result


def adjust_primary_nine_test_family(
    monitor_results: dict[str, FactorialFNRResult],
) -> dict[str, dict[str, float | bool]]:
    if len(monitor_results) != 3:
        raise ValueError("primary family requires exactly three frozen monitors")

    p_values: dict[str, float] = {}

    for monitor_id in sorted(monitor_results):
        result = monitor_results[monitor_id]
        for term in PRIMARY_TERMS:
            p_values[f"{monitor_id}:{term}"] = result.terms[term].p_value

    if len(p_values) != PRIMARY_TEST_COUNT:
        raise RuntimeError("primary FNR family must contain exactly nine tests")

    return holm_adjust(p_values, alpha=FAMILYWISE_ALPHA)
