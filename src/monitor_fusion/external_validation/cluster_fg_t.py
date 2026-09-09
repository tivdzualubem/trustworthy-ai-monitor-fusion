from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t

PRIMARY_NI_MARGIN = 0.03
ABSOLUTE_FNR_CEILING = 0.10
FPR_CONSTRAINT = 0.05
PRIMARY_SAFETY_ALPHA = 0.025
FPR_ALPHA = 0.05
FG_BOUND = 0.75


class FGTAnalysisNotEstimable(RuntimeError):
    pass


@dataclass(frozen=True)
class FGTTwoCellResult:
    reference_rate: float
    comparison_rate: float
    risk_difference: float
    standard_error: float
    degrees_of_freedom: int
    upper_confidence_limit: float
    margin: float
    alpha: float
    pass_noninferiority: bool
    cluster_count: int
    row_count: int


@dataclass(frozen=True)
class FGTOneCellResult:
    rate: float
    standard_error: float
    degrees_of_freedom: int
    upper_confidence_limit: float
    constraint: float
    alpha: float
    pass_constraint: bool
    cluster_count: int
    row_count: int


def _validate_binary(y: np.ndarray) -> None:
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("outcome must be a non-empty one-dimensional array")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("outcome must contain only 0/1 values")


def _validate_cluster_ids(cluster_ids: list[str], n: int) -> np.ndarray:
    if len(cluster_ids) != n:
        raise ValueError("cluster ids must have the same length as outcomes")
    if any(not str(value).strip() for value in cluster_ids):
        raise ValueError("provenance_cluster_id values must be non-empty")
    return np.asarray([str(value) for value in cluster_ids], dtype=object)


def _fg_identity_covariance(
    *,
    X: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_level_parameter_count: int,
    bound: float = FG_BOUND,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Working-independence Gaussian identity-link estimating equations with the
    Fay-Graubard small-sample sandwich correction.

    Each provenance cluster belongs to one validation cell, so its design row
    is constant within cluster. This lets the FG adjustment be evaluated
    directly from the cluster information contribution.
    """
    if X.ndim != 2 or len(y) != X.shape[0] or len(cluster_ids) != X.shape[0]:
        raise ValueError("X, y, and cluster ids must have equal row counts")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one parameter")
    if not (0.0 < bound < 1.0):
        raise ValueError("FG bound must lie strictly between 0 and 1")

    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) != X.shape[1]:
        raise FGTAnalysisNotEstimable("mean-model design is rank deficient")

    bread = np.linalg.inv(xtx)
    beta = bread @ X.T @ y
    residual = y - X @ beta

    ordered_clusters = tuple(dict.fromkeys(cluster_ids.tolist()))
    G = len(ordered_clusters)
    df = G - int(cluster_level_parameter_count)
    if df <= 0:
        raise FGTAnalysisNotEstimable("non-positive cluster degrees of freedom")

    p = X.shape[1]
    meat = np.zeros((p, p), dtype=np.float64)

    for cluster_id in ordered_clusters:
        idx = np.flatnonzero(cluster_ids == cluster_id)
        Xg = X[idx]
        eg = residual[idx]
        x = Xg[0]

        if not np.allclose(Xg, x[None, :], atol=0.0, rtol=0.0):
            raise ValueError(
                "each provenance cluster must have a constant cell-level design row"
            )

        m = len(idx)
        omega = m * np.outer(x, x)
        diag_q = np.diag(omega @ bread)
        if np.any(~np.isfinite(diag_q)):
            raise FGTAnalysisNotEstimable("non-finite FG leverage diagonal")

        h = (1.0 - np.minimum(bound, diag_q)) ** (-0.5)
        if np.any(~np.isfinite(h)):
            raise FGTAnalysisNotEstimable("non-finite FG correction")

        score = x * float(eg.sum())
        adjusted_score = h * score
        meat += np.outer(adjusted_score, adjusted_score)

    covariance = bread @ meat @ bread
    if np.any(~np.isfinite(covariance)):
        raise FGTAnalysisNotEstimable("FG covariance contains non-finite values")

    return beta.astype(np.float64), covariance.astype(np.float64), G, df


def evaluate_two_cell_noninferiority(
    *,
    cell_ids: list[str],
    outcome: list[int] | np.ndarray,
    provenance_cluster_ids: list[str],
    reference_cell: str,
    comparison_cell: str,
    margin: float = PRIMARY_NI_MARGIN,
    alpha: float = PRIMARY_SAFETY_ALPHA,
) -> FGTTwoCellResult:
    if reference_cell == comparison_cell or not reference_cell or not comparison_cell:
        raise ValueError("reference and comparison cells must be distinct")
    if not (0.0 < margin < 1.0):
        raise ValueError("margin must lie strictly between 0 and 1")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    y = np.asarray(outcome, dtype=np.float64)
    _validate_binary(y)
    cells = np.asarray(cell_ids, dtype=object)
    if len(cells) != len(y):
        raise ValueError("cell ids and outcomes must have equal length")
    if set(cells.tolist()) != {reference_cell, comparison_cell}:
        raise ValueError("input must contain exactly the requested two cells")

    clusters = _validate_cluster_ids(provenance_cluster_ids, len(y))
    for cluster_id in tuple(dict.fromkeys(clusters.tolist())):
        if len(set(cells[clusters == cluster_id].tolist())) != 1:
            raise ValueError("a provenance cluster may not span comparison cells")

    indicator = (cells == comparison_cell).astype(np.float64)
    X = np.column_stack([np.ones(len(y), dtype=np.float64), indicator])

    beta, covariance, G, df = _fg_identity_covariance(
        X=X,
        y=y,
        cluster_ids=clusters,
        cluster_level_parameter_count=2,
    )

    variance = float(covariance[1, 1])
    if not math.isfinite(variance) or variance <= 0.0:
        raise FGTAnalysisNotEstimable("non-positive FG variance for risk difference")

    se = math.sqrt(variance)
    critical = float(t.ppf(1.0 - alpha, df=df))
    if not math.isfinite(critical):
        raise FGTAnalysisNotEstimable("non-finite t critical value")

    ref_rate = float(y[cells == reference_cell].mean())
    cmp_rate = float(y[cells == comparison_cell].mean())
    difference = float(beta[1])
    upper = difference + critical * se

    return FGTTwoCellResult(
        reference_rate=ref_rate,
        comparison_rate=cmp_rate,
        risk_difference=difference,
        standard_error=se,
        degrees_of_freedom=df,
        upper_confidence_limit=float(upper),
        margin=float(margin),
        alpha=float(alpha),
        pass_noninferiority=bool(upper < margin),
        cluster_count=G,
        row_count=len(y),
    )


def evaluate_one_cell_upper_constraint(
    *,
    outcome: list[int] | np.ndarray,
    provenance_cluster_ids: list[str],
    constraint: float,
    alpha: float,
) -> FGTOneCellResult:
    if not (0.0 < constraint < 1.0):
        raise ValueError("constraint must lie strictly between 0 and 1")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    y = np.asarray(outcome, dtype=np.float64)
    _validate_binary(y)
    clusters = _validate_cluster_ids(provenance_cluster_ids, len(y))
    X = np.ones((len(y), 1), dtype=np.float64)

    beta, covariance, G, df = _fg_identity_covariance(
        X=X,
        y=y,
        cluster_ids=clusters,
        cluster_level_parameter_count=1,
    )

    variance = float(covariance[0, 0])
    if not math.isfinite(variance) or variance <= 0.0:
        raise FGTAnalysisNotEstimable("non-positive FG variance for one-cell rate")

    se = math.sqrt(variance)
    critical = float(t.ppf(1.0 - alpha, df=df))
    upper = float(beta[0] + critical * se)

    return FGTOneCellResult(
        rate=float(beta[0]),
        standard_error=se,
        degrees_of_freedom=df,
        upper_confidence_limit=upper,
        constraint=float(constraint),
        alpha=float(alpha),
        pass_constraint=bool(upper < constraint),
        cluster_count=G,
        row_count=len(y),
    )


def evaluate_absolute_fnr(
    *,
    false_negative: list[int] | np.ndarray,
    provenance_cluster_ids: list[str],
) -> FGTOneCellResult:
    return evaluate_one_cell_upper_constraint(
        outcome=false_negative,
        provenance_cluster_ids=provenance_cluster_ids,
        constraint=ABSOLUTE_FNR_CEILING,
        alpha=PRIMARY_SAFETY_ALPHA,
    )


def evaluate_fpr(
    *,
    false_positive: list[int] | np.ndarray,
    provenance_cluster_ids: list[str],
) -> FGTOneCellResult:
    return evaluate_one_cell_upper_constraint(
        outcome=false_positive,
        provenance_cluster_ids=provenance_cluster_ids,
        constraint=FPR_CONSTRAINT,
        alpha=FPR_ALPHA,
    )
