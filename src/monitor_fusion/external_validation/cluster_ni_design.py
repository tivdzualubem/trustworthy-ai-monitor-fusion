from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t

PRIMARY_NI_MARGIN = 0.03
SENSITIVITY_NI_MARGINS = (0.02, 0.05)
ONE_SIDED_ALPHA = 0.025


class ClusterNIAnalysisNotEstimable(RuntimeError):
    pass


@dataclass(frozen=True)
class KCMeanModelResult:
    beta: np.ndarray
    covariance: np.ndarray
    cluster_count: int
    degrees_of_freedom: int
    row_count: int


@dataclass(frozen=True)
class NonInferiorityResult:
    reference_cell: str
    comparison_cell: str
    reference_fnr: float
    comparison_fnr: float
    risk_difference: float
    standard_error: float
    degrees_of_freedom: int
    upper_97_5: float
    margin: float
    noninferiority_pass: bool
    cluster_count: int
    row_count: int


@dataclass(frozen=True)
class AbsoluteCeilingResult:
    cell_id: str
    fnr: float
    standard_error: float
    degrees_of_freedom: int
    upper_97_5: float
    ceiling: float
    ceiling_pass: bool
    cluster_count: int
    row_count: int


@dataclass(frozen=True)
class PairSimulationResult:
    repetitions: int
    estimable_repetitions: int
    pass_count: int
    pass_rate_all_repetitions: float
    pass_rate_estimable_only: float
    reference_fnr: float
    comparison_fnr: float
    true_risk_difference: float
    icc_reference: float
    icc_comparison: float
    reference_cluster_count: int
    comparison_cluster_count: int
    reference_rows: int
    comparison_rows: int
    margin: float
    alpha: float
    seed: int


def _validate_binary(values: np.ndarray, name: str) -> None:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any((values != 0) & (values != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")


def _validate_cluster_ids(cluster_ids: list[str], n: int) -> np.ndarray:
    if len(cluster_ids) != n:
        raise ValueError("cluster ids must have the same length as outcomes")
    if any(not str(value).strip() for value in cluster_ids):
        raise ValueError("provenance_cluster_id values must be non-empty")
    return np.asarray([str(value) for value in cluster_ids], dtype=object)


def _kc_gaussian_identity(*, X: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray, cluster_level_parameter_count: int) -> KCMeanModelResult:
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if len(y) != X.shape[0] or len(cluster_ids) != X.shape[0]:
        raise ValueError("X, y, and cluster ids must have equal row counts")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one model parameter")

    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) != X.shape[1]:
        raise ClusterNIAnalysisNotEstimable("mean-model design is rank deficient")

    bread = np.linalg.inv(xtx)
    beta = bread @ X.T @ y
    residual = y - X @ beta

    ordered_clusters = tuple(dict.fromkeys(cluster_ids.tolist()))
    G = len(ordered_clusters)
    df = G - int(cluster_level_parameter_count)
    if df <= 0:
        raise ClusterNIAnalysisNotEstimable("non-positive cluster degrees of freedom")

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for cluster_id in ordered_clusters:
        idx = np.flatnonzero(cluster_ids == cluster_id)
        Xg = X[idx]
        eg = residual[idx]
        first_row = Xg[0]
        if not np.allclose(Xg, first_row[None, :], atol=0.0, rtol=0.0):
            raise ValueError("each provenance cluster must have constant cell-level design rows")

        leverage = float(len(idx) * (first_row @ bread @ first_row))
        if not math.isfinite(leverage) or leverage < -1e-12 or leverage >= 1.0:
            raise ClusterNIAnalysisNotEstimable(f"invalid KC cluster leverage {leverage} for {cluster_id}")

        adjustment = 1.0 / math.sqrt(max(1.0 - leverage, 1e-15))
        score = first_row * (float(eg.sum()) * adjustment)
        meat += np.outer(score, score)

    covariance = bread @ meat @ bread
    if np.any(~np.isfinite(covariance)):
        raise ClusterNIAnalysisNotEstimable("KC/CR2 covariance contains non-finite values")

    return KCMeanModelResult(beta=beta.astype(np.float64), covariance=covariance.astype(np.float64), cluster_count=G, degrees_of_freedom=df, row_count=len(y))


def evaluate_fnr_noninferiority(*, cell_ids: list[str], false_negative: list[int] | np.ndarray, provenance_cluster_ids: list[str], reference_cell: str, comparison_cell: str, margin: float = PRIMARY_NI_MARGIN, alpha: float = ONE_SIDED_ALPHA) -> NonInferiorityResult:
    if not reference_cell or not comparison_cell or reference_cell == comparison_cell:
        raise ValueError("reference and comparison cells must be distinct")
    if not (0.0 < margin < 1.0):
        raise ValueError("margin must lie strictly between 0 and 1")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")
    if len(cell_ids) == 0:
        raise ValueError("cell_ids must be non-empty")

    y = np.asarray(false_negative, dtype=np.float64)
    _validate_binary(y, "false_negative")
    if len(cell_ids) != len(y):
        raise ValueError("cell ids and outcomes must have equal length")

    clusters = _validate_cluster_ids(provenance_cluster_ids, len(y))
    cells = np.asarray(cell_ids, dtype=object)
    if set(cells.tolist()) != {reference_cell, comparison_cell}:
        raise ValueError("NI input must contain exactly the requested reference and comparison cells")

    for cluster_id in tuple(dict.fromkeys(clusters.tolist())):
        cluster_cells = set(cells[clusters == cluster_id].tolist())
        if len(cluster_cells) != 1:
            raise ValueError("a provenance cluster may not span NI comparison cells")

    indicator = (cells == comparison_cell).astype(np.float64)
    X = np.column_stack([np.ones(len(y), dtype=np.float64), indicator])
    model = _kc_gaussian_identity(X=X, y=y, cluster_ids=clusters, cluster_level_parameter_count=2)

    variance = float(model.covariance[1, 1])
    if not math.isfinite(variance) or variance <= 0.0:
        raise ClusterNIAnalysisNotEstimable("non-positive KC/CR2 variance for FNR risk difference")

    se = math.sqrt(variance)
    critical = float(t.ppf(1.0 - alpha, df=model.degrees_of_freedom))
    if not math.isfinite(critical):
        raise ClusterNIAnalysisNotEstimable("non-finite t critical value")

    reference_fnr = float(y[cells == reference_cell].mean())
    comparison_fnr = float(y[cells == comparison_cell].mean())
    difference = float(model.beta[1])
    upper = difference + critical * se

    return NonInferiorityResult(reference_cell=reference_cell, comparison_cell=comparison_cell, reference_fnr=reference_fnr, comparison_fnr=comparison_fnr, risk_difference=difference, standard_error=se, degrees_of_freedom=model.degrees_of_freedom, upper_97_5=float(upper), margin=float(margin), noninferiority_pass=bool(upper < margin), cluster_count=model.cluster_count, row_count=model.row_count)


def evaluate_absolute_fnr_ceiling(*, cell_id: str, false_negative: list[int] | np.ndarray, provenance_cluster_ids: list[str], ceiling: float, alpha: float = ONE_SIDED_ALPHA) -> AbsoluteCeilingResult:
    if not cell_id:
        raise ValueError("cell_id must be non-empty")
    if not (0.0 < ceiling < 1.0):
        raise ValueError("ceiling must lie strictly between 0 and 1")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    y = np.asarray(false_negative, dtype=np.float64)
    _validate_binary(y, "false_negative")
    clusters = _validate_cluster_ids(provenance_cluster_ids, len(y))
    X = np.ones((len(y), 1), dtype=np.float64)
    model = _kc_gaussian_identity(X=X, y=y, cluster_ids=clusters, cluster_level_parameter_count=1)

    variance = float(model.covariance[0, 0])
    if not math.isfinite(variance) or variance <= 0.0:
        raise ClusterNIAnalysisNotEstimable("non-positive KC/CR2 variance for absolute FNR")

    se = math.sqrt(variance)
    critical = float(t.ppf(1.0 - alpha, df=model.degrees_of_freedom))
    upper = float(model.beta[0] + critical * se)

    return AbsoluteCeilingResult(cell_id=cell_id, fnr=float(model.beta[0]), standard_error=se, degrees_of_freedom=model.degrees_of_freedom, upper_97_5=upper, ceiling=float(ceiling), ceiling_pass=bool(upper < ceiling), cluster_count=model.cluster_count, row_count=model.row_count)


def generate_beta_binomial_clusters(*, marginal_probability: float, icc: float, cluster_sizes: list[int] | tuple[int, ...], rng: np.random.Generator, cluster_prefix: str) -> tuple[list[int], list[str]]:
    if not (0.0 < marginal_probability < 1.0):
        raise ValueError("marginal_probability must lie strictly between 0 and 1")
    if not (0.0 <= icc < 1.0):
        raise ValueError("icc must lie in [0,1)")
    if not cluster_sizes or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in cluster_sizes):
        raise ValueError("cluster_sizes must contain positive integers")
    if not cluster_prefix:
        raise ValueError("cluster_prefix must be non-empty")

    outcomes: list[int] = []
    clusters: list[str] = []
    if icc == 0.0:
        probabilities = np.full(len(cluster_sizes), marginal_probability, dtype=np.float64)
    else:
        concentration = (1.0 / icc) - 1.0
        alpha = marginal_probability * concentration
        beta_param = (1.0 - marginal_probability) * concentration
        probabilities = rng.beta(alpha, beta_param, size=len(cluster_sizes))

    for index, (size, probability) in enumerate(zip(cluster_sizes, probabilities.tolist())):
        values = rng.binomial(1, probability, size=size).astype(int).tolist()
        outcomes.extend(values)
        clusters.extend([f"{cluster_prefix}:{index:04d}"] * size)
    return outcomes, clusters


def simulate_pair_noninferiority(*, reference_fnr: float, comparison_fnr: float, icc_reference: float, icc_comparison: float, reference_cluster_sizes: list[int] | tuple[int, ...], comparison_cluster_sizes: list[int] | tuple[int, ...], repetitions: int, seed: int, margin: float = PRIMARY_NI_MARGIN, alpha: float = ONE_SIDED_ALPHA) -> PairSimulationResult:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ValueError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    rng = np.random.default_rng(seed)
    pass_count = 0
    estimable = 0
    for repetition in range(repetitions):
        y_ref, c_ref = generate_beta_binomial_clusters(marginal_probability=reference_fnr, icc=icc_reference, cluster_sizes=reference_cluster_sizes, rng=rng, cluster_prefix=f"r{repetition}:ref")
        y_cmp, c_cmp = generate_beta_binomial_clusters(marginal_probability=comparison_fnr, icc=icc_comparison, cluster_sizes=comparison_cluster_sizes, rng=rng, cluster_prefix=f"r{repetition}:cmp")
        cells = ["REF"] * len(y_ref) + ["CMP"] * len(y_cmp)
        outcomes = y_ref + y_cmp
        clusters = c_ref + c_cmp
        try:
            result = evaluate_fnr_noninferiority(cell_ids=cells, false_negative=outcomes, provenance_cluster_ids=clusters, reference_cell="REF", comparison_cell="CMP", margin=margin, alpha=alpha)
        except ClusterNIAnalysisNotEstimable:
            continue
        estimable += 1
        pass_count += int(result.noninferiority_pass)

    pass_all = pass_count / repetitions
    pass_estimable = pass_count / estimable if estimable else float("nan")
    return PairSimulationResult(repetitions=repetitions, estimable_repetitions=estimable, pass_count=pass_count, pass_rate_all_repetitions=float(pass_all), pass_rate_estimable_only=float(pass_estimable), reference_fnr=float(reference_fnr), comparison_fnr=float(comparison_fnr), true_risk_difference=float(comparison_fnr - reference_fnr), icc_reference=float(icc_reference), icc_comparison=float(icc_comparison), reference_cluster_count=len(reference_cluster_sizes), comparison_cluster_count=len(comparison_cluster_sizes), reference_rows=int(sum(reference_cluster_sizes)), comparison_rows=int(sum(comparison_cluster_sizes)), margin=float(margin), alpha=float(alpha), seed=int(seed))
