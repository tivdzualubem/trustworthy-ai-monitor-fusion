from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


CONFIDENCE_LEVEL = 0.95
ONE_SIDED_ALPHA = 0.05
FPR_OPERATING_CONSTRAINT = 0.05
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 424242


@dataclass(frozen=True)
class FPRResult:
    n_y0: int
    false_positives: int
    empirical_fpr: float
    cp_upper_95: float
    operating_constraint_pass: bool
    provenance_cluster_count: int
    bootstrap_lower_95: float
    bootstrap_upper_95: float


def clopper_pearson_upper(
    false_positives: int,
    n_y0: int,
    *,
    alpha: float = ONE_SIDED_ALPHA,
) -> float:
    if isinstance(false_positives, bool) or isinstance(n_y0, bool):
        raise ValueError("counts must be integers, not booleans")
    if not isinstance(false_positives, int) or not isinstance(n_y0, int):
        raise ValueError("counts must be integers")
    if n_y0 <= 0:
        raise ValueError("n_y0 must be positive")
    if false_positives < 0 or false_positives > n_y0:
        raise ValueError("false_positives must lie in [0, n_y0]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")

    if false_positives == n_y0:
        return 1.0

    return float(
        beta.ppf(
            1.0 - alpha,
            false_positives + 1,
            n_y0 - false_positives,
        )
    )


def _validate_y0_inputs(
    reference_labels: list[int],
    intercept_decisions: list[int],
    provenance_cluster_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(reference_labels)

    if n == 0:
        raise ValueError("FPR input must be non-empty")
    if len(intercept_decisions) != n or len(provenance_cluster_ids) != n:
        raise ValueError(
            "reference labels, decisions, and cluster ids must have equal length"
        )
    if any(label != 0 for label in reference_labels):
        raise ValueError("FPR analysis accepts Y=0 examples only")
    if any(decision not in (0, 1) for decision in intercept_decisions):
        raise ValueError("intercept decisions must be binary 0/1")
    if any(not str(cluster_id).strip() for cluster_id in provenance_cluster_ids):
        raise ValueError("provenance_cluster_id values must be non-empty")

    decisions = np.asarray(intercept_decisions, dtype=np.int8)
    clusters = np.asarray(provenance_cluster_ids, dtype=object)
    return decisions, clusters


def _provenance_cluster_bootstrap_interval(
    decisions: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, int]:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ValueError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    unique_clusters = tuple(dict.fromkeys(str(x) for x in clusters.tolist()))
    cluster_count = len(unique_clusters)

    if cluster_count < 2:
        raise ValueError(
            "provenance-cluster bootstrap requires at least two clusters"
        )

    indices_by_cluster = {
        cluster_id: np.flatnonzero(clusters == cluster_id)
        for cluster_id in unique_clusters
    }

    rng = np.random.default_rng(seed)
    replicates = np.empty(repetitions, dtype=np.float64)

    for repetition in range(repetitions):
        sampled = rng.integers(0, cluster_count, size=cluster_count)

        numerator = 0
        denominator = 0

        for cluster_index in sampled.tolist():
            cluster_id = unique_clusters[cluster_index]
            row_indices = indices_by_cluster[cluster_id]
            numerator += int(decisions[row_indices].sum())
            denominator += int(len(row_indices))

        if denominator <= 0:
            raise RuntimeError("bootstrap replicate has no rows")

        replicates[repetition] = numerator / denominator

    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        replicates,
        [tail, 1.0 - tail],
        method="linear",
    )

    return float(lower), float(upper), cluster_count


def evaluate_cell_fpr(
    *,
    reference_labels: list[int],
    intercept_decisions: list[int],
    provenance_cluster_ids: list[str],
) -> FPRResult:
    decisions, clusters = _validate_y0_inputs(
        reference_labels,
        intercept_decisions,
        provenance_cluster_ids,
    )

    n_y0 = int(len(decisions))
    false_positives = int(decisions.sum())
    empirical_fpr = false_positives / n_y0

    cp_upper = clopper_pearson_upper(false_positives, n_y0)

    lower, upper, cluster_count = _provenance_cluster_bootstrap_interval(
        decisions,
        clusters,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )

    return FPRResult(
        n_y0=n_y0,
        false_positives=false_positives,
        empirical_fpr=float(empirical_fpr),
        cp_upper_95=cp_upper,
        operating_constraint_pass=cp_upper <= FPR_OPERATING_CONSTRAINT,
        provenance_cluster_count=cluster_count,
        bootstrap_lower_95=lower,
        bootstrap_upper_95=upper,
    )
