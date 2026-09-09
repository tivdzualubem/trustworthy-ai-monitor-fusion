from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

PRIMARY_NI_MARGIN = 0.03
ABSOLUTE_FNR_CEILING = 0.10
FPR_CONSTRAINT = 0.05
PRIMARY_SAFETY_ALPHA = 0.025
FPR_ALPHA = 0.05


class ClusterScoreNotEstimable(RuntimeError):
    pass


@dataclass(frozen=True)
class ClusterWilsonResult:
    estimate: float
    lower: float
    upper: float
    icc_hat: float
    variance_inflation: float
    cluster_count: int
    row_count: int


@dataclass(frozen=True)
class ClusterMoverRDResult:
    reference_rate: float
    comparison_rate: float
    risk_difference: float
    upper_confidence_limit: float
    margin: float
    pass_noninferiority: bool
    reference_icc_hat: float
    comparison_icc_hat: float
    reference_variance_inflation: float
    comparison_variance_inflation: float


def _validate_counts(
    cluster_events: list[int] | np.ndarray,
    cluster_sizes: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(cluster_events, dtype=np.float64)
    n = np.asarray(cluster_sizes, dtype=np.float64)

    if x.ndim != 1 or n.ndim != 1 or len(x) != len(n) or len(x) < 2:
        raise ValueError("cluster event and size vectors must have equal length >= 2")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(n)):
        raise ValueError("cluster inputs must be finite")
    if np.any(n <= 0) or np.any(np.floor(n) != n):
        raise ValueError("cluster sizes must be positive integers")
    if np.any(x < 0) or np.any(np.floor(x) != x) or np.any(x > n):
        raise ValueError("cluster events must be integer counts in [0, cluster size]")
    if float(np.sum(n - 1.0)) <= 0.0:
        raise ClusterScoreNotEstimable(
            "clustered-proportion ICC requires at least one cluster with size > 1"
        )

    return x, n


def cluster_variance_inflation(
    *,
    cluster_events: list[int] | np.ndarray,
    cluster_sizes: list[int] | np.ndarray,
) -> tuple[float, float]:
    """
    Moment ICC and variance-inflation calculation used by Saha et al. and
    ratesci::clusterpci. This is ported directly for reproducible validation.
    """
    x, n = _validate_counts(cluster_events, cluster_sizes)

    totx = float(x.sum())
    totn = float(n.sum())
    k = len(n)

    sum_x2_over_n = float(np.sum((x * x) / n))
    bms = (sum_x2_over_n - (totx * totx) / totn) / (k - 1.0)
    wms = (totx - sum_x2_over_n) / float(np.sum(n - 1.0))
    nstar = (totn * totn - float(np.sum(n * n))) / ((k - 1.0) * totn)

    denominator = bms + (nstar - 1.0) * wms
    if not math.isfinite(denominator) or abs(denominator) <= 1e-15:
        raise ClusterScoreNotEstimable("cluster ICC moment denominator is degenerate")

    icc_hat = (bms - wms) / denominator
    variance_inflation = float(
        np.sum(n * (1.0 + (n - 1.0) * icc_hat)) / totn
    )

    if not math.isfinite(icc_hat) or not math.isfinite(variance_inflation):
        raise ClusterScoreNotEstimable("non-finite clustered-proportion moment estimate")
    if variance_inflation <= 0.0:
        raise ClusterScoreNotEstimable("non-positive variance inflation")

    return float(icc_hat), float(variance_inflation)


def cluster_wilson_interval(
    *,
    cluster_events: list[int] | np.ndarray,
    cluster_sizes: list[int] | np.ndarray,
    one_sided_alpha: float,
) -> ClusterWilsonResult:
    """
    Cluster-adjusted Wilson score interval. For a one-sided upper test at alpha,
    z = Phi^{-1}(1-alpha). This is equivalent to using a central interval whose
    upper-tail probability is alpha.
    """
    if not (0.0 < one_sided_alpha < 0.5):
        raise ValueError("one_sided_alpha must lie strictly between 0 and 0.5")

    x, n = _validate_counts(cluster_events, cluster_sizes)
    total_events = float(x.sum())
    total_rows = float(n.sum())
    estimate = total_events / total_rows

    icc_hat, xihat = cluster_variance_inflation(
        cluster_events=x,
        cluster_sizes=n,
    )

    z = float(norm.ppf(1.0 - one_sided_alpha))
    za = z * math.sqrt(xihat)
    za2 = za * za

    denom = 1.0 + za2 / total_rows
    center = (estimate + za2 / (2.0 * total_rows)) / denom
    half = (
        za
        * math.sqrt(
            estimate * (1.0 - estimate) / total_rows
            + za2 / (4.0 * total_rows * total_rows)
        )
        / denom
    )

    lower = max(0.0, center - half)
    upper = min(1.0, center + half)

    if total_events == 0.0:
        lower = 0.0
    if total_events == total_rows:
        upper = 1.0

    return ClusterWilsonResult(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        icc_hat=float(icc_hat),
        variance_inflation=float(xihat),
        cluster_count=len(n),
        row_count=int(total_rows),
    )


def cluster_mover_rd_upper(
    *,
    reference_events: list[int] | np.ndarray,
    reference_sizes: list[int] | np.ndarray,
    comparison_events: list[int] | np.ndarray,
    comparison_sizes: list[int] | np.ndarray,
    margin: float = PRIMARY_NI_MARGIN,
    one_sided_alpha: float = PRIMARY_SAFETY_ALPHA,
) -> ClusterMoverRDResult:
    """
    Newcombe Method-10 MOVER upper bound for comparison-reference RD, using
    cluster-adjusted Wilson score intervals for the two marginal proportions.
    """
    if not (0.0 < margin < 1.0):
        raise ValueError("margin must lie strictly between 0 and 1")

    ref = cluster_wilson_interval(
        cluster_events=reference_events,
        cluster_sizes=reference_sizes,
        one_sided_alpha=one_sided_alpha,
    )
    cmp_ = cluster_wilson_interval(
        cluster_events=comparison_events,
        cluster_sizes=comparison_sizes,
        one_sided_alpha=one_sided_alpha,
    )

    rd = cmp_.estimate - ref.estimate
    upper = rd + math.sqrt(
        max(
            0.0,
            (cmp_.upper - cmp_.estimate) ** 2
            + (ref.estimate - ref.lower) ** 2,
        )
    )

    return ClusterMoverRDResult(
        reference_rate=ref.estimate,
        comparison_rate=cmp_.estimate,
        risk_difference=float(rd),
        upper_confidence_limit=float(upper),
        margin=float(margin),
        pass_noninferiority=bool(upper < margin),
        reference_icc_hat=ref.icc_hat,
        comparison_icc_hat=cmp_.icc_hat,
        reference_variance_inflation=ref.variance_inflation,
        comparison_variance_inflation=cmp_.variance_inflation,
    )
