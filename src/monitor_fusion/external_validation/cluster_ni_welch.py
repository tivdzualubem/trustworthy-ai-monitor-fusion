from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import t

from monitor_fusion.external_validation.cluster_ni_design import (
    ONE_SIDED_ALPHA,
    PRIMARY_NI_MARGIN,
)
from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    _validate_sizes,
)


@dataclass(frozen=True)
class WelchClusterCalibration:
    repetitions: int
    reject_count: int
    estimable_count: int
    type_I_rate: float
    mc_se: float
    cp_lower_95: float
    cp_upper_95: float
    df_min: float
    df_median: float
    df_max: float


def _cp_interval(k: int, n: int) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    upper = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lower, upper


def _group_cluster_variance(
    counts: np.ndarray,
    cluster_sizes: np.ndarray,
    *,
    correction: str = "CRV1",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Row-marginal mean and independent-cluster variance component for one cell.

    The estimand remains row-marginal FNR:
        p_hat = sum_g k_g / sum_g m_g

    For CRV1, the cluster variance component is:
        V = G/(G-1) * sum_g (k_g - m_g p_hat)^2 / N^2

    For CRV3, each cluster score is divided by (1 - m_g/N) before squaring.

    Returns (mean, variance, nominal_group_df).
    """
    sizes = _validate_sizes(np.asarray(cluster_sizes)).astype(np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(sizes):
        raise ValueError("counts must have one column per cluster")

    G = len(sizes)
    if G < 2:
        raise ValueError("each cell requires at least two provenance clusters")

    N = float(sizes.sum())
    mean = counts.sum(axis=1) / N
    residual_sum = counts - mean[:, None] * sizes[None, :]

    if correction == "CRV1":
        adjusted = residual_sum
    elif correction == "CRV3":
        leverage = sizes / N
        if np.any(leverage >= 1.0):
            raise ValueError("cluster leverage must be below one")
        adjusted = residual_sum / (1.0 - leverage[None, :])
    else:
        raise ValueError("correction must be CRV1 or CRV3")

    variance = np.sum(adjusted * adjusted, axis=1) / (N * N)

    # Cell-specific finite-cluster correction. For CRV3 we do not multiply by
    # G/(G-1), because the leave-one-cluster leverage adjustment is already the
    # explicit small-sample correction being evaluated.
    if correction == "CRV1":
        variance *= G / (G - 1.0)

    return mean, variance, float(G - 1)


def _welch_df(
    variance_ref: np.ndarray,
    variance_cmp: np.ndarray,
    df_ref: float,
    df_cmp: float,
) -> np.ndarray:
    numerator = (variance_ref + variance_cmp) ** 2
    denominator = (
        (variance_ref**2) / df_ref
        + (variance_cmp**2) / df_cmp
    )
    df = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    df[valid] = numerator[valid] / denominator[valid]
    return df


def simulate_welch_cluster_boundary(
    *,
    reference_fnr: float,
    icc_reference: float,
    icc_comparison: float,
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    repetitions: int,
    seed: int,
    correction: str = "CRV1",
    margin: float = PRIMARY_NI_MARGIN,
    alpha: float = ONE_SIDED_ALPHA,
) -> WelchClusterCalibration:
    """
    Welch-Satterthwaite test for a difference of two independent row-marginal
    FNRs using cell-specific cluster-robust variance components.

    This is intentionally different from the earlier pooled-cluster df rules:
    the smaller/noisier cell contributes more strongly to the final degrees of
    freedom through the Welch-Satterthwaite denominator.
    """
    if not (0.0 < reference_fnr < 1.0 - margin):
        raise ValueError("reference_fnr must leave room for the NI boundary")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    ref_sizes = _validate_sizes(np.asarray(reference_cluster_sizes))
    cmp_sizes = _validate_sizes(np.asarray(comparison_cluster_sizes))
    comparison_fnr = reference_fnr + margin

    rng = np.random.default_rng(seed)

    ref_counts = _beta_binomial_counts(
        marginal_probability=reference_fnr,
        icc=icc_reference,
        cluster_sizes=ref_sizes,
        repetitions=repetitions,
        rng=rng,
    )
    cmp_counts = _beta_binomial_counts(
        marginal_probability=comparison_fnr,
        icc=icc_comparison,
        cluster_sizes=cmp_sizes,
        repetitions=repetitions,
        rng=rng,
    )

    ref_mean, ref_var, ref_df = _group_cluster_variance(
        ref_counts,
        ref_sizes,
        correction=correction,
    )
    cmp_mean, cmp_var, cmp_df = _group_cluster_variance(
        cmp_counts,
        cmp_sizes,
        correction=correction,
    )

    difference = cmp_mean - ref_mean
    variance = ref_var + cmp_var
    df = _welch_df(ref_var, cmp_var, ref_df, cmp_df)

    estimable = (
        np.isfinite(variance)
        & (variance > 0.0)
        & np.isfinite(df)
        & (df >= 1.0)
    )

    upper = np.full(repetitions, np.inf, dtype=np.float64)
    critical = np.full(repetitions, np.nan, dtype=np.float64)
    critical[estimable] = t.ppf(1.0 - alpha, df=df[estimable])

    upper[estimable] = (
        difference[estimable]
        + critical[estimable] * np.sqrt(variance[estimable])
    )

    reject = estimable & (upper < margin)
    reject_count = int(reject.sum())
    estimable_count = int(estimable.sum())
    rate = reject_count / repetitions
    mc_se = math.sqrt(rate * (1.0 - rate) / repetitions)
    lower, upper_ci = _cp_interval(reject_count, repetitions)

    finite_df = df[estimable]
    return WelchClusterCalibration(
        repetitions=repetitions,
        reject_count=reject_count,
        estimable_count=estimable_count,
        type_I_rate=float(rate),
        mc_se=float(mc_se),
        cp_lower_95=lower,
        cp_upper_95=upper_ci,
        df_min=float(np.min(finite_df)) if len(finite_df) else float("nan"),
        df_median=float(np.median(finite_df)) if len(finite_df) else float("nan"),
        df_max=float(np.max(finite_df)) if len(finite_df) else float("nan"),
    )
