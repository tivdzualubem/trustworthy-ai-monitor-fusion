from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from monitor_fusion.external_validation.cluster_ni_power import _validate_sizes
from monitor_fusion.external_validation.cluster_ni_wcr import (
    _bootstrap_group_variance_fast,
    _cp_interval,
    _group_variance_from_cluster_residual_sums,
    _lower_tail_p_values_fail_closed,
    _webb_weights,
)


@dataclass(frozen=True)
class WCRBatchCalibration:
    repetitions: int
    reject_count: int
    estimable_count: int
    type_I_rate: float
    cp_lower_95: float
    cp_upper_95: float


def _validate_bootstrap_grid(alpha: float, bootstrap_repetitions: int) -> None:
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    target = alpha * (bootstrap_repetitions + 1)
    if not math.isclose(target, round(target), abs_tol=1e-12):
        raise ValueError(
            "choose B so alpha*(B+1) is an integer; B=999 works for 0.025 and 0.05"
        )


def _finalize(reject: np.ndarray, estimable: np.ndarray) -> WCRBatchCalibration:
    repetitions = int(len(reject))
    reject_count = int(np.sum(reject))
    estimable_count = int(np.sum(estimable))
    rate = reject_count / repetitions
    lo, hi = _cp_interval(reject_count, repetitions)
    return WCRBatchCalibration(
        repetitions=repetitions,
        reject_count=reject_count,
        estimable_count=estimable_count,
        type_I_rate=float(rate),
        cp_lower_95=float(lo),
        cp_upper_95=float(hi),
    )


def wcr_c_two_sample_batch(
    *,
    reference_counts: np.ndarray,
    comparison_counts: np.ndarray,
    reference_cluster_sizes: np.ndarray,
    comparison_cluster_sizes: np.ndarray,
    margin: float,
    alpha: float,
    bootstrap_repetitions: int,
    reference_weights: np.ndarray,
    comparison_weights: np.ndarray,
) -> WCRBatchCalibration:
    """
    WCR-C lower-tail test for H0: RD >= margin at the least-favorable
    boundary RD=margin, evaluated on already-generated outer datasets.

    Each row of *_counts is one outer dataset; each column is one independent
    provenance cluster. The estimand is the response-average marginal risk
    difference.
    """
    _validate_bootstrap_grid(alpha, bootstrap_repetitions)
    ref_sizes = _validate_sizes(np.asarray(reference_cluster_sizes)).astype(float)
    cmp_sizes = _validate_sizes(np.asarray(comparison_cluster_sizes)).astype(float)
    ref = np.asarray(reference_counts, dtype=float)
    cmp_ = np.asarray(comparison_counts, dtype=float)

    if ref.ndim != 2 or cmp_.ndim != 2 or ref.shape[0] != cmp_.shape[0]:
        raise ValueError("reference/comparison counts must be 2D with equal repetitions")
    if ref.shape[1] != len(ref_sizes) or cmp_.shape[1] != len(cmp_sizes):
        raise ValueError("count columns must match cluster-size vectors")
    if reference_weights.shape != (bootstrap_repetitions, len(ref_sizes)):
        raise ValueError("reference weight matrix has wrong shape")
    if comparison_weights.shape != (bootstrap_repetitions, len(cmp_sizes)):
        raise ValueError("comparison weight matrix has wrong shape")

    n_ref = float(ref_sizes.sum())
    n_cmp = float(cmp_sizes.sum())
    n_total = n_ref + n_cmp
    g_total = len(ref_sizes) + len(cmp_sizes)

    ref_mean = ref.sum(axis=1) / n_ref
    cmp_mean = cmp_.sum(axis=1) / n_cmp
    difference = cmp_mean - ref_mean

    ref_u = ref - ref_mean[:, None] * ref_sizes[None, :]
    cmp_u = cmp_ - cmp_mean[:, None] * cmp_sizes[None, :]

    scalar = (g_total / (g_total - 1.0)) * ((n_total - 1.0) / (n_total - 2.0))
    obs_var = scalar * (
        _group_variance_from_cluster_residual_sums(
            ref_u, ref_sizes, method="CRV1"
        )
        + _group_variance_from_cluster_residual_sums(
            cmp_u, cmp_sizes, method="CRV1"
        )
    )

    obs_estimable = np.isfinite(obs_var) & (obs_var > 0.0)
    t_obs = np.full(len(ref), np.nan, dtype=float)
    t_obs[obs_estimable] = (
        difference[obs_estimable] - margin
    ) / np.sqrt(obs_var[obs_estimable])

    beta0_tilde = (
        ref.sum(axis=1) + cmp_.sum(axis=1) - margin * n_cmp
    ) / n_total
    ref_r = ref - beta0_tilde[:, None] * ref_sizes[None, :]
    cmp_r = cmp_ - (
        beta0_tilde[:, None] + margin
    ) * cmp_sizes[None, :]

    ref_shift, ref_bvar = _bootstrap_group_variance_fast(
        ref_r, ref_sizes, reference_weights, method="CRV1"
    )
    cmp_shift, cmp_bvar = _bootstrap_group_variance_fast(
        cmp_r, cmp_sizes, comparison_weights, method="CRV1"
    )

    b_num = cmp_shift - ref_shift
    b_var = scalar * (ref_bvar + cmp_bvar)
    b_estimable = np.isfinite(b_var) & (b_var > 0.0)

    t_boot = np.full_like(b_num, np.nan)
    t_boot[b_estimable] = b_num[b_estimable] / np.sqrt(b_var[b_estimable])

    _, reject = _lower_tail_p_values_fail_closed(
        t_obs=t_obs,
        t_boot=t_boot,
        bootstrap_estimable=b_estimable,
        observed_estimable=obs_estimable,
        bootstrap_repetitions=bootstrap_repetitions,
        alpha=alpha,
    )
    return _finalize(reject, obs_estimable)


def wcr_c_one_sample_batch(
    *,
    counts: np.ndarray,
    cluster_sizes: np.ndarray,
    boundary: float,
    alpha: float,
    bootstrap_repetitions: int,
    weights: np.ndarray,
) -> WCRBatchCalibration:
    """
    Intercept-only WCR-C lower-tail test of H0: marginal proportion >= boundary
    at the least-favorable boundary. Used for absolute FNR=0.10 and FPR=0.05.
    """
    _validate_bootstrap_grid(alpha, bootstrap_repetitions)
    sizes = _validate_sizes(np.asarray(cluster_sizes)).astype(float)
    x = np.asarray(counts, dtype=float)

    if x.ndim != 2 or x.shape[1] != len(sizes):
        raise ValueError("counts must be 2D with one column per cluster")
    if weights.shape != (bootstrap_repetitions, len(sizes)):
        raise ValueError("weight matrix has wrong shape")

    n = float(sizes.sum())
    g = len(sizes)
    mean = x.sum(axis=1) / n

    unrestricted = x - mean[:, None] * sizes[None, :]
    # Standard CRV1 scalar for p=1:
    # G/(G-1) * (N-1)/(N-1) = G/(G-1).
    scalar = g / (g - 1.0)
    obs_var = scalar * _group_variance_from_cluster_residual_sums(
        unrestricted, sizes, method="CRV1"
    )

    obs_estimable = np.isfinite(obs_var) & (obs_var > 0.0)
    t_obs = np.full(len(x), np.nan, dtype=float)
    t_obs[obs_estimable] = (
        mean[obs_estimable] - boundary
    ) / np.sqrt(obs_var[obs_estimable])

    restricted = x - boundary * sizes[None, :]
    shift, bvar = _bootstrap_group_variance_fast(
        restricted, sizes, weights, method="CRV1"
    )
    b_var = scalar * bvar
    b_estimable = np.isfinite(b_var) & (b_var > 0.0)

    t_boot = np.full_like(shift, np.nan)
    t_boot[b_estimable] = shift[b_estimable] / np.sqrt(b_var[b_estimable])

    _, reject = _lower_tail_p_values_fail_closed(
        t_obs=t_obs,
        t_boot=t_boot,
        bootstrap_estimable=b_estimable,
        observed_estimable=obs_estimable,
        bootstrap_repetitions=bootstrap_repetitions,
        alpha=alpha,
    )
    return _finalize(reject, obs_estimable)


def fixed_webb_weights(
    *,
    bootstrap_repetitions: int,
    cluster_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _webb_weights(
        bootstrap_repetitions=bootstrap_repetitions,
        cluster_count=cluster_count,
        rng=rng,
    )
