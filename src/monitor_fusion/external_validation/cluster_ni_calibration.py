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


SUPPORTED_CORRECTIONS = ("KC", "MD")


@dataclass(frozen=True)
class CalibrationRate:
    correction: str
    repetitions: int
    pass_count: int
    estimable_count: int
    pass_rate: float
    mc_se: float
    cp_lower_95: float
    cp_upper_95: float


def _group_mean_and_variance(
    counts: np.ndarray,
    cluster_sizes: np.ndarray,
    *,
    correction: str,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = _validate_sizes(cluster_sizes).astype(np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(sizes):
        raise ValueError("counts must have one column per cluster")
    if correction not in SUPPORTED_CORRECTIONS:
        raise ValueError(f"unknown correction: {correction}")

    total_n = float(sizes.sum())
    means = counts.sum(axis=1) / total_n
    leverage = sizes / total_n

    if np.any(leverage >= 1.0):
        raise ValueError("cluster leverage must be below one")

    residual_sums = counts - means[:, None] * sizes[None, :]

    if correction == "KC":
        adjusted = residual_sums / np.sqrt(1.0 - leverage[None, :])
    else:
        # Mancl-DeRouen: (I-H_g)^(-1) residual adjustment.
        # In the two-cell identity-link marginal-mean model each provenance
        # cluster has a constant design row, giving this scalar closed form.
        adjusted = residual_sums / (1.0 - leverage[None, :])

    variances = np.sum(adjusted * adjusted, axis=1) / (total_n * total_n)
    return means, variances


def simulate_boundary_calibration(
    *,
    reference_fnr: float,
    icc_reference: float,
    icc_comparison: float,
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    repetitions: int,
    seed: int,
    correction: str,
    margin: float = PRIMARY_NI_MARGIN,
    alpha: float = ONE_SIDED_ALPHA,
) -> CalibrationRate:
    if not (0.0 < reference_fnr < 1.0 - margin):
        raise ValueError("reference_fnr must leave room for the NI boundary")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if correction not in SUPPORTED_CORRECTIONS:
        raise ValueError(f"unknown correction: {correction}")

    comparison_fnr = reference_fnr + margin
    ref_sizes = _validate_sizes(np.asarray(reference_cluster_sizes))
    cmp_sizes = _validate_sizes(np.asarray(comparison_cluster_sizes))

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

    ref_mean, ref_var = _group_mean_and_variance(
        ref_counts, ref_sizes, correction=correction
    )
    cmp_mean, cmp_var = _group_mean_and_variance(
        cmp_counts, cmp_sizes, correction=correction
    )

    difference = cmp_mean - ref_mean
    variance = ref_var + cmp_var

    total_clusters = len(ref_sizes) + len(cmp_sizes)
    df = total_clusters - 2
    if df <= 0:
        raise ValueError("cluster degrees of freedom must be positive")

    critical = float(t.ppf(1.0 - alpha, df=df))
    estimable = np.isfinite(variance) & (variance > 0.0)

    upper = np.full(repetitions, np.inf, dtype=np.float64)
    upper[estimable] = difference[estimable] + critical * np.sqrt(
        variance[estimable]
    )
    passes = estimable & (upper < margin)

    pass_count = int(passes.sum())
    estimable_count = int(estimable.sum())
    rate = pass_count / repetitions
    mc_se = math.sqrt(rate * (1.0 - rate) / repetitions)

    if pass_count == 0:
        lower = 0.0
    else:
        lower = float(
            beta_dist.ppf(
                0.025,
                pass_count,
                repetitions - pass_count + 1,
            )
        )

    if pass_count == repetitions:
        upper_ci = 1.0
    else:
        upper_ci = float(
            beta_dist.ppf(
                0.975,
                pass_count + 1,
                repetitions - pass_count,
            )
        )

    return CalibrationRate(
        correction=correction,
        repetitions=repetitions,
        pass_count=pass_count,
        estimable_count=estimable_count,
        pass_rate=float(rate),
        mc_se=float(mc_se),
        cp_lower_95=lower,
        cp_upper_95=upper_ci,
    )
