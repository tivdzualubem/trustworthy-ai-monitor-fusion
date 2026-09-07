from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import t

from monitor_fusion.external_validation.cluster_ni_design import (
    ONE_SIDED_ALPHA,
    PRIMARY_NI_MARGIN,
)


@dataclass(frozen=True)
class MonteCarloRate:
    repetitions: int
    pass_count: int
    estimable_count: int
    pass_rate_all: float
    pass_rate_estimable: float
    mc_se_all: float
    cp_lower_95: float
    cp_upper_95: float


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _validate_sizes(sizes: np.ndarray) -> np.ndarray:
    arr = np.asarray(sizes, dtype=np.int64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("cluster sizes must be a 1D array with at least two clusters")
    if np.any(arr <= 0):
        raise ValueError("cluster sizes must be positive")
    return arr


def _beta_binomial_counts(
    *,
    marginal_probability: float,
    icc: float,
    cluster_sizes: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (0.0 < marginal_probability < 1.0):
        raise ValueError("marginal_probability must lie strictly between 0 and 1")
    if not (0.0 <= icc < 1.0):
        raise ValueError("icc must lie in [0,1)")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    sizes = _validate_sizes(cluster_sizes)
    G = len(sizes)

    if icc == 0.0:
        probabilities = np.full(
            (repetitions, G),
            marginal_probability,
            dtype=np.float64,
        )
    else:
        concentration = (1.0 / icc) - 1.0
        a = marginal_probability * concentration
        b = (1.0 - marginal_probability) * concentration
        probabilities = rng.beta(a, b, size=(repetitions, G))

    return rng.binomial(
        sizes[None, :],
        probabilities,
    ).astype(np.float64)


def _kc_group_variance(
    counts: np.ndarray,
    cluster_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = _validate_sizes(cluster_sizes).astype(np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(sizes):
        raise ValueError("counts must have one column per cluster")

    total_n = float(sizes.sum())
    means = counts.sum(axis=1) / total_n

    leverage = sizes / total_n
    if np.any(leverage >= 1.0):
        raise ValueError("cluster leverage must be below one")

    residual_sums = counts - means[:, None] * sizes[None, :]
    adjusted = residual_sums / np.sqrt(1.0 - leverage[None, :])
    variances = np.sum(adjusted * adjusted, axis=1) / (total_n * total_n)

    return means, variances


def simulate_pair_pass_rate_vectorized(
    *,
    reference_fnr: float,
    comparison_fnr: float,
    icc_reference: float,
    icc_comparison: float,
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    repetitions: int,
    seed: int,
    margin: float = PRIMARY_NI_MARGIN,
    alpha: float = ONE_SIDED_ALPHA,
) -> MonteCarloRate:
    ref_sizes = _validate_sizes(np.asarray(reference_cluster_sizes))
    cmp_sizes = _validate_sizes(np.asarray(comparison_cluster_sizes))

    if not (0.0 < margin < 1.0):
        raise ValueError("margin must lie strictly between 0 and 1")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

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

    ref_mean, ref_var = _kc_group_variance(ref_counts, ref_sizes)
    cmp_mean, cmp_var = _kc_group_variance(cmp_counts, cmp_sizes)

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
    rate_all = pass_count / repetitions
    rate_estimable = (
        pass_count / estimable_count if estimable_count else float("nan")
    )
    mc_se = math.sqrt(rate_all * (1.0 - rate_all) / repetitions)

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

    return MonteCarloRate(
        repetitions=repetitions,
        pass_count=pass_count,
        estimable_count=estimable_count,
        pass_rate_all=float(rate_all),
        pass_rate_estimable=float(rate_estimable),
        mc_se_all=float(mc_se),
        cp_lower_95=lower,
        cp_upper_95=upper_ci,
    )
