from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

from monitor_fusion.external_validation.cluster_ni_design import (
    ONE_SIDED_ALPHA,
    PRIMARY_NI_MARGIN,
)
from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    _validate_sizes,
)


WEBB_SUPPORT = np.asarray(
    [
        -math.sqrt(1.5),
        -1.0,
        -math.sqrt(0.5),
        math.sqrt(0.5),
        1.0,
        math.sqrt(1.5),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class WCRCalibrationRate:
    method: str
    repetitions: int
    bootstrap_repetitions: int
    reject_count: int
    estimable_count: int
    type_I_rate: float
    mc_se: float
    cp_lower_95: float
    cp_upper_95: float


def _cp_interval(k: int, n: int) -> tuple[float, float]:
    if k == 0:
        lower = 0.0
    else:
        lower = float(beta_dist.ppf(0.025, k, n - k + 1))

    if k == n:
        upper = 1.0
    else:
        upper = float(beta_dist.ppf(0.975, k + 1, n - k))

    return lower, upper


def _group_variance_from_cluster_residual_sums(
    residual_sums: np.ndarray,
    sizes: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """
    residual_sums: (R,G), unrestricted cluster residual sums within one cell.
    Returns the cell contribution to Var(row-marginal mean), before CRV1's
    whole-model finite-sample scalar.
    """
    sizes = _validate_sizes(np.asarray(sizes)).astype(np.float64)
    if residual_sums.ndim != 2 or residual_sums.shape[1] != len(sizes):
        raise ValueError("residual_sums must have one column per cluster")

    n = float(sizes.sum())

    if method == "CRV1":
        return np.sum(residual_sums**2, axis=1) / (n * n)

    if method == "CRV3":
        leverage = sizes / n
        if np.any(leverage >= 1.0):
            raise ValueError("cluster leverage must be below one")
        adjusted = residual_sums / (1.0 - leverage[None, :])
        return np.sum(adjusted**2, axis=1) / (n * n)

    raise ValueError(f"unknown variance method: {method}")


def _bootstrap_group_variance_fast(
    restricted_residual_sums: np.ndarray,
    sizes: np.ndarray,
    weights: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized group contribution for all outer datasets and bootstrap draws.

    restricted_residual_sums: (R,G)
    sizes: (G,)
    weights: (B,G)

    Returns:
      mean_shift: (R,B), the bootstrap shift in the fitted group mean.
      variance:   (R,B), the group contribution to bootstrap variance.
    """
    sizes = _validate_sizes(np.asarray(sizes)).astype(np.float64)
    r = np.asarray(restricted_residual_sums, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if r.ndim != 2 or r.shape[1] != len(sizes):
        raise ValueError("restricted residuals must have one column per cluster")
    if w.ndim != 2 or w.shape[1] != len(sizes):
        raise ValueError("weights must have one column per cluster")

    n = float(sizes.sum())
    weighted_sum = r @ w.T
    mean_shift = weighted_sum / n

    if method == "CRV1":
        a2 = np.ones_like(sizes)
    elif method == "CRV3":
        leverage = sizes / n
        if np.any(leverage >= 1.0):
            raise ValueError("cluster leverage must be below one")
        a2 = 1.0 / ((1.0 - leverage) ** 2)
    else:
        raise ValueError(f"unknown variance method: {method}")

    # For bootstrap unrestricted residual cluster sum:
    # e*_g = w_g r_g - n_g * mean_shift.
    #
    # Sum a_g^2 (e*_g)^2 is expanded so we never materialize R x B x G.
    first = (r**2 * a2[None, :]) @ (w**2).T
    second_core = (r * (a2 * sizes)[None, :]) @ w.T
    third_constant = float(np.sum(a2 * sizes**2))

    ss = (
        first
        - 2.0 * mean_shift * second_core
        + (mean_shift**2) * third_constant
    )
    ss = np.maximum(ss, 0.0)
    return mean_shift, ss / (n * n)


def _webb_weights(
    *,
    bootstrap_repetitions: int,
    cluster_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    if cluster_count < 2:
        raise ValueError("cluster_count must be at least two")

    draw = rng.integers(
        0,
        len(WEBB_SUPPORT),
        size=(bootstrap_repetitions, cluster_count),
    )
    return WEBB_SUPPORT[draw]



def _lower_tail_p_values_fail_closed(
    *,
    t_obs: np.ndarray,
    t_boot: np.ndarray,
    bootstrap_estimable: np.ndarray,
    observed_estimable: np.ndarray,
    bootstrap_repetitions: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Conservative lower-tail bootstrap p-values.
    #
    # For H1 in the lower tail, omitting an invalid bootstrap draw from the
    # numerator while keeping it in the fixed B denominator makes p-values
    # artificially smaller. Fail closed by counting every invalid bootstrap
    # draw in the p-value numerator.
    if t_boot.shape != bootstrap_estimable.shape:
        raise ValueError("t_boot and bootstrap_estimable must have equal shape")
    if t_boot.shape[0] != len(t_obs) or len(t_obs) != len(observed_estimable):
        raise ValueError("outer repetition dimensions must agree")
    if t_boot.shape[1] != bootstrap_repetitions:
        raise ValueError("bootstrap repetition dimension mismatch")

    tail_or_invalid = (~bootstrap_estimable) | (
        bootstrap_estimable & (t_boot <= t_obs[:, None])
    )
    tail_count = np.sum(tail_or_invalid, axis=1)
    p_value = (1.0 + tail_count) / (bootstrap_repetitions + 1.0)
    reject = observed_estimable & (p_value <= alpha)
    return p_value, reject


def simulate_wcr_boundary_calibration(
    *,
    reference_fnr: float,
    icc_reference: float,
    icc_comparison: float,
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    repetitions: int,
    bootstrap_repetitions: int,
    seed: int,
    method: str,
    margin: float = PRIMARY_NI_MARGIN,
    alpha: float = ONE_SIDED_ALPHA,
) -> WCRCalibrationRate:
    """
    Wild-cluster restricted bootstrap test of H0: RD >= margin at the least
    favorable boundary RD=margin, specialized to the two-cell identity-link
    marginal risk-difference model.

    method:
      WCR-C -> classic restricted bootstrap, CRV1 studentization.
      WCR-V -> restricted bootstrap, CRV3 studentization.

    One-sided NI p-value uses the lower tail:
      p = (1 + #{t*_b <= t_obs}) / (B+1)
    and rejects H0 when p <= alpha.
    """
    if method not in ("WCR-C", "WCR-V"):
        raise ValueError("method must be WCR-C or WCR-V")
    if not (0.0 < reference_fnr < 1.0 - margin):
        raise ValueError("reference_fnr must leave room for the NI boundary")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    # Ensure the bootstrap p-value grid contains alpha exactly.
    target = alpha * (bootstrap_repetitions + 1)
    if not math.isclose(target, round(target), abs_tol=1e-12):
        raise ValueError(
            "choose B so alpha*(B+1) is an integer "
            "(e.g. B=999 for alpha=0.025)"
        )

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

    n_ref = float(ref_sizes.sum())
    n_cmp = float(cmp_sizes.sum())
    n_total = n_ref + n_cmp
    G_total = len(ref_sizes) + len(cmp_sizes)

    ref_mean = ref_counts.sum(axis=1) / n_ref
    cmp_mean = cmp_counts.sum(axis=1) / n_cmp
    difference = cmp_mean - ref_mean

    # Unrestricted residual cluster sums for observed studentization.
    ref_u = ref_counts - ref_mean[:, None] * ref_sizes[None, :]
    cmp_u = cmp_counts - cmp_mean[:, None] * cmp_sizes[None, :]

    variance_method = "CRV1" if method == "WCR-C" else "CRV3"
    obs_var = (
        _group_variance_from_cluster_residual_sums(
            ref_u, ref_sizes, method=variance_method
        )
        + _group_variance_from_cluster_residual_sums(
            cmp_u, cmp_sizes, method=variance_method
        )
    )

    if method == "WCR-C":
        # Standard CV1 finite-sample scalar for p=2.
        scalar = (G_total / (G_total - 1.0)) * (
            (n_total - 1.0) / (n_total - 2.0)
        )
        obs_var = scalar * obs_var
    else:
        scalar = 1.0

    obs_estimable = np.isfinite(obs_var) & (obs_var > 0.0)
    t_obs = np.full(repetitions, np.nan, dtype=np.float64)
    t_obs[obs_estimable] = (
        difference[obs_estimable] - margin
    ) / np.sqrt(obs_var[obs_estimable])

    # Restricted OLS fit under beta1=margin.
    beta0_tilde = (
        ref_counts.sum(axis=1)
        + cmp_counts.sum(axis=1)
        - margin * n_cmp
    ) / n_total

    ref_r = ref_counts - beta0_tilde[:, None] * ref_sizes[None, :]
    cmp_r = cmp_counts - (
        beta0_tilde[:, None] + margin
    ) * cmp_sizes[None, :]

    # Use one deterministic Webb-weight set across outer Monte Carlo datasets.
    # Outer datasets remain independent; fixing the bootstrap randomization
    # removes an unnecessary second source of Monte Carlo noise.
    w_ref = _webb_weights(
        bootstrap_repetitions=bootstrap_repetitions,
        cluster_count=len(ref_sizes),
        rng=rng,
    )
    w_cmp = _webb_weights(
        bootstrap_repetitions=bootstrap_repetitions,
        cluster_count=len(cmp_sizes),
        rng=rng,
    )

    ref_shift, ref_bvar = _bootstrap_group_variance_fast(
        ref_r,
        ref_sizes,
        w_ref,
        method=variance_method,
    )
    cmp_shift, cmp_bvar = _bootstrap_group_variance_fast(
        cmp_r,
        cmp_sizes,
        w_cmp,
        method=variance_method,
    )

    # Under the restricted DGP, beta1* - margin is the difference in the two
    # bootstrap group-mean shifts.
    b_num = cmp_shift - ref_shift
    b_var = scalar * (ref_bvar + cmp_bvar)
    b_estimable = np.isfinite(b_var) & (b_var > 0.0)

    t_boot = np.full_like(b_num, np.nan, dtype=np.float64)
    t_boot[b_estimable] = b_num[b_estimable] / np.sqrt(b_var[b_estimable])

    # Fail closed for observed and bootstrap non-estimability.
    # Invalid bootstrap draws count toward the lower-tail p-value numerator;
    # omitting them while retaining fixed B would make the test more liberal.
    p_value, reject = _lower_tail_p_values_fail_closed(
        t_obs=t_obs,
        t_boot=t_boot,
        bootstrap_estimable=b_estimable,
        observed_estimable=obs_estimable,
        bootstrap_repetitions=bootstrap_repetitions,
        alpha=alpha,
    )

    reject_count = int(reject.sum())
    estimable_count = int(obs_estimable.sum())
    rate = reject_count / repetitions
    mc_se = math.sqrt(rate * (1.0 - rate) / repetitions)
    lower, upper = _cp_interval(reject_count, repetitions)

    return WCRCalibrationRate(
        method=method,
        repetitions=repetitions,
        bootstrap_repetitions=bootstrap_repetitions,
        reject_count=reject_count,
        estimable_count=estimable_count,
        type_I_rate=float(rate),
        mc_se=float(mc_se),
        cp_lower_95=lower,
        cp_upper_95=upper,
    )
