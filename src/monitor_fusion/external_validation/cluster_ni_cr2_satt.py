from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import t

from monitor_fusion.external_validation.cluster_ni_calibration import (
    _group_mean_and_variance,
)
from monitor_fusion.external_validation.cluster_ni_design import (
    ONE_SIDED_ALPHA,
    PRIMARY_NI_MARGIN,
)
from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    _validate_sizes,
)


@dataclass(frozen=True)
class CR2SatterthwaiteDesign:
    reference_sizes: np.ndarray
    comparison_sizes: np.ndarray
    degrees_of_freedom: float
    P_matrix: np.ndarray


@dataclass(frozen=True)
class CR2SatterthwaiteCalibration:
    repetitions: int
    reject_count: int
    estimable_count: int
    type_I_rate: float
    mc_se: float
    cp_lower_95: float
    cp_upper_95: float
    degrees_of_freedom: float


def _cp_interval(k: int, n: int) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    upper = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lower, upper


def build_cr2_satterthwaite_design(
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
) -> CR2SatterthwaiteDesign:
    """
    Exact specialization of clubSandwich CR2 + Satterthwaite for an
    unweighted OLS two-cell identity-link model with identity working target.

    Model:
        E[FN] = beta0 + beta1 * I(comparison)

    The CR2 variance for beta1 is the same bias-reduced linearization
    adjustment used in the project's existing KC candidate. The key difference
    here is the coefficient-specific Satterthwaite degrees of freedom from the
    clubSandwich P-array construction, rather than total_clusters - 2.

    clubSandwich uses:
        df = trace(P)^2 / sum(P^2)
    for a one-coefficient Satterthwaite test.
    """
    ref = _validate_sizes(np.asarray(reference_cluster_sizes)).astype(np.int64)
    cmp_ = _validate_sizes(np.asarray(comparison_cluster_sizes)).astype(np.int64)

    sizes = np.concatenate([ref, cmp_]).astype(np.float64)
    group = np.concatenate(
        [
            np.zeros(len(ref), dtype=np.float64),
            np.ones(len(cmp_), dtype=np.float64),
        ]
    )
    Xc = np.column_stack([np.ones(len(sizes)), group])

    xtx = np.einsum("i,ij,ik->jk", sizes, Xc, Xc)
    if np.linalg.matrix_rank(xtx) != 2:
        raise ValueError("two-cell design is rank deficient")

    M = np.linalg.inv(xtx)
    # R's t(chol(M)) is the lower Cholesky factor.
    L = np.linalg.cholesky(M)

    J = len(sizes)
    g_norm2 = np.empty(J, dtype=np.float64)
    H_rows = np.empty((J, 2), dtype=np.float64)

    for i, (m, x) in enumerate(zip(sizes.tolist(), Xc)):
        leverage = float(m * (x @ M @ x))
        if not math.isfinite(leverage) or leverage < 0.0 or leverage >= 1.0:
            raise ValueError(f"invalid CR2 cluster leverage: {leverage}")

        adjustment = 1.0 / math.sqrt(1.0 - leverage)

        # E_i = X_i' A_i. Because every row of X_i is x and the CR2
        # adjustment acts on the cluster-one direction by adjustment,
        # M E_i has identical columns z_i = M x * adjustment.
        z = (M @ x) * adjustment

        # Diagonal contribution in clubSandwich get_P_array:
        # || G_i[slope, :] ||^2 = m * z_slope^2.
        g_norm2[i] = m * (z[1] ** 2)

        # H_i = (M E_i X_i) chol_lower(M).
        # Extract the slope row.
        H_rows[i] = m * z[1] * (x @ L)

    P = np.diag(g_norm2) - H_rows @ H_rows.T

    numerator = float(np.trace(P) ** 2)
    denominator = float(np.sum(P * P))

    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise ValueError("invalid CR2 Satterthwaite P matrix")

    df = numerator / denominator
    if not math.isfinite(df) or df < 1.0:
        raise ValueError(f"invalid CR2 Satterthwaite df: {df}")

    return CR2SatterthwaiteDesign(
        reference_sizes=ref,
        comparison_sizes=cmp_,
        degrees_of_freedom=float(df),
        P_matrix=P,
    )


def simulate_cr2_satterthwaite_boundary(
    *,
    reference_fnr: float,
    icc_reference: float,
    icc_comparison: float,
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    repetitions: int,
    seed: int,
    margin: float = PRIMARY_NI_MARGIN,
    alpha: float = ONE_SIDED_ALPHA,
) -> CR2SatterthwaiteCalibration:
    if not (0.0 < reference_fnr < 1.0 - margin):
        raise ValueError("reference_fnr must leave room for the NI boundary")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    comparison_fnr = reference_fnr + margin
    design = build_cr2_satterthwaite_design(
        reference_cluster_sizes,
        comparison_cluster_sizes,
    )

    rng = np.random.default_rng(seed)

    ref_counts = _beta_binomial_counts(
        marginal_probability=reference_fnr,
        icc=icc_reference,
        cluster_sizes=design.reference_sizes,
        repetitions=repetitions,
        rng=rng,
    )
    cmp_counts = _beta_binomial_counts(
        marginal_probability=comparison_fnr,
        icc=icc_comparison,
        cluster_sizes=design.comparison_sizes,
        repetitions=repetitions,
        rng=rng,
    )

    ref_mean, ref_var = _group_mean_and_variance(
        ref_counts,
        design.reference_sizes,
        correction="KC",
    )
    cmp_mean, cmp_var = _group_mean_and_variance(
        cmp_counts,
        design.comparison_sizes,
        correction="KC",
    )

    difference = cmp_mean - ref_mean
    variance = ref_var + cmp_var
    estimable = np.isfinite(variance) & (variance > 0.0)

    critical = float(t.ppf(1.0 - alpha, df=design.degrees_of_freedom))
    upper = np.full(repetitions, np.inf, dtype=np.float64)
    upper[estimable] = (
        difference[estimable]
        + critical * np.sqrt(variance[estimable])
    )

    reject = estimable & (upper < margin)
    reject_count = int(reject.sum())
    estimable_count = int(estimable.sum())
    rate = reject_count / repetitions
    mc_se = math.sqrt(rate * (1.0 - rate) / repetitions)
    lower, upper_ci = _cp_interval(reject_count, repetitions)

    return CR2SatterthwaiteCalibration(
        repetitions=repetitions,
        reject_count=reject_count,
        estimable_count=estimable_count,
        type_I_rate=float(rate),
        mc_se=float(mc_se),
        cp_lower_95=lower,
        cp_upper_95=upper_ci,
        degrees_of_freedom=design.degrees_of_freedom,
    )
