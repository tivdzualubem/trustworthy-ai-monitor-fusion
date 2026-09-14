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
class FGD5Design:
    reference_sizes: np.ndarray
    comparison_sizes: np.ndarray
    X_cluster: np.ndarray
    omega: np.ndarray
    vm: np.ndarray
    H: np.ndarray
    w: np.ndarray
    trace_linear_matrix: np.ndarray
    trace_quadratic_matrix: np.ndarray
    total_rows_reference: int
    total_rows_comparison: int


@dataclass(frozen=True)
class FGD5SimulationRate:
    repetitions: int
    pass_count: int
    estimable_count: int
    pass_rate: float
    mc_se: float
    cp_lower_95: float
    cp_upper_95: float
    min_df: float
    median_df: float
    max_df: float


def _build_design(
    reference_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    comparison_cluster_sizes: list[int] | tuple[int, ...] | np.ndarray,
    *,
    bound: float = 0.75,
) -> FGD5Design:
    ref = _validate_sizes(np.asarray(reference_cluster_sizes))
    cmp_ = _validate_sizes(np.asarray(comparison_cluster_sizes))

    if not (0.0 < bound < 1.0):
        raise ValueError("FG bound must lie strictly between 0 and 1")

    sizes = np.concatenate([ref, cmp_]).astype(np.float64)
    group = np.concatenate(
        [
            np.zeros(len(ref), dtype=np.float64),
            np.ones(len(cmp_), dtype=np.float64),
        ]
    )
    Xc = np.column_stack([np.ones(len(sizes)), group])

    # omega_i = X_i' X_i for a cluster whose row-level design is constant.
    omega = np.einsum("i,ij,ik->ijk", sizes, Xc, Xc)
    vminv = omega.sum(axis=0)
    if np.linalg.matrix_rank(vminv) != 2:
        raise ValueError("two-cell design is rank deficient")
    vm = np.linalg.inv(vminv)

    # Fay-Graubard bias correction from saws::sawsCalc(method='d5'):
    # H_ij = [1 - min(bound, diag(omega_i vm)_j)]^(-1/2).
    diag_q = np.diagonal(np.einsum("iab,bc->iac", omega, vm), axis1=1, axis2=2)
    H = (1.0 - np.minimum(bound, diag_q)) ** (-0.5)

    test = np.asarray([0.0, 1.0], dtype=np.float64)

    # w_i from saws::PsiTildeCalc for the single slope contrast.
    w = np.empty(len(sizes), dtype=np.float64)
    for i in range(len(sizes)):
        leave_one_out = vminv - omega[i]
        if np.linalg.matrix_rank(leave_one_out) != 2:
            raise ValueError(
                "FG d5 requires the design to remain full rank after removing "
                "any one provenance cluster"
            )
        delta = np.linalg.inv(leave_one_out) - vm
        w[i] = float(test @ delta @ test)

    # dfCalc uses:
    # G_{i,k} = delta_{ik} I - omega_i vm
    # M_i = diag(H_i) vm test' test vm diag(H_i)
    # and Psi_i = w_i * S, with S = sum (H_i*u_i)(H_i*u_i)'.
    #
    # We precompute the design-only trace operators so every Monte Carlo
    # replicate needs only its 2x2 S matrix.
    A = np.einsum("iab,bc->iac", omega, vm)

    a_vec = np.empty((len(sizes), 2), dtype=np.float64)
    vm_test = vm @ test
    for i in range(len(sizes)):
        a_vec[i] = H[i] * vm_test

    M = np.einsum("ia,ib->iab", a_vec, a_vec)
    C = np.einsum(
        "iab,ibc,icd->ad",
        np.transpose(A, (0, 2, 1)),
        M,
        A,
    )

    K = len(sizes)
    B = np.empty((K, K, 2, 2), dtype=np.float64)
    for k in range(K):
        for l in range(K):
            block = -M[k] @ A[k] - A[l].T @ M[l] + C
            if k == l:
                block = block + M[k]
            B[k, l] = block

    # Numerator trace = trace(S * L), where L=sum_i w_i B_ii.
    L = np.zeros((2, 2), dtype=np.float64)
    for i in range(K):
        L += w[i] * B[i, i]

    # Denominator trace = vec(S)' Q vec(S). Build Q from four 2x2 bases.
    bases = []
    for a in range(2):
        for b in range(2):
            E = np.zeros((2, 2), dtype=np.float64)
            E[a, b] = 1.0
            bases.append(E)

    Q = np.zeros((4, 4), dtype=np.float64)
    ww = np.outer(w, w)

    for r, Er in enumerate(bases):
        for s, Es in enumerate(bases):
            total = 0.0
            for k in range(K):
                for l in range(K):
                    total += (
                        ww[k, l]
                        * np.trace(Er @ B[k, l] @ Es @ B[l, k])
                    )
            Q[r, s] = total

    return FGD5Design(
        reference_sizes=ref.astype(np.int64),
        comparison_sizes=cmp_.astype(np.int64),
        X_cluster=Xc,
        omega=omega,
        vm=vm,
        H=H,
        w=w,
        trace_linear_matrix=L,
        trace_quadratic_matrix=Q,
        total_rows_reference=int(ref.sum()),
        total_rows_comparison=int(cmp_.sum()),
    )


def _fg_d5_from_counts(
    *,
    reference_counts: np.ndarray,
    comparison_counts: np.ndarray,
    design: FGD5Design,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if reference_counts.ndim != 2 or comparison_counts.ndim != 2:
        raise ValueError("count arrays must be two-dimensional")
    if reference_counts.shape[0] != comparison_counts.shape[0]:
        raise ValueError("reference and comparison repetitions must match")
    if reference_counts.shape[1] != len(design.reference_sizes):
        raise ValueError("reference cluster count does not match design")
    if comparison_counts.shape[1] != len(design.comparison_sizes):
        raise ValueError("comparison cluster count does not match design")

    R = reference_counts.shape[0]

    ref_mean = (
        reference_counts.sum(axis=1) / design.total_rows_reference
    )
    cmp_mean = (
        comparison_counts.sum(axis=1) / design.total_rows_comparison
    )
    difference = cmp_mean - ref_mean

    ref_resid_sum = (
        reference_counts
        - ref_mean[:, None] * design.reference_sizes[None, :]
    )
    cmp_resid_sum = (
        comparison_counts
        - cmp_mean[:, None] * design.comparison_sizes[None, :]
    )

    residual_sums = np.concatenate([ref_resid_sum, cmp_resid_sum], axis=1)

    # u_i = x_i * residual_sum_i; H*u is the FG-adjusted cluster score.
    u = residual_sums[:, :, None] * design.X_cluster[None, :, :]
    hu = u * design.H[None, :, :]

    # S = sum_i (H_i u_i)(H_i u_i)'.
    S = np.einsum("ria,rib->rab", hu, hu)

    # V_FG = vm S vm; extract slope variance.
    V = np.einsum("ab,rbc,cd->rad", design.vm, S, design.vm)
    variance = V[:, 1, 1]

    # Exact Fay-Graubard d5 df calculation, algebraically reduced from
    # saws::PsiTildeCalc + saws::dfCalc for p=2 and one slope contrast.
    numerator_trace = np.einsum(
        "rab,ba->r",
        S,
        design.trace_linear_matrix,
    )

    Svec = S.reshape(R, 4)
    denominator_trace = np.einsum(
        "ri,ij,rj->r",
        Svec,
        design.trace_quadratic_matrix,
        Svec,
    )

    df = np.full(R, np.nan, dtype=np.float64)
    valid_df = (
        np.isfinite(numerator_trace)
        & np.isfinite(denominator_trace)
        & (denominator_trace > 0.0)
    )
    df[valid_df] = (
        numerator_trace[valid_df] ** 2
        / denominator_trace[valid_df]
    )
    # saws caps df below 1 at 1.
    df[valid_df] = np.maximum(df[valid_df], 1.0)

    return difference, variance, df, S


def simulate_fg_d5_boundary_calibration(
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
    bound: float = 0.75,
) -> FGD5SimulationRate:
    if not (0.0 < reference_fnr < 1.0 - margin):
        raise ValueError("reference_fnr must leave room for the NI boundary")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie strictly between 0 and 0.5")

    comparison_fnr = reference_fnr + margin
    design = _build_design(
        reference_cluster_sizes,
        comparison_cluster_sizes,
        bound=bound,
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

    difference, variance, df, _ = _fg_d5_from_counts(
        reference_counts=ref_counts,
        comparison_counts=cmp_counts,
        design=design,
    )

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

    finite_df = df[estimable]
    return FGD5SimulationRate(
        repetitions=repetitions,
        pass_count=pass_count,
        estimable_count=estimable_count,
        pass_rate=float(rate),
        mc_se=float(mc_se),
        cp_lower_95=lower,
        cp_upper_95=upper_ci,
        min_df=float(np.min(finite_df)) if len(finite_df) else float("nan"),
        median_df=float(np.median(finite_df)) if len(finite_df) else float("nan"),
        max_df=float(np.max(finite_df)) if len(finite_df) else float("nan"),
    )
