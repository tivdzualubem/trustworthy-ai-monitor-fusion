from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammaln
from scipy.stats import norm


class BetaBinomialProfileNotEstimable(RuntimeError):
    pass


@dataclass(frozen=True)
class BBFit:
    probability: float
    icc: float
    loglik: float


@dataclass(frozen=True)
class BBOneSampleTest:
    probability_hat: float
    icc_hat: float
    signed_root_lr: float
    p_value_one_sided: float
    reject_below_boundary: bool


@dataclass(frozen=True)
class BBTwoSampleNITest:
    reference_probability_hat: float
    comparison_probability_hat: float
    risk_difference_hat: float
    reference_icc_hat: float
    comparison_icc_hat: float
    signed_root_lr: float
    p_value_one_sided: float
    reject_noninferiority_null: bool


EPS = 1e-7
RHO_MAX = 0.95


def _validate_counts(
    events: list[int] | np.ndarray,
    sizes: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(events, dtype=np.float64)
    n = np.asarray(sizes, dtype=np.float64)
    if k.ndim != 1 or n.ndim != 1 or len(k) != len(n) or len(k) < 2:
        raise ValueError("events and sizes must be equal-length vectors with >=2 clusters")
    if np.any(n <= 0) or np.any(np.floor(n) != n):
        raise ValueError("sizes must be positive integers")
    if np.any(k < 0) or np.any(np.floor(k) != k) or np.any(k > n):
        raise ValueError("events must be integer counts in [0,size]")
    return k, n


def _log_choose(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def beta_binomial_loglik(
    *,
    events: list[int] | np.ndarray,
    sizes: list[int] | np.ndarray,
    probability: float,
    icc: float,
) -> float:
    k, n = _validate_counts(events, sizes)
    p = float(probability)
    rho = float(icc)

    if not (EPS <= p <= 1.0 - EPS):
        return -math.inf
    if not (EPS <= rho <= RHO_MAX):
        return -math.inf

    concentration = (1.0 / rho) - 1.0
    a = p * concentration
    b = (1.0 - p) * concentration

    terms = (
        _log_choose(n, k)
        + betaln(k + a, n - k + b)
        - betaln(a, b)
    )
    value = float(np.sum(terms))
    return value if math.isfinite(value) else -math.inf


def fit_beta_binomial(
    *,
    events: list[int] | np.ndarray,
    sizes: list[int] | np.ndarray,
) -> BBFit:
    k, n = _validate_counts(events, sizes)
    p0 = float(np.clip(k.sum() / n.sum(), 0.001, 0.999))

    def objective(x):
        value = beta_binomial_loglik(
            events=k,
            sizes=n,
            probability=float(x[0]),
            icc=float(x[1]),
        )
        return 1e100 if not math.isfinite(value) else -value

    starts = [
        (p0, 0.01),
        (p0, 0.05),
        (p0, 0.10),
        (p0, 0.25),
    ]
    best = None
    for start in starts:
        res = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(EPS, 1.0 - EPS), (EPS, RHO_MAX)],
        )
        if res.success and math.isfinite(float(res.fun)):
            if best is None or res.fun < best.fun:
                best = res

    if best is None:
        raise BetaBinomialProfileNotEstimable("unconstrained beta-binomial fit failed")

    return BBFit(
        probability=float(best.x[0]),
        icc=float(best.x[1]),
        loglik=float(-best.fun),
    )


def _fit_icc_given_probability(
    *,
    events: np.ndarray,
    sizes: np.ndarray,
    probability: float,
) -> BBFit:
    p = float(probability)
    if not (EPS <= p <= 1.0 - EPS):
        raise BetaBinomialProfileNotEstimable("fixed probability outside admissible range")

    def objective(rho):
        value = beta_binomial_loglik(
            events=events,
            sizes=sizes,
            probability=p,
            icc=float(rho),
        )
        return 1e100 if not math.isfinite(value) else -value

    res = minimize_scalar(
        objective,
        bounds=(EPS, RHO_MAX),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not res.success or not math.isfinite(float(res.fun)):
        raise BetaBinomialProfileNotEstimable("conditional ICC fit failed")

    return BBFit(probability=p, icc=float(res.x), loglik=float(-res.fun))


def one_sample_boundary_test(
    *,
    events: list[int] | np.ndarray,
    sizes: list[int] | np.ndarray,
    boundary: float,
    alpha: float,
) -> BBOneSampleTest:
    if not (0.0 < boundary < 1.0):
        raise ValueError("boundary must lie in (0,1)")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie in (0,0.5)")

    k, n = _validate_counts(events, sizes)
    unrestricted = fit_beta_binomial(events=k, sizes=n)
    null = _fit_icc_given_probability(events=k, sizes=n, probability=boundary)

    lr = max(0.0, 2.0 * (unrestricted.loglik - null.loglik))
    signed_root = math.copysign(math.sqrt(lr), unrestricted.probability - boundary)

    # H1: p < boundary.
    p_value = float(norm.cdf(signed_root))
    reject = bool(unrestricted.probability < boundary and p_value < alpha)

    return BBOneSampleTest(
        probability_hat=unrestricted.probability,
        icc_hat=unrestricted.icc,
        signed_root_lr=float(signed_root),
        p_value_one_sided=p_value,
        reject_below_boundary=reject,
    )


def two_sample_ni_boundary_test(
    *,
    reference_events: list[int] | np.ndarray,
    reference_sizes: list[int] | np.ndarray,
    comparison_events: list[int] | np.ndarray,
    comparison_sizes: list[int] | np.ndarray,
    margin: float,
    alpha: float,
) -> BBTwoSampleNITest:
    if not (0.0 < margin < 1.0):
        raise ValueError("margin must lie in (0,1)")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie in (0,0.5)")

    kr, nr = _validate_counts(reference_events, reference_sizes)
    kc, nc = _validate_counts(comparison_events, comparison_sizes)

    ref_fit = fit_beta_binomial(events=kr, sizes=nr)
    cmp_fit = fit_beta_binomial(events=kc, sizes=nc)
    unrestricted_ll = ref_fit.loglik + cmp_fit.loglik
    rd_hat = cmp_fit.probability - ref_fit.probability

    upper_ref = 1.0 - margin - EPS
    if upper_ref <= EPS:
        raise ValueError("margin leaves no admissible constrained probabilities")

    cache: dict[float, tuple[float, float, float]] = {}

    def objective(p_ref):
        p_ref = float(p_ref)
        p_cmp = p_ref + margin
        try:
            rf = _fit_icc_given_probability(
                events=kr, sizes=nr, probability=p_ref
            )
            cf = _fit_icc_given_probability(
                events=kc, sizes=nc, probability=p_cmp
            )
        except BetaBinomialProfileNotEstimable:
            return 1e100

        cache[p_ref] = (rf.icc, cf.icc, rf.loglik + cf.loglik)
        return -(rf.loglik + cf.loglik)

    res = minimize_scalar(
        objective,
        bounds=(EPS, upper_ref),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not res.success or not math.isfinite(float(res.fun)):
        raise BetaBinomialProfileNotEstimable("constrained NI profile fit failed")

    constrained_ll = float(-res.fun)
    lr = max(0.0, 2.0 * (unrestricted_ll - constrained_ll))
    signed_root = math.copysign(math.sqrt(lr), rd_hat - margin)

    # H1: RD < margin.
    p_value = float(norm.cdf(signed_root))
    reject = bool(rd_hat < margin and p_value < alpha)

    return BBTwoSampleNITest(
        reference_probability_hat=ref_fit.probability,
        comparison_probability_hat=cmp_fit.probability,
        risk_difference_hat=float(rd_hat),
        reference_icc_hat=ref_fit.icc,
        comparison_icc_hat=cmp_fit.icc,
        signed_root_lr=float(signed_root),
        p_value_one_sided=p_value,
        reject_noninferiority_null=reject,
    )
