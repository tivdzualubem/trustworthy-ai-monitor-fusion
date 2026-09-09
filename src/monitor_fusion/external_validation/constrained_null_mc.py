from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from monitor_fusion.external_validation.beta_binomial_profile import (
    BetaBinomialProfileNotEstimable,
    EPS,
    _fit_icc_given_probability,
    _validate_counts,
)


@dataclass(frozen=True)
class TwoSampleNullFit:
    reference_probability: float
    comparison_probability: float
    reference_icc: float
    comparison_icc: float
    loglik: float
    margin: float


@dataclass(frozen=True)
class OneSampleMonteCarloResult:
    observed_rate: float
    null_icc_hat: float
    p_value: float
    reject_below_boundary: bool
    bootstrap_repetitions: int


@dataclass(frozen=True)
class TwoSampleMonteCarloResult:
    observed_reference_rate: float
    observed_comparison_rate: float
    observed_risk_difference: float
    null_reference_probability: float
    null_comparison_probability: float
    null_reference_icc_hat: float
    null_comparison_icc_hat: float
    p_value: float
    reject_noninferiority_null: bool
    bootstrap_repetitions: int


def fit_two_sample_ni_null(
    *,
    reference_events,
    reference_sizes,
    comparison_events,
    comparison_sizes,
    margin: float,
) -> TwoSampleNullFit:
    kr, nr = _validate_counts(reference_events, reference_sizes)
    kc, nc = _validate_counts(comparison_events, comparison_sizes)

    if not (0.0 < margin < 1.0):
        raise ValueError("margin must lie in (0,1)")

    upper_ref = 1.0 - margin - EPS
    if upper_ref <= EPS:
        raise ValueError("margin leaves no admissible null probabilities")

    def objective(p_ref: float) -> float:
        p_ref = float(p_ref)
        p_cmp = p_ref + margin
        try:
            rf = _fit_icc_given_probability(
                events=kr,
                sizes=nr,
                probability=p_ref,
            )
            cf = _fit_icc_given_probability(
                events=kc,
                sizes=nc,
                probability=p_cmp,
            )
        except BetaBinomialProfileNotEstimable:
            return 1e100

        return -(rf.loglik + cf.loglik)

    res = minimize_scalar(
        objective,
        bounds=(EPS, upper_ref),
        method="bounded",
        options={"xatol": 1e-7},
    )

    if not res.success or not math.isfinite(float(res.fun)):
        raise BetaBinomialProfileNotEstimable(
            "restricted two-sample NI null fit failed"
        )

    p_ref = float(res.x)
    p_cmp = p_ref + margin
    rf = _fit_icc_given_probability(events=kr, sizes=nr, probability=p_ref)
    cf = _fit_icc_given_probability(events=kc, sizes=nc, probability=p_cmp)

    return TwoSampleNullFit(
        reference_probability=p_ref,
        comparison_probability=p_cmp,
        reference_icc=rf.icc,
        comparison_icc=cf.icc,
        loglik=rf.loglik + cf.loglik,
        margin=float(margin),
    )


def _draw_beta_binomial_counts(
    *,
    rng: np.random.Generator,
    probability: float,
    icc: float,
    sizes: np.ndarray,
    repetitions: int,
) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.int64)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    concentration = (1.0 / icc) - 1.0
    a = probability * concentration
    b = (1.0 - probability) * concentration
    probs = rng.beta(a, b, size=(repetitions, len(sizes)))
    return rng.binomial(sizes[None, :], probs).astype(np.int64)


def one_sample_constrained_null_mc(
    *,
    events,
    sizes,
    boundary: float,
    alpha: float,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> OneSampleMonteCarloResult:
    k, n = _validate_counts(events, sizes)
    if not (0.0 < boundary < 1.0):
        raise ValueError("boundary must lie in (0,1)")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must lie in (0,0.5)")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")

    null_fit = _fit_icc_given_probability(
        events=k,
        sizes=n,
        probability=boundary,
    )

    observed = float(k.sum() / n.sum())
    boot = _draw_beta_binomial_counts(
        rng=rng,
        probability=boundary,
        icc=null_fit.icc,
        sizes=n.astype(np.int64),
        repetitions=bootstrap_repetitions,
    )
    boot_rate = boot.sum(axis=1) / float(n.sum())

    favorable = int(np.sum(boot_rate <= observed + 1e-15))
    p_value = (1.0 + favorable) / (bootstrap_repetitions + 1.0)
    reject = bool(observed < boundary and p_value <= alpha)

    return OneSampleMonteCarloResult(
        observed_rate=observed,
        null_icc_hat=null_fit.icc,
        p_value=float(p_value),
        reject_below_boundary=reject,
        bootstrap_repetitions=bootstrap_repetitions,
    )


def two_sample_ni_constrained_null_mc(
    *,
    reference_events,
    reference_sizes,
    comparison_events,
    comparison_sizes,
    margin: float,
    alpha: float,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> TwoSampleMonteCarloResult:
    kr, nr = _validate_counts(reference_events, reference_sizes)
    kc, nc = _validate_counts(comparison_events, comparison_sizes)

    null_fit = fit_two_sample_ni_null(
        reference_events=kr,
        reference_sizes=nr,
        comparison_events=kc,
        comparison_sizes=nc,
        margin=margin,
    )

    observed_ref = float(kr.sum() / nr.sum())
    observed_cmp = float(kc.sum() / nc.sum())
    observed_rd = observed_cmp - observed_ref

    br = _draw_beta_binomial_counts(
        rng=rng,
        probability=null_fit.reference_probability,
        icc=null_fit.reference_icc,
        sizes=nr.astype(np.int64),
        repetitions=bootstrap_repetitions,
    )
    bc = _draw_beta_binomial_counts(
        rng=rng,
        probability=null_fit.comparison_probability,
        icc=null_fit.comparison_icc,
        sizes=nc.astype(np.int64),
        repetitions=bootstrap_repetitions,
    )

    boot_rd = (
        bc.sum(axis=1) / float(nc.sum())
        - br.sum(axis=1) / float(nr.sum())
    )

    favorable = int(np.sum(boot_rd <= observed_rd + 1e-15))
    p_value = (1.0 + favorable) / (bootstrap_repetitions + 1.0)
    reject = bool(observed_rd < margin and p_value <= alpha)

    return TwoSampleMonteCarloResult(
        observed_reference_rate=observed_ref,
        observed_comparison_rate=observed_cmp,
        observed_risk_difference=float(observed_rd),
        null_reference_probability=null_fit.reference_probability,
        null_comparison_probability=null_fit.comparison_probability,
        null_reference_icc_hat=null_fit.reference_icc,
        null_comparison_icc_hat=null_fit.comparison_icc,
        p_value=float(p_value),
        reject_noninferiority_null=reject,
        bootstrap_repetitions=bootstrap_repetitions,
    )
