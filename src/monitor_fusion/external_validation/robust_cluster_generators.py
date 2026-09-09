from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import least_squares
from scipy.special import expit, logit


@dataclass(frozen=True)
class LogitNormalCalibration:
    marginal_probability: float
    response_scale_icc: float
    intercept: float
    random_intercept_sd: float
    quadrature_mean: float
    quadrature_icc: float


_NODES, _WEIGHTS = hermgauss(60)
_Z = np.sqrt(2.0) * _NODES
_W = _WEIGHTS / math.sqrt(math.pi)


def _logit_normal_moments(intercept: float, sd: float) -> tuple[float, float]:
    probs = expit(intercept + sd * _Z)
    mean = float(np.sum(_W * probs))
    second = float(np.sum(_W * probs * probs))
    variance = second - mean * mean
    return mean, variance


def calibrate_logit_normal(
    *,
    marginal_probability: float,
    response_scale_icc: float,
) -> LogitNormalCalibration:
    p = float(marginal_probability)
    rho = float(response_scale_icc)
    if not (0.0 < p < 1.0):
        raise ValueError("marginal_probability must lie in (0,1)")
    if not (0.0 < rho < 1.0):
        raise ValueError("response_scale_icc must lie in (0,1)")

    target_variance = rho * p * (1.0 - p)
    latent_sd_guess = math.sqrt(
        (rho / (1.0 - rho)) * (math.pi ** 2 / 3.0)
    )

    def residual(params: np.ndarray) -> np.ndarray:
        intercept = float(params[0])
        sd = math.exp(float(params[1]))
        mean, variance = _logit_normal_moments(intercept, sd)
        return np.asarray(
            [
                (mean - p) / max(p, 1e-3),
                (variance - target_variance) / max(target_variance, 1e-5),
            ],
            dtype=float,
        )

    result = least_squares(
        residual,
        x0=np.asarray(
            [float(logit(p)), math.log(max(latent_sd_guess, 1e-4))],
            dtype=float,
        ),
        bounds=([-20.0, -10.0], [20.0, 5.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError("logit-normal calibration failed")

    intercept = float(result.x[0])
    sd = math.exp(float(result.x[1]))
    mean, variance = _logit_normal_moments(intercept, sd)
    achieved_icc = variance / (mean * (1.0 - mean))

    if abs(mean - p) > 1e-8 or abs(achieved_icc - rho) > 1e-8:
        raise RuntimeError(
            "logit-normal calibration did not reproduce target marginal moments"
        )

    return LogitNormalCalibration(
        marginal_probability=p,
        response_scale_icc=rho,
        intercept=intercept,
        random_intercept_sd=sd,
        quadrature_mean=mean,
        quadrature_icc=achieved_icc,
    )


def draw_logit_normal_counts(
    *,
    rng: np.random.Generator,
    marginal_probability: float,
    response_scale_icc: float,
    cluster_sizes: np.ndarray,
    calibration: LogitNormalCalibration | None = None,
) -> np.ndarray:
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    if np.any(sizes <= 0):
        raise ValueError("cluster sizes must be positive")

    cal = calibration or calibrate_logit_normal(
        marginal_probability=marginal_probability,
        response_scale_icc=response_scale_icc,
    )

    z = rng.normal(size=len(sizes))
    probabilities = expit(cal.intercept + cal.random_intercept_sd * z)
    return rng.binomial(sizes, probabilities).astype(np.int64)
