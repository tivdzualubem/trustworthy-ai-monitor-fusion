import numpy as np
import pytest

from monitor_fusion.external_validation.beta_binomial_profile import (
    beta_binomial_loglik,
    fit_beta_binomial,
    one_sample_boundary_test,
    two_sample_ni_boundary_test,
)


def test_loglik_is_finite_for_valid_cluster_counts():
    value = beta_binomial_loglik(
        events=[0, 1, 2, 1],
        sizes=[5, 5, 5, 5],
        probability=0.10,
        icc=0.05,
    )
    assert np.isfinite(value)


def test_fit_recovers_reasonable_probability():
    fit = fit_beta_binomial(
        events=[0, 1, 0, 1, 1, 0, 2, 0, 1, 0],
        sizes=[10] * 10,
    )
    empirical = 6 / 100
    assert abs(fit.probability - empirical) < 0.04
    assert 0.0 < fit.icc < 0.95


def test_one_sample_boundary_direction():
    result = one_sample_boundary_test(
        events=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        sizes=[10] * 10,
        boundary=0.10,
        alpha=0.025,
    )
    assert result.probability_hat < 0.10
    assert 0.0 <= result.p_value_one_sided <= 1.0


def test_two_sample_ni_direction_and_estimate():
    result = two_sample_ni_boundary_test(
        reference_events=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        reference_sizes=[10] * 10,
        comparison_events=[1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        comparison_sizes=[10] * 10,
        margin=0.03,
        alpha=0.025,
    )
    assert result.risk_difference_hat < 0.03
    assert 0.0 <= result.p_value_one_sided <= 1.0
