import numpy as np

from monitor_fusion.external_validation.constrained_null_mc import (
    fit_two_sample_ni_null,
    one_sample_constrained_null_mc,
    two_sample_ni_constrained_null_mc,
)


def test_restricted_two_sample_fit_respects_3pp_boundary():
    fit = fit_two_sample_ni_null(
        reference_events=[0, 1, 0, 1, 0, 1, 0, 1],
        reference_sizes=[10] * 8,
        comparison_events=[1, 1, 0, 1, 1, 0, 1, 0],
        comparison_sizes=[10] * 8,
        margin=0.03,
    )
    assert abs(
        (fit.comparison_probability - fit.reference_probability) - 0.03
    ) < 1e-10


def test_one_sample_mc_is_reproducible_and_on_exact_p_grid():
    kwargs = dict(
        events=[0, 0, 1, 0, 0, 1, 0, 0],
        sizes=[10] * 8,
        boundary=0.10,
        alpha=0.025,
        bootstrap_repetitions=399,
    )
    a = one_sample_constrained_null_mc(
        **kwargs,
        rng=np.random.default_rng(123),
    )
    b = one_sample_constrained_null_mc(
        **kwargs,
        rng=np.random.default_rng(123),
    )
    assert a == b
    assert abs(a.p_value * 400 - round(a.p_value * 400)) < 1e-12


def test_two_sample_mc_is_reproducible_and_respects_direction():
    kwargs = dict(
        reference_events=[0, 1, 0, 1, 0, 1, 0, 1],
        reference_sizes=[10] * 8,
        comparison_events=[1, 1, 0, 1, 1, 0, 1, 0],
        comparison_sizes=[10] * 8,
        margin=0.03,
        alpha=0.025,
        bootstrap_repetitions=399,
    )
    a = two_sample_ni_constrained_null_mc(
        **kwargs,
        rng=np.random.default_rng(456),
    )
    b = two_sample_ni_constrained_null_mc(
        **kwargs,
        rng=np.random.default_rng(456),
    )
    assert a == b
    assert abs(a.p_value * 400 - round(a.p_value * 400)) < 1e-12
    if a.reject_noninferiority_null:
        assert a.observed_risk_difference < 0.03
