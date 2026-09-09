import numpy as np
import pytest

from monitor_fusion.external_validation.cluster_wcr_current import (
    fixed_webb_weights,
    wcr_c_one_sample_batch,
    wcr_c_two_sample_batch,
)


def test_invalid_bootstrap_draws_make_lower_tail_p_value_more_conservative():
    from monitor_fusion.external_validation.cluster_ni_wcr import (
        _lower_tail_p_values_fail_closed,
    )

    t_obs = np.asarray([-1.0])
    t_boot = np.asarray([[-2.0, np.nan, 0.0]])
    estimable = np.asarray([[True, False, True]])
    observed = np.asarray([True])

    p_value, reject = _lower_tail_p_values_fail_closed(
        t_obs=t_obs,
        t_boot=t_boot,
        bootstrap_estimable=estimable,
        observed_estimable=observed,
        bootstrap_repetitions=3,
        alpha=0.50,
    )

    # One valid lower-tail hit plus one invalid draw:
    # p = (1 + 2) / (3 + 1) = 0.75.
    assert p_value[0] == pytest.approx(0.75)
    assert not reject[0]


def test_one_sample_bootstrap_grid_and_reproducibility():
    counts = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    sizes = np.asarray([10, 10, 10, 10])
    weights = fixed_webb_weights(
        bootstrap_repetitions=199,
        cluster_count=4,
        seed=123,
    )
    a = wcr_c_one_sample_batch(
        counts=counts,
        cluster_sizes=sizes,
        boundary=0.10,
        alpha=0.025,
        bootstrap_repetitions=199,
        weights=weights,
    )
    b = wcr_c_one_sample_batch(
        counts=counts,
        cluster_sizes=sizes,
        boundary=0.10,
        alpha=0.025,
        bootstrap_repetitions=199,
        weights=weights,
    )
    assert a == b
    assert a.repetitions == 3


def test_two_sample_requires_matching_count_columns_to_sizes():
    weights = fixed_webb_weights(
        bootstrap_repetitions=199,
        cluster_count=3,
        seed=1,
    )
    with pytest.raises(ValueError):
        wcr_c_two_sample_batch(
            reference_counts=np.zeros((2, 2)),
            comparison_counts=np.zeros((2, 3)),
            reference_cluster_sizes=np.asarray([10, 10, 10]),
            comparison_cluster_sizes=np.asarray([10, 10, 10]),
            margin=0.03,
            alpha=0.025,
            bootstrap_repetitions=199,
            reference_weights=weights,
            comparison_weights=weights,
        )


def test_one_sample_fails_closed_when_variance_zero():
    counts = np.zeros((5, 4), dtype=float)
    sizes = np.asarray([10, 10, 10, 10])
    weights = fixed_webb_weights(
        bootstrap_repetitions=199,
        cluster_count=4,
        seed=2,
    )
    result = wcr_c_one_sample_batch(
        counts=counts,
        cluster_sizes=sizes,
        boundary=0.10,
        alpha=0.025,
        bootstrap_repetitions=199,
        weights=weights,
    )
    assert result.reject_count == 0
    assert result.estimable_count == 0
