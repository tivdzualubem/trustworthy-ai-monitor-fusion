import math

import numpy as np
import pytest

from monitor_fusion.external_validation.cluster_ni_wcr import (
    WEBB_SUPPORT,
    _bootstrap_group_variance_fast,
    _group_variance_from_cluster_residual_sums,
    simulate_wcr_boundary_calibration,
)


def test_webb_support_has_zero_mean_unit_variance():
    assert math.isclose(float(WEBB_SUPPORT.mean()), 0.0, abs_tol=1e-15)
    assert math.isclose(float(np.mean(WEBB_SUPPORT**2)), 1.0, abs_tol=1e-15)


def test_fast_bootstrap_variance_matches_direct_materialization():
    r = np.asarray(
        [
            [-1.0, 0.5, 0.5],
            [0.0, -1.0, 1.0],
        ]
    )
    sizes = np.asarray([4, 5, 6])
    weights = np.asarray(
        [
            [-1.0, 1.0, math.sqrt(0.5)],
            [math.sqrt(1.5), -1.0, 1.0],
        ]
    )

    for method in ("CRV1", "CRV3"):
        shift, fast = _bootstrap_group_variance_fast(
            r, sizes, weights, method=method
        )
        direct = np.empty_like(fast)

        n = sizes.sum()
        leverage = sizes / n

        for i in range(r.shape[0]):
            for b in range(weights.shape[0]):
                e = weights[b] * r[i] - sizes * shift[i, b]
                if method == "CRV3":
                    e = e / (1.0 - leverage)
                direct[i, b] = np.sum(e**2) / (n * n)

        assert np.allclose(fast, direct, rtol=1e-12, atol=1e-12)


def test_observed_group_variance_requires_valid_method():
    with pytest.raises(ValueError):
        _group_variance_from_cluster_residual_sums(
            np.zeros((2, 3)),
            np.asarray([3, 3, 3]),
            method="BAD",
        )


def test_wcr_calibration_reproducible():
    kwargs = dict(
        reference_fnr=0.05,
        icc_reference=0.05,
        icc_comparison=0.10,
        reference_cluster_sizes=[20] * 20,
        comparison_cluster_sizes=[40] * 10,
        repetitions=100,
        bootstrap_repetitions=199,
        seed=20260907,
        method="WCR-V",
    )
    # 0.025*(199+1)=5 is an integer.
    first = simulate_wcr_boundary_calibration(**kwargs)
    second = simulate_wcr_boundary_calibration(**kwargs)
    assert first == second
    assert 0.0 <= first.type_I_rate <= 1.0


def test_bad_bootstrap_grid_rejected():
    with pytest.raises(ValueError):
        simulate_wcr_boundary_calibration(
            reference_fnr=0.05,
            icc_reference=0.0,
            icc_comparison=0.0,
            reference_cluster_sizes=[10, 10],
            comparison_cluster_sizes=[10, 10],
            repetitions=10,
            bootstrap_repetitions=100,
            seed=1,
            method="WCR-C",
        )
