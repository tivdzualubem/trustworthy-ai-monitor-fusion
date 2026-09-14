import numpy as np

from monitor_fusion.external_validation.cluster_ni_welch import (
    _group_cluster_variance,
    _welch_df,
    simulate_welch_cluster_boundary,
)


def test_welch_df_reduces_to_sum_like_df_when_variances_and_G_match():
    v1 = np.asarray([0.01])
    v2 = np.asarray([0.01])
    df = _welch_df(v1, v2, 99.0, 99.0)
    assert np.isclose(df[0], 198.0)


def test_welch_df_drops_when_smaller_group_dominates_variance():
    v1 = np.asarray([0.005])
    v2 = np.asarray([0.020])
    df = _welch_df(v1, v2, 99.0, 49.0)
    assert df[0] < 100.0


def test_group_cluster_variance_preserves_row_marginal_mean():
    sizes = np.asarray([2, 4, 6])
    counts = np.asarray([[1, 1, 2]], dtype=float)
    mean, var, df = _group_cluster_variance(counts, sizes, correction="CRV1")
    assert np.isclose(mean[0], 4 / 12)
    assert var[0] >= 0.0
    assert df == 2.0


def test_boundary_simulation_reproducible():
    kwargs = dict(
        reference_fnr=0.05,
        icc_reference=0.05,
        icc_comparison=0.10,
        reference_cluster_sizes=[20] * 30,
        comparison_cluster_sizes=[40] * 15,
        repetitions=300,
        seed=20260907,
        correction="CRV1",
    )
    first = simulate_welch_cluster_boundary(**kwargs)
    second = simulate_welch_cluster_boundary(**kwargs)
    assert first == second
    assert 0.0 <= first.type_I_rate <= 1.0
