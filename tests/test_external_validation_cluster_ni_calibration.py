import numpy as np
import pytest

from monitor_fusion.external_validation.cluster_ni_calibration import (
    _group_mean_and_variance,
    simulate_boundary_calibration,
)


def test_md_variance_is_at_least_kc_for_same_clusters():
    sizes = np.asarray([5, 8, 7, 10])
    counts = np.asarray(
        [
            [0, 1, 2, 1],
            [1, 0, 1, 3],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    mean_kc, var_kc = _group_mean_and_variance(
        counts, sizes, correction="KC"
    )
    mean_md, var_md = _group_mean_and_variance(
        counts, sizes, correction="MD"
    )
    assert np.allclose(mean_kc, mean_md)
    assert np.all(var_md >= var_kc)


def test_equal_cluster_closed_form_md_to_kc_ratio():
    sizes = np.asarray([10] * 20)
    counts = np.asarray([[0, 1, 0, 2, 1] * 4], dtype=float)
    _, kc = _group_mean_and_variance(counts, sizes, correction="KC")
    _, md = _group_mean_and_variance(counts, sizes, correction="MD")
    expected = 20 / 19
    assert np.isclose(float(md[0] / kc[0]), expected)


def test_boundary_simulation_is_reproducible():
    kwargs = dict(
        reference_fnr=0.05,
        icc_reference=0.01,
        icc_comparison=0.01,
        reference_cluster_sizes=[20] * 30,
        comparison_cluster_sizes=[20] * 30,
        repetitions=400,
        seed=20260907,
        correction="MD",
    )
    first = simulate_boundary_calibration(**kwargs)
    second = simulate_boundary_calibration(**kwargs)
    assert first == second
    assert 0.0 <= first.pass_rate <= 1.0
    assert 0.0 <= first.cp_lower_95 <= first.cp_upper_95 <= 1.0


def test_unknown_correction_rejected():
    with pytest.raises(ValueError):
        simulate_boundary_calibration(
            reference_fnr=0.05,
            icc_reference=0.0,
            icc_comparison=0.0,
            reference_cluster_sizes=[10, 10],
            comparison_cluster_sizes=[10, 10],
            repetitions=10,
            seed=1,
            correction="BAD",
        )
