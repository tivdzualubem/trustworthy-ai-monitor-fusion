import numpy as np

from monitor_fusion.external_validation.cluster_ni_fg_d5 import (
    _build_design,
    _fg_d5_from_counts,
    simulate_fg_d5_boundary_calibration,
)


def _direct_dfcalc_single(S, design):
    K = len(design.X_cluster)
    p = 2
    test = np.asarray([0.0, 1.0])

    Psimat = np.zeros((p * K, p * K))
    G = np.zeros((p * K, p * K))
    M = np.zeros((p * K, p * K))

    for i in range(K):
        idx = slice(i * p, (i + 1) * p)
        A_i = design.omega[i] @ design.vm

        G[idx, idx] = np.eye(p)
        for k in range(K):
            kdx = slice(k * p, (k + 1) * p)
            G[idx, kdx] -= A_i

        Psimat[idx, idx] = design.w[i] * S

        DHi = np.diag(design.H[i])
        a = DHi @ design.vm @ test
        M[idx, idx] = np.outer(a, a)

    psi_b1 = Psimat @ G.T @ M @ G
    num = np.trace(psi_b1) ** 2
    den = np.sum(psi_b1 * psi_b1.T)
    value = num / den
    return max(value, 1.0)


def test_fast_df_matches_direct_saws_matrix_formula():
    design = _build_design([3, 4, 5], [4, 5, 6])
    ref_counts = np.asarray([[0, 1, 1], [1, 0, 2]], dtype=float)
    cmp_counts = np.asarray([[1, 1, 2], [0, 2, 2]], dtype=float)

    _, _, df, S = _fg_d5_from_counts(
        reference_counts=ref_counts,
        comparison_counts=cmp_counts,
        design=design,
    )

    for r in range(len(df)):
        direct = _direct_dfcalc_single(S[r], design)
        assert np.isclose(df[r], direct, rtol=1e-10, atol=1e-10)


def test_fg_design_is_full_rank_and_positive_corrections():
    design = _build_design([10] * 5, [10] * 5)
    assert np.linalg.matrix_rank(design.vm) == 2
    assert np.all(design.H >= 1.0)
    assert np.all(np.isfinite(design.w))


def test_fg_d5_boundary_simulation_reproducible():
    kwargs = dict(
        reference_fnr=0.05,
        icc_reference=0.05,
        icc_comparison=0.10,
        reference_cluster_sizes=[20] * 20,
        comparison_cluster_sizes=[40] * 10,
        repetitions=200,
        seed=20260907,
    )
    first = simulate_fg_d5_boundary_calibration(**kwargs)
    second = simulate_fg_d5_boundary_calibration(**kwargs)
    assert first == second
    assert 0.0 <= first.pass_rate <= 1.0
    assert first.estimable_count <= first.repetitions
    assert first.min_df >= 1.0
