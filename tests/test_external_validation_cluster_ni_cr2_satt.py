import numpy as np

from monitor_fusion.external_validation.cluster_ni_cr2_satt import (
    build_cr2_satterthwaite_design,
    simulate_cr2_satterthwaite_boundary,
)


def _matrix_inv_sqrt(A):
    values, vectors = np.linalg.eigh(A)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def _direct_clubsandwich_p(ref_sizes, cmp_sizes):
    sizes = list(ref_sizes) + list(cmp_sizes)
    groups = [0] * len(ref_sizes) + [1] * len(cmp_sizes)

    rows = []
    cluster_slices = []
    start = 0
    for m, g in zip(sizes, groups):
        Xi = np.column_stack(
            [np.ones(m), np.full(m, float(g))]
        )
        rows.append(Xi)
        cluster_slices.append((start, start + m))
        start += m

    X = np.vstack(rows)
    M = np.linalg.inv(X.T @ X)
    L = np.linalg.cholesky(M)

    G_slope_norm2 = []
    H_slope_rows = []

    for Xi in rows:
        Hi = Xi @ M @ Xi.T
        Ai = _matrix_inv_sqrt(np.eye(len(Xi)) - Hi)
        Ei = Xi.T @ Ai
        MEi = M @ Ei

        Gi = MEi
        G_slope_norm2.append(float(np.sum(Gi[1] ** 2)))

        Hi_small = (MEi @ Xi) @ L
        H_slope_rows.append(Hi_small[1])

    H = np.asarray(H_slope_rows)
    return np.diag(G_slope_norm2) - H @ H.T


def test_specialized_p_matches_direct_clubsandwich_matrix_formula():
    ref = [3, 4, 5]
    cmp_ = [4, 6, 5, 3]

    specialized = build_cr2_satterthwaite_design(ref, cmp_)
    direct = _direct_clubsandwich_p(ref, cmp_)

    assert np.allclose(
        specialized.P_matrix,
        direct,
        rtol=1e-10,
        atol=1e-10,
    )

    direct_df = np.trace(direct) ** 2 / np.sum(direct * direct)
    assert np.isclose(
        specialized.degrees_of_freedom,
        direct_df,
        rtol=1e-12,
        atol=1e-12,
    )


def test_balanced_design_has_positive_df():
    design = build_cr2_satterthwaite_design([20] * 30, [20] * 30)
    assert design.degrees_of_freedom >= 1.0
    assert np.all(np.isfinite(design.P_matrix))


def test_asymmetric_design_df_is_not_naive_total_clusters_minus_two():
    design = build_cr2_satterthwaite_design([20] * 100, [40] * 50)
    naive = 100 + 50 - 2
    assert not np.isclose(design.degrees_of_freedom, naive)


def test_boundary_simulation_reproducible():
    kwargs = dict(
        reference_fnr=0.05,
        icc_reference=0.05,
        icc_comparison=0.10,
        reference_cluster_sizes=[20] * 30,
        comparison_cluster_sizes=[40] * 15,
        repetitions=300,
        seed=20260907,
    )
    first = simulate_cr2_satterthwaite_boundary(**kwargs)
    second = simulate_cr2_satterthwaite_boundary(**kwargs)
    assert first == second
    assert 0.0 <= first.type_I_rate <= 1.0
