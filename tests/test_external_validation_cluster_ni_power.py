import math

import numpy as np

from monitor_fusion.external_validation.cluster_ni_design import (
    evaluate_fnr_noninferiority,
    generate_beta_binomial_clusters,
)
from monitor_fusion.external_validation.cluster_ni_power import (
    _kc_group_variance,
    simulate_pair_pass_rate_vectorized,
    stable_seed,
)


def test_stable_seed_is_deterministic_and_context_specific():
    a = stable_seed(20260907, "p", 0.10, "icc", 0.05)
    b = stable_seed(20260907, "p", 0.10, "icc", 0.05)
    c = stable_seed(20260907, "p", 0.20, "icc", 0.05)
    assert a == b
    assert a != c


def test_aggregated_group_variance_matches_row_level_ni():
    ref_sizes = [4, 5, 6, 5]
    cmp_sizes = [5, 4, 6, 5]
    rng = np.random.default_rng(123)

    y_ref, c_ref = generate_beta_binomial_clusters(
        marginal_probability=0.15,
        icc=0.04,
        cluster_sizes=ref_sizes,
        rng=rng,
        cluster_prefix="ref",
    )
    y_cmp, c_cmp = generate_beta_binomial_clusters(
        marginal_probability=0.18,
        icc=0.04,
        cluster_sizes=cmp_sizes,
        rng=rng,
        cluster_prefix="cmp",
    )

    row = evaluate_fnr_noninferiority(
        cell_ids=["R"] * len(y_ref) + ["C"] * len(y_cmp),
        false_negative=y_ref + y_cmp,
        provenance_cluster_ids=c_ref + c_cmp,
        reference_cell="R",
        comparison_cell="C",
        margin=0.50,
    )

    def counts_by_cluster(y, clusters):
        order = tuple(dict.fromkeys(clusters))
        return np.asarray(
            [[sum(v for v, c in zip(y, clusters) if c == cluster) for cluster in order]],
            dtype=float,
        )

    ref_counts = counts_by_cluster(y_ref, c_ref)
    cmp_counts = counts_by_cluster(y_cmp, c_cmp)

    ref_mean, ref_var = _kc_group_variance(ref_counts, np.asarray(ref_sizes))
    cmp_mean, cmp_var = _kc_group_variance(cmp_counts, np.asarray(cmp_sizes))

    assert math.isclose(float(cmp_mean[0] - ref_mean[0]), row.risk_difference)
    assert math.isclose(
        math.sqrt(float(ref_var[0] + cmp_var[0])),
        row.standard_error,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_vectorized_simulation_is_reproducible():
    kwargs = dict(
        reference_fnr=0.10,
        comparison_fnr=0.10,
        icc_reference=0.03,
        icc_comparison=0.03,
        reference_cluster_sizes=[10] * 20,
        comparison_cluster_sizes=[10] * 20,
        repetitions=200,
        seed=44,
    )
    first = simulate_pair_pass_rate_vectorized(**kwargs)
    second = simulate_pair_pass_rate_vectorized(**kwargs)
    assert first == second
    assert first.estimable_count <= first.repetitions
    assert 0.0 <= first.pass_rate_all <= 1.0
    assert 0.0 <= first.cp_lower_95 <= first.cp_upper_95 <= 1.0


def test_boundary_and_preservation_scenarios_are_distinct():
    common = dict(
        reference_fnr=0.10,
        icc_reference=0.02,
        icc_comparison=0.02,
        reference_cluster_sizes=[10] * 30,
        comparison_cluster_sizes=[10] * 30,
        repetitions=300,
        seed=99,
    )
    preserve = simulate_pair_pass_rate_vectorized(
        comparison_fnr=0.10,
        **common,
    )
    boundary = simulate_pair_pass_rate_vectorized(
        comparison_fnr=0.13,
        **common,
    )
    assert preserve.pass_rate_all > boundary.pass_rate_all
