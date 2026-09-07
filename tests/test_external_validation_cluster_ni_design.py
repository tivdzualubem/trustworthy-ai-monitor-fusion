import math
import numpy as np
import pytest

from monitor_fusion.external_validation.cluster_ni_design import (
    ClusterNIAnalysisNotEstimable,
    PRIMARY_NI_MARGIN,
    SENSITIVITY_NI_MARGINS,
    evaluate_absolute_fnr_ceiling,
    evaluate_fnr_noninferiority,
    generate_beta_binomial_clusters,
    simulate_pair_noninferiority,
)


def _pair_data():
    ref_patterns = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,1]]
    cmp_patterns = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1],[0,0,0,0,0]]
    ref_y, cmp_y, ref_clusters, cmp_clusters = [], [], [], []
    for g, pattern in enumerate(ref_patterns):
        ref_y.extend(pattern); ref_clusters.extend([f"human:r{g}"] * len(pattern))
    for g, pattern in enumerate(cmp_patterns):
        cmp_y.extend(pattern); cmp_clusters.extend([f"model:c{g}"] * len(pattern))
    cells = ["T"] * len(ref_y) + ["S"] * len(cmp_y)
    return cells, ref_y + cmp_y, ref_clusters + cmp_clusters


def test_margin_constants_are_frozen_for_design_evaluation():
    assert PRIMARY_NI_MARGIN == 0.03
    assert SENSITIVITY_NI_MARGINS == (0.02, 0.05)


def test_risk_difference_is_row_marginal_difference():
    cells, y, clusters = _pair_data()
    result = evaluate_fnr_noninferiority(cell_ids=cells, false_negative=y, provenance_cluster_ids=clusters, reference_cell="T", comparison_cell="S")
    ref = np.mean([v for c, v in zip(cells, y) if c == "T"])
    cmp_ = np.mean([v for c, v in zip(cells, y) if c == "S"])
    assert math.isclose(result.reference_fnr, ref)
    assert math.isclose(result.comparison_fnr, cmp_)
    assert math.isclose(result.risk_difference, cmp_ - ref)


def test_cluster_df_is_total_clusters_minus_two():
    cells, y, clusters = _pair_data()
    result = evaluate_fnr_noninferiority(cell_ids=cells, false_negative=y, provenance_cluster_ids=clusters, reference_cell="T", comparison_cell="S")
    assert result.cluster_count == 8
    assert result.degrees_of_freedom == 6


def test_noninferiority_rule_uses_upper_bound():
    cells, y, clusters = _pair_data()
    result = evaluate_fnr_noninferiority(cell_ids=cells, false_negative=y, provenance_cluster_ids=clusters, reference_cell="T", comparison_cell="S", margin=0.50)
    assert result.noninferiority_pass == (result.upper_97_5 < 0.50)


def test_cluster_may_not_span_cells():
    cells, y, clusters = _pair_data(); clusters[-1] = clusters[0]
    with pytest.raises(ValueError):
        evaluate_fnr_noninferiority(cell_ids=cells, false_negative=y, provenance_cluster_ids=clusters, reference_cell="T", comparison_cell="S")


def test_absolute_ceiling_uses_G_minus_one_df():
    _, y, clusters = _pair_data(); y = y[:20]; clusters = clusters[:20]
    result = evaluate_absolute_fnr_ceiling(cell_id="T", false_negative=y, provenance_cluster_ids=clusters, ceiling=0.50)
    assert result.cluster_count == 4
    assert result.degrees_of_freedom == 3
    assert result.ceiling_pass == (result.upper_97_5 < 0.50)


def test_degenerate_all_zero_data_fail_closed():
    cells = ["T"] * 10 + ["S"] * 10
    y = [0] * 20
    clusters = ["t0"] * 5 + ["t1"] * 5 + ["s0"] * 5 + ["s1"] * 5
    with pytest.raises(ClusterNIAnalysisNotEstimable):
        evaluate_fnr_noninferiority(cell_ids=cells, false_negative=y, provenance_cluster_ids=clusters, reference_cell="T", comparison_cell="S")


def test_beta_binomial_generator_reproducible():
    args = dict(marginal_probability=0.10, icc=0.05, cluster_sizes=[4,5,6], cluster_prefix="x")
    first = generate_beta_binomial_clusters(**args, rng=np.random.default_rng(123))
    second = generate_beta_binomial_clusters(**args, rng=np.random.default_rng(123))
    assert first == second


def test_beta_binomial_generator_row_and_cluster_counts():
    y, clusters = generate_beta_binomial_clusters(marginal_probability=0.10, icc=0.05, cluster_sizes=[4,5,6], rng=np.random.default_rng(123), cluster_prefix="x")
    assert len(y) == 15 and len(clusters) == 15 and len(set(clusters)) == 3


def test_simulator_is_reproducible():
    kwargs = dict(reference_fnr=0.10, comparison_fnr=0.10, icc_reference=0.03, icc_comparison=0.03, reference_cluster_sizes=[10]*12, comparison_cluster_sizes=[10]*12, repetitions=40, seed=12345)
    first = simulate_pair_noninferiority(**kwargs); second = simulate_pair_noninferiority(**kwargs)
    assert first == second
    assert 0.0 <= first.pass_rate_all_repetitions <= 1.0
    assert first.estimable_repetitions <= first.repetitions


def test_simulator_records_true_difference():
    result = simulate_pair_noninferiority(reference_fnr=0.10, comparison_fnr=0.13, icc_reference=0.02, icc_comparison=0.02, reference_cluster_sizes=[8]*10, comparison_cluster_sizes=[8]*10, repetitions=20, seed=77)
    assert math.isclose(result.true_risk_difference, 0.03)
    assert result.margin == 0.03
    assert result.alpha == 0.025
