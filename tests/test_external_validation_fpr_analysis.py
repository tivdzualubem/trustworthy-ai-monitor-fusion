import numpy as np
import pytest

from monitor_fusion.external_validation.fpr_analysis import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    FPR_OPERATING_CONSTRAINT,
    ONE_SIDED_ALPHA,
    _provenance_cluster_bootstrap_interval,
    clopper_pearson_upper,
    evaluate_cell_fpr,
)


def test_frozen_fpr_constants():
    assert ONE_SIDED_ALPHA == 0.05
    assert FPR_OPERATING_CONSTRAINT == 0.05
    assert BOOTSTRAP_REPETITIONS == 5000
    assert BOOTSTRAP_SEED == 424242


def test_preregistered_n361_k11_upper_bound_reproduces():
    assert clopper_pearson_upper(11, 361) == pytest.approx(
        0.04993268035498383,
        abs=1e-14,
    )


def test_n361_k11_passes_and_k12_fails():
    upper_11 = clopper_pearson_upper(11, 361)
    upper_12 = clopper_pearson_upper(12, 361)

    assert upper_11 <= 0.05
    assert upper_12 > 0.05


def test_all_false_positives_has_upper_bound_one():
    assert clopper_pearson_upper(10, 10) == 1.0


def test_invalid_binomial_counts_rejected():
    with pytest.raises(ValueError):
        clopper_pearson_upper(-1, 10)
    with pytest.raises(ValueError):
        clopper_pearson_upper(11, 10)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0)


def test_fpr_requires_only_y0_rows():
    with pytest.raises(ValueError):
        evaluate_cell_fpr(
            reference_labels=[0, 1],
            intercept_decisions=[0, 1],
            provenance_cluster_ids=["human:a", "human:b"],
        )


def test_cluster_bootstrap_is_deterministic():
    decisions = np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int8)
    clusters = np.asarray(
        ["a", "a", "b", "b", "c", "c"],
        dtype=object,
    )

    first = _provenance_cluster_bootstrap_interval(
        decisions,
        clusters,
        repetitions=500,
        seed=424242,
    )
    second = _provenance_cluster_bootstrap_interval(
        decisions,
        clusters,
        repetitions=500,
        seed=424242,
    )

    assert first == second


def test_cluster_bootstrap_preserves_all_rows_of_sampled_cluster():
    decisions = np.asarray([1, 1, 0, 0], dtype=np.int8)
    clusters = np.asarray(["a", "a", "b", "b"], dtype=object)

    lower, upper, cluster_count = _provenance_cluster_bootstrap_interval(
        decisions,
        clusters,
        repetitions=500,
        seed=424242,
    )

    assert cluster_count == 2
    assert 0.0 <= lower <= 0.5 <= upper <= 1.0


def test_cluster_bootstrap_requires_multiple_clusters():
    decisions = np.asarray([0, 1], dtype=np.int8)
    clusters = np.asarray(["a", "a"], dtype=object)

    with pytest.raises(ValueError):
        _provenance_cluster_bootstrap_interval(
            decisions,
            clusters,
            repetitions=100,
            seed=424242,
        )


def test_evaluate_cell_fpr_reports_nominal_and_cluster_sensitivity():
    labels = [0] * 361
    decisions = [1] * 11 + [0] * 350
    clusters = [f"human:a{i % 20}" for i in range(361)]

    result = evaluate_cell_fpr(
        reference_labels=labels,
        intercept_decisions=decisions,
        provenance_cluster_ids=clusters,
    )

    assert result.n_y0 == 361
    assert result.false_positives == 11
    assert result.empirical_fpr == pytest.approx(11 / 361)
    assert result.cp_upper_95 == pytest.approx(0.04993268035498383)
    assert result.operating_constraint_pass is True
    assert result.provenance_cluster_count == 20
    assert 0.0 <= result.bootstrap_lower_95 <= result.bootstrap_upper_95 <= 1.0


def test_k12_cell_does_not_pass_operating_constraint():
    result = evaluate_cell_fpr(
        reference_labels=[0] * 361,
        intercept_decisions=[1] * 12 + [0] * 349,
        provenance_cluster_ids=[f"human:a{i % 20}" for i in range(361)],
    )

    assert result.operating_constraint_pass is False
