import math

import numpy as np
import pytest

from monitor_fusion.external_validation.cluster_fg_t import (
    ABSOLUTE_FNR_CEILING,
    FG_BOUND,
    FPR_ALPHA,
    FPR_CONSTRAINT,
    PRIMARY_NI_MARGIN,
    PRIMARY_SAFETY_ALPHA,
    FGTAnalysisNotEstimable,
    _fg_identity_covariance,
    evaluate_absolute_fnr,
    evaluate_fpr,
    evaluate_two_cell_noninferiority,
)
from monitor_fusion.external_validation.cluster_ni_fg_d5 import (
    _build_design,
    _fg_d5_from_counts,
)


def _pair_fixture():
    ref_patterns = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]
    cmp_patterns = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    ref_y, cmp_y, ref_clusters, cmp_clusters = [], [], [], []
    for g, pattern in enumerate(ref_patterns):
        ref_y.extend(pattern)
        ref_clusters.extend([f"r{g}"] * len(pattern))
    for g, pattern in enumerate(cmp_patterns):
        cmp_y.extend(pattern)
        cmp_clusters.extend([f"c{g}"] * len(pattern))
    cells = ["T"] * len(ref_y) + ["S"] * len(cmp_y)
    return cells, ref_y + cmp_y, ref_clusters + cmp_clusters


def test_frozen_candidate_constants():
    assert PRIMARY_NI_MARGIN == 0.03
    assert ABSOLUTE_FNR_CEILING == 0.10
    assert FPR_CONSTRAINT == 0.05
    assert PRIMARY_SAFETY_ALPHA == 0.025
    assert FPR_ALPHA == 0.05
    assert FG_BOUND == 0.75


def test_two_cell_estimate_is_response_average_risk_difference():
    cells, y, clusters = _pair_fixture()
    result = evaluate_two_cell_noninferiority(
        cell_ids=cells,
        outcome=y,
        provenance_cluster_ids=clusters,
        reference_cell="T",
        comparison_cell="S",
        margin=0.50,
    )
    ref = np.mean([v for c, v in zip(cells, y) if c == "T"])
    cmp_ = np.mean([v for c, v in zip(cells, y) if c == "S"])
    assert math.isclose(result.reference_rate, ref)
    assert math.isclose(result.comparison_rate, cmp_)
    assert math.isclose(result.risk_difference, cmp_ - ref)
    assert result.degrees_of_freedom == 6
    assert result.pass_noninferiority == (
        result.upper_confidence_limit < 0.50
    )


def test_fg_covariance_matches_existing_fg_d5_variance_before_df_choice():
    ref_sizes = [3, 4, 5]
    cmp_sizes = [4, 5, 6]
    ref_counts = np.asarray([[0, 1, 1]], dtype=float)
    cmp_counts = np.asarray([[1, 1, 2]], dtype=float)

    design = _build_design(ref_sizes, cmp_sizes)
    _, old_var, _, _ = _fg_d5_from_counts(
        reference_counts=ref_counts,
        comparison_counts=cmp_counts,
        design=design,
    )

    outcomes = []
    cells = []
    clusters = []

    for i, (m, k) in enumerate(zip(ref_sizes, ref_counts[0].astype(int))):
        outcomes.extend([1] * int(k) + [0] * (m - int(k)))
        cells.extend(["R"] * m)
        clusters.extend([f"r{i}"] * m)

    for i, (m, k) in enumerate(zip(cmp_sizes, cmp_counts[0].astype(int))):
        outcomes.extend([1] * int(k) + [0] * (m - int(k)))
        cells.extend(["C"] * m)
        clusters.extend([f"c{i}"] * m)

    y = np.asarray(outcomes, dtype=float)
    X = np.column_stack(
        [
            np.ones(len(y)),
            (np.asarray(cells, dtype=object) == "C").astype(float),
        ]
    )
    _, covariance, _, _ = _fg_identity_covariance(
        X=X,
        y=y,
        cluster_ids=np.asarray(clusters, dtype=object),
        cluster_level_parameter_count=2,
    )
    assert covariance[1, 1] == pytest.approx(old_var[0], rel=1e-12, abs=1e-12)


def test_one_cell_absolute_fnr_and_fpr_use_different_alpha_and_constraints():
    y = [0, 0, 1, 0, 0] * 6
    clusters = []
    for i in range(6):
        clusters.extend([f"g{i}"] * 5)

    fnr = evaluate_absolute_fnr(
        false_negative=y,
        provenance_cluster_ids=clusters,
    )
    fpr = evaluate_fpr(
        false_positive=y,
        provenance_cluster_ids=clusters,
    )
    assert fnr.constraint == 0.10
    assert fnr.alpha == 0.025
    assert fpr.constraint == 0.05
    assert fpr.alpha == 0.05
    assert fnr.degrees_of_freedom == 5
    assert fpr.degrees_of_freedom == 5


def test_cluster_may_not_span_two_cells():
    cells, y, clusters = _pair_fixture()
    clusters[-1] = clusters[0]
    with pytest.raises(ValueError):
        evaluate_two_cell_noninferiority(
            cell_ids=cells,
            outcome=y,
            provenance_cluster_ids=clusters,
            reference_cell="T",
            comparison_cell="S",
        )


def test_degenerate_one_cell_data_fail_closed():
    y = [0] * 20
    clusters = ["a"] * 10 + ["b"] * 10
    with pytest.raises(FGTAnalysisNotEstimable):
        evaluate_absolute_fnr(
            false_negative=y,
            provenance_cluster_ids=clusters,
        )
