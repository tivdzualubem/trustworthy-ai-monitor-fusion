import numpy as np
import pytest

from monitor_fusion.external_validation.fnr_analysis import (
    FAMILYWISE_ALPHA,
    PRIMARY_TEST_COUNT,
    FactorialModelNotEstimable,
    adjust_primary_nine_test_family,
    design_row,
    false_negative_indicator,
    fit_w1_fnr_factorial,
    holm_adjust,
)


def _synthetic_w1():
    cell_ids = []
    endpoint = []
    clusters = []

    counts = {
        "T": 8,
        "S": 16,
        "F": 24,
        "SF": 40,
    }

    for cell_id in ("T", "S", "F", "SF"):
        for i in range(80):
            cell_ids.append(cell_id)
            endpoint.append(1 if i < counts[cell_id] else 0)

            if cell_id in ("T", "F"):
                clusters.append(f"human:a{i % 10}")
            else:
                clusters.append(f"model:b{i % 10}")

    return cell_ids, endpoint, clusters


def test_primary_family_is_frozen():
    assert FAMILYWISE_ALPHA == 0.05
    assert PRIMARY_TEST_COUNT == 9


def test_effect_coding_matches_preregistered_design():
    assert np.allclose(design_row("T"), [1.0, -0.5, -0.5, 0.25])
    assert np.allclose(design_row("S"), [1.0, 0.5, -0.5, -0.25])
    assert np.allclose(design_row("F"), [1.0, -0.5, 0.5, -0.25])
    assert np.allclose(design_row("SF"), [1.0, 0.5, 0.5, 0.25])


def test_false_negative_indicator():
    result = false_negative_indicator(
        [1, 1, 1],
        [1, 0, 1],
    )
    assert result.tolist() == [0, 1, 0]


def test_fnr_rejects_y0_rows():
    with pytest.raises(ValueError):
        false_negative_indicator([1, 0], [1, 1])


def test_factorial_fit_reproduces_cell_fnr():
    cell_ids, endpoint, clusters = _synthetic_w1()

    result = fit_w1_fnr_factorial(
        cell_ids=cell_ids,
        false_negative=endpoint,
        provenance_cluster_ids=clusters,
    )

    assert result.cells["T"].fnr == pytest.approx(0.10)
    assert result.cells["S"].fnr == pytest.approx(0.20)
    assert result.cells["F"].fnr == pytest.approx(0.30)
    assert result.cells["SF"].fnr == pytest.approx(0.50)
    assert result.cluster_count == 20
    assert result.row_count == 320


def test_expected_main_effect_directions():
    cell_ids, endpoint, clusters = _synthetic_w1()

    result = fit_w1_fnr_factorial(
        cell_ids=cell_ids,
        false_negative=endpoint,
        provenance_cluster_ids=clusters,
    )

    assert result.terms["source"].estimate > 0
    assert result.terms["attack_family"].estimate > 0


def test_cluster_robust_intervals_are_finite_and_ordered():
    cell_ids, endpoint, clusters = _synthetic_w1()

    result = fit_w1_fnr_factorial(
        cell_ids=cell_ids,
        false_negative=endpoint,
        provenance_cluster_ids=clusters,
    )

    for term in result.terms.values():
        assert np.isfinite(term.standard_error)
        assert np.isfinite(term.p_value)
        assert term.ci_lower < term.ci_upper

    for cell in result.cells.values():
        assert 0.0 <= cell.ci_lower <= cell.fnr <= cell.ci_upper <= 1.0


def test_complete_cell_separation_is_not_silently_repaired():
    cell_ids, endpoint, clusters = _synthetic_w1()

    endpoint = list(endpoint)
    for i, cell_id in enumerate(cell_ids):
        if cell_id == "T":
            endpoint[i] = 0

    with pytest.raises(FactorialModelNotEstimable):
        fit_w1_fnr_factorial(
            cell_ids=cell_ids,
            false_negative=endpoint,
            provenance_cluster_ids=clusters,
        )


def test_holm_known_example():
    result = holm_adjust(
        {
            "a": 0.001,
            "b": 0.010,
            "c": 0.040,
        }
    )

    assert result["a"]["holm_adjusted_p"] == pytest.approx(0.003)
    assert result["b"]["holm_adjusted_p"] == pytest.approx(0.020)
    assert result["c"]["holm_adjusted_p"] == pytest.approx(0.040)


def test_holm_adjusted_p_values_are_monotone_in_sorted_order():
    raw = {
        "a": 0.03,
        "b": 0.001,
        "c": 0.02,
        "d": 0.20,
    }
    adjusted = holm_adjust(raw)
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    values = [adjusted[key]["holm_adjusted_p"] for key in ordered]
    assert values == sorted(values)


def test_primary_adjustment_requires_exactly_three_monitors():
    cell_ids, endpoint, clusters = _synthetic_w1()
    result = fit_w1_fnr_factorial(
        cell_ids=cell_ids,
        false_negative=endpoint,
        provenance_cluster_ids=clusters,
    )

    with pytest.raises(ValueError):
        adjust_primary_nine_test_family({"m1": result, "m2": result})


def test_primary_adjustment_contains_exactly_nine_tests():
    cell_ids, endpoint, clusters = _synthetic_w1()
    result = fit_w1_fnr_factorial(
        cell_ids=cell_ids,
        false_negative=endpoint,
        provenance_cluster_ids=clusters,
    )

    adjusted = adjust_primary_nine_test_family(
        {
            "llama_guard": result,
            "shieldgemma": result,
            "granite_guardian": result,
        }
    )

    assert len(adjusted) == 9
