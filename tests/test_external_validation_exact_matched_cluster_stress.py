from scripts.run_external_validation_exact_matched_cluster_stress import (
    PAIR_SCENARIOS,
    sizes_from_pattern,
)


def test_every_relative_scenario_has_exactly_matched_cluster_counts():
    for _, g, ref_pattern, cmp_pattern, *_ in PAIR_SCENARIOS:
        ref = sizes_from_pattern(g, ref_pattern)
        cmp_ = sizes_from_pattern(g, cmp_pattern)
        assert len(ref) == g
        assert len(cmp_) == g
        assert len(ref) == len(cmp_)


def test_stress_contains_different_cluster_size_distributions():
    assert any(
        ref_pattern != cmp_pattern
        for _, _, ref_pattern, cmp_pattern, *_ in PAIR_SCENARIOS
    )


def test_stress_contains_low_fnr_boundary_case():
    assert any(
        baseline == 0.02
        for _, _, _, _, baseline, _, _ in PAIR_SCENARIOS
    )
