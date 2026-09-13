import json
from pathlib import Path

CONFIG = Path("configs/external_validation_targeted_quota_confirmation_v1.json")


def load():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_claim_boundaries_are_preserved():
    cfg = load()
    b = cfg["boundaries"]
    assert b["relative_FNR_NI_margin"] == 0.03
    assert b["absolute_FNR_ceiling"] == 0.10
    assert b["FPR_constraint"] == 0.05


def test_candidate_patterns_have_equal_mean_and_bounded_sizes():
    cfg = load()
    d = cfg["cluster_design"]
    ref = d["reference_pattern"]
    cmp_ = d["comparison_stress_pattern"]
    assert min(ref + cmp_) >= 15
    assert max(ref + cmp_) <= 25
    assert sum(ref) / len(ref) == 20
    assert sum(cmp_) / len(cmp_) == 20


def test_relative_ni_keeps_exact_equal_cluster_counts():
    cfg = load()
    assert cfg["cluster_design"]["exact_equal_independent_provenance_cluster_counts_for_relative_NI"] is True


def test_final_power_rule_uses_lower_confidence_bound():
    cfg = load()
    p = cfg["power_gate"]
    assert p["planning_target"] == 0.80
    assert "lower confidence bound at least 0.80" in p["final_rule"]
    assert p["two_point_role"].startswith("stress sensitivity")


def test_w0_remains_blocked():
    assert load()["W0_remains_blocked"] is True
