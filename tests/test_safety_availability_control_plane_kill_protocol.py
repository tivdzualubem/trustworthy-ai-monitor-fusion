import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "configs/safety_availability_control_plane_kill_study_v1.json"


def load():
    return json.loads(P.read_text(encoding="utf-8"))


def test_ordered_prerequisites_and_scope():
    p = load()
    assert all(p["ordered_prerequisites"].values())
    assert p["scope"]["development_only"] is True
    assert p["scope"]["protected_legacy_splits_used"] is False
    assert p["scope"]["new_model_training"] is False
    assert p["scope"]["router_tuning_or_rescue"] is False


def test_both_adversarial_pressure_families_are_present():
    p = load()
    assert "routing_bypass_pressure" in p
    assert "routing_escalation_pressure" in p
    assert "load_escalation_pressure" in p


def test_three_control_strategies_are_frozen():
    p = load()
    assert p["load_escalation_pressure"]["strategies"] == [
        "fail_open_budget_cap",
        "fail_closed_no_admission",
        "fail_closed_defer_reject",
    ]


def test_internal_novelty_rule_is_prespecified_and_not_literature_claim():
    p = load()
    k = p["internal_novelty_kill_rule"]
    assert k["deadband_scale_incremental_bypass_threshold"] == 0.01
    assert k["minimum_stacks_for_nontrivial_signal"] == 2
    assert k["literature_novelty_claim"] is False
