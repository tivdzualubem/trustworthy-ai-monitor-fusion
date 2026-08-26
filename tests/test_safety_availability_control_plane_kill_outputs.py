import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/safety_availability_control_plane_kill_study_v1"


def test_required_outputs_cover_bypass_and_load():
    bypass = pd.read_csv(OUT / "routing_bypass_sweep.csv")
    escalation = pd.read_csv(OUT / "routing_escalation_sweep.csv")
    load = pd.read_csv(OUT / "load_capacity_sweep.csv")
    assert set(bypass["stack"]) == {
        "rule_compact_to_qwen",
        "compact_to_qwen",
        "rule_to_compact",
    }
    assert set(escalation["stack"]) == set(bypass["stack"])
    assert set(load["strategy"]) == {
        "fail_open_budget_cap",
        "fail_closed_no_admission",
        "fail_closed_defer_reject",
    }


def test_overload_exposes_the_expected_control_tradeoff():
    load = pd.read_csv(OUT / "load_capacity_sweep.csv")
    worst = load[load["attack_mode"] == "worst_case_all_escalate"]
    no_admission = worst[worst["strategy"] == "fail_closed_no_admission"]
    fail_open = worst[worst["strategy"] == "fail_open_budget_cap"]
    deferred = worst[worst["strategy"] == "fail_closed_defer_reject"]

    assert (~no_admission["queue_stable"].astype(bool)).any()
    assert (fail_open["expensive_bypass_fraction"] > 0).any()
    assert (deferred["defer_reject_fraction"] > 0).any()
    assert deferred["fail_closed_safety_semantics_preserved"].astype(bool).all()
    assert deferred["resource_bound_preserved"].astype(bool).all()


def test_summary_keeps_claim_boundary_and_internal_decision_scope():
    s = json.loads((OUT / "summary.json").read_text())
    assert s["status"] == "completed_development_only"
    assert s["routing_bypass_pressure_completed"] is True
    assert s["load_escalation_pressure_completed"] is True
    assert s["literature_novelty_claim"] is False
    assert s["claim_boundary"]["protected_legacy_splits_used"] is False
    assert s["next_step"] == "integrate_ordered_studies_and_reassess_paper_direction"
