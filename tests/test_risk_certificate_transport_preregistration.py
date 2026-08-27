import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads(
    (ROOT / "configs/risk_certificate_transport_preregistration_v1.json").read_text(
        encoding="utf-8"
    )
)
DOC = (ROOT / "docs/risk_certificate_transport_preregistration.md").read_text(
    encoding="utf-8"
).lower()
SCRIPT = (ROOT / "scripts/calculate_risk_certificate_sample_size.py").read_text(
    encoding="utf-8"
).lower()


def test_protocol_is_precollection_and_existing_discovery_is_closed():
    assert CFG["status"] == "prepared_for_review_before_fresh_data"
    assert CFG["base_commit"] == "06324465e5fe1c7fb614241fbc0c82f18e61dea3"
    assert CFG["existing_data_boundary"]["discovery_closed"] is True
    forbidden = set(CFG["existing_data_boundary"]["forbidden_uses"])
    for item in [
        "new hypothesis selection",
        "threshold tuning",
        "router rescue",
    ]:
        assert item in forbidden

    execution = CFG["execution_boundary"]
    assert execution["fresh_data_collection_started"] is False
    assert execution["new_labels_collected"] is False
    assert execution["fresh_monitor_scoring_started"] is False
    assert execution["transport_evaluation_started"] is False


def test_source_attack_family_and_time_domains_are_prespecified():
    d = CFG["domains"]
    assert d["A"] == {
        "source": "S_human",
        "attack_family": "F_direct",
        "window": "W0",
        "role": "optimization_and_certificate",
    }
    assert d["B"]["source"] == "S_model"
    assert d["B"]["attack_family"] == "F_direct"
    assert d["C"]["source"] == "S_human"
    assert d["C"]["attack_family"] == "F_obfuscated"
    assert d["D"]["source"] == "S_model"
    assert d["D"]["attack_family"] == "F_obfuscated"
    assert "non-overlapping" in CFG["time_separation"]
    assert "frozen before collection" in CFG["time_separation"]


def test_independent_multirater_labeling_is_frozen():
    labels = CFG["labeling"]
    assert labels["independent_raters_per_example"] == 3
    assert labels["primary_label"] == "majority_vote"
    assert "monitor_outputs" in labels["blinded_to"]
    assert labels["rater_cannot_label_own_authored_example"] is True
    assert "2_to_1_disagreement_rate" in labels["agreement_reporting"]


def test_dependency_groups_define_primary_independence():
    dep = CFG["dependency_handling"]
    assert dep["primary_independence_unit"] == "dependency_group_id/base_intent_id"
    assert dep["cross_domain_dependency_allowed"] is False
    assert dep["grouping_before_monitor_scoring"] is True
    assert dep["primary_certificate_uses_independent_groups"] is True
    assert dep["row_level_results_secondary_only"] is True


def test_monitor_families_are_heterogeneous_and_prescoring_lock_is_required():
    assert [x["model_id"] for x in CFG["monitor_panel"]] == [
        "meta-llama/Llama-Guard-3-1B",
        "google/shieldgemma-2b",
        "ibm-granite/granite-guardian-3.3-8b",
    ]
    assert len({x["family"] for x in CFG["monitor_panel"]}) == 3

    lock = CFG["pre_scoring_monitor_lock"]
    for value in lock.values():
        assert value is True


def test_fpr_calibration_certificate_and_transport_are_frozen():
    opt = CFG["calibration"]["A_optimization"]
    cert = CFG["calibration"]["A_certificate"]
    assert opt["minimum_independent_negative_units"] == 250
    assert "fpr <= 0.025" in opt["threshold_rule"].lower()
    assert opt["model_fitting_or_finetuning"] is False
    assert cert["minimum_independent_negative_units"] == 361
    assert cert["threshold_changes_allowed"] is False
    assert "clopper-pearson" in cert["certificate_rule"].lower()

    tr = CFG["transport"]
    assert tr["target_domains"] == ["B", "C", "D"]
    assert tr["minimum_independent_negative_units_per_domain"] == 361
    assert tr["monitor_and_threshold_applied_unchanged"] is True


def test_no_confirmation_retuning_is_hard_boundary():
    e = CFG["execution_boundary"]
    assert e["confirmation_data_retuning_allowed"] is False
    assert e["substantive_post_scoring_change_primary_status"] == "exploratory_only"


def test_protocol_document_contains_only_needed_methodological_sections():
    for heading in [
        "## 1. research question and study objective",
        "## 2. prospective domains and time separation",
        "## 3. independent multi-rater labels",
        "## 4. dependency and template-generated examples",
        "## 5. genuinely different monitor families",
        "## 6. calibration and 5% fpr certificate",
        "## 7. sample-size and power calculation",
        "## 8. certificate-transport analysis",
        "## 9. frozen execution and no-retuning rule",
    ]:
        assert heading in DOC

    for removed in [
        "protocol id:",
        "design stage:",
        "## 2. claim boundary",
        "## 11. technical failures and amendments",
        "## 12. current authorization state",
        "fresh data collected: no",
        "not authorized for data collection or evaluation",
    ]:
        assert removed not in DOC


def test_sample_size_script_is_analytic_only():
    assert "scipy.stats" in SCRIPT
    for forbidden in [
        "read_parquet",
        "monitor_score_cache",
        "unified_dataset",
        "final_test",
        "held_out_shift",
        "policy_latency_raw",
    ]:
        assert forbidden not in SCRIPT
