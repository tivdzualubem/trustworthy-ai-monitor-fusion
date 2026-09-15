import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(rel):
    return json.loads((ROOT / rel).read_text())

def test_scientific_blockers_are_restored_before_W0():
    x = load("configs/external_validation_confirmatory_redesign_v2.json")
    blockers = set(x["mandatory_pre_W0_blockers"])

    required = {
        "freeze_cluster_aware_primary_FNR_inference_after_calibration",
        "freeze_cluster_aware_primary_FPR_inference_after_calibration",
        "complete_full_cluster_aware_type_I_coverage_and_power_grid",
        "freeze_Y1_Y0_quotas_cluster_minima_cluster_caps_and_validation_candidate_caps",
        "freeze_terminal_batch_rule",
        "freeze_required_score_CDF_margin_and_threshold_jitter_diagnostics",
        "complete_final_precollection_manifest",
        "freeze_exact_human_sampling_frame_and_collection_windows",
        "resolve_human_recruitment_ethics_consent_and_pool_assignment_before_recruitment",
    }
    assert required <= blockers

def test_tracker_does_not_call_unvalidated_inference_power_or_quotas_frozen():
    x = load("configs/external_validation_confirmatory_redesign_v2.json")
    done = x["completed_pre_W0_components"]

    assert done["cluster_aware_inference_calibration_contract"]["status"] == \
        "pending_primary_method_validation"
    assert done["cluster_aware_type_I_power_grid"]["status"] == \
        "not_validated_historical_noninferential_artifacts_only"
    assert done["cluster_aware_type_I_power_grid"]["do_not_use_for_W0"] is True
    assert done["Y1_Y0_quota_cap_contract"]["frozen"] is False
    assert done["terminal_batch_rule_contract"]["frozen"] is False
    assert done["score_cdf_threshold_jitter_contract"]["frozen"] is False
    assert done["final_precollection_manifest_contract"]["frozen"] is False

def test_repo_still_records_quota_and_primary_inference_as_pending():
    prereg = load("configs/safety_monitor_external_validation_preregistration_v1.json")
    text = json.dumps(prereg)
    assert '"Y1_quota": null' in text
    assert '"Y0_quota": null' in text
    assert "pending_pre_W0_calibration" in text
    assert "blocked until Y1/Y0 quotas" in text

    stopping = load("configs/external_validation_sampling_stopping_contract_v1.json")
    stext = json.dumps(stopping)
    assert "pending_full_cluster_aware_design_grid" in stext
    assert "no validation collection authorization" in stext

def test_failed_inference_validation_and_historical_power_are_respected():
    inf = load("configs/external_validation_final_inference_validation_v1.json")
    assert "not frozen" in json.dumps(inf).lower()

    hist = load("configs/external_validation_exploratory_power_artifact_status_v1.json")
    assert hist["status"] == "historical_noninferential_artifacts"
    assert "W0 authorization" in hist["do_not_use_for"]

def test_human_sampling_and_recruitment_gates_are_not_false_frozen():
    pop = load("configs/external_validation_population_sampling_contract_v1.json")
    assert pop["status"] == "population_definition_frozen_sampling_frames_pending"

    raters = load("configs/external_validation_rater_pool_contract_v1.json")
    assert raters["status"] == "role_separation_frozen_recruitment_ethics_pending"

def test_readiness_is_retracted_until_scientific_gates_pass():
    readiness = load("data/metadata/external_validation_final_precollection_readiness_v1.json")
    assert readiness["status"] == "not_ready_for_W0_unresolved_scientific_blockers"
    assert readiness["freeze_contracts_verified"] is False
    assert readiness["readiness_retracted_by_truth_audit"] is True

    repair = load("data/metadata/external_validation_preW0_truth_repair_v1.json")
    assert repair["W0_authorized"] is False
    assert repair["qwen3guard_preflight_status"] == "already_passed_no_rerun_required"
