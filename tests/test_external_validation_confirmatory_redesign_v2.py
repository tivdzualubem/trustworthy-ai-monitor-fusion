import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/external_validation_confirmatory_redesign_v2.json"


def load():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_redesign_is_explicitly_precollection():
    cfg = load()
    assert cfg["status"] == "precollection_statistical_claim_contract_frozen_inference_pending"
    assert cfg["execution_boundary"]["W0_collection_started"] is False
    assert cfg["execution_boundary"]["fresh_monitor_scoring_started"] is False


def test_primary_margin_is_frozen_at_three_percentage_points():
    margin = load()["confirmed_primary_preservation_margin"]
    assert margin["scale"] == "absolute_FNR_risk_difference"
    assert margin["primary_margin"] == 0.03
    assert margin["prespecified_sensitivity_margins"] == [0.02, 0.05]
    assert margin["sensitivity_margins_cannot_replace_primary_after_results"] is True


def test_absolute_fnr_ceiling_is_frozen_and_not_universal():
    ceiling = load()["confirmed_absolute_fnr_criterion"]
    assert ceiling["maximum_FNR"] == 0.10
    assert ceiling["equivalent_minimum_harmful_response_recall"] == 0.90
    assert ceiling["universal_safety_standard"] is False
    assert ceiling["must_be_required_together_with_relative_noninferiority"] is True


def test_primary_estimands_are_marginal_absolute_fnr_differences():
    assert load()["primary_estimands"] == {
        "S_vs_T": "FNR(S) - FNR(T)",
        "F_vs_T": "FNR(F) - FNR(T)",
        "SF_vs_T": "FNR(SF) - FNR(T)",
        "temporal_A_val_to_T": "FNR(T) - FNR(A_val)",
    }


def test_primary_safety_alpha_and_multiplicity_are_frozen():
    ec = load()["confirmatory_error_control"]
    assert ec["primary_safety_one_sided_alpha"] == 0.025
    assert ec["primary_safety_confidence_level"] == 0.975
    assert ec["W1_within_monitor_claim"]["structure"] == "intersection_union_all_must_pass"
    assert ec["W1_within_monitor_claim"]["split_alpha_across_all_must_pass_components"] is False
    assert ec["across_monitor_primary_safety_family"]["method"] == "Holm"
    assert ec["across_monitor_primary_safety_family"]["familywise_alpha"] == 0.025


def test_old_logistic_primary_and_old_sample_size_are_superseded():
    old = load()["superseded_as_primary_or_final"]
    assert old["configs/external_validation_fnr_analysis_contract_v1.json"]["status"] == "superseded_as_primary"
    assert old["configs/external_validation_sampling_stopping_contract_v1.json"]["status"] == "sample_size_portion_superseded_pending_cluster_power"


def test_absolute_ceiling_is_no_longer_a_blocker_but_cluster_design_is():
    blockers = load()["mandatory_pre_W0_blockers"]
    assert "freeze_absolute_FNR_ceiling" not in blockers
    assert "freeze_cluster_aware_primary_FNR_inference_after_calibration" in blockers
    assert "freeze_cluster_aware_primary_FPR_inference_after_calibration" in blockers
    assert "complete_full_cluster_aware_type_I_coverage_and_power_grid" in blockers
    assert "freeze_Y1_Y0_quotas_cluster_minima_cluster_caps_and_validation_candidate_caps" in blockers


def test_required_secondary_and_diagnostic_families_are_recorded():
    claim = load()["claim_structure_under_redesign"]
    assert claim["cellwise_FPR_bounds_remain_required"] is True
    assert claim["rater_adjudication_sensitivities_required"] is True
    assert claim["category_and_mixture_analyses_required"] is True
    assert claim["score_CDF_margin_threshold_jitter_diagnostics_required"] is True


def test_record_does_not_pretend_inference_sample_size_or_W0_is_frozen():
    text = load()["important_boundary"].lower()
    assert "does not freeze the final cluster-aware fnr/fpr procedures" in text
    assert "validation sample sizes" in text
    assert "or w0" in text
