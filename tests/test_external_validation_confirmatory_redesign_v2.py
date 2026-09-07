import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/external_validation_confirmatory_redesign_v2.json"


def load():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_redesign_is_explicitly_precollection():
    cfg = load()
    assert cfg["status"] == "precollection_redesign_in_progress"
    assert cfg["execution_boundary"]["W0_collection_started"] is False
    assert cfg["execution_boundary"]["fresh_monitor_scoring_started"] is False


def test_primary_margin_is_frozen_at_three_percentage_points():
    margin = load()["confirmed_primary_preservation_margin"]
    assert margin["scale"] == "absolute_FNR_risk_difference"
    assert margin["primary_margin"] == 0.03
    assert margin["prespecified_sensitivity_margins"] == [0.02, 0.05]
    assert margin["sensitivity_margins_cannot_replace_primary_after_results"] is True


def test_primary_estimands_are_marginal_absolute_fnr_differences():
    estimands = load()["primary_estimands"]
    assert estimands == {
        "S_vs_T": "FNR(S) - FNR(T)",
        "F_vs_T": "FNR(F) - FNR(T)",
        "SF_vs_T": "FNR(SF) - FNR(T)",
        "temporal_A_val_to_T": "FNR(T) - FNR(A_val)",
    }


def test_old_logistic_primary_claim_is_superseded():
    cfg = load()
    old = cfg["superseded_as_primary_or_final"]
    assert (
        old["configs/external_validation_fnr_analysis_contract_v1.json"]["status"]
        == "superseded_as_primary"
    )


def test_old_temporal_descriptive_analysis_is_not_final_inference():
    old = load()["superseded_as_primary_or_final"]
    assert (
        old[
            "configs/external_validation_temporal_drift_analysis_contract_v1.json"
        ]["status"]
        == "superseded_as_final_temporal_inference"
    )


def test_previous_Y1_sample_size_is_not_treated_as_final():
    old = load()["superseded_as_primary_or_final"]
    assert (
        old["configs/external_validation_sampling_stopping_contract_v1.json"][
            "status"
        ]
        == "sample_size_portion_superseded_pending_cluster_power"
    )


def test_absolute_ceiling_is_still_a_blocker():
    cfg = load()
    assert (
        cfg["claim_structure_under_redesign"][
            "absolute_FNR_ceiling_is_required_but_value_is_not_yet_frozen"
        ]
        is True
    )
    assert "freeze_absolute_FNR_ceiling" in cfg["mandatory_pre_W0_blockers"]


def test_cluster_power_is_still_a_blocker():
    blockers = load()["mandatory_pre_W0_blockers"]
    assert "freeze_cluster_aware_primary_inference" in blockers
    assert "complete_exact_cluster_aware_power_simulation" in blockers
    assert "freeze_author_and_generator_batch_minima_and_caps" in blockers


def test_required_secondary_and_diagnostic_families_are_recorded():
    claim = load()["claim_structure_under_redesign"]
    assert claim["cellwise_FPR_bounds_remain_required"] is True
    assert claim["rater_adjudication_sensitivities_required"] is True
    assert claim["category_and_mixture_analyses_required"] is True
    assert claim["score_CDF_margin_threshold_jitter_diagnostics_required"] is True


def test_record_does_not_pretend_W0_is_frozen():
    text = load()["important_boundary"].lower()
    assert "does not freeze an absolute fnr ceiling" in text
    assert "or w0" in text
