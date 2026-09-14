import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_y0_population_is_cell_specific_safe_response_population():
    c = load("external_validation_y0_sampling_contract_v1.json")
    assert c["status"] == "Y0_strata_and_proportions_frozen_quota_pending"
    assert "Y=0" in c["Y0_population"]["definition"]
    assert c["Y0_population"]["not_a_separate_benign_prompt_population"] is True


def test_y0_category_mix_is_uniform_and_common_across_cells():
    c = load("external_validation_y0_sampling_contract_v1.json")
    s = c["primary_stratification"]
    assert s["categories"] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    assert s["target_proportion_per_category"] == 0.125
    assert s["same_category_vector_across_cells"] is True


def test_response_style_is_not_a_posthoc_quota_dimension():
    c = load("external_validation_y0_sampling_contract_v1.json")
    assert c["response_style"]["separate_quota_strata"] is False
    assert "not retrospectively quota-balanced" in c["response_style"]["rule"]


def test_y0_topup_cannot_use_monitor_results():
    c = load("external_validation_y0_sampling_contract_v1.json")
    t = c["counting_and_topup"]
    assert t["top_up_only_before_fresh_monitor_scoring"] is True
    assert t["monitor_output_based_top_up_forbidden"] is True
    assert t["FPR_based_top_up_forbidden"] is True
    assert t["post_scoring_replacement_forbidden"] is True


def test_sampling_stopping_contract_points_to_y0_contract():
    s = load("external_validation_sampling_stopping_contract_v1.json")
    y0 = s["Y0_sampling_contract"]
    assert y0["path"] == "configs/external_validation_y0_sampling_contract_v1.json"
    assert y0["target_proportion_per_category"] == 0.125
    assert y0["same_category_vector_across_cells"] is True
