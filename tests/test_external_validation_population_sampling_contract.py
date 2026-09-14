import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_population_and_estimand_are_explicit():
    c = load("external_validation_population_sampling_contract_v1.json")
    assert c["status"] == "population_definition_frozen_sampling_frames_pending"
    assert c["scientific_estimands"]["weighting"] == "response_average_marginal"
    assert "Y=1" in c["scientific_estimands"]["primary_FNR_population"]
    assert "Y=0" in c["scientific_estimands"]["FPR_population"]


def test_block_identifiers_do_not_automatically_establish_independence():
    c = load("external_validation_population_sampling_contract_v1.json")
    units = c["independent_primary_sampling_units"]
    assert units["row_level_independence_assumption_allowed"] is False
    assert units["cross_cell_reuse_of_primary_sampling_unit_allowed"] is False
    assert units["human"]["author_id_alone_is_sufficient_proof_of_independence"] is False
    assert units["model_generated"]["generator_batch_id_alone_is_sufficient_proof_of_independence"] is False


def test_collection_remains_blocked_until_sampling_and_analysis_are_frozen():
    c = load("external_validation_population_sampling_contract_v1.json")
    gate = c["freeze_gate"]
    assert gate["W0_collection_authorized"] is False
    assert gate["fresh_monitor_scoring_authorized"] is False
    assert gate["sampling_frame_must_be_frozen_before_W0"] is True
    assert gate["primary_analysis_and_joint_power_must_be_validated_before_quota_freeze"] is True


def test_recent_raw_inequality_power_artifacts_are_not_confirmatory():
    c = load("external_validation_exploratory_power_artifact_status_v1.json")
    assert c["status"] == "historical_noninferential_artifacts"
    assert "confirmatory sample-size freeze" in c["do_not_use_for"]
    scripts = {x["script"]: x for x in c["artifacts"]}
    assert scripts["scripts/run_external_validation_ni_operating_characteristics_v1.py"]["status"] == "noninferential_exploratory"
    assert scripts["scripts/run_external_validation_joint_operating_characteristics_v1.py"]["status"] == "noninferential_exploratory"


def test_confirmatory_design_points_to_population_sampling_contract():
    c = load("external_validation_confirmatory_design_v3.json")
    assert c["population_sampling_contract"]["path"] == "configs/external_validation_population_sampling_contract_v1.json"
    assert c["independent_sampling"]["candidate_identifier_alone_is_sufficient_proof_of_independence"] is False
    assert c["analysis_hierarchy"]["exploratory_raw_inequality_power_artifacts"]["status"] == "noninferential_do_not_use_for_quota_freeze"
