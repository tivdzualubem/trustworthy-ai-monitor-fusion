import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_obfuscation_contract_is_frozen_and_registry_backed():
    c = load("external_validation_obfuscation_generation_contract_v1.json")
    assert c["status"] == "frozen_precollection_component"
    assert c["authoritative_registry"]["registry_id"] == "external_validation_attack_family_registry_v1"
    assert c["generation_rule"]["transform_set"] == ["O1", "O2", "O3"]


def test_assignment_is_deterministic_and_nonadaptive():
    c = load("external_validation_obfuscation_generation_contract_v1.json")
    g = c["generation_rule"]
    assert "SHA256(base_intent_id)" in g["selection_rule"]
    assert g["exactly_one_transform_required"] is True
    assert g["adaptive_transform_selection_allowed"] is False
    assert g["monitor_output_based_selection_allowed"] is False
    assert g["target_response_based_selection_allowed"] is False
    assert g["transform_stacking_allowed"] is False
    assert g["manual_post_transform_editing_allowed"] is False


def test_dependency_text_remains_pre_obfuscation_text():
    c = load("external_validation_obfuscation_generation_contract_v1.json")
    assert c["eligibility_and_provenance"]["dependency_text_is_pre_obfuscation_text"] is True
    assert c["eligibility_and_provenance"]["failed_or_invalid_transform_excluded_before_monitor_scoring"] is True


def test_redesign_marks_obfuscation_generation_complete():
    c = load("external_validation_confirmatory_redesign_v2.json")
    done = c["completed_pre_W0_components"]["obfuscation_generation_contract"]
    assert done["status"] == "frozen_precollection_component"
    assert "freeze_obfuscation_generation_contract" not in c["mandatory_pre_W0_blockers"]
