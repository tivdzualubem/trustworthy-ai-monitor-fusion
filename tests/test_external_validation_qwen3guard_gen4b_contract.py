import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_qwen3guard_identity_and_revision_are_exact():
    c = load("external_validation_qwen3guard_gen4b_contract_v1.json")
    assert c["model"]["model_id"] == "Qwen/Qwen3Guard-Gen-4B"
    assert c["model"]["revision"] == "6ec42827da0c1ff11e7a49dc269d2e810d27e108"


def test_qwen3guard_confirmatory_input_is_response_moderation():
    c = load("external_validation_qwen3guard_gen4b_contract_v1.json")
    i = c["input_contract"]
    assert i["mode"] == "prompt_response"
    assert [m["role"] for m in i["messages"]] == ["user", "assistant"]
    assert i["response_only_mode_allowed_for_confirmatory_scoring"] is False
    assert i["prompt_only_mode_allowed_for_confirmatory_scoring"] is False


def test_qwen3guard_parser_and_score_mapping_are_frozen():
    c = load("external_validation_qwen3guard_gen4b_contract_v1.json")
    assert "Safe|Unsafe|Controversial" in c["output_parser"]["safety_pattern"]
    assert c["score_contract"]["mapping"] == {
        "Safe": 0.0,
        "Controversial": 0.5,
        "Unsafe": 1.0,
    }
    assert c["score_contract"]["direction"] == "larger_is_more_harmful"
    assert c["score_contract"]["threshold_retuning_on_validation_cells_allowed"] is False


def test_qwen3guard_runtime_preflight_passed_and_scoring_still_globally_blocked():
    c = load("external_validation_qwen3guard_gen4b_contract_v1.json")
    assert c["status"] == "frozen_precollection_component"
    assert c["runtime"]["preflight_passed"] is True
    assert c["runtime"]["runtime_dtype"] == "float16"
    assert c["runtime"]["gpu_layout"] == "single_gpu_cuda0"
    assert c["gate"]["runtime_preflight_satisfied"] is True
    assert c["gate"]["fresh_confirmatory_scoring_authorized"] is False


def test_runtime_registry_contains_verified_qwen3guard():
    r = load("external_validation_model_runtime_registry_v1.json")
    hits = [
        m for m in r["models"]["safety_monitors"]
        if m["model_id"] == "Qwen/Qwen3Guard-Gen-4B"
    ]
    assert len(hits) == 1
    assert hits[0]["revision"] == "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
    assert hits[0]["preflight_passed"] is True
    assert hits[0]["status"] == "verified_precollection_component"


def test_prereg_panel_is_frozen_with_qwen3guard():
    p = load("safety_monitor_external_validation_preregistration_v1.json")
    ids = [m["model_id"] for m in p["monitors"]]
    assert "Qwen/Qwen3Guard-Gen-4B" in ids
    assert p["monitor_panel_status"] == "frozen_with_contemporary_qwen3guard_crosswalk_pending"


def test_redesign_closes_comparator_blocker_but_keeps_crosswalk_blocker():
    r = load("external_validation_confirmatory_redesign_v2.json")
    assert "add_and_pin_contemporary_comparator_monitor_or_monitors" not in r["mandatory_pre_W0_blockers"]
    assert "review_and_freeze_monitor_native_to_common_ontology_crosswalk" in r["mandatory_pre_W0_blockers"]
    done = r["completed_pre_W0_components"]["contemporary_monitor_panel"]
    assert done["status"] == "frozen_qwen3guard_comparator"
