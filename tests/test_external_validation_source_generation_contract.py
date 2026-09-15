import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_source_model_identity_matches_runtime_registry():
    c = load("external_validation_source_generation_contract_v1.json")
    r = load("external_validation_model_runtime_registry_v1.json")
    expected = r["models"]["model_generated_source"]
    assert c["model"]["model_id"] == expected["model_id"]
    assert c["model"]["revision"] == expected["revision"]


def test_source_batches_are_stateless_and_never_cross_cells():
    c = load("external_validation_source_generation_contract_v1.json")
    b = c["independent_batch_mechanism"]
    assert b["rows_per_generator_batch"] == 1
    assert b["generation_batch_size"] == 1
    assert b["cross_cell_batch_reuse_allowed"] is False
    assert b["stateless_generation_required"] is True
    assert b["cross_batch_conversation_state_allowed"] is False
    assert b["source_output_from_one_batch_may_condition_another_batch"] is False


def test_decoding_and_instruction_are_now_explicitly_frozen():
    c = load("external_validation_source_generation_contract_v1.json")
    assert c["status"] == "frozen_precollection_component"

    d = c["decoding"]
    assert d["max_new_tokens"] == 256
    assert d["do_sample"] is True
    assert d["temperature"] == 0.35
    assert d["top_p"] == 1.0
    assert d["top_k"] == 0
    assert d["repetition_penalty"] == 1.0
    assert d["num_return_sequences"] == 1

    s = c["source_instruction"]
    assert s["system"]
    assert s["user_template"]
    assert s["one_candidate_per_generation_call"] is True


def test_source_generation_is_fully_frozen_but_global_W0_gate_remains():
    c = load("external_validation_source_generation_contract_v1.json")
    g = c["freeze_gate"]
    assert g["source_generation_fully_frozen"] is True
    assert g["source_generation_contract_blocker_closed"] is True
    assert g["W0_collection_authorized_by_this_contract"] is False
    assert g["S_or_SF_generation_authorized_only_after_all_global_preW0_blockers_close"] is True


def test_population_contract_links_final_source_generation_freeze():
    p = load("external_validation_population_sampling_contract_v1.json")
    m = p["independent_primary_sampling_units"]["model_generated"]
    assert m["source_generation_contract"] == "configs/external_validation_source_generation_contract_v1.json"
    assert m["source_generation_contract_status"] == "frozen_precollection_component"
    assert m["count_as_independent_after_contract_validation"] is True
    assert m["rows_per_generator_batch"] == 1


def test_redesign_closes_only_source_generation_blocker():
    r = load("external_validation_confirmatory_redesign_v2.json")
    assert "freeze_source_generation_contract" not in r["mandatory_pre_W0_blockers"]
    done = r["completed_pre_W0_components"]["source_generation_contract"]
    assert done["status"] == "frozen_precollection_component"

    # Other unresolved scientific blockers must remain; this contract alone never authorizes W0.
    assert "freeze_cluster_aware_primary_FNR_inference_after_calibration" in r["mandatory_pre_W0_blockers"]
    assert "freeze_cluster_aware_primary_FPR_inference_after_calibration" in r["mandatory_pre_W0_blockers"]

    # The contemporary comparator blocker was legitimately closed later by the
    # exact-revision Qwen3Guard contract plus passing Kaggle preflight evidence.
    assert "add_and_pin_contemporary_comparator_monitor_or_monitors" not in r["mandatory_pre_W0_blockers"]
    panel = r["completed_pre_W0_components"]["contemporary_monitor_panel"]
    assert panel["status"] == "frozen_qwen3guard_comparator"
    assert panel["revision"] == "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
