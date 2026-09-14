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
    u = c["generation_unit"]
    i = c["independent_batch_mechanism"]
    assert u["cross_cell_batch_reuse_allowed"] is False
    assert u["batch_continuation_or_hidden_state_carryover_allowed"] is False
    assert i["stateless_generation_required"] is True
    assert i["generator_batch_id_alone_is_not_sufficient_proof_of_independence"] is True


def test_unresolved_decoding_values_are_explicitly_not_invented():
    c = load("external_validation_source_generation_contract_v1.json")
    pending = c["unresolved_values_that_must_be_frozen_before_W0"]
    for key in [
        "source_instruction_text",
        "chat_template_contract",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "generation_seed_derivation",
        "rows_per_generator_batch",
    ]:
        assert key in pending
        assert pending[key] is None


def test_source_generation_remains_blocked_until_full_freeze():
    c = load("external_validation_source_generation_contract_v1.json")
    g = c["freeze_gate"]
    assert g["source_generation_structure_frozen"] is True
    assert g["source_generation_fully_frozen"] is False
    assert g["S_or_SF_generation_authorized"] is False
    assert g["blocker_remains_until_instruction_decoding_seed_and_batch_size_are_frozen"] is True


def test_population_contract_requires_final_source_generation_freeze():
    p = load("external_validation_population_sampling_contract_v1.json")
    m = p["independent_primary_sampling_units"]["model_generated"]
    assert m["source_generation_contract"] == "configs/external_validation_source_generation_contract_v1.json"
    assert m["count_as_independent_before_source_contract_final_freeze"] is False


def test_redesign_keeps_source_generation_blocker():
    r = load("external_validation_confirmatory_redesign_v2.json")
    assert "freeze_source_generation_contract" in r["mandatory_pre_W0_blockers"]
    item = r["in_progress_pre_W0_components"]["source_generation_contract"]
    assert item["blocker_remains"] is True
