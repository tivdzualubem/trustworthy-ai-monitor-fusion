import json
from pathlib import Path

import pytest

from monitor_fusion.external_validation.source_generation import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    NUM_RETURN_SEQUENCES,
    REPETITION_PENALTY,
    ROWS_PER_GENERATOR_BATCH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    build_source_generation_spec,
    source_seed,
    validate_batch_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_contract_is_fully_frozen():
    c = load("external_validation_source_generation_contract_v1.json")
    assert c["status"] == "frozen_precollection_component"
    assert c["freeze_gate"]["source_generation_fully_frozen"] is True
    assert c["freeze_gate"]["source_generation_contract_blocker_closed"] is True


def test_decoding_constants_match_contract():
    c = load("external_validation_source_generation_contract_v1.json")
    d = c["decoding"]
    assert MAX_NEW_TOKENS == d["max_new_tokens"] == 256
    assert DO_SAMPLE is d["do_sample"] is True
    assert TEMPERATURE == d["temperature"] == 0.35
    assert TOP_P == d["top_p"] == 1.0
    assert TOP_K == d["top_k"] == 0
    assert REPETITION_PENALTY == d["repetition_penalty"] == 1.0
    assert NUM_RETURN_SEQUENCES == d["num_return_sequences"] == 1
    assert ROWS_PER_GENERATOR_BATCH == c["independent_batch_mechanism"]["rows_per_generator_batch"] == 1


def test_source_seed_is_deterministic_and_batch_specific():
    a = source_seed("S-batch-000001")
    b = source_seed("S-batch-000001")
    c = source_seed("S-batch-000002")
    assert a == b
    assert a != c
    assert 0 <= a < 2_147_483_647


def test_generation_spec_uses_one_system_and_one_user_message():
    spec = build_source_generation_spec(
        generator_batch_id="S-batch-000001",
        category_id="C4",
        category_definition="Cyber abuse category definition.",
    )
    assert spec.seed == source_seed("S-batch-000001")
    assert [m["role"] for m in spec.messages] == ["system", "user"]
    assert "C4" in spec.messages[1]["content"]
    assert "Output exactly one user request" in spec.messages[1]["content"]


def test_batch_ids_must_be_unique():
    validate_batch_ids(["S-batch-1", "SF-batch-1"])
    with pytest.raises(ValueError):
        validate_batch_ids(["S-batch-1", "S-batch-1"])


def test_population_contract_now_recognizes_validated_single_row_model_blocks():
    p = load("external_validation_population_sampling_contract_v1.json")
    m = p["independent_primary_sampling_units"]["model_generated"]
    assert m["source_generation_contract_status"] == "frozen_precollection_component"
    assert m["count_as_independent_after_contract_validation"] is True
    assert m["rows_per_generator_batch"] == 1


def test_redesign_source_generation_blocker_is_closed():
    r = load("external_validation_confirmatory_redesign_v2.json")
    assert "freeze_source_generation_contract" not in r["mandatory_pre_W0_blockers"]
    done = r["completed_pre_W0_components"]["source_generation_contract"]
    assert done["status"] == "frozen_precollection_component"
