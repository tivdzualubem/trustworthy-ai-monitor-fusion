import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    return json.loads(
        (ROOT / "configs" / "external_validation_ontology_crosswalk_contract_v1.json").read_text()
    )


def test_ontology_contract_is_frozen():
    x = load()
    assert x["status"] == "frozen_precollection_component"
    assert x["common_ontology"]["label_space"]["Y0"] == "not_harmful"
    assert x["common_ontology"]["label_space"]["Y1"] == "harmful"


def test_qwen3guard_mapping_is_fixed():
    x = load()
    q = x["monitor_crosswalks"]["Qwen/Qwen3Guard-Gen-4B"]
    assert q["mapping"]["Safe"]["Y"] == 0
    assert q["mapping"]["Controversial"]["Y"] == 1
    assert q["mapping"]["Unsafe"]["Y"] == 1
    assert q["score_mapping"]["Unsafe"] == 1.0


def test_post_collection_reinterpretation_is_blocked():
    x = load()
    assert x["freeze_rules"]["ontology_changes_after_W0"] is False
    assert x["freeze_rules"]["monitor_specific_mapping_changes_after_W0"] is False
