import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load():
    return json.loads(
        (ROOT / "configs/external_validation_annotation_readiness_contract_v1.json").read_text()
    )

def test_annotation_contract_frozen():
    x = load()
    assert x["status"] == "frozen_precollection_component"
    assert x["annotation_protocol"]["rater_blinding_required"] is True

def test_label_redefinition_is_blocked():
    x = load()
    assert x["annotation_protocol"]["post_collection_label_redefinition_allowed"] is False

def test_disagreement_policy_preserves_original_labels():
    x = load()
    assert x["disagreement_policy"]["disagreement_recorded"] is True
    assert x["disagreement_policy"]["adjudication_changes_original_labels"] is False
