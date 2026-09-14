import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_annotation_adjudication_contract_frozen():
    x = json.loads((ROOT / "configs/external_validation_confirmatory_redesign_v2.json").read_text())

    assert "annotation_adjudication_consequence_contract" in x["completed_pre_W0_components"]
    assert "revise_annotation_adjudication_and_unanimous_audit_consequences" not in x["mandatory_pre_W0_blockers"]
