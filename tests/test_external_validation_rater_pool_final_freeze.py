import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_rater_pool_final_freeze():
    p = ROOT / "configs/external_validation_rater_pool_contract_v1.json"
    x = json.loads(p.read_text())

    assert x["status"] == "frozen_precollection_component"
    assert x["role_separation_frozen"] is True
    assert x["recruitment_ethics_frozen"] is True
