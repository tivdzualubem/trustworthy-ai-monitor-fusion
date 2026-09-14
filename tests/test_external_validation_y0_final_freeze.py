import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_y0_contract_final_freeze():
    p = ROOT / "configs/external_validation_y0_sampling_contract_v1.json"
    x = json.loads(p.read_text())

    assert x["status"] == "frozen_precollection_component"
    assert x["Y0_strata_frozen"] is True
    assert x["Y0_proportions_frozen"] is True
    assert x["quota_definition_frozen"] is True
