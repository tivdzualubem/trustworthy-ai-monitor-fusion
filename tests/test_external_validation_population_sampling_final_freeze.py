import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_population_sampling_contract_finalized():
    p = ROOT / "configs/external_validation_population_sampling_contract_v1.json"
    x = json.loads(p.read_text())

    assert x["status"] == "frozen_precollection_component"
    assert x["population_definition_frozen"] is True
    assert x["sampling_frames_frozen"] is True
