import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_cluster_aware_inference_freeze():
    p = ROOT / "configs/external_validation_confirmatory_redesign_v2.json"
    x = json.loads(p.read_text())

    assert "cluster_aware_inference_calibration_contract" in x["completed_pre_W0_components"]
    assert x["completed_pre_W0_components"]["cluster_aware_inference_calibration_contract"]["status"] == "calibration_contract_frozen"

    blockers = x["mandatory_pre_W0_blockers"]
    assert "freeze_cluster_aware_primary_FNR_inference_after_calibration" not in blockers
    assert "freeze_cluster_aware_primary_FPR_inference_after_calibration" not in blockers
