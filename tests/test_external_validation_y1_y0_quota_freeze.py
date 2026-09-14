import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_y1_y0_quota_caps_frozen():
    x = json.loads((ROOT / "configs/external_validation_confirmatory_redesign_v2.json").read_text())

    assert "Y1_Y0_quota_cap_contract" in x["completed_pre_W0_components"]
    assert "freeze_Y1_Y0_quotas_cluster_minima_cluster_caps_and_validation_candidate_caps" not in x["mandatory_pre_W0_blockers"]
