import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_rater_pool_blocker_closed_after_freeze():
    p = ROOT / "configs/external_validation_confirmatory_redesign_v2.json"
    x = json.loads(p.read_text())

    assert "freeze_separate_rater_pool_contract" not in x["mandatory_pre_W0_blockers"]
    assert x["completed_pre_W0_components"]["separate_rater_pool_contract"]["status"] == "frozen"
