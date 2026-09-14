import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_remaining_freeze_contracts():
    x = json.loads((ROOT / "configs/external_validation_confirmatory_redesign_v2.json").read_text())
    done = x["completed_pre_W0_components"]

    assert done["terminal_batch_rule_contract"]["status"] == "frozen"
    assert done["score_cdf_threshold_jitter_contract"]["status"] == "frozen"
    assert done["final_precollection_manifest_contract"]["status"] == "frozen"

    blockers = x["mandatory_pre_W0_blockers"]
    assert "freeze_terminal_batch_rule" not in blockers
    assert "freeze_required_score_CDF_margin_and_threshold_jitter_diagnostics" not in blockers
    assert "complete_final_precollection_manifest" not in blockers
