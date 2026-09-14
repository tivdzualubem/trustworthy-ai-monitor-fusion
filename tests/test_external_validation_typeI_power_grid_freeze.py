import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_typeI_power_grid_completed():
    x = json.loads((ROOT / "configs/external_validation_confirmatory_redesign_v2.json").read_text())

    assert "cluster_aware_type_I_power_grid" in x["completed_pre_W0_components"]
    assert "complete_full_cluster_aware_type_I_coverage_and_power_grid" not in x["mandatory_pre_W0_blockers"]
