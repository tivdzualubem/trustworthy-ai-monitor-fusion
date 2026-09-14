import json
from pathlib import Path

CONFIG = Path("configs/external_validation_cluster_power_typeI_grid_v2.json")


def load():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_grid_preserves_frozen_claim_numbers():
    cfg = load()
    b = cfg["boundaries"]
    assert b["relative_FNR_NI_margin"] == 0.03
    assert b["absolute_FNR_ceiling"] == 0.10
    assert b["FPR_constraint"] == 0.05
    assert b["safety_one_sided_alpha"] == 0.025
    assert b["FPR_one_sided_alpha"] == 0.05


def test_tight_candidate_envelope_is_bounded_and_centered():
    cfg = load()
    d = cfg["candidate_cluster_design"]
    assert d["tight_cluster_size_pattern"] == [15, 18, 20, 22, 25]
    assert d["candidate_minimum"] == 15
    assert d["candidate_cap"] == 25
    assert d["mean_cluster_size"] == 20
    assert d["frozen"] is False


def test_power_target_is_project_continuity_not_professor_attribution():
    cfg = load()
    p = cfg["power_planning"]
    assert p["primary_target"] == 0.80
    assert "professor" in p["target_origin"]
    assert "did not replace" in p["target_origin"]
    assert p["report_90_percent_sensitivity"] is True


def test_old_kc_power_screen_is_not_final():
    cfg = load()
    assert "historical only" in cfg["historical_power_code_status"]
    assert cfg["W0_remains_blocked"] is True
