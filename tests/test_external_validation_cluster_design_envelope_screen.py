import json
from pathlib import Path

CONFIG = Path("configs/external_validation_cluster_design_envelope_screen_v1.json")


def load():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_screen_preserves_professor_approved_claim():
    cfg = load()
    b = cfg["screen_boundaries"]
    assert abs(
        (b["relative_NI_lowrisk"]["comparison_FNR"] - b["relative_NI_lowrisk"]["reference_FNR"])
        - 0.03
    ) < 1e-12
    assert abs(
        (b["relative_NI_typical"]["comparison_FNR"] - b["relative_NI_typical"]["reference_FNR"])
        - 0.03
    ) < 1e-12
    assert b["absolute_FNR"]["boundary"] == 0.10
    assert b["FPR"]["boundary"] == 0.05


def test_envelopes_are_bounded_and_centered_near_twenty():
    cfg = load()
    for values in cfg["candidate_cluster_size_envelopes"].values():
        assert min(values) >= 8
        assert max(values) <= 32
        assert abs(sum(values) / len(values) - 20.0) < 1e-12


def test_screen_is_not_a_freeze():
    cfg = load()
    assert cfg["status"] == "simulation_design_screen_not_frozen"
    assert cfg["W0_remains_blocked"] is True
