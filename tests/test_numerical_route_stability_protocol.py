import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "configs/numerical_route_stability_v1.json"


def load():
    return json.loads(P.read_text(encoding="utf-8"))


def test_scope_and_order():
    p = load()
    assert p["scope"]["development_only"] is True
    assert p["scope"]["protected_legacy_splits_used"] is False
    assert p["scope"]["fresh_confirmatory_claim"] is False
    assert p["scope"]["security_control_plane_study_started"] is False
    assert p["prerequisite"]["factorial_measurement_decomposition_complete"] is True


def test_exact_mismatch_taxonomy():
    p = load()
    assert p["mismatches"]["route_mismatch_rows"] == 2
    assert p["mismatches"]["prediction_mismatch_rows"] == 5
    assert p["mismatches"]["pure_decision_threshold_rows"] == 3
    assert p["mismatches"]["route_induced_prediction_rows"] == 2


def test_three_axes_are_separate():
    p = load()
    axes = p["stability_axes"]
    assert "precision" in axes
    assert "runtime_implementation" in axes
    assert "hardware" in axes
    assert axes["hardware"]["controlled_comparison"] == "same_float32_compact_model_cpu_vs_nvidia_t4"


def test_deadband_is_prespecified_not_posthoc():
    p = load()
    assert p["deadband"]["epsilon_grid"] == [0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    assert p["deadband"]["select_epsilon_to_eliminate_mismatches"] is False
    assert p["deadband"]["final_invariance_envelope_requires_hardware_results"] is True
