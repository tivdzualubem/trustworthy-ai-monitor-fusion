import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports/numerical_route_stability_v1/final"
T4 = ROOT / "reports/numerical_route_stability_v1/t4"
GRID = [0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


def test_final_stability_summary_closes_step3_only():
    s = json.loads((FINAL / "summary.json").read_text())
    assert s["status"] == "completed_development_only"
    assert s["deadband"]["reporting_epsilon"] in GRID
    assert s["deadband"]["selected_to_eliminate_mismatches"] is False
    assert s["claim_boundary"]["protected_legacy_splits_used"] is False
    assert s["claim_boundary"]["security_control_plane_study_started"] is False
    assert s["next_step"] == "small_safety_availability_control_plane_kill_study"


def test_controlled_float32_hardware_has_no_policy_flips():
    h = pd.read_csv(FINAL / "hardware_axis_summary.csv")
    r = h[h["precision"] == "float32"].iloc[0]
    assert int(r["route_flip_n_cpu_vs_t4"]) == 0
    assert int(r["prediction_flip_n_cpu_vs_t4"]) == 0


def test_exploratory_float16_is_not_supported():
    s = json.loads((FINAL / "summary.json").read_text())
    assert s["precision"]["float16_supported_path"] is False
    assert s["precision"]["t4_float16_exploratory_route_flips_on_five_cases"] == 2
    assert s["precision"]["t4_float16_exploratory_prediction_flips_on_five_cases"] == 5


def test_deadband_is_first_prespecified_grid_point_covering_envelope():
    d = pd.read_csv(FINAL / "final_deadband_summary.csv").iloc[0]
    assert bool(d["posthoc_mismatch_elimination_used"]) is False

    epsilon = float(d["reporting_epsilon"])
    required = float(d["required_nonexploratory_envelope"])
    assert epsilon in GRID
    assert epsilon >= required

    index = GRID.index(epsilon)
    if index > 0:
        assert GRID[index - 1] < required

    assert 0.0 < float(d["route_ambiguous_fraction"]) < 0.1
    assert 0.0 < float(d["decision_ambiguous_fraction"]) < 0.01


def test_all_float32_diagnostic_flips_are_covered_by_selected_deadband():
    s = json.loads((FINAL / "summary.json").read_text())
    coverage = s["deadband"]["diagnostic_flip_coverage"]
    for variant in ("numpy_float32", "torch_cpu_float32"):
        v = coverage[variant]
        assert v["route_flip_n"] == v["route_flip_covered_n"]
        assert (
            v["same_route_prediction_flip_n"]
            == v["same_route_prediction_flip_covered_n"]
        )


def test_t4_environment_is_controlled_target():
    env = json.loads((T4 / "environment.json").read_text())
    assert "T4" in env["gpu_name"]
    assert env["compact_revision"] == "838ade0edb66dcffc5532d08ff6ed5c899abfb5c"
    assert env["tf32_disabled"] is True
    assert env["cudnn_deterministic"] is True
