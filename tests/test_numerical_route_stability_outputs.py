import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/numerical_route_stability_v1/cpu"


def test_taxonomy_and_summary():
    tax = pd.read_csv(OUT / "mismatch_taxonomy.csv")
    summary = json.loads((OUT / "summary.json").read_text())
    assert len(tax) == 5
    assert int(tax["route_mismatch"].sum()) == 2
    assert (tax["prediction_mismatch_type"] == "pure_decision_threshold").sum() == 3
    assert (tax["prediction_mismatch_type"] == "downstream_of_route_mismatch").sum() == 2
    assert summary["hardware_phase_required"] is True
    assert summary["final_deadband_pending_controlled_hardware_phase"] is True


def test_deadband_grid_and_no_posthoc_selection():
    dead = pd.read_csv(OUT / "deadband_sweep.csv")
    assert dead["epsilon"].tolist() == [0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


def test_runtime_precision_matrix_contains_independent_runtime_paths():
    frame = pd.read_csv(OUT / "runtime_precision_matrix.csv")
    assert {"python_scalar", "numpy", "torch_cpu"}.issubset(set(frame["implementation"]))
    assert {"float64", "float32"}.issubset(set(frame["precision"]))
