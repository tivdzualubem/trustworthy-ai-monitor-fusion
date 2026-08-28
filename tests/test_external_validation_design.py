import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/safety_monitor_external_validation_preregistration_v1"


def test_fpr_design_reproduces():
    s = json.loads((OUT / "design_summary.json").read_text(encoding="utf-8"))
    f = s["FPR"]
    assert f["n_negative_per_validation_cell"] == 361
    assert f["maximum_false_positives_for_constraint"] == 11
    assert abs(f["cp_upper_at_kmax"] - 0.04993268035498383) < 1e-14
    assert abs(f["pass_probability_at_true_fpr"] - 0.8030314156589886) < 1e-14
    assert "not source/attack power" in f["interpretation"]


def test_fnr_design_and_sensitivity_grid_reproduce():
    s = json.loads((OUT / "design_summary.json").read_text(encoding="utf-8"))
    f = s["FNR"]
    assert f["n_positive_per_validation_cell"] == 600
    assert f["primary_test_count"] == 9
    assert f["multiplicity"] == "Holm"
    assert abs(f["planning_alpha"] - (0.05 / 9)) < 1e-15
    assert f["target_power"] == 0.8
    assert f["design_effect_sensitivity"] == 1.2

    frame = pd.read_csv(OUT / "design_sensitivity.csv")
    assert frame.shape[0] == 6
    assert sorted(frame["baseline_fnr"].unique().tolist()) == [0.05, 0.1, 0.2]
    assert sorted(frame["design_effect"].unique().tolist()) == [1.0, 1.2]
    assert (frame["n_positive_per_W1_cell"] == 600).all()
