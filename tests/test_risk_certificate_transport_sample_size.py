import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/risk_certificate_transport_preregistration_v1"


def test_sample_size_design_reproduces_exactly():
    s = json.loads((OUT / "sample_size_summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "analytic_design_only_no_fresh_data"
    assert s["risk_limit"] == 0.05
    assert s["one_sided_alpha"] == 0.05
    assert s["design_true_fpr"] == 0.025
    assert s["target_power"] == 0.8
    assert s["minimum_independent_negative_units_per_certificate_cell"] == 361
    assert s["maximum_false_positives_at_n_for_certificate"] == 11
    assert abs(
        s["exact_upper_bound_at_max_false_positives"] - 0.04993268035498383
    ) < 1e-14
    assert abs(
        s["achieved_power_at_design_fpr"] - 0.8030314156589886
    ) < 1e-14
    assert s["fresh_data_read"] is False
    assert s["historical_outcome_data_read"] is False
    assert s["monitor_scoring_performed"] is False


def test_power_sensitivity_table_is_complete():
    frame = pd.read_csv(OUT / "sample_size_power.csv")
    assert frame["true_fpr"].tolist() == [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
    assert (frame["n_negative_independent_units"] == 361).all()
    assert (frame["max_false_positives_for_95pct_certificate"] == 11).all()
    p025 = frame.loc[
        frame["true_fpr"].eq(0.025), "certificate_power"
    ].iloc[0]
    assert abs(float(p025) - 0.8030314156589886) < 1e-14
