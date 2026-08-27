import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/threshold_tie_diagnostic_v1"


def as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def test_threshold_tie_outputs_match_archived_deadband():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_development_only_archived_diagnostic"
    assert summary["reporting_epsilon"] == 1e-6
    assert summary["route_rows"] == 3267
    assert summary["deadband_ambiguous_n"] == 200
    assert abs(summary["deadband_ambiguous_fraction"] - (200 / 3267)) < 1e-15

    # The epsilon=0 and epsilon=1e-6 archived counts already establish that the
    # reporting dead-band contains no additional non-tied reference-margin rows.
    assert summary["exact_threshold_tie_n"] == 200
    assert summary["exact_threshold_tie_share_of_deadband"] == 1.0
    assert summary["nonexact_deadband_n"] == 0

    assert summary["route_mismatch_n"] == 2
    assert summary["route_mismatch_at_exact_tie_n"] == 2
    assert summary["route_mismatch_nonexact_n"] == 0

    # Preserve the separate controlled hardware result.
    assert summary["controlled_cpu_t4_float32_route_flips"] == 0
    assert summary["controlled_cpu_t4_float32_prediction_flips"] == 0


def test_deadband_row_table_contains_only_exact_reference_ties():
    rows = pd.read_csv(OUT / "deadband_rows.csv")
    assert len(rows) == 200
    assert as_bool(rows["exact_threshold_tie"]).all()
    assert (rows["reference_route_margin"].astype(float) == 0.0).all()
    assert not as_bool(rows["non_tied_route_crossing"]).any()


def test_discrete_output_diagnostics_are_reported():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    for key in [
        "exact_ties_with_repeated_raw_cheap_state_n",
        "exact_ties_with_repeated_raw_cheap_state_share",
        "exact_ties_with_repeated_route_distance_mass_n",
        "exact_ties_with_repeated_route_distance_mass_share",
        "runtime_exact_threshold_tie_n_in_deadband",
        "reference_ties_shifted_off_exact_runtime_n",
    ]:
        assert key in summary

    states = pd.read_csv(OUT / "cheap_state_summary.csv")
    assert set(states["stack"]) == {
        "rule_to_compact",
        "compact_to_qwen",
        "rule_compact_to_qwen",
    }
    assert states["cheap_monitor_features"].notna().all()
    assert (states["raw_state_support_size"] >= 1).all()
    assert (states["route_distance_support_size"] >= 1).all()

    policies = pd.read_csv(OUT / "policy_summary.csv")
    assert "acquisition_threshold" in policies.columns
    assert policies["acquisition_threshold"].notna().all()


def test_existing_data_discovery_is_closed_after_diagnostic():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    boundary = summary["workflow_boundary"]
    assert boundary["threshold_tie_diagnostic_complete"] is True
    assert boundary["existing_data_discovery_closed"] is True
    assert boundary["new_discovery_on_existing_data_allowed"] is False
    assert boundary["fresh_data_collection_started"] is False
    assert boundary["fresh_transport_protocol_run"] is False
    assert (
        boundary["next_step"]
        == "prepare_preregistered_fresh_risk_certificate_transport_protocol"
    )
