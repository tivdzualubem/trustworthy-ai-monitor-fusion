import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "configs/threshold_tie_diagnostic_v1.json").read_text(encoding="utf-8")
)
SCRIPT = (
    ROOT / "scripts/run_threshold_tie_diagnostic.py"
).read_text(encoding="utf-8")


def test_threshold_tie_protocol_is_narrow_and_frozen():
    assert PROTOCOL["protocol_id"] == "threshold_tie_diagnostic_v1"
    assert PROTOCOL["status"] == "frozen_before_diagnostic_outcomes"
    assert PROTOCOL["frozen_reporting"]["reporting_epsilon"] == 1e-6
    assert PROTOCOL["scope"]["new_data_collection"] is False
    assert PROTOCOL["scope"]["model_fitting"] is False
    assert PROTOCOL["scope"]["threshold_reselection"] is False
    assert PROTOCOL["scope"]["router_retuning"] is False
    assert PROTOCOL["scope"]["protected_legacy_splits_used"] is False


def test_threshold_tie_categories_are_prespecified():
    frozen = PROTOCOL["frozen_reporting"]
    assert frozen["exact_threshold_tie_rule"] == "reference_route_margin == 0.0 exactly"
    assert "float.hex" in frozen["raw_cheap_state_rule"]
    assert "distinct examples" in frozen["raw_cheap_state_rule"]
    assert "reference_distance" in frozen["route_distance_mass_rule"]
    assert "route_mismatch" in frozen["observed_route_crossing_rule"]
    assert "exact_threshold_tie == false" in frozen["non_tied_crossing_rule"]


def test_diagnostic_cannot_retrain_or_reselect():
    lower = SCRIPT.lower()
    forbidden = [
        "logisticregression(",
        ".fit(",
        "select_distance_threshold(",
        "select_classification_threshold(",
        "final_test",
        "held_out_shift",
    ]
    for phrase in forbidden:
        assert phrase not in lower


def test_completion_boundary_closes_existing_data_discovery():
    boundary = PROTOCOL["completion_boundary"]
    assert boundary["after_diagnostic_existing_data_discovery_allowed"] is False
    assert boundary["fresh_data_collection_before_protocol_review"] is False
    assert boundary["confirmation_router_retuning_allowed"] is False
    assert (
        boundary["next_step"]
        == "prepare_preregistered_fresh_risk_certificate_transport_protocol"
    )


def test_prespecified_epsilon_rows_use_exact_lookup():
    assert 'np.isclose(prior_sweep["epsilon"]' not in SCRIPT
    assert 'prior_sweep["epsilon"].astype(float).eq(0.0)' in SCRIPT
    assert 'prior_sweep["epsilon"].astype(float).eq(epsilon)' in SCRIPT


def test_implementation_correction_does_not_change_frozen_definitions():
    correction = PROTOCOL["implementation_correction"]
    assert correction["diagnostic_outcomes_observed_before_correction"] is False
    assert correction["frozen_definitions_changed"] is False
    assert correction["reporting_epsilon_changed"] is False
    assert correction["thresholds_or_models_changed"] is False
