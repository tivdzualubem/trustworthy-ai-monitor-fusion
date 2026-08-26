import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/factorial_measurement_decomposition_v1.json"


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_protocol_is_frozen_and_development_only():
    protocol = load_protocol()
    assert protocol["status"] == "frozen_before_decomposition_outcomes"
    assert protocol["scope"]["development_only"] is True
    assert protocol["scope"]["fresh_confirmatory_claim"] is False
    assert protocol["scope"]["router_superiority_claim"] is False
    assert protocol["scope"]["protected_legacy_splits"] == [
        "final_test",
        "held_out_shift",
    ]
    assert protocol["scope"]["protected_split_action"] == "raise_before_read"


def test_protocol_decomposes_both_factors_into_required_layers():
    protocol = load_protocol()
    assert set(protocol["factors"]["label"]["conditions"]) == {
        "audited",
        "original",
    }
    assert set(protocol["factors"]["grouping"]["conditions"]) == {
        "dependency_primary",
        "singleton_weak",
    }
    assert protocol["factorial_cells"] == [
        "audited__dependency_primary",
        "audited__singleton_weak",
        "original__dependency_primary",
        "original__singleton_weak",
    ]
    assert set(protocol["layers"]) == {
        "fixed_policy_measurement",
        "retraining_reselection",
        "full_protocol",
    }


def test_fixed_policy_layer_changes_measurement_only():
    layer = load_protocol()["layers"]["fixed_policy_measurement"]
    assert layer["policy_source"] == "audited__dependency_primary"
    assert layer["freeze"] == [
        "evaluation_rows",
        "fitted_models",
        "acquisition_thresholds",
        "decision_thresholds",
        "routes",
        "predictions",
    ]
    assert layer["vary_only"] == [
        "evaluation_label",
        "risk_grouping",
    ]


def test_retraining_reselection_layer_has_common_outer_holdout():
    layer = load_protocol()["layers"]["retraining_reselection"]
    assert layer["outer_holdout_source"] == (
        "audited__dependency_primary_internal_risk_holdout"
    )
    assert layer["outer_holdout_fixed_across_factorial_cells"] is True
    assert layer["outer_measurement_label"] == "audited"
    assert layer["outer_measurement_grouping"] == "dependency_primary"
    assert layer["outer_dependency_group_closed"] is True
    assert layer["remaining_data_splitter"] == "StratifiedGroupKFold"
    assert layer["remaining_data_n_splits"] == 3
    assert layer["remaining_fold_roles"] == {
        "0": "policy_selection",
        "1": "policy_train",
        "2": "policy_train",
    }


def test_full_protocol_layer_is_explicitly_as_implemented():
    layer = load_protocol()["layers"]["full_protocol"]
    assert layer["source"] == (
        "reports/evaluation_measurement_pilot_v1/cpu/policy_summary.csv"
    )
    assert layer["reproduce_existing_grouping_flip_n"] == 19
    assert layer["reproduce_existing_label_flip_n"] == 14
    assert layer["risk_grouping_as_implemented"] == (
        "dependency_primary_for_all_factorial_conditions"
    )


def test_seed_counts_are_descriptive_not_pooled_inference():
    protocol = load_protocol()
    assert protocol["seeds"] == [1729, 2718, 3141, 5772, 8111]
    assert protocol["policy_universe"]["deployable_policies_per_seed"] == 15
    assert protocol["policy_universe"]["aggregate_descriptive_n"] == 75
    assert protocol["inference"]["pool_across_seeds"] is False
    assert protocol["inference"]["aggregate_flip_counts_role"] == (
        "descriptive_comparison_to_existing_19_of_75_and_14_of_75"
    )
