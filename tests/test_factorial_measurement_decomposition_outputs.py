import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/factorial_measurement_decomposition_v1"


def test_decomposition_outputs_exist_and_are_complete():
    summary = json.loads(
        (OUT / "decomposition_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "completed_development_only"
    assert summary["protected_legacy_splits_used"] is False
    assert summary["fresh_confirmatory_claim"] is False
    assert summary["router_superiority_claim"] is False
    assert summary["inferential_pooling_across_seeds"] is False
    assert summary["policy_universe_per_primary_contrast"] == 75
    assert set(summary["layers"]) == {
        "fixed_policy_measurement",
        "retraining_reselection",
        "full_protocol",
    }
    assert summary["next_step"] == "numerical_route_stability"


def test_full_protocol_contrasts_reproduce_preexisting_counts():
    summary = json.loads(
        (OUT / "decomposition_summary.json").read_text(encoding="utf-8")
    )
    check = summary["full_protocol_validation"]
    assert check["grouping_flip_n"] == 19
    assert check["grouping_eligible_n"] == 75
    assert check["label_flip_n"] == 14
    assert check["label_eligible_n"] == 75


def test_each_layer_has_complete_two_by_two_cells():
    cells = pd.read_csv(OUT / "factorial_cells.csv")
    expected_layers = {
        "fixed_policy_measurement",
        "retraining_reselection",
        "full_protocol",
    }
    expected_cells = {
        ("audited", "dependency_primary"),
        ("audited", "singleton_weak"),
        ("original", "dependency_primary"),
        ("original", "singleton_weak"),
    }

    assert set(cells["layer"]) == expected_layers
    for layer in expected_layers:
        part = cells[cells["layer"].eq(layer)]
        observed = set(
            zip(
                part["factor_label_condition"],
                part["factor_grouping_condition"],
            )
        )
        assert observed == expected_cells
        counts = (
            part.groupby(
                ["factor_label_condition", "factor_grouping_condition"]
            )
            .size()
            .to_dict()
        )
        assert set(counts.values()) == {75}


def test_contrasts_are_paired_and_seed_counts_are_not_pooled_inference():
    contrasts = pd.read_csv(OUT / "contrast_summary.csv")
    by_seed = pd.read_csv(OUT / "contrast_summary_by_seed.csv")

    assert len(contrasts) == 12
    assert set(contrasts["eligible_n"]) == {75}
    assert contrasts["descriptive_seed_pooling_only"].astype(bool).all()

    assert len(by_seed) == 60
    assert set(by_seed["eligible_n"]) == {15}
    assert set(by_seed["seed"]) == {1729, 2718, 3141, 5772, 8111}
