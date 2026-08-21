from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evaluation_measurement_pilot_v1.json"
META = ROOT / "data/metadata/evaluation_measurement_pilot_v1"
GROUPS = META / "development_dependency_groups.csv"
LOCK = META / "protocol_lock.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_pilot_protocol_is_locked_and_fail_closed():
    protocol = json.loads(PROTOCOL.read_text())

    assert protocol["status"] == "frozen_before_pilot_outcomes"
    assert protocol["scope"]["router_optimization_goal"] is False
    assert protocol["scope"]["confirmatory_claims_allowed"] is False
    assert set(protocol["scope"]["protected_legacy_splits"]) == {
        "final_test",
        "held_out_shift",
    }
    assert protocol["scope"]["protected_split_action"] == "raise_before_read"

    grouping = protocol["dependency_grouping"]
    assert grouping["primary"]["semantic_threshold"] == 0.90
    assert grouping["primary"]["lexical_jaccard_threshold"] == 0.75
    assert grouping["selection_used_labels_or_monitor_outcomes"] is False

    assert len(protocol["monitor_stacks"]) == 3
    assert (
        protocol["routing_comparison"]["offline_ranked_diagnostic"]["deployable"]
        is False
    )
    assert (
        protocol["routing_comparison"]["deployable_primary"][
            "threshold_reused_unchanged"
        ]
        is True
    )

    cost = protocol["cost_measurement"]
    assert (
        cost["primary_estimand"]
        == "direct_wall_clock_end_to_end_policy_latency_ms"
    )
    assert cost["historical_35s_cap_applied"] is False
    assert cost["posthoc_latency_clipping_allowed"] is False
    assert cost["bounded_mean_cost_certification"] is False

    objectives = protocol["comparison_objectives"]
    assert objectives["primary"] == "pareto"
    assert objectives["confirmatory_iso_cost_equivalence"] is False
    assert (
        objectives["historical_one_percent_margin"]["externally_justified"]
        is False
    )


def test_grouping_artifact_matches_frozen_structure():
    frame = pd.read_csv(GROUPS)
    assert len(frame) == 1687
    assert frame["example_id"].is_unique

    counts = frame["primary_dependency_group"].value_counts()
    assert len(counts) == 1642
    assert int((counts > 1).sum()) == 9
    assert int(counts[counts > 1].sum()) == 54
    assert int(counts.max()) == 34


def test_protocol_lock_hashes_match():
    lock = json.loads(LOCK.read_text())
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        assert path.is_file()
        assert sha256(path) == expected


def test_preoutcome_hardening_is_frozen():
    protocol = json.loads(PROTOCOL.read_text())

    assert protocol["protocol_version"] == "1.1.0"
    assert (
        protocol["preoutcome_hardening"][
            "performed_before_any_pilot_result_namespace_existed"
        ]
        is True
    )

    resampling = protocol["resampling"]
    assert resampling["primary_inference_seed"] == 1729
    assert resampling["pool_repeated_holdouts_for_inference"] is False

    risk = protocol["decision_operating_point"]["internal_risk_bound"]
    assert (
        risk["primary_dependency_aware"]["risk_unit"]
        == "dependency_group_with_at_least_one_negative_row"
    )
    assert len(risk["primary_gate_requires"]) == 2

    timing = protocol["cost_measurement"]
    assert "same policy call" in timing["component_sum_definition"]
    assert timing["timing_sample"]["group_closure_required"] is True
    assert timing["timing_sample"]["labels_used_for_sampling"] is False

    inference = protocol["inference"]
    assert inference["bootstrap_repetitions"] == 5000
    assert inference["repeated_seed_pooling_allowed"] is False

    objectives = protocol["comparison_objectives"]
    assert objectives["pairwise_pareto_matrix"] is True
    assert (
        objectives["offline_ranked_excluded_from_deployable_pareto_claims"]
        is True
    )
