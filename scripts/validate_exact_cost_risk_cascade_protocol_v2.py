#!/usr/bin/env python3
"""Validate the frozen v2 cascade protocol without opening project data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/exact_cost_risk_cascade_protocol_v2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    assert protocol["artifact"] == "exact_cost_risk_cascade_protocol_v2"
    assert protocol["status"] == "frozen_before_v2_modeling_or_threshold_tuning"
    assert protocol["data_boundary"]["default_action"] == "deny"
    assert protocol["data_boundary"]["report_files_may_be_changed_during_implementation"] is False

    scope = protocol["scope"]
    development = set(scope["legacy_development_splits"])
    protected = set(scope["protected_legacy_splits"])
    fresh = set(scope["fresh_split_ids"])
    assert development == {"policy_train", "policy_selection", "calibration"}
    assert protected == {"final_test", "held_out_shift"}
    assert development.isdisjoint(protected | fresh)
    assert protected.isdisjoint(fresh)

    boundary = protocol["data_boundary"]
    allowed_paths = boundary["permitted_existing_development_artifacts"]
    assert all(
        not any(token in path for token in ("final_test", "held_out_shift", "final_evaluation"))
        for path in allowed_paths
    )
    assert len(boundary["sealed_mixed_split_containers"]) == 2

    value = protocol["decision_value"]
    assert value["negative_values_allowed"] is True
    assert value["policy_specific"] is True
    assert value["zero_one_loss_values"] == [-1, 0, 1]
    assert value["bayes_nonnegativity_claim_for_fitted_policies_forbidden"] is True

    resampling = protocol["development_resampling"]
    assert resampling["outer_folds"] >= 5
    assert resampling["inner_folds"] >= 4
    assert len(set(resampling["fold_seeds"])) >= 5

    models = protocol["model_families"]
    assert len(models["signed_value_regressors"]) >= 3
    assert len(models["current_error_classifiers"]) >= 3
    assert len(models["downstream_fusion_classifiers"]) >= 3

    baseline_ids = {
        item["policy_id"]
        for item in protocol["policies"]["required_baselines"]
    }
    assert baseline_ids == {
        "threshold_distance",
        "current_error_prediction",
        "random_acquisition",
        "direct_fusion",
    }
    assert protocol["policies"]["proposed"]["batch_rank_or_test_rank_allowed"] is False

    costs = protocol["heterogeneous_cost"]
    assert costs["per_example_measurement_required"] is True
    assert costs["post_acquisition_latency_is_forbidden_as_a_router_feature"] is True
    assert costs["end_to_end_timeout_ms"] > costs["optional_monitor_timeout_ms"]

    exact = protocol["exact_cost_comparison"]
    assert exact["same_absolute_budget_for_every_selective_policy"] is True
    assert exact["evaluation_ranks_used"] is False
    assert exact["recall_comparison_allowed_only_when_cost_equivalence_passes"] is True
    assert exact["cost_ceiling_substitution_for_exact_matching_forbidden"] is True

    calibration = protocol["independent_calibration"]
    assert abs(
        calibration["optimization_fraction"]
        + calibration["risk_testing_fraction"]
        - 1.0
    ) < 1e-12
    assert calibration["confirmatory_information_used"] is False

    joint = protocol["joint_risk_control"]
    assert joint["familywise_error_rate"] == 0.05
    risk_names = {item["risk"] for item in joint["constraints"]}
    assert risk_names == {
        "false_positive_rate",
        "mean_total_end_to_end_cost_ms",
    }
    assert joint["point_estimate_only_pass_forbidden"] is True
    assert joint["separate_random_repetition_metrics_may_not_be_combined_into_one_policy"] is True

    latency_stats = set(protocol["latency_measurement"]["required_statistics"])
    assert {"mean", "median", "p95", "p99", "maximum", "timeout_rate"}.issubset(latency_stats)

    fresh_data = protocol["fresh_data"]
    assert fresh_data["required"] is True
    assert fresh_data["legacy_overlap_allowed"] is False
    assert fresh_data["minimum_distinct_sources"] >= 3
    assert fresh_data["labeling"]["minimum_independent_raters_per_example"] >= 3
    assert fresh_data["one_shot_evaluation"]["maximum_confirmatory_runs"] == 1

    gates = protocol["claim_gate"][
        "router_improves_recall_under_controlled_FPR_and_cost"
    ]["all_required"]
    assert len(gates) >= 8
    assert protocol["legacy_frontier"]["status"] == "invalid_for_router_superiority_claims"
    assert protocol["legacy_frontier"]["legacy_outputs_may_be_used_as_v2_evidence"] is False
    assert protocol["protocol_change_control"]["tuning_after_freeze_allowed"] is False

    return {
        "status": "PASS",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_split_count": len(development),
        "protected_split_count": len(protected),
        "fold_seed_count": len(resampling["fold_seeds"]),
        "signed_value_model_family_count": len(models["signed_value_regressors"]),
        "required_baselines": sorted(baseline_ids),
        "joint_risks": sorted(risk_names),
        "report_files_changed": False,
        "project_data_opened": False,
    }


def main() -> None:
    result = validate_protocol(load_protocol())
    print("exact-cost risk-cascade protocol validation passed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
