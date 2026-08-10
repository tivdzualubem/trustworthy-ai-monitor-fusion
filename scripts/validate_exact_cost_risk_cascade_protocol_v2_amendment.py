#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "exact_cost_risk_cascade_protocol_v2.json"
)

AMENDMENT_PATH = (
    ROOT
    / "configs"
    / "exact_cost_risk_cascade_protocol_v2_amendment_001.json"
)

OLD_SHA = (
    "bde3a29f655c4b133beeae34f90f57a6"
    "cd8035afa0ce37688581f5dccce1f87d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    amendment = load_json(AMENDMENT_PATH)

    actual_sha = sha256_file(PROTOCOL_PATH)

    assert protocol["version"] == "2.1.0"
    assert (
        protocol["status"]
        == "frozen_before_v2_modeling_or_threshold_tuning"
    )

    pre = protocol["pre_tuning_amendment"]

    assert (
        pre["amendment_id"]
        == "complete_model_selection_contracts_v2_1"
    )
    assert pre["old_protocol_sha256"] == OLD_SHA

    seen = pre[
        "whether_any_development_calibration_or_confirmatory_result_was_seen"
    ]

    assert (
        seen[
            "v2_repeated_grouped_model_selection_results_seen"
        ]
        is False
    )
    assert (
        seen[
            "permitted_legacy_development_material_inspected_during_v2_implementation"
        ]
        is True
    )
    assert seen["fresh_calibration_results_seen"] is False
    assert seen["fresh_confirmatory_results_seen"] is False
    assert (
        seen[
            "protected_legacy_final_or_shift_opened_during_v2"
        ]
        is False
    )

    assert amendment["old_protocol_hash"] == OLD_SHA
    assert amendment["new_protocol_hash"] == actual_sha
    assert amendment["v2_model_tuning_started"] is False
    assert (
        amendment["fresh_calibration_evaluation_started"]
        is False
    )
    assert (
        amendment["fresh_confirmatory_evaluation_started"]
        is False
    )
    assert (
        amendment["confirmatory_access_status"]
        == "still_sealed"
    )

    families = protocol["model_families"]

    for role in (
        "current_error_classifiers",
        "downstream_fusion_classifiers",
        "cost_predictors",
    ):
        specs = families[role]

        assert specs
        assert all(
            isinstance(spec, dict)
            and spec.get("candidate_grid")
            for spec in specs
        )

    assert {
        spec["family"]
        for spec in families["current_error_classifiers"]
    } == {
        "LogisticRegression",
        "HistGradientBoostingClassifier",
        "RandomForestClassifier",
    }

    assert {
        spec["family"]
        for spec in families[
            "downstream_fusion_classifiers"
        ]
    } == {
        "LogisticRegression",
        "HistGradientBoostingClassifier",
        "RandomForestClassifier",
    }

    assert {
        spec["family"]
        for spec in families["cost_predictors"]
    } == {
        "Ridge_on_log_latency",
        "HistGradientBoostingRegressor_on_log_latency",
    }

    selections = families["candidate_selection"]

    assert (
        selections["current_error_classifier"]["metric"]
        == "pooled_binary_log_loss"
    )
    assert (
        selections["incremental_cost_predictor"]["metric"]
        == (
            "pooled_mean_squared_error_on_"
            "log1p_optional_monitor_latency"
        )
    )

    inputs = protocol["model_input_contract"]

    assert (
        inputs["pre_acquisition_feature_family"]
        == "all_features"
    )
    assert (
        inputs["same_pre_acquisition_information_for"]
        == [
            "signed_value_regressor",
            "current_error_classifier",
            "incremental_cost_predictor",
        ]
    )
    assert inputs["direct_fusion_features"] == [
        "rule_score",
        "compact_unsafe_score",
        "qwen_prompt_response_score",
    ]

    exact_matrix = inputs[
        "exact_pre_acquisition_feature_matrix"
    ]

    assert exact_matrix["final_dimension"] == 49
    assert len(exact_matrix["numeric_feature_order"]) == 17
    assert exact_matrix["embedding_pca"]["n_components"] == 32

    cost = protocol["heterogeneous_cost"][
        "cost_predictor_contract"
    ]

    assert (
        cost["predicted_quantity"]
        == "optional_monitor_execution_latency_ms"
    )
    assert (
        cost["training_target"]
        == "log1p(optional_monitor_stage_latency_ms)"
    )
    assert (
        cost["inverse_transform"]
        == (
            "max(0.0, "
            "expm1(predicted_log1p_optional_monitor_latency))"
        )
    )
    assert cost["post_acquisition_latency_as_feature"] is False
    assert (
        cost[
            "cost_prediction_does_not_replace_actual_cost_measurement"
        ]
        is True
    )

    measurement = protocol["heterogeneous_cost"][
        "development_cost_target_measurement"
    ]

    assert measurement["required_before_cost_predictor_fit"] is True
    assert measurement["warmup_requests"] == 20
    assert measurement["labels_required"] is False
    assert measurement["protected_legacy_rows_allowed"] is False
    assert measurement["fresh_rows_allowed"] is False

    direct = [
        item
        for item in protocol["policies"]["required_baselines"]
        if item["policy_id"] == "direct_fusion"
    ]

    assert len(direct) == 1
    assert "decision_threshold_rule" in direct[0]

    change = protocol["protocol_change_control"]

    assert change["tuning_after_freeze_allowed"] is False
    assert (
        change[
            "further_model_contract_changes_after_v2_development_results"
        ]
        == "forbidden"
    )

    return {
        "status": "PASS",
        "old_protocol_sha256": OLD_SHA,
        "new_protocol_sha256": actual_sha,
        "version": protocol["version"],
        "amendment_id": pre["amendment_id"],
    }


def main() -> None:
    print(
        json.dumps(
            validate(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
