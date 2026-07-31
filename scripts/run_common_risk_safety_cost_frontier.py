#!/usr/bin/env python3
"""Evaluate the frozen common-risk incremental safety-cost frontier."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)
VALUE_RUNNER_PATH = (
    ROOT / "scripts/run_cross_fitted_value_predictability.py"
)
NESTED_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "nested_value_training_targets.parquet"
)
OUTER_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "cross_fitted_decision_value_targets.parquet"
)
FOLD_METRICS_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "value_estimator_fold_metrics.csv"
)
CURVES_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "matched_budget_value_curves.csv"
)
RANDOM_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "matched_budget_random_repetitions.csv"
)
EMBEDDING_MANIFEST_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "frozen_prompt_response_embedding_manifest.json"
)
VALUE_MANIFEST_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "value_predictability_manifest.json"
)
OUTPUT_DIR = ROOT / "reports/decision_value_real_data"

RUNTIME_PATH = (
    OUTPUT_DIR / "value_policy_inference_runtime.csv"
)
RANDOM_SUMMARY_PATH = (
    OUTPUT_DIR / "common_risk_random_summary.csv"
)
FRONTIER_PATH = (
    OUTPUT_DIR / "common_risk_safety_cost_frontier.csv"
)
CANDIDATE_PATH = (
    OUTPUT_DIR / "common_risk_selective_candidates.csv"
)
SUMMARY_PATH = (
    OUTPUT_DIR / "common_risk_frontier_summary.json"
)
REPORT_PATH = (
    OUTPUT_DIR / "common_risk_frontier_summary.md"
)


def load_value_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "frozen_value_predictability_runner_for_frontier",
        VALUE_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {VALUE_RUNNER_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_primary_inference(
    module: Any,
    protocol: dict[str, Any],
    diagnostic: dict[str, Any],
) -> pd.DataFrame:
    setup_id = diagnostic["primary_setup"]
    family = diagnostic["primary_feature_family"]
    runtime_protocol = diagnostic[
        "inference_runtime_benchmark"
    ]

    warmups = int(
        runtime_protocol["warmup_repetitions"]
    )
    repetitions = int(
        runtime_protocol["timed_repetitions"]
    )
    random_state = int(
        protocol["cross_fitting"]["random_state"]
    )

    feature_lookup, embedding_columns = (
        module.load_feature_frame(protocol)
    )
    nested = pd.read_parquet(NESTED_PATH)
    outer = pd.read_parquet(OUTER_PATH)
    fold_metrics = pd.read_csv(FOLD_METRICS_PATH)

    numeric_features, use_embedding = (
        module.family_feature_names(
            protocol,
            setup_id,
            family,
        )
    )
    if not use_embedding:
        raise RuntimeError(
            "Frozen primary family must use embeddings"
        )

    pca_components = int(
        protocol["predictor_families"][
            "frozen_embedding"
        ]["pca_components"]
    )

    rows: list[dict[str, Any]] = []

    for outer_fold in range(
        int(protocol["cross_fitting"]["outer_folds"])
    ):
        train_rows = nested.loc[
            (nested["setup_id"] == setup_id)
            & (
                nested["value_outer_fold"]
                == outer_fold
            )
        ].copy()
        eval_rows = outer.loc[
            (outer["setup_id"] == setup_id)
            & (outer["outer_fold"] == outer_fold)
        ].copy()

        if set(train_rows["example_id"]).intersection(
            set(eval_rows["example_id"])
        ):
            raise RuntimeError(
                "Outer-evaluation leakage in runtime rebuild"
            )

        selected = fold_metrics.loc[
            (fold_metrics["setup_id"] == setup_id)
            & (fold_metrics["outer_fold"] == outer_fold)
            & (
                fold_metrics["feature_family"]
                == family
            )
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Missing unique selected model configuration"
            )
        selected = selected.iloc[0]

        candidate = module.Candidate(
            candidate_id=int(
                selected["selected_candidate_id"]
            ),
            params=json.loads(
                selected["selected_params_json"]
            ),
        )

        pca = module.fit_pca(
            rows=train_rows,
            feature_lookup=feature_lookup,
            embedding_columns=embedding_columns,
            components=pca_components,
            random_state=(
                random_state
                + outer_fold * 10000
                + 5 * 1000
                + 701
            ),
        )

        x_train = module.matrix_for_rows(
            rows=train_rows,
            feature_lookup=feature_lookup,
            numeric_features=numeric_features,
            embedding_columns=embedding_columns,
            pca=pca,
        )

        model = module.build_estimator(
            candidate=candidate,
            random_state=(
                random_state
                + outer_fold * 10000
                + 5 * 1000
                + 911
            ),
        )
        model.fit(
            x_train,
            train_rows[
                "realized_decision_value"
            ].to_numpy(dtype=float),
        )

        joined_eval = module.join_features(
            eval_rows,
            feature_lookup,
        )
        numeric_eval = joined_eval[
            numeric_features
        ].to_numpy(dtype=np.float64)
        embedding_eval = joined_eval[
            embedding_columns
        ].to_numpy(dtype=np.float64)

        def predict_once() -> np.ndarray:
            reduced = pca.transform(embedding_eval)
            matrix = np.concatenate(
                [numeric_eval, reduced],
                axis=1,
            )
            return model.predict(matrix)

        with threadpool_limits(limits=1):
            for _ in range(warmups):
                prediction = predict_once()

            start = time.perf_counter()
            for _ in range(repetitions):
                prediction = predict_once()
            elapsed = time.perf_counter() - start

        if not np.isfinite(prediction).all():
            raise RuntimeError(
                "Non-finite benchmark prediction"
            )

        per_batch_ms = (
            elapsed * 1000.0 / repetitions
        )
        per_example_ms = (
            per_batch_ms / len(eval_rows)
        )

        rows.append(
            {
                "setup_id": setup_id,
                "feature_family": family,
                "outer_fold": outer_fold,
                "eval_n": len(eval_rows),
                "warmup_repetitions": warmups,
                "timed_repetitions": repetitions,
                "pca_components": pca_components,
                "selected_candidate_id": (
                    candidate.candidate_id
                ),
                "elapsed_seconds": elapsed,
                "mean_batch_ms": per_batch_ms,
                "mean_per_example_ms": (
                    per_example_ms
                ),
            }
        )

        print(
            "runtime benchmark "
            f"outer_fold={outer_fold + 1}/5 "
            f"per_example_ms={per_example_ms:.6f}",
            flush=True,
        )

    return pd.DataFrame(rows)


def summarize_random(
    random: pd.DataFrame,
    optional_cost_ms: float,
    risk_limit: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for budget, group in random.groupby(
        "budget",
        sort=True,
    ):
        acquisition_rate = float(
            group["actual_acquisition_rate"].mean()
        )
        recall = group["recall"].to_numpy(
            dtype=float
        )
        fpr = group["fpr"].to_numpy(dtype=float)

        rows.append(
            {
                "policy": "random",
                "budget": float(budget),
                "repetitions": len(group),
                "actual_acquisition_rate": (
                    acquisition_rate
                ),
                "incremental_cost_ms_per_example": (
                    acquisition_rate
                    * optional_cost_ms
                ),
                "mean_recall": float(recall.mean()),
                "recall_lower95": float(
                    np.quantile(recall, 0.025)
                ),
                "recall_upper95": float(
                    np.quantile(recall, 0.975)
                ),
                "mean_fpr": float(fpr.mean()),
                "fpr_lower95": float(
                    np.quantile(fpr, 0.025)
                ),
                "fpr_upper95": float(
                    np.quantile(fpr, 0.975)
                ),
                "risk_feasible": bool(
                    fpr.mean() <= risk_limit
                ),
            }
        )

    return pd.DataFrame(rows)


def build_deterministic_frontier(
    curves: pd.DataFrame,
    optional_cost_ms: float,
    learned_fixed_overhead_ms: float,
    risk_limit: float,
) -> pd.DataFrame:
    frame = curves.loc[
        curves["policy"].isin(
            [
                "learned_decision_value",
                "ordinary_uncertainty",
            ]
        )
    ].copy()

    frame[
        "fixed_policy_overhead_ms_per_example"
    ] = np.where(
        frame["policy"]
        == "learned_decision_value",
        learned_fixed_overhead_ms,
        0.0,
    )

    frame[
        "optional_monitor_cost_ms_per_example"
    ] = (
        frame["actual_acquisition_rate"]
        * optional_cost_ms
    )

    frame[
        "incremental_cost_ms_per_example"
    ] = (
        frame[
            "fixed_policy_overhead_ms_per_example"
        ]
        + frame[
            "optional_monitor_cost_ms_per_example"
        ]
    )

    frame["risk_feasible"] = (
        frame["fpr"] <= risk_limit
    )

    return frame.sort_values(
        ["policy", "incremental_cost_ms_per_example"]
    ).reset_index(drop=True)


def best_uncertainty_under_cost(
    frontier: pd.DataFrame,
    cost_ceiling: float,
) -> pd.Series | None:
    candidates = frontier.loc[
        (
            frontier["policy"]
            == "ordinary_uncertainty"
        )
        & frontier["risk_feasible"]
        & (
            frontier[
                "incremental_cost_ms_per_example"
            ]
            <= cost_ceiling + 1e-12
        )
    ].copy()

    if candidates.empty:
        return None

    return candidates.sort_values(
        [
            "recall",
            "incremental_cost_ms_per_example",
            "budget",
        ],
        ascending=[False, True, True],
    ).iloc[0]


def best_random_under_cost(
    random_summary: pd.DataFrame,
    cost_ceiling: float,
) -> pd.Series | None:
    candidates = random_summary.loc[
        random_summary["risk_feasible"]
        & (
            random_summary[
                "incremental_cost_ms_per_example"
            ]
            <= cost_ceiling + 1e-12
        )
    ].copy()

    if candidates.empty:
        return None

    return candidates.sort_values(
        [
            "recall_upper95",
            "incremental_cost_ms_per_example",
            "budget",
        ],
        ascending=[False, True, True],
    ).iloc[0]


def build_candidates(
    frontier: pd.DataFrame,
    random_summary: pd.DataFrame,
    risk_limit: float,
) -> pd.DataFrame:
    learned = frontier.loc[
        (
            frontier["policy"]
            == "learned_decision_value"
        )
        & (frontier["budget"] < 1.0)
    ].copy()

    rows: list[dict[str, Any]] = []

    for _, point in learned.iterrows():
        cost = float(
            point[
                "incremental_cost_ms_per_example"
            ]
        )
        uncertainty = best_uncertainty_under_cost(
            frontier,
            cost,
        )
        random = best_random_under_cost(
            random_summary,
            cost,
        )

        uncertainty_recall = (
            float(uncertainty["recall"])
            if uncertainty is not None
            else float("nan")
        )
        uncertainty_budget = (
            float(uncertainty["budget"])
            if uncertainty is not None
            else float("nan")
        )
        uncertainty_cost = (
            float(
                uncertainty[
                    "incremental_cost_ms_per_example"
                ]
            )
            if uncertainty is not None
            else float("nan")
        )

        random_upper = (
            float(random["recall_upper95"])
            if random is not None
            else float("nan")
        )
        random_budget = (
            float(random["budget"])
            if random is not None
            else float("nan")
        )
        random_cost = (
            float(
                random[
                    "incremental_cost_ms_per_example"
                ]
            )
            if random is not None
            else float("nan")
        )

        risk_pass = bool(
            float(point["fpr"]) <= risk_limit
        )
        uncertainty_pass = bool(
            uncertainty is not None
            and float(point["recall"])
            > uncertainty_recall
        )
        random_pass = bool(
            random is not None
            and float(point["recall"])
            > random_upper
        )
        candidate_pass = bool(
            risk_pass
            and uncertainty_pass
            and random_pass
        )

        rows.append(
            {
                "learned_budget": float(
                    point["budget"]
                ),
                "learned_acquisition_rate": float(
                    point["actual_acquisition_rate"]
                ),
                "learned_incremental_cost_ms_per_example": (
                    cost
                ),
                "learned_recall": float(
                    point["recall"]
                ),
                "learned_fpr": float(point["fpr"]),
                "risk_pass": risk_pass,
                "best_uncertainty_budget_under_cost": (
                    uncertainty_budget
                ),
                "best_uncertainty_cost_ms_per_example": (
                    uncertainty_cost
                ),
                "best_uncertainty_recall_under_cost": (
                    uncertainty_recall
                ),
                "recall_margin_over_uncertainty": (
                    float(point["recall"])
                    - uncertainty_recall
                ),
                "uncertainty_dominance_pass": (
                    uncertainty_pass
                ),
                "best_random_budget_under_cost": (
                    random_budget
                ),
                "best_random_cost_ms_per_example": (
                    random_cost
                ),
                "best_random_recall_upper95_under_cost": (
                    random_upper
                ),
                "recall_margin_over_random_upper95": (
                    float(point["recall"])
                    - random_upper
                ),
                "random_dominance_pass": random_pass,
                "selective_point_pass": candidate_pass,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    diagnostic = protocol[
        "safety_cost_frontier_diagnostic"
    ]

    if diagnostic["status"] != (
        "frozen_before_frontier_evaluation"
    ):
        raise RuntimeError(
            "Frontier diagnostic was not frozen"
        )
    if diagnostic["overall_milestone"][
        "overall_pass_allowed"
    ]:
        raise RuntimeError(
            "Overall pass must remain disabled"
        )

    setup_id = diagnostic["primary_setup"]
    family = diagnostic[
        "primary_feature_family"
    ]
    risk_limit = float(
        diagnostic["risk"]["maximum"]
    )

    setup = next(
        item
        for item in protocol["optional_monitor_setups"]
        if item["setup_id"] == setup_id
    )
    optional_cost_ms = float(
        setup["measured_mean_cost_ms"]
    )

    embedding_manifest = json.loads(
        EMBEDDING_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    embedding_runtime_ms = float(
        embedding_manifest["runtime"][
            "end_to_end_ms_per_example"
        ]
    )

    value_manifest = json.loads(
        VALUE_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    if value_manifest["primary_comparison"][
        "predictability_criterion_pass"
    ]:
        raise RuntimeError(
            "Expected frozen predictability no-go"
        )

    module = load_value_module()

    runtime = benchmark_primary_inference(
        module=module,
        protocol=protocol,
        diagnostic=diagnostic,
    )
    runtime.to_csv(
        RUNTIME_PATH,
        index=False,
    )

    inference_runtime_ms = float(
        np.average(
            runtime["mean_per_example_ms"],
            weights=runtime["eval_n"],
        )
    )
    learned_fixed_overhead_ms = (
        embedding_runtime_ms
        + inference_runtime_ms
    )

    curves = pd.read_csv(CURVES_PATH)
    random = pd.read_csv(RANDOM_PATH)

    curves = curves.loc[
        (curves["setup_id"] == setup_id)
        & (curves["feature_family"] == family)
    ].copy()
    random = random.loc[
        (random["setup_id"] == setup_id)
        & (random["feature_family"] == family)
    ].copy()

    random_summary = summarize_random(
        random=random,
        optional_cost_ms=optional_cost_ms,
        risk_limit=risk_limit,
    )
    random_summary.to_csv(
        RANDOM_SUMMARY_PATH,
        index=False,
    )

    frontier = build_deterministic_frontier(
        curves=curves,
        optional_cost_ms=optional_cost_ms,
        learned_fixed_overhead_ms=(
            learned_fixed_overhead_ms
        ),
        risk_limit=risk_limit,
    )
    frontier.to_csv(
        FRONTIER_PATH,
        index=False,
    )

    candidates = build_candidates(
        frontier=frontier,
        random_summary=random_summary,
        risk_limit=risk_limit,
    )
    candidates.to_csv(
        CANDIDATE_PATH,
        index=False,
    )

    passing = candidates.loc[
        candidates["selective_point_pass"]
    ].copy()

    if passing.empty:
        selected: dict[str, Any] | None = None
        frontier_condition_pass = False
    else:
        selected_row = passing.sort_values(
            [
                "learned_incremental_cost_ms_per_example",
                "learned_recall",
                "learned_acquisition_rate",
            ],
            ascending=[True, False, True],
        ).iloc[0]
        selected = selected_row.to_dict()
        frontier_condition_pass = True

    overall_milestone_pass = bool(
        frontier_condition_pass
        and value_manifest["primary_comparison"][
            "predictability_criterion_pass"
        ]
    )

    summary = {
        "artifact": (
            "common_risk_safety_cost_frontier_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "development_only_frontier_completed"
        ),
        "primary_setup": setup_id,
        "primary_feature_family": family,
        "risk_limit": risk_limit,
        "cost_accounting": {
            "optional_monitor_mean_cost_ms": (
                optional_cost_ms
            ),
            "embedding_mean_runtime_ms_per_example": (
                embedding_runtime_ms
            ),
            "pca_and_value_inference_mean_ms_per_example": (
                inference_runtime_ms
            ),
            "learned_fixed_overhead_ms_per_example": (
                learned_fixed_overhead_ms
            ),
            "common_base_monitor_cost_excluded": True,
        },
        "frontier_condition_pass": (
            frontier_condition_pass
        ),
        "selected_selective_point": selected,
        "predictability_condition_pass": False,
        "overall_milestone_pass": (
            overall_milestone_pass
        ),
        "overall_project_status": "no-go",
        "reason": (
            "the prespecified value-predictability "
            "confidence interval includes zero"
        ),
        "scope": {
            "development_rows": 1687,
            "final_test_used": False,
            "held_out_shift_used": False,
        },
        "outputs": {
            "runtime": str(
                RUNTIME_PATH.relative_to(ROOT)
            ),
            "random_summary": str(
                RANDOM_SUMMARY_PATH.relative_to(ROOT)
            ),
            "frontier": str(
                FRONTIER_PATH.relative_to(ROOT)
            ),
            "candidates": str(
                CANDIDATE_PATH.relative_to(ROOT)
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Common-Risk Safety-Cost Frontier",
        "",
        "This is a development-only diagnostic.",
        "",
        "## Cost accounting",
        "",
        (
            f"- Optional Qwen mean cost: "
            f"{optional_cost_ms:.6f} ms/acquired example"
        ),
        (
            f"- Frozen embedding mean runtime: "
            f"{embedding_runtime_ms:.6f} ms/example"
        ),
        (
            f"- PCA plus value-estimator inference: "
            f"{inference_runtime_ms:.6f} ms/example"
        ),
        (
            f"- Learned fixed overhead: "
            f"{learned_fixed_overhead_ms:.6f} ms/example"
        ),
        "",
        "## Selective-point candidates",
        "",
        "```text",
        candidates.to_string(index=False),
        "```",
        "",
        (
            "- Frontier condition: "
            f"{'PASS' if frontier_condition_pass else 'NO-GO'}"
        ),
        "- Predictability condition: NO-GO",
        "- Overall milestone: NO-GO",
        "",
        "The overall result remains no-go regardless of the frontier "
        "condition because the prespecified value-predictability confidence "
        "interval included zero.",
        "",
    ]
    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=== COMMON-RISK SAFETY-COST SUMMARY ===")
    print(
        "optional monitor mean cost ms:",
        f"{optional_cost_ms:.6f}",
    )
    print(
        "embedding mean runtime ms/example:",
        f"{embedding_runtime_ms:.6f}",
    )
    print(
        "PCA + value inference ms/example:",
        f"{inference_runtime_ms:.6f}",
    )
    print(
        "learned fixed overhead ms/example:",
        f"{learned_fixed_overhead_ms:.6f}",
    )
    print()
    print(candidates.to_string(index=False))
    print()
    print(
        "frontier condition:",
        (
            "PASS"
            if frontier_condition_pass
            else "NO-GO"
        ),
    )
    print("predictability condition: NO-GO")
    print("overall milestone: NO-GO")
    print("overall project status: no-go")


if __name__ == "__main__":
    main()
