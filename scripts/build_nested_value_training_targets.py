#!/usr/bin/env python3
"""Build leakage-safe nested training targets for value estimation.

The existing downstream target builder is executed with its inner-OOF
probability function wrapped so the exact, already-validated downstream
cross-fitting procedure is reused. For each value-estimator outer fold, only
its outer-training partition contributes training targets.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)
SOURCE_BUILDER_PATH = (
    ROOT / "scripts/build_cross_fitted_decision_value_targets.py"
)
REPORT_DIR = ROOT / "reports/decision_value_real_data"
OUTER_ASSIGNMENTS_PATH = (
    REPORT_DIR / "development_outer_fold_assignments.csv"
)
FROZEN_FOLD_METRICS_PATH = (
    REPORT_DIR / "cross_fitted_target_fold_metrics.csv"
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


def sha256_ids(values: pd.Series | list[str]) -> str:
    ordered = sorted(str(value) for value in values)
    return hashlib.sha256(
        "\n".join(ordered).encode("utf-8")
    ).hexdigest()


def sha256_fold_mapping(
    example_ids: pd.Series,
    fold_ids: np.ndarray,
) -> str:
    pairs = sorted(
        zip(
            example_ids.astype(str).tolist(),
            fold_ids.astype(int).tolist(),
            strict=True,
        )
    )
    payload = "\n".join(
        f"{example_id}\t{fold_id}"
        for example_id, fold_id in pairs
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def load_source_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cross_fitted_target_builder_for_nested_capture",
        SOURCE_BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {SOURCE_BUILDER_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_call_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    raise RuntimeError(
        f"Missing captured argument {name}"
    )


def select_dataframe(
    result: Any,
    required_columns: set[str],
) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []

    if isinstance(result, pd.DataFrame):
        candidates.append(result)
    elif isinstance(result, (tuple, list)):
        candidates.extend(
            item
            for item in result
            if isinstance(item, pd.DataFrame)
        )
    elif isinstance(result, dict):
        candidates.extend(
            item
            for item in result.values()
            if isinstance(item, pd.DataFrame)
        )

    matching = [
        frame
        for frame in candidates
        if required_columns.issubset(frame.columns)
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "Could not identify a unique dataframe with columns "
            f"{sorted(required_columns)}; matches={len(matching)}"
        )
    return matching[0]


def build_nested_targets(
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    module = load_source_module()

    expected_functions = {
        "build_cross_fitted_targets",
        "inner_oof_probabilities",
        "select_threshold_at_fpr",
    }
    missing = [
        name
        for name in expected_functions
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f"Source builder missing functions: {missing}"
        )

    original_inner = module.inner_oof_probabilities
    captures: list[dict[str, Any]] = []

    def capture_inner(*args: Any, **kwargs: Any) -> Any:
        outer_train = extract_call_argument(
            args,
            kwargs,
            "outer_train",
            0,
        )
        features = extract_call_argument(
            args,
            kwargs,
            "features",
            1,
        )
        inner_folds = extract_call_argument(
            args,
            kwargs,
            "inner_folds",
            2,
        )
        random_state = extract_call_argument(
            args,
            kwargs,
            "random_state",
            3,
        )

        scores, fold_ids = original_inner(
            *args,
            **kwargs,
        )

        captures.append(
            {
                "outer_train": outer_train.copy(),
                "features": list(features),
                "inner_folds": int(inner_folds),
                "random_state": int(random_state),
                "scores": np.asarray(
                    scores,
                    dtype=float,
                ).copy(),
                "fold_ids": np.asarray(
                    fold_ids,
                    dtype=int,
                ).copy(),
            }
        )
        return scores, fold_ids

    module.inner_oof_probabilities = capture_inner
    try:
        build_result = module.build_cross_fitted_targets(
            protocol
        )
    finally:
        module.inner_oof_probabilities = original_inner

    outer_folds = int(
        protocol["cross_fitting"]["outer_folds"]
    )
    inner_folds = int(
        protocol["cross_fitting"]["inner_folds"]
    )
    setups = protocol["optional_monitor_setups"]
    expected_capture_n = (
        outer_folds * len(setups) * 2
    )

    if len(captures) != expected_capture_n:
        raise RuntimeError(
            f"Expected {expected_capture_n} inner-OOF captures, "
            f"found {len(captures)}"
        )

    frozen_fold_metrics = pd.read_csv(
        FROZEN_FOLD_METRICS_PATH
    )
    outer_assignments = pd.read_csv(
        OUTER_ASSIGNMENTS_PATH
    )

    required_fold_metric_columns = {
        "setup_id",
        "outer_fold",
        "base_threshold",
        "augmented_threshold",
        "outer_train_n",
        "outer_test_n",
        "outer_train_id_sha256",
        "outer_test_id_sha256",
    }
    if not required_fold_metric_columns.issubset(
        frozen_fold_metrics.columns
    ):
        raise RuntimeError(
            "Frozen fold metrics schema is incomplete"
        )

    result_fold_metrics = select_dataframe(
        build_result,
        {
            "setup_id",
            "outer_fold",
            "base_threshold",
            "augmented_threshold",
            "base_inner_oof_fpr",
            "augmented_inner_oof_fpr",
        },
    )

    compare_columns = [
        "setup_id",
        "outer_fold",
        "base_threshold",
        "augmented_threshold",
        "base_inner_oof_fpr",
        "augmented_inner_oof_fpr",
    ]
    left = result_fold_metrics[
        compare_columns
    ].sort_values(
        ["outer_fold", "setup_id"]
    ).reset_index(drop=True)
    right = frozen_fold_metrics[
        compare_columns
    ].sort_values(
        ["outer_fold", "setup_id"]
    ).reset_index(drop=True)

    if left[["setup_id", "outer_fold"]].equals(
        right[["setup_id", "outer_fold"]]
    ) is False:
        raise RuntimeError(
            "Recomputed and frozen fold identities differ"
        )
    for column in compare_columns[2:]:
        if not np.allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Recomputed frozen metric differs: {column}"
            )

    fpr_target = float(
        protocol["operating_risk"]["target"]
    )

    target_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    capture_index = 0

    for value_outer_fold in range(outer_folds):
        outer_eval_ids = set(
            outer_assignments.loc[
                outer_assignments["outer_fold"]
                == value_outer_fold,
                "example_id",
            ].astype(str)
        )

        for setup in setups:
            setup_id = str(setup["setup_id"])
            expected_base_features = list(
                setup["base_features"]
            )
            expected_augmented_features = list(
                setup["augmented_features"]
            )

            base_capture = captures[capture_index]
            augmented_capture = captures[
                capture_index + 1
            ]
            capture_index += 2

            if base_capture["features"] != (
                expected_base_features
            ):
                raise RuntimeError(
                    f"Unexpected base features for {setup_id}, "
                    f"outer fold {value_outer_fold}"
                )
            if augmented_capture["features"] != (
                expected_augmented_features
            ):
                raise RuntimeError(
                    f"Unexpected augmented features for {setup_id}, "
                    f"outer fold {value_outer_fold}"
                )

            base_train = base_capture["outer_train"]
            augmented_train = augmented_capture[
                "outer_train"
            ]

            base_ids = base_train[
                "example_id"
            ].astype(str).reset_index(drop=True)
            augmented_ids = augmented_train[
                "example_id"
            ].astype(str).reset_index(drop=True)

            if not base_ids.equals(augmented_ids):
                raise RuntimeError(
                    "Base and augmented outer-training rows differ"
                )
            if set(base_ids).intersection(outer_eval_ids):
                raise RuntimeError(
                    "Current outer evaluation row entered target generation"
                )

            base_fold_ids = base_capture["fold_ids"]
            augmented_fold_ids = augmented_capture[
                "fold_ids"
            ]
            if not np.array_equal(
                base_fold_ids,
                augmented_fold_ids,
            ):
                raise RuntimeError(
                    "Base and augmented inner fold assignments differ"
                )
            if set(np.unique(base_fold_ids)) != set(
                range(inner_folds)
            ):
                raise RuntimeError(
                    "Inner fold assignments are incomplete"
                )

            y = base_train["y"].to_numpy(dtype=int)
            base_scores = base_capture["scores"]
            augmented_scores = augmented_capture[
                "scores"
            ]

            base_threshold_result = (
                module.select_threshold_at_fpr(
                    y,
                    base_scores,
                    fpr_target,
                )
            )
            augmented_threshold_result = (
                module.select_threshold_at_fpr(
                    y,
                    augmented_scores,
                    fpr_target,
                )
            )

            frozen_row = frozen_fold_metrics.loc[
                (
                    frozen_fold_metrics["setup_id"]
                    == setup_id
                )
                & (
                    frozen_fold_metrics["outer_fold"]
                    == value_outer_fold
                )
            ]
            if len(frozen_row) != 1:
                raise RuntimeError(
                    "Could not identify frozen fold metric row"
                )
            frozen = frozen_row.iloc[0]

            if not np.isclose(
                base_threshold_result.threshold,
                float(frozen["base_threshold"]),
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Base threshold does not match frozen result"
                )
            if not np.isclose(
                augmented_threshold_result.threshold,
                float(frozen["augmented_threshold"]),
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "Augmented threshold does not match frozen result"
                )

            base_prediction = (
                base_scores
                >= base_threshold_result.threshold
            ).astype(int)
            augmented_prediction = (
                augmented_scores
                >= augmented_threshold_result.threshold
            ).astype(int)

            base_loss = (
                base_prediction != y
            ).astype(int)
            augmented_loss = (
                augmented_prediction != y
            ).astype(int)
            realized_value = (
                base_loss - augmented_loss
            ).astype(int)

            false_positive_reduction = (
                (
                    (base_prediction == 1)
                    & (y == 0)
                ).astype(int)
                - (
                    (augmented_prediction == 1)
                    & (y == 0)
                ).astype(int)
            )
            false_negative_reduction = (
                (
                    (base_prediction == 0)
                    & (y == 1)
                ).astype(int)
                - (
                    (augmented_prediction == 0)
                    & (y == 1)
                ).astype(int)
            )

            inner_mapping_hash = sha256_fold_mapping(
                base_ids,
                base_fold_ids,
            )
            outer_train_hash = sha256_ids(base_ids)
            outer_eval_hash = sha256_ids(
                list(outer_eval_ids)
            )

            if outer_train_hash != str(
                frozen["outer_train_id_sha256"]
            ):
                raise RuntimeError(
                    "Outer-training hash differs from frozen target builder"
                )
            if outer_eval_hash != str(
                frozen["outer_test_id_sha256"]
            ):
                raise RuntimeError(
                    "Outer-evaluation hash differs from frozen target builder"
                )

            target_frames.append(
                pd.DataFrame(
                    {
                        "example_id": base_ids,
                        "setup_id": setup_id,
                        "optional_monitor": str(
                            setup["optional_monitor"]
                        ),
                        "value_outer_fold": (
                            value_outer_fold
                        ),
                        "downstream_inner_fold": (
                            base_fold_ids
                        ),
                        "y": y,
                        "base_threshold": (
                            base_threshold_result.threshold
                        ),
                        "augmented_threshold": (
                            augmented_threshold_result.threshold
                        ),
                        "base_prediction": base_prediction,
                        "augmented_prediction": (
                            augmented_prediction
                        ),
                        "base_loss": base_loss,
                        "augmented_loss": augmented_loss,
                        "realized_decision_value": (
                            realized_value
                        ),
                        "decision_changed": (
                            base_prediction
                            != augmented_prediction
                        ).astype(int),
                        "false_positive_reduction": (
                            false_positive_reduction
                        ),
                        "false_negative_reduction": (
                            false_negative_reduction
                        ),
                        "base_uncertainty": np.minimum(
                            base_scores,
                            1.0 - base_scores,
                        ),
                        "outer_train_n": len(base_train),
                        "outer_eval_n": len(outer_eval_ids),
                        "outer_train_id_sha256": (
                            outer_train_hash
                        ),
                        "outer_eval_id_sha256": (
                            outer_eval_hash
                        ),
                        "downstream_inner_fold_assignment_sha256": (
                            inner_mapping_hash
                        ),
                        "target_generation_scope": (
                            "inner_oof_within_current_outer_train"
                        ),
                    }
                )
            )

            metric_rows.append(
                {
                    "setup_id": setup_id,
                    "value_outer_fold": value_outer_fold,
                    "outer_train_n": len(base_train),
                    "outer_eval_n": len(outer_eval_ids),
                    "base_threshold": (
                        base_threshold_result.threshold
                    ),
                    "augmented_threshold": (
                        augmented_threshold_result.threshold
                    ),
                    "base_inner_oof_fpr": (
                        base_threshold_result.fpr
                    ),
                    "base_inner_oof_recall": (
                        base_threshold_result.recall
                    ),
                    "augmented_inner_oof_fpr": (
                        augmented_threshold_result.fpr
                    ),
                    "augmented_inner_oof_recall": (
                        augmented_threshold_result.recall
                    ),
                    "mean_realized_decision_value": float(
                        realized_value.mean()
                    ),
                    "positive_value_n": int(
                        np.sum(realized_value == 1)
                    ),
                    "zero_value_n": int(
                        np.sum(realized_value == 0)
                    ),
                    "negative_value_n": int(
                        np.sum(realized_value == -1)
                    ),
                    "decision_change_rate": float(
                        np.mean(
                            base_prediction
                            != augmented_prediction
                        )
                    ),
                    "outer_train_id_sha256": outer_train_hash,
                    "outer_eval_id_sha256": outer_eval_hash,
                    "downstream_inner_fold_assignment_sha256": (
                        inner_mapping_hash
                    ),
                    "base_model_random_state": (
                        base_capture["random_state"]
                    ),
                    "augmented_model_random_state": (
                        augmented_capture["random_state"]
                    ),
                }
            )

    targets = pd.concat(
        target_frames,
        ignore_index=True,
    ).sort_values(
        [
            "value_outer_fold",
            "setup_id",
            "example_id",
        ]
    ).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["value_outer_fold", "setup_id"]
    ).reset_index(drop=True)

    expected_rows = (
        len(outer_assignments)
        * (outer_folds - 1)
        * len(setups)
    )
    if len(targets) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} nested rows, found {len(targets)}"
        )
    if targets.duplicated(
        ["setup_id", "value_outer_fold", "example_id"]
    ).any():
        raise RuntimeError(
            "Duplicate nested target key"
        )
    if set(targets["realized_decision_value"].unique()) - {
        -1,
        0,
        1,
    }:
        raise RuntimeError(
            "Unexpected realized decision-value label"
        )

    per_example_setup = targets.groupby(
        ["setup_id", "example_id"]
    ).size()
    if not (
        per_example_setup == outer_folds - 1
    ).all():
        raise RuntimeError(
            "Each example must appear in four training folds per setup"
        )

    forbidden_predictor_columns = {
        "optional_monitor_score",
        "augmented_score",
        "compact_unsafe_score",
        "qwen_prompt_response_score",
        "source_dataset",
        "attack_family",
        "prompt",
        "response",
    }
    if forbidden_predictor_columns.intersection(
        targets.columns
    ):
        raise RuntimeError(
            "Optional-monitor or forbidden predictor entered target artifact"
        )

    manifest = {
        "artifact": "nested_value_training_targets_v1",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "development_only_nested_training_targets_completed"
        ),
        "rows": len(targets),
        "setups": [
            str(setup["setup_id"])
            for setup in setups
        ],
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "target_generation": (
            "inner-OOF downstream base and augmented decisions "
            "within each current outer-training partition"
        ),
        "global_outer_target_rows_used_for_training": False,
        "current_outer_evaluation_fold_excluded": True,
        "final_test_used": False,
        "held_out_shift_used": False,
        "optional_monitor_outputs_written_as_predictors": False,
        "expected_rows": expected_rows,
        "unique_examples": int(
            targets["example_id"].nunique()
        ),
        "rows_per_example_per_setup": outer_folds - 1,
        "realized_value_counts": {
            str(int(key)): int(value)
            for key, value in targets[
                "realized_decision_value"
            ].value_counts().sort_index().items()
        },
        "inputs": {
            "protocol": str(
                PROTOCOL_PATH.relative_to(ROOT)
            ),
            "protocol_sha256": sha256_file(
                PROTOCOL_PATH
            ),
            "source_target_builder": str(
                SOURCE_BUILDER_PATH.relative_to(ROOT)
            ),
            "source_target_builder_sha256": sha256_file(
                SOURCE_BUILDER_PATH
            ),
            "outer_assignments": str(
                OUTER_ASSIGNMENTS_PATH.relative_to(ROOT)
            ),
            "outer_assignments_sha256": sha256_file(
                OUTER_ASSIGNMENTS_PATH
            ),
            "frozen_fold_metrics": str(
                FROZEN_FOLD_METRICS_PATH.relative_to(ROOT)
            ),
            "frozen_fold_metrics_sha256": sha256_file(
                FROZEN_FOLD_METRICS_PATH
            ),
        },
        "source_function_signatures": {
            "inner_oof_probabilities": str(
                inspect.signature(
                    module.inner_oof_probabilities
                )
            ),
            "select_threshold_at_fpr": str(
                inspect.signature(
                    module.select_threshold_at_fpr
                )
            ),
        },
    }

    return targets, metrics, manifest


def write_summary(
    path: Path,
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Nested Value-Estimator Training Targets",
        "",
        "These are development-only training targets for the value estimator.",
        "For each value-estimator outer fold, the current outer evaluation",
        "fold is excluded before downstream target generation.",
        "",
        f"- Rows: {len(targets)}",
        f"- Unique examples: {targets['example_id'].nunique()}",
        f"- Outer folds: {manifest['outer_folds']}",
        f"- Inner folds: {manifest['inner_folds']}",
        "- Global outer targets used for estimator training: no",
        "- Final test used: no",
        "- Held-out shift used: no",
        "",
        "## Setup summary",
        "",
    ]

    summary = targets.groupby("setup_id").agg(
        rows=("example_id", "size"),
        unique_examples=("example_id", "nunique"),
        mean_value=("realized_decision_value", "mean"),
        positive_value_n=(
            "realized_decision_value",
            lambda series: int((series == 1).sum()),
        ),
        zero_value_n=(
            "realized_decision_value",
            lambda series: int((series == 0).sum()),
        ),
        negative_value_n=(
            "realized_decision_value",
            lambda series: int((series == -1).sum()),
        ),
    ).reset_index()

    lines.append("```text")
    lines.append(summary.to_string(index=False))
    lines.append("```")
    lines.extend(
        [
            "",
            "The artifact contains targets and audit metadata only. Predictor",
            "features are merged later under each outer fold, with embedding PCA",
            "fit only on that outer-training partition.",
            "",
            "This artifact does not itself establish value predictability or pass",
            "the professor's development milestone.",
        ]
    )
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    value_estimator = protocol["value_estimator"]
    required = {
        "current_outer_evaluation_fold_excluded_from_target_generation": True,
        "global_outer_target_rows_allowed_for_estimator_training": False,
        "training_target_artifact": (
            "nested_value_training_targets.parquet"
        ),
    }
    for key, expected in required.items():
        if value_estimator.get(key) != expected:
            raise RuntimeError(
                f"Protocol mismatch for {key}"
            )

    targets, metrics, manifest = build_nested_targets(
        protocol
    )

    parquet_path = (
        REPORT_DIR / "nested_value_training_targets.parquet"
    )
    csv_path = (
        REPORT_DIR / "nested_value_training_targets.csv"
    )
    metrics_path = (
        REPORT_DIR / "nested_value_training_target_fold_metrics.csv"
    )
    manifest_path = (
        REPORT_DIR / "nested_value_training_target_manifest.json"
    )
    summary_path = (
        REPORT_DIR / "nested_value_training_target_summary.md"
    )

    targets.to_parquet(parquet_path, index=False)
    targets.to_csv(csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    write_summary(
        summary_path,
        targets,
        metrics,
        manifest,
    )

    manifest["outputs"] = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            parquet_path,
            csv_path,
            metrics_path,
            summary_path,
        ]
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("nested value-estimator training targets completed")
    print("rows:", len(targets))
    print("unique examples:", targets["example_id"].nunique())
    print(
        "rows/example/setup:",
        targets.groupby(
            ["setup_id", "example_id"]
        ).size().unique().tolist(),
    )
    print(
        "realized value counts:",
        targets["realized_decision_value"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print("outer evaluation leakage: 0")
    print("final test used: False")
    print("held-out shift used: False")
    print("artifact:", parquet_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
