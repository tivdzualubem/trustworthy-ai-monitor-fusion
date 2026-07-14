#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("data/processed/monitor_score_cache_v3.parquet")
POLICY_SCRIPT = Path("scripts/run_policy_evaluation_v3.py")

ARTIFACT_DIR = Path("artifacts/nested_leave_group_out_v1")
REPORT_DIR = Path("reports/nested_leave_group_out_v1")
MANIFEST = Path(
    "data/metadata/nested_leave_group_out_v1_manifest.json"
)

CHEAP_FEATURES = ["rule_score", "compact_unsafe_score"]
FULL_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]
TARGET_FPRS = [0.01, 0.025, 0.05, 0.10]
INNER_SELECTION_FRACTION = 1.0 / 3.0
BASE_SEED = 20260714

SOURCE_FOLDS = [
    "jailbreakbench_judge_comparison",
    "wildguardtest",
    "xstest_safe_gpt4",
]
FAMILY_FOLDS = ["GCG", "PAIR", "random_search"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_policy_module():
    spec = importlib.util.spec_from_file_location(
        "policy_evaluation_v3",
        POLICY_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {POLICY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_calibrated_logistic() -> CalibratedClassifierCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=5,
        ensemble=True,
    )


def deterministic_inner_split(
    development: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = development["y"].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_SELECTION_FRACTION,
        random_state=seed,
    )
    train_index, selection_index = next(
        splitter.split(np.zeros(len(development)), y)
    )
    inner_train = development.iloc[train_index].copy()
    inner_selection = development.iloc[selection_index].copy()

    for name, frame in [
        ("inner_train", inner_train),
        ("inner_selection", inner_selection),
    ]:
        if frame["y"].nunique() != 2:
            raise RuntimeError(f"{name} does not contain both classes")

    return inner_train, inner_selection


def serialize_fold_bundle(
    *,
    scheme: str,
    fold_name: str,
    seed: int,
    inner_train: pd.DataFrame,
    inner_selection: pd.DataFrame,
    outer_test: pd.DataFrame,
) -> tuple[dict, Path, dict]:
    safe_name = fold_name.replace("/", "_").replace(" ", "_")
    fold_dir = ARTIFACT_DIR / scheme / safe_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    cheap_model = build_calibrated_logistic()
    full_model = build_calibrated_logistic()

    cheap_model.fit(
        inner_train[CHEAP_FEATURES],
        inner_train["y"].astype(int),
    )
    full_model.fit(
        inner_train[FULL_FEATURES],
        inner_train["y"].astype(int),
    )

    cheap_path = fold_dir / "cheap_router.joblib"
    full_path = fold_dir / "full_information_fusion.joblib"
    bundle_path = fold_dir / "fusion_bundle.joblib"

    joblib.dump(cheap_model, cheap_path, compress=3)
    joblib.dump(full_model, full_path, compress=3)

    provenance = {
        "artifact": "nested_fold_fusion_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scheme": scheme,
        "outer_fold": fold_name,
        "seed": seed,
        "cache_path": str(CACHE),
        "cache_sha256": sha256(CACHE),
        "inner_train_rows": int(len(inner_train)),
        "inner_selection_rows": int(len(inner_selection)),
        "outer_test_rows": int(len(outer_test)),
        "inner_train_label_counts": {
            str(key): int(value)
            for key, value in inner_train["y"]
            .value_counts()
            .sort_index()
            .items()
        },
        "inner_selection_label_counts": {
            str(key): int(value)
            for key, value in inner_selection["y"]
            .value_counts()
            .sort_index()
            .items()
        },
        "outer_test_label_counts": {
            str(key): int(value)
            for key, value in outer_test["y"]
            .value_counts()
            .sort_index()
            .items()
        },
        "inner_train_id_sha256": id_hash(
            inner_train["example_id"]
        ),
        "inner_selection_id_sha256": id_hash(
            inner_selection["example_id"]
        ),
        "outer_test_id_sha256": id_hash(
            outer_test["example_id"]
        ),
        "cheap_features": CHEAP_FEATURES,
        "full_features": FULL_FEATURES,
        "model_family": (
            "5-fold sigmoid-calibrated class-balanced logistic regression"
        ),
        "outer_rows_used_for_fit_or_selection": False,
        "existing_split_column_used": False,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }

    bundle = {
        "cheap_router": cheap_model,
        "full_information_fusion": full_model,
        "cheap_features": CHEAP_FEATURES,
        "full_features": FULL_FEATURES,
        "provenance": provenance,
    }
    joblib.dump(bundle, bundle_path, compress=3)

    loaded_cheap = joblib.load(cheap_path)
    loaded_full = joblib.load(full_path)
    loaded_bundle = joblib.load(bundle_path)

    for frame_name, frame in [
        ("inner_train", inner_train),
        ("inner_selection", inner_selection),
        ("outer_test", outer_test),
    ]:
        cheap_memory = cheap_model.predict_proba(
            frame[CHEAP_FEATURES]
        )[:, 1]
        cheap_standalone = loaded_cheap.predict_proba(
            frame[CHEAP_FEATURES]
        )[:, 1]
        cheap_bundle = loaded_bundle[
            "cheap_router"
        ].predict_proba(
            frame[loaded_bundle["cheap_features"]]
        )[:, 1]

        full_memory = full_model.predict_proba(
            frame[FULL_FEATURES]
        )[:, 1]
        full_standalone = loaded_full.predict_proba(
            frame[FULL_FEATURES]
        )[:, 1]
        full_bundle = loaded_bundle[
            "full_information_fusion"
        ].predict_proba(
            frame[loaded_bundle["full_features"]]
        )[:, 1]

        if not np.array_equal(cheap_memory, cheap_standalone):
            raise RuntimeError(
                f"Cheap standalone mismatch: {scheme}/{fold_name}/{frame_name}"
            )
        if not np.array_equal(cheap_memory, cheap_bundle):
            raise RuntimeError(
                f"Cheap bundle mismatch: {scheme}/{fold_name}/{frame_name}"
            )
        if not np.array_equal(full_memory, full_standalone):
            raise RuntimeError(
                f"Full standalone mismatch: {scheme}/{fold_name}/{frame_name}"
            )
        if not np.array_equal(full_memory, full_bundle):
            raise RuntimeError(
                f"Full bundle mismatch: {scheme}/{fold_name}/{frame_name}"
            )

    artifact_manifest = {
        **provenance,
        "artifacts": {
            "cheap_router": {
                "path": str(cheap_path),
                "sha256": sha256(cheap_path),
            },
            "full_information_fusion": {
                "path": str(full_path),
                "sha256": sha256(full_path),
            },
            "fusion_bundle": {
                "path": str(bundle_path),
                "sha256": sha256(bundle_path),
            },
        },
        "serialization_verification": {
            "exact_probability_match": True,
            "checked_on": [
                "inner_train",
                "inner_selection",
                "outer_test",
            ],
        },
    }

    manifest_path = fold_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2),
        encoding="utf-8",
    )

    return loaded_bundle, bundle_path, artifact_manifest


def select_prespecified_policies(
    *,
    policy_module,
    inner_train: pd.DataFrame,
    inner_selection: pd.DataFrame,
    bundle: dict,
    bundle_path: Path,
    timing: dict[str, float],
) -> tuple[pd.DataFrame, list[dict], int]:
    # Keep the policy-details provenance correct for this fold.
    policy_module.BUNDLE = bundle_path

    candidates = policy_module.build_baseline_candidates(
        inner_train,
        inner_selection,
        bundle,
        timing,
    )

    selected_rows = [
        policy_module.choose_family_candidate(group)
        for _, group in candidates.groupby("family", sort=True)
    ]
    selected = pd.DataFrame(selected_rows).sort_values(
        ["recall", "avg_cost_ms", "fpr"],
        ascending=[False, True, True],
    )
    primary_id = str(selected.iloc[0]["policy_id"])

    policies = []
    for row in selected.to_dict(orient="records"):
        policies.append(
            {
                "family": row["family"],
                "policy_id": row["policy_id"],
                "frozen_role": (
                    "primary_budget_aware_policy"
                    if str(row["policy_id"]) == primary_id
                    else "required_baseline"
                ),
                "policy_details": json.loads(
                    row["policy_details_json"]
                ),
            }
        )

    return selected, policies, int(len(candidates))


def evaluate_prespecified_policies(
    *,
    policy_module,
    scheme: str,
    fold_name: str,
    outer_test: pd.DataFrame,
    policies: list[dict],
    bundle: dict,
    timing: dict[str, float],
) -> tuple[list[dict], list[pd.DataFrame]]:
    y = outer_test["y"].astype(int).to_numpy()
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for policy in policies:
        pred, cost, runtime = policy_module.apply_frozen_policy(
            policy,
            outer_test,
            bundle,
            timing,
        )
        result = policy_module.finite_sample_metrics(y, pred)
        metric_rows.append(
            {
                "scheme": scheme,
                "outer_fold": fold_name,
                "family": policy["family"],
                "policy_id": policy["policy_id"],
                "frozen_role": policy["frozen_role"],
                **result,
                "avg_estimated_cost_ms": float(np.mean(cost)),
                "median_estimated_cost_ms": float(np.median(cost)),
                "p95_estimated_cost_ms": float(
                    np.quantile(cost, 0.95)
                ),
                **runtime,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "example_id": outer_test[
                        "example_id"
                    ].astype(str),
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "y": y,
                    "family": policy["family"],
                    "policy_id": policy["policy_id"],
                    "intercept_pred": pred.astype(int),
                    "estimated_cost_ms": cost,
                    "qwen_called": (
                        np.ones(len(outer_test), dtype=int)
                        if runtime["qwen_call_rate"] == 1.0
                        else np.nan
                    ),
                }
            )
        )

    return metric_rows, prediction_frames



def select_selective_policy_feasible(
    policy_module,
    y: np.ndarray,
    cheap_probability: np.ndarray,
    full_probability: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    """Select a feasible policy, including route-all boundary candidates."""
    low_grid = np.unique(
        np.concatenate(
            [
                np.quantile(
                    cheap_probability,
                    np.linspace(0.02, 0.75, 18),
                ),
                [
                    np.nextafter(
                        float(np.min(cheap_probability)),
                        -np.inf,
                    )
                ],
            ]
        )
    )
    high_grid = np.unique(
        np.concatenate(
            [
                np.quantile(
                    cheap_probability,
                    np.linspace(0.25, 0.98, 18),
                ),
                [
                    np.nextafter(
                        float(np.max(cheap_probability)),
                        np.inf,
                    )
                ],
            ]
        )
    )
    final_grid = np.unique(
        np.concatenate(
            [
                np.quantile(
                    full_probability,
                    np.linspace(0.0, 1.0, 121),
                ),
                [
                    0.0,
                    1.0,
                    np.nextafter(
                        float(np.max(full_probability)),
                        np.inf,
                    ),
                ],
            ]
        )
    )

    best_key = None
    best_payload = None

    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue

            cheap_allow = cheap_probability <= low
            cheap_intercept = cheap_probability >= high
            call_qwen = ~(cheap_allow | cheap_intercept)

            for final_threshold in final_grid:
                pred = cheap_intercept.copy()
                pred[call_qwen] = (
                    full_probability[call_qwen]
                    >= final_threshold
                )

                result = policy_module.binary_metrics(y, pred)
                if result["fpr"] > target_fpr + 1e-12:
                    continue

                qwen_call_rate = float(np.mean(call_qwen))
                precision = (
                    float(result["precision"])
                    if np.isfinite(result["precision"])
                    else -1.0
                )

                key = (
                    float(result["recall"]),
                    -qwen_call_rate,
                    precision,
                    -float(result["fpr"]),
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        "router_low_allow_threshold": float(low),
                        "router_high_intercept_threshold": float(high),
                        "full_fusion_threshold": float(
                            final_threshold
                        ),
                        "selection_recall": float(
                            result["recall"]
                        ),
                        "selection_fpr": float(result["fpr"]),
                        "selection_precision": float(
                            result["precision"]
                        ),
                        "selection_qwen_call_rate": (
                            qwen_call_rate
                        ),
                    }

    if best_payload is None:
        raise RuntimeError(
            "No selective policy even after adding "
            f"boundary candidates for target {target_fpr}"
        )

    return best_payload


def select_and_evaluate_fusion(
    *,
    policy_module,
    scheme: str,
    fold_name: str,
    inner_selection: pd.DataFrame,
    outer_test: pd.DataFrame,
    bundle: dict,
    timing: dict[str, float],
) -> tuple[list[dict], list[dict], list[pd.DataFrame]]:
    y_selection = inner_selection["y"].astype(int).to_numpy()
    y_test = outer_test["y"].astype(int).to_numpy()

    cheap_selection = bundle["cheap_router"].predict_proba(
        inner_selection[bundle["cheap_features"]]
    )[:, 1]
    full_selection = bundle[
        "full_information_fusion"
    ].predict_proba(
        inner_selection[bundle["full_features"]]
    )[:, 1]

    cheap_test = bundle["cheap_router"].predict_proba(
        outer_test[bundle["cheap_features"]]
    )[:, 1]
    full_test = bundle[
        "full_information_fusion"
    ].predict_proba(
        outer_test[bundle["full_features"]]
    )[:, 1]

    selection_rows: list[dict] = []
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    full_constant_cost = (
        timing["rule"]
        + timing["compact"]
        + timing["qwen_prompt_response"]
    )

    for target in TARGET_FPRS:
        full_op = policy_module.select_full_threshold(
            y_selection,
            full_selection,
            target,
        )
        selective_op = select_selective_policy_feasible(
            policy_module,
            y_selection,
            cheap_selection,
            full_selection,
            target,
        )

        selection_rows.extend(
            [
                {
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "target_fpr": target,
                    "policy": "full_information_always_on",
                    **full_op,
                    "selection_qwen_call_rate": 1.0,
                },
                {
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "target_fpr": target,
                    "policy": "selective_acquisition",
                    **selective_op,
                },
            ]
        )

        full_pred = full_test >= float(full_op["threshold"])

        low = float(
            selective_op["router_low_allow_threshold"]
        )
        high = float(
            selective_op["router_high_intercept_threshold"]
        )
        final_threshold = float(
            selective_op["full_fusion_threshold"]
        )

        cheap_allow = cheap_test <= low
        cheap_intercept = cheap_test >= high
        call_qwen = ~(cheap_allow | cheap_intercept)

        selective_pred = cheap_intercept.copy()
        selective_pred[call_qwen] = (
            full_test[call_qwen] >= final_threshold
        )

        full_cost = np.full(
            len(outer_test),
            full_constant_cost,
            dtype=float,
        )
        selective_cost = policy_module.policy_costs(
            len(outer_test),
            timing,
            rule=True,
            compact=True,
            qwen_mask=call_qwen,
        )

        full_metrics = policy_module.finite_sample_metrics(
            y_test,
            full_pred,
        )
        selective_metrics = policy_module.finite_sample_metrics(
            y_test,
            selective_pred,
        )

        metric_rows.extend(
            [
                {
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "target_fpr": target,
                    "policy": "full_information_always_on",
                    **full_metrics,
                    "qwen_call_rate": 1.0,
                    "avg_estimated_cost_ms": float(
                        np.mean(full_cost)
                    ),
                    "estimated_cost_reduction": 0.0,
                },
                {
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "target_fpr": target,
                    "policy": "selective_acquisition",
                    **selective_metrics,
                    "qwen_call_rate": float(
                        np.mean(call_qwen)
                    ),
                    "avg_estimated_cost_ms": float(
                        np.mean(selective_cost)
                    ),
                    "estimated_cost_reduction": float(
                        1.0
                        - np.mean(selective_cost)
                        / np.mean(full_cost)
                    ),
                },
            ]
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "example_id": outer_test[
                        "example_id"
                    ].astype(str),
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "y": y_test,
                    "target_fpr": target,
                    "cheap_probability": cheap_test,
                    "full_probability": full_test,
                    "full_intercept_pred": full_pred.astype(int),
                    "selective_intercept_pred": (
                        selective_pred.astype(int)
                    ),
                    "selective_qwen_called": call_qwen.astype(int),
                    "full_estimated_cost_ms": full_cost,
                    "selective_estimated_cost_ms": selective_cost,
                }
            )
        )

    return selection_rows, metric_rows, prediction_frames


def pooled_baseline_metrics(
    policy_module,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (scheme, family), frame in predictions.groupby(
        ["scheme", "family"],
        sort=True,
    ):
        result = policy_module.finite_sample_metrics(
            frame["y"].astype(int).to_numpy(),
            frame["intercept_pred"].astype(bool).to_numpy(),
        )
        rows.append(
            {
                "scheme": scheme,
                "family": family,
                **result,
                "avg_estimated_cost_ms": float(
                    frame["estimated_cost_ms"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def pooled_fusion_metrics(
    policy_module,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (scheme, target), frame in predictions.groupby(
        ["scheme", "target_fpr"],
        sort=True,
    ):
        y = frame["y"].astype(int).to_numpy()

        full = policy_module.finite_sample_metrics(
            y,
            frame["full_intercept_pred"]
            .astype(bool)
            .to_numpy(),
        )
        selective = policy_module.finite_sample_metrics(
            y,
            frame["selective_intercept_pred"]
            .astype(bool)
            .to_numpy(),
        )

        rows.extend(
            [
                {
                    "scheme": scheme,
                    "target_fpr": target,
                    "policy": "full_information_always_on",
                    **full,
                    "qwen_call_rate": 1.0,
                    "avg_estimated_cost_ms": float(
                        frame["full_estimated_cost_ms"].mean()
                    ),
                    "estimated_cost_reduction": 0.0,
                },
                {
                    "scheme": scheme,
                    "target_fpr": target,
                    "policy": "selective_acquisition",
                    **selective,
                    "qwen_call_rate": float(
                        frame["selective_qwen_called"].mean()
                    ),
                    "avg_estimated_cost_ms": float(
                        frame[
                            "selective_estimated_cost_ms"
                        ].mean()
                    ),
                    "estimated_cost_reduction": float(
                        1.0
                        - frame[
                            "selective_estimated_cost_ms"
                        ].mean()
                        / frame[
                            "full_estimated_cost_ms"
                        ].mean()
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def fold_definitions(cache: pd.DataFrame):
    definitions = []

    for index, source in enumerate(SOURCE_FOLDS):
        test_mask = cache["source_dataset"].eq(source)
        definitions.append(
            {
                "scheme": "leave_source_out",
                "fold_name": source,
                "seed": BASE_SEED + 100 + index,
                "test_mask": test_mask,
            }
        )

    family_series = cache["attack_family"].astype("string")
    for index, family in enumerate(FAMILY_FOLDS):
        test_mask = family_series.eq(family).fillna(False)
        definitions.append(
            {
                "scheme": "leave_family_out",
                "fold_name": family,
                "seed": BASE_SEED + 200 + index,
                "test_mask": test_mask,
            }
        )

    return definitions


def main() -> None:
    for path in [CACHE, POLICY_SCRIPT]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    cache = pd.read_parquet(CACHE).copy()
    cache["example_id"] = cache["example_id"].astype(str)

    if len(cache) != 2159:
        raise SystemExit("Expected 2159 rows")
    if cache["example_id"].nunique() != 2159:
        raise SystemExit("example_id is not unique")

    for column in [
        "source_dataset",
        "y",
        *CHEAP_FEATURES,
        *FULL_FEATURES,
    ]:
        if column not in cache.columns:
            raise SystemExit(f"Missing column: {column}")

    policy_module = load_policy_module()
    timing, timing_provenance = (
        policy_module.controlled_timing_medians()
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fold_manifest_rows = []
    selected_policy_frames = []
    operating_point_frames = []
    baseline_metric_rows = []
    baseline_prediction_frames = []
    fusion_metric_rows = []
    fusion_prediction_frames = []
    artifact_manifests = {}

    definitions = fold_definitions(cache)

    for fold_number, definition in enumerate(definitions, start=1):
        scheme = definition["scheme"]
        fold_name = definition["fold_name"]
        seed = int(definition["seed"])
        test_mask = definition["test_mask"].to_numpy(bool)

        outer_test = cache.loc[test_mask].copy()
        development = cache.loc[~test_mask].copy()

        if outer_test.empty:
            raise RuntimeError(
                f"Outer test is empty: {scheme}/{fold_name}"
            )
        if set(outer_test["example_id"]) & set(
            development["example_id"]
        ):
            raise RuntimeError("Outer/development ID overlap")

        inner_train, inner_selection = (
            deterministic_inner_split(development, seed)
        )

        print(
            f"\n[{fold_number}/{len(definitions)}] "
            f"{scheme} / {fold_name}"
        )
        print(
            "rows:",
            {
                "inner_train": len(inner_train),
                "inner_selection": len(inner_selection),
                "outer_test": len(outer_test),
            },
        )
        print(
            "outer labels:",
            outer_test["y"]
            .value_counts()
            .sort_index()
            .to_dict(),
        )

        bundle, bundle_path, artifact_manifest = (
            serialize_fold_bundle(
                scheme=scheme,
                fold_name=fold_name,
                seed=seed,
                inner_train=inner_train,
                inner_selection=inner_selection,
                outer_test=outer_test,
            )
        )
        artifact_manifests[
            f"{scheme}/{fold_name}"
        ] = artifact_manifest

        selected, policies, candidate_count = (
            select_prespecified_policies(
                policy_module=policy_module,
                inner_train=inner_train,
                inner_selection=inner_selection,
                bundle=bundle,
                bundle_path=bundle_path,
                timing=timing,
            )
        )
        selected.insert(0, "outer_fold", fold_name)
        selected.insert(0, "scheme", scheme)
        selected_policy_frames.append(selected)

        fold_baseline_metrics, fold_baseline_predictions = (
            evaluate_prespecified_policies(
                policy_module=policy_module,
                scheme=scheme,
                fold_name=fold_name,
                outer_test=outer_test,
                policies=policies,
                bundle=bundle,
                timing=timing,
            )
        )
        baseline_metric_rows.extend(fold_baseline_metrics)
        baseline_prediction_frames.extend(
            fold_baseline_predictions
        )

        (
            fold_operating_points,
            fold_fusion_metrics,
            fold_fusion_predictions,
        ) = select_and_evaluate_fusion(
            policy_module=policy_module,
            scheme=scheme,
            fold_name=fold_name,
            inner_selection=inner_selection,
            outer_test=outer_test,
            bundle=bundle,
            timing=timing,
        )
        operating_point_frames.append(
            pd.DataFrame(fold_operating_points)
        )
        fusion_metric_rows.extend(fold_fusion_metrics)
        fusion_prediction_frames.extend(
            fold_fusion_predictions
        )

        fold_manifest_rows.append(
            {
                "scheme": scheme,
                "outer_fold": fold_name,
                "seed": seed,
                "inner_train_rows": len(inner_train),
                "inner_selection_rows": len(inner_selection),
                "outer_test_rows": len(outer_test),
                "outer_positive_n": int(
                    (outer_test["y"] == 1).sum()
                ),
                "outer_negative_n": int(
                    (outer_test["y"] == 0).sum()
                ),
                "candidate_policy_count": candidate_count,
                "selected_policy_count": len(policies),
                "bundle_path": str(bundle_path),
                "bundle_sha256": sha256(bundle_path),
                "outer_test_id_sha256": id_hash(
                    outer_test["example_id"]
                ),
            }
        )

    fold_manifest = pd.DataFrame(fold_manifest_rows)
    selected_policies = pd.concat(
        selected_policy_frames,
        ignore_index=True,
    )
    operating_points = pd.concat(
        operating_point_frames,
        ignore_index=True,
    )
    baseline_fold_metrics = pd.DataFrame(
        baseline_metric_rows
    )
    baseline_predictions = pd.concat(
        baseline_prediction_frames,
        ignore_index=True,
    )
    fusion_fold_metrics = pd.DataFrame(
        fusion_metric_rows
    )
    fusion_predictions = pd.concat(
        fusion_prediction_frames,
        ignore_index=True,
    )

    baseline_pooled = pooled_baseline_metrics(
        policy_module,
        baseline_predictions,
    )
    fusion_pooled = pooled_fusion_metrics(
        policy_module,
        fusion_predictions,
    )

    # Verify source outer folds cover every example exactly once.
    source_predictions = fusion_predictions[
        fusion_predictions["scheme"].eq("leave_source_out")
        & fusion_predictions["target_fpr"].eq(0.05)
    ]
    source_counts = source_predictions[
        "example_id"
    ].value_counts()
    if len(source_counts) != 2159 or not (
        source_counts == 1
    ).all():
        raise RuntimeError(
            "Leave-source-out predictions do not cover each example once"
        )

    # Verify family outer folds cover the 200 JBB family-labelled rows once.
    family_predictions = fusion_predictions[
        fusion_predictions["scheme"].eq("leave_family_out")
        & fusion_predictions["target_fpr"].eq(0.05)
    ]
    family_counts = family_predictions[
        "example_id"
    ].value_counts()
    if len(family_counts) != 200 or not (
        family_counts == 1
    ).all():
        raise RuntimeError(
            "Leave-family-out predictions do not cover 200 JBB rows once"
        )

    output_paths = {
        "fold_manifest": REPORT_DIR / "fold_manifest.csv",
        "selected_policies": REPORT_DIR / "selected_policies_by_fold.csv",
        "fusion_operating_points": REPORT_DIR
        / "fusion_operating_points_by_fold.csv",
        "baseline_fold_metrics": REPORT_DIR
        / "prespecified_policy_metrics_by_fold.csv",
        "baseline_predictions": REPORT_DIR
        / "prespecified_policy_predictions.csv",
        "baseline_pooled": REPORT_DIR
        / "prespecified_policy_pooled_metrics.csv",
        "fusion_fold_metrics": REPORT_DIR
        / "fusion_metrics_by_fold.csv",
        "fusion_predictions": REPORT_DIR
        / "fusion_predictions.csv",
        "fusion_pooled": REPORT_DIR
        / "fusion_pooled_metrics.csv",
    }

    fold_manifest.to_csv(
        output_paths["fold_manifest"],
        index=False,
    )
    selected_policies.to_csv(
        output_paths["selected_policies"],
        index=False,
    )
    operating_points.to_csv(
        output_paths["fusion_operating_points"],
        index=False,
    )
    baseline_fold_metrics.to_csv(
        output_paths["baseline_fold_metrics"],
        index=False,
    )
    baseline_predictions.to_csv(
        output_paths["baseline_predictions"],
        index=False,
    )
    baseline_pooled.to_csv(
        output_paths["baseline_pooled"],
        index=False,
    )
    fusion_fold_metrics.to_csv(
        output_paths["fusion_fold_metrics"],
        index=False,
    )
    fusion_predictions.to_csv(
        output_paths["fusion_predictions"],
        index=False,
    )
    fusion_pooled.to_csv(
        output_paths["fusion_pooled"],
        index=False,
    )

    target_005 = fusion_fold_metrics[
        fusion_fold_metrics["target_fpr"].eq(0.05)
    ].copy()
    pooled_005 = fusion_pooled[
        fusion_pooled["target_fpr"].eq(0.05)
    ].copy()

    summary = f"""# Nested leave-group-out evaluation v1

This is the professor-required replacement for relying on the previously
influenced final-test split.

## Design

- Outer leave-source-out folds: {SOURCE_FOLDS}
- Outer leave-family-out folds: {FAMILY_FOLDS}
- Each outer fold is excluded from both model fitting and threshold selection.
- Remaining rows are deterministically split into inner training and inner
  policy-selection sets with stratification.
- The existing `split` column is ignored.
- Complete calibrated cheap-router and full-fusion pipelines are fitted,
  serialized, reloaded, and verified independently inside each outer fold.
- All six prespecified policy families are selected within each fold.
- Full-information and selective acquisition are compared at common selection
  target FPRs of {TARGET_FPRS}.

## Interpretation limit

This is nested cross-fitted robustness evaluation, not a genuinely new
untouched external test set. The XSTest source fold contains only negative
examples, so it supports FPR evaluation but not source-specific recall.

## Target FPR 0.05 fold results

{target_005.to_string(index=False)}

## Target FPR 0.05 pooled out-of-fold results

{pooled_005.to_string(index=False)}

## Validity statement

No outer-fold label or feature row is used for fitting or threshold selection
within that fold. Formal Neyman-Pearson or Learn-then-Test risk certification
is not performed here and remains the next professor-required stage.
"""
    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text(
        summary,
        encoding="utf-8",
    )
    output_paths["summary"] = summary_path

    manifest = {
        "artifact": "nested_leave_group_out_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_cross_fitted_not_formally_risk_certified",
        "purpose": (
            "Replacement for claims based only on the previously influenced "
            "final-test split"
        ),
        "cache": {
            "path": str(CACHE),
            "sha256": sha256(CACHE),
            "rows": int(len(cache)),
            "unique_example_id": int(
                cache["example_id"].nunique()
            ),
        },
        "policy_implementation": {
            "path": str(POLICY_SCRIPT),
            "sha256": sha256(POLICY_SCRIPT),
        },
        "schemes": {
            "leave_source_out": SOURCE_FOLDS,
            "leave_family_out": FAMILY_FOLDS,
        },
        "inner_selection_fraction": INNER_SELECTION_FRACTION,
        "base_seed": BASE_SEED,
        "existing_split_column_used": False,
        "outer_rows_used_for_fit_or_selection": False,
        "source_oof_coverage": {
            "rows": 2159,
            "each_example_once": True,
        },
        "family_oof_coverage": {
            "rows": 200,
            "each_family_labelled_jbb_example_once": True,
        },
        "timing": timing_provenance,
        "fold_artifacts": artifact_manifests,
        "outputs": {
            str(path): sha256(path)
            for path in output_paths.values()
        },
        "risk_control_status": (
            "not formally certified; formal risk control remains next"
        ),
        "interpretation_limit": (
            "Nested cross-fitted robustness estimate, not a new untouched "
            "external test set"
        ),
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== FOLD MANIFEST ===")
    print(fold_manifest.to_string(index=False))

    print("\n=== FULL VS SELECTIVE, TARGET FPR 0.05 BY FOLD ===")
    print(
        target_005[
            [
                "scheme",
                "outer_fold",
                "policy",
                "positive_n",
                "negative_n",
                "recall",
                "fpr",
                "fpr_one_sided95_upper",
                "qwen_call_rate",
                "estimated_cost_reduction",
            ]
        ].to_string(index=False)
    )

    print("\n=== POOLED OOF TARGET FPR 0.05 ===")
    print(
        pooled_005[
            [
                "scheme",
                "policy",
                "positive_n",
                "negative_n",
                "recall",
                "fpr",
                "fpr_one_sided95_upper",
                "qwen_call_rate",
                "estimated_cost_reduction",
            ]
        ].to_string(index=False)
    )

    print("\nmanifest:", MANIFEST)


if __name__ == "__main__":
    main()
