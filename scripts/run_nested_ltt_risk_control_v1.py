#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import beta, binom
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("data/processed/monitor_score_cache_v3.parquet")
POLICY_SCRIPT = Path("scripts/run_policy_evaluation_v3.py")

ARTIFACT_DIR = Path("artifacts/nested_ltt_risk_control_v1")
REPORT_DIR = Path("reports/nested_ltt_risk_control_v1")
MANIFEST = Path(
    "data/metadata/nested_ltt_risk_control_v1_manifest.json"
)

CHEAP_FEATURES = ["rule_score", "compact_unsafe_score"]
FULL_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]

RISK_LIMIT = 0.05
CONFIDENCE_DELTA = 0.05
TARGET_LADDER = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
RISK_CONTROL_FRACTION = 0.40
SELECTION_FRACTION_OF_REMAINDER = 1.0 / 3.0
BASE_SEED = 20261000

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


def exact_upper(k: int, n: int, delta: float) -> float:
    if n == 0:
        return math.nan
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def ltt_p_value(k: int, n: int, risk_limit: float) -> float:
    """Exact lower-tail p-value for H0: FPR >= risk_limit."""
    if n == 0:
        return math.nan
    return float(binom.cdf(k, n, risk_limit))


def three_way_split(
    development: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = development["y"].astype(int).to_numpy()

    first = StratifiedShuffleSplit(
        n_splits=1,
        test_size=RISK_CONTROL_FRACTION,
        random_state=seed,
    )
    fit_select_idx, risk_idx = next(
        first.split(np.zeros(len(development)), y)
    )

    fit_select = development.iloc[fit_select_idx].reset_index(drop=True)
    risk_control = development.iloc[risk_idx].reset_index(drop=True)

    second = StratifiedShuffleSplit(
        n_splits=1,
        test_size=SELECTION_FRACTION_OF_REMAINDER,
        random_state=seed + 1,
    )
    train_idx, selection_idx = next(
        second.split(
            np.zeros(len(fit_select)),
            fit_select["y"].astype(int).to_numpy(),
        )
    )

    train = fit_select.iloc[train_idx].copy()
    selection = fit_select.iloc[selection_idx].copy()

    partitions = {
        "train": train,
        "selection": selection,
        "risk_control": risk_control,
    }
    for name, frame in partitions.items():
        if frame["y"].nunique() != 2:
            raise RuntimeError(
                f"{name} lacks both classes for seed={seed}"
            )

    id_sets = {
        name: set(frame["example_id"].astype(str))
        for name, frame in partitions.items()
    }
    names = list(id_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if id_sets[left] & id_sets[right]:
                raise RuntimeError(
                    f"Partition overlap: {left} and {right}"
                )

    return train, selection, risk_control


def fit_and_serialize_bundle(
    *,
    scheme: str,
    fold_name: str,
    seed: int,
    train: pd.DataFrame,
    selection: pd.DataFrame,
    risk_control: pd.DataFrame,
    outer_test: pd.DataFrame,
) -> tuple[dict, Path, dict]:
    safe_fold = fold_name.replace("/", "_").replace(" ", "_")
    fold_dir = ARTIFACT_DIR / scheme / safe_fold
    fold_dir.mkdir(parents=True, exist_ok=True)

    cheap_model = build_calibrated_logistic()
    full_model = build_calibrated_logistic()

    cheap_model.fit(
        train[CHEAP_FEATURES],
        train["y"].astype(int),
    )
    full_model.fit(
        train[FULL_FEATURES],
        train["y"].astype(int),
    )

    cheap_path = fold_dir / "cheap_router.joblib"
    full_path = fold_dir / "full_information_fusion.joblib"
    bundle_path = fold_dir / "fusion_bundle.joblib"

    joblib.dump(cheap_model, cheap_path, compress=3)
    joblib.dump(full_model, full_path, compress=3)

    provenance = {
        "artifact": "nested_ltt_fold_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scheme": scheme,
        "outer_fold": fold_name,
        "seed": seed,
        "cache_path": str(CACHE),
        "cache_sha256": sha256(CACHE),
        "train_rows": int(len(train)),
        "selection_rows": int(len(selection)),
        "risk_control_rows": int(len(risk_control)),
        "outer_test_rows": int(len(outer_test)),
        "train_id_sha256": id_hash(train["example_id"]),
        "selection_id_sha256": id_hash(selection["example_id"]),
        "risk_control_id_sha256": id_hash(
            risk_control["example_id"]
        ),
        "outer_test_id_sha256": id_hash(
            outer_test["example_id"]
        ),
        "cheap_features": CHEAP_FEATURES,
        "full_features": FULL_FEATURES,
        "model_family": (
            "5-fold sigmoid-calibrated class-balanced logistic regression"
        ),
        "existing_split_column_used": False,
        "risk_control_rows_used_for_fit_or_candidate_learning": False,
        "outer_test_rows_used_for_fit_selection_or_certification": False,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
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

    for partition_name, frame in [
        ("train", train),
        ("selection", selection),
        ("risk_control", risk_control),
        ("outer_test", outer_test),
    ]:
        cheap_memory = cheap_model.predict_proba(
            frame[CHEAP_FEATURES]
        )[:, 1]
        cheap_disk = loaded_cheap.predict_proba(
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
        full_disk = loaded_full.predict_proba(
            frame[FULL_FEATURES]
        )[:, 1]
        full_bundle = loaded_bundle[
            "full_information_fusion"
        ].predict_proba(
            frame[loaded_bundle["full_features"]]
        )[:, 1]

        if not np.array_equal(cheap_memory, cheap_disk):
            raise RuntimeError(
                f"Cheap standalone mismatch: {partition_name}"
            )
        if not np.array_equal(cheap_memory, cheap_bundle):
            raise RuntimeError(
                f"Cheap bundle mismatch: {partition_name}"
            )
        if not np.array_equal(full_memory, full_disk):
            raise RuntimeError(
                f"Full standalone mismatch: {partition_name}"
            )
        if not np.array_equal(full_memory, full_bundle):
            raise RuntimeError(
                f"Full bundle mismatch: {partition_name}"
            )

    artifact_manifest = {
        **provenance,
        "partition_label_counts": {
            name: {
                str(key): int(value)
                for key, value in frame["y"]
                .value_counts()
                .sort_index()
                .items()
            }
            for name, frame in [
                ("train", train),
                ("selection", selection),
                ("risk_control", risk_control),
                ("outer_test", outer_test),
            ]
        },
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
            "checked_partitions": [
                "train",
                "selection",
                "risk_control",
                "outer_test",
            ],
        },
    }

    (fold_dir / "manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2),
        encoding="utf-8",
    )

    return loaded_bundle, bundle_path, artifact_manifest


def selective_candidate_for_target(
    *,
    policy_module,
    y: np.ndarray,
    cheap_probability: np.ndarray,
    full_probability: np.ndarray,
    target_fpr: float,
) -> dict[str, Any]:
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
                        "candidate_source_target_fpr": target_fpr,
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
                        "selection_qwen_call_rate": qwen_call_rate,
                    }

    if best_payload is None:
        raise RuntimeError(
            f"No feasible selective candidate for target {target_fpr}"
        )
    return best_payload


def learn_full_candidates(
    *,
    policy_module,
    selection: pd.DataFrame,
    bundle: dict,
) -> list[dict[str, Any]]:
    y = selection["y"].astype(int).to_numpy()
    probability = bundle[
        "full_information_fusion"
    ].predict_proba(
        selection[bundle["full_features"]]
    )[:, 1]

    candidates: list[dict[str, Any]] = [
        {
            "candidate_order_label": "selection_zero_intercept_fallback",
            "candidate_source_target_fpr": 0.0,
            "threshold": float(
                np.nextafter(
                    float(np.max(probability)),
                    np.inf,
                )
            ),
            "selection_recall": 0.0,
            "selection_fpr": 0.0,
            "selection_precision": math.nan,
            "selection_qwen_call_rate": 1.0,
        }
    ]

    for target in TARGET_LADDER:
        result = policy_module.select_full_threshold(
            y,
            probability,
            target,
        )
        candidates.append(
            {
                "candidate_order_label": f"target_{target:.3f}",
                "candidate_source_target_fpr": target,
                "threshold": float(result["threshold"]),
                "selection_recall": float(
                    result["selection_recall"]
                ),
                "selection_fpr": float(
                    result["selection_fpr"]
                ),
                "selection_precision": float(
                    result["selection_precision"]
                ),
                "selection_qwen_call_rate": 1.0,
            }
        )

    deduplicated: list[dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = candidate["threshold"]
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return deduplicated


def learn_selective_candidates(
    *,
    policy_module,
    selection: pd.DataFrame,
    bundle: dict,
) -> list[dict[str, Any]]:
    y = selection["y"].astype(int).to_numpy()
    cheap_probability = bundle["cheap_router"].predict_proba(
        selection[bundle["cheap_features"]]
    )[:, 1]
    full_probability = bundle[
        "full_information_fusion"
    ].predict_proba(
        selection[bundle["full_features"]]
    )[:, 1]

    trivial_low = float(
        np.nextafter(
            float(np.max(cheap_probability)),
            np.inf,
        )
    )
    trivial_high = float(np.nextafter(trivial_low, np.inf))

    candidates: list[dict[str, Any]] = [
        {
            "candidate_order_label": "selection_zero_intercept_fallback",
            "candidate_source_target_fpr": 0.0,
            "router_low_allow_threshold": trivial_low,
            "router_high_intercept_threshold": trivial_high,
            "full_fusion_threshold": float(
                np.nextafter(
                    float(np.max(full_probability)),
                    np.inf,
                )
            ),
            "selection_recall": 0.0,
            "selection_fpr": 0.0,
            "selection_precision": math.nan,
            "selection_qwen_call_rate": 0.0,
        }
    ]

    for target in TARGET_LADDER:
        candidate = selective_candidate_for_target(
            policy_module=policy_module,
            y=y,
            cheap_probability=cheap_probability,
            full_probability=full_probability,
            target_fpr=target,
        )
        candidate["candidate_order_label"] = (
            f"target_{target:.3f}"
        )
        candidates.append(candidate)

    deduplicated: list[dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate["router_low_allow_threshold"],
            candidate["router_high_intercept_threshold"],
            candidate["full_fusion_threshold"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return deduplicated


def apply_candidate(
    *,
    method: str,
    candidate: dict[str, Any],
    frame: pd.DataFrame,
    bundle: dict,
    policy_module,
    timing: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frame)
    full_probability = bundle[
        "full_information_fusion"
    ].predict_proba(
        frame[bundle["full_features"]]
    )[:, 1]

    if method == "full_information_always_on":
        pred = full_probability >= float(candidate["threshold"])
        qwen_called = np.ones(n, dtype=bool)
        cost = np.full(
            n,
            timing["rule"]
            + timing["compact"]
            + timing["qwen_prompt_response"],
            dtype=float,
        )
        return pred, qwen_called, cost

    if method != "selective_acquisition":
        raise ValueError(f"Unknown method: {method}")

    cheap_probability = bundle["cheap_router"].predict_proba(
        frame[bundle["cheap_features"]]
    )[:, 1]

    low = float(candidate["router_low_allow_threshold"])
    high = float(candidate["router_high_intercept_threshold"])
    final_threshold = float(candidate["full_fusion_threshold"])

    cheap_allow = cheap_probability <= low
    cheap_intercept = cheap_probability >= high
    qwen_called = ~(cheap_allow | cheap_intercept)

    pred = cheap_intercept.copy()
    pred[qwen_called] = (
        full_probability[qwen_called] >= final_threshold
    )
    cost = policy_module.policy_costs(
        n,
        timing,
        rule=True,
        compact=True,
        qwen_mask=qwen_called,
    )
    return pred, qwen_called, cost


def fixed_sequence_ltt(
    *,
    scheme: str,
    fold_name: str,
    method: str,
    candidates: list[dict[str, Any]],
    risk_control: pd.DataFrame,
    bundle: dict,
    policy_module,
    timing: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = risk_control["y"].astype(int).to_numpy()
    negative_mask = y == 0
    negative_n = int(np.sum(negative_mask))
    if negative_n == 0:
        raise RuntimeError("Risk-control split has no negatives")

    test_rows: list[dict[str, Any]] = []
    certified_candidates: list[dict[str, Any]] = []
    sequence_open = True
    stopped_at = None

    for index, candidate in enumerate(candidates):
        parameters = {
            key: value
            for key, value in candidate.items()
            if key
            not in {
                "selection_recall",
                "selection_fpr",
                "selection_precision",
                "selection_qwen_call_rate",
            }
        }

        if not sequence_open:
            test_rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold_name,
                    "method": method,
                    "sequence_index": index,
                    "candidate_order_label": candidate[
                        "candidate_order_label"
                    ],
                    "candidate_source_target_fpr": candidate[
                        "candidate_source_target_fpr"
                    ],
                    "tested": False,
                    "certified": False,
                    "sequence_stopped_here": False,
                    "risk_negative_n": negative_n,
                    "risk_false_positive_n": math.nan,
                    "risk_observed_fpr": math.nan,
                    "ltt_p_value": math.nan,
                    "one_sided95_upper": math.nan,
                    "selection_recall": candidate[
                        "selection_recall"
                    ],
                    "selection_fpr": candidate["selection_fpr"],
                    "selection_qwen_call_rate": candidate[
                        "selection_qwen_call_rate"
                    ],
                    "candidate_parameters_json": json.dumps(
                        parameters,
                        sort_keys=True,
                    ),
                }
            )
            continue

        pred, _, _ = apply_candidate(
            method=method,
            candidate=candidate,
            frame=risk_control,
            bundle=bundle,
            policy_module=policy_module,
            timing=timing,
        )

        false_positive_n = int(
            np.sum(pred[negative_mask])
        )
        observed_fpr = false_positive_n / negative_n
        p_value = ltt_p_value(
            false_positive_n,
            negative_n,
            RISK_LIMIT,
        )
        upper = exact_upper(
            false_positive_n,
            negative_n,
            CONFIDENCE_DELTA,
        )
        certified = bool(p_value <= CONFIDENCE_DELTA)

        # These are equivalent exact one-sided statements up to
        # floating-point tolerance.
        if certified != bool(upper <= RISK_LIMIT + 1e-12):
            raise RuntimeError(
                "Exact p-value and upper-bound decisions disagree"
            )

        stop_here = not certified
        test_rows.append(
            {
                "scheme": scheme,
                "outer_fold": fold_name,
                "method": method,
                "sequence_index": index,
                "candidate_order_label": candidate[
                    "candidate_order_label"
                ],
                "candidate_source_target_fpr": candidate[
                    "candidate_source_target_fpr"
                ],
                "tested": True,
                "certified": certified,
                "sequence_stopped_here": stop_here,
                "risk_negative_n": negative_n,
                "risk_false_positive_n": false_positive_n,
                "risk_observed_fpr": observed_fpr,
                "ltt_p_value": p_value,
                "one_sided95_upper": upper,
                "selection_recall": candidate[
                    "selection_recall"
                ],
                "selection_fpr": candidate["selection_fpr"],
                "selection_qwen_call_rate": candidate[
                    "selection_qwen_call_rate"
                ],
                "candidate_parameters_json": json.dumps(
                    parameters,
                    sort_keys=True,
                ),
            }
        )

        if certified:
            certified_candidates.append(candidate)
        else:
            sequence_open = False
            stopped_at = candidate["candidate_order_label"]

    if not certified_candidates:
        raise RuntimeError(
            f"No candidate certified for {scheme}/{fold_name}/{method}"
        )

    selected = certified_candidates[-1]
    certificate_row = {
        "scheme": scheme,
        "outer_fold": fold_name,
        "method": method,
        "risk_limit": RISK_LIMIT,
        "confidence_delta": CONFIDENCE_DELTA,
        "confidence_level": 1.0 - CONFIDENCE_DELTA,
        "fixed_sequence_candidate_count": len(candidates),
        "tested_candidate_count": int(
            sum(row["tested"] for row in test_rows)
        ),
        "certified_candidate_count": len(
            certified_candidates
        ),
        "sequence_stopped_at": stopped_at,
        "selected_candidate_order_label": selected[
            "candidate_order_label"
        ],
        "selected_candidate_source_target_fpr": selected[
            "candidate_source_target_fpr"
        ],
        "selected_selection_recall": selected[
            "selection_recall"
        ],
        "selected_selection_fpr": selected["selection_fpr"],
        "selected_selection_qwen_call_rate": selected[
            "selection_qwen_call_rate"
        ],
        "selected_candidate_parameters_json": json.dumps(
            {
                key: value
                for key, value in selected.items()
                if key
                not in {
                    "selection_recall",
                    "selection_fpr",
                    "selection_precision",
                    "selection_qwen_call_rate",
                }
            },
            sort_keys=True,
        ),
    }

    selected_test = next(
        row
        for row in test_rows
        if row["tested"]
        and row["certified"]
        and row["candidate_order_label"]
        == selected["candidate_order_label"]
    )
    certificate_row.update(
        {
            "risk_negative_n": selected_test[
                "risk_negative_n"
            ],
            "risk_false_positive_n": selected_test[
                "risk_false_positive_n"
            ],
            "risk_observed_fpr": selected_test[
                "risk_observed_fpr"
            ],
            "ltt_p_value": selected_test["ltt_p_value"],
            "one_sided95_upper": selected_test[
                "one_sided95_upper"
            ],
            "certificate_pass": True,
        }
    )
    certificate_row["_selected_candidate"] = selected
    return test_rows, certificate_row


def fold_definitions(cache: pd.DataFrame):
    definitions = []
    for index, source in enumerate(SOURCE_FOLDS):
        definitions.append(
            {
                "scheme": "leave_source_out",
                "fold_name": source,
                "seed": BASE_SEED + index,
                "test_mask": cache["source_dataset"].eq(source),
            }
        )

    family_series = cache["attack_family"].astype("string")
    for index, family in enumerate(FAMILY_FOLDS):
        definitions.append(
            {
                "scheme": "leave_family_out",
                "fold_name": family,
                "seed": BASE_SEED + 100 + index,
                "test_mask": family_series.eq(family).fillna(False),
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
        raise SystemExit("Expected 2159 unique example IDs")

    policy_module = load_policy_module()
    timing, timing_provenance = (
        policy_module.controlled_timing_medians()
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    candidate_test_rows = []
    certificate_rows = []
    outer_metric_rows = []
    outer_prediction_frames = []
    artifact_manifests = {}

    definitions = fold_definitions(cache)

    for fold_index, definition in enumerate(
        definitions,
        start=1,
    ):
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
            raise RuntimeError("Outer/development overlap")

        train, selection, risk_control = three_way_split(
            development,
            seed,
        )

        for name, frame in [
            ("train", train),
            ("selection", selection),
            ("risk_control", risk_control),
        ]:
            if set(frame["example_id"]) & set(
                outer_test["example_id"]
            ):
                raise RuntimeError(
                    f"Outer overlap with {name}"
                )

        print(
            f"\n[{fold_index}/{len(definitions)}] "
            f"{scheme} / {fold_name}"
        )
        print(
            "rows:",
            {
                "train": len(train),
                "selection": len(selection),
                "risk_control": len(risk_control),
                "outer_test": len(outer_test),
            },
        )
        print(
            "risk-control labels:",
            risk_control["y"]
            .value_counts()
            .sort_index()
            .to_dict(),
        )
        print(
            "outer labels:",
            outer_test["y"]
            .value_counts()
            .sort_index()
            .to_dict(),
        )

        bundle, bundle_path, artifact_manifest = (
            fit_and_serialize_bundle(
                scheme=scheme,
                fold_name=fold_name,
                seed=seed,
                train=train,
                selection=selection,
                risk_control=risk_control,
                outer_test=outer_test,
            )
        )
        artifact_manifests[
            f"{scheme}/{fold_name}"
        ] = artifact_manifest

        candidate_sets = {
            "full_information_always_on": (
                learn_full_candidates(
                    policy_module=policy_module,
                    selection=selection,
                    bundle=bundle,
                )
            ),
            "selective_acquisition": (
                learn_selective_candidates(
                    policy_module=policy_module,
                    selection=selection,
                    bundle=bundle,
                )
            ),
        }

        fold_rows.append(
            {
                "scheme": scheme,
                "outer_fold": fold_name,
                "seed": seed,
                "train_rows": len(train),
                "selection_rows": len(selection),
                "risk_control_rows": len(risk_control),
                "risk_control_negative_n": int(
                    (risk_control["y"] == 0).sum()
                ),
                "risk_control_positive_n": int(
                    (risk_control["y"] == 1).sum()
                ),
                "outer_test_rows": len(outer_test),
                "outer_negative_n": int(
                    (outer_test["y"] == 0).sum()
                ),
                "outer_positive_n": int(
                    (outer_test["y"] == 1).sum()
                ),
                "bundle_path": str(bundle_path),
                "bundle_sha256": sha256(bundle_path),
                "train_id_sha256": id_hash(
                    train["example_id"]
                ),
                "selection_id_sha256": id_hash(
                    selection["example_id"]
                ),
                "risk_control_id_sha256": id_hash(
                    risk_control["example_id"]
                ),
                "outer_test_id_sha256": id_hash(
                    outer_test["example_id"]
                ),
            }
        )

        for method, candidates in candidate_sets.items():
            tests, certificate = fixed_sequence_ltt(
                scheme=scheme,
                fold_name=fold_name,
                method=method,
                candidates=candidates,
                risk_control=risk_control,
                bundle=bundle,
                policy_module=policy_module,
                timing=timing,
            )
            candidate_test_rows.extend(tests)

            selected_candidate = certificate.pop(
                "_selected_candidate"
            )

            pred, qwen_called, cost = apply_candidate(
                method=method,
                candidate=selected_candidate,
                frame=outer_test,
                bundle=bundle,
                policy_module=policy_module,
                timing=timing,
            )
            outer_metrics = policy_module.finite_sample_metrics(
                outer_test["y"].astype(int).to_numpy(),
                pred,
            )

            metric_row = {
                "scheme": scheme,
                "outer_fold": fold_name,
                "method": method,
                **outer_metrics,
                "qwen_call_rate": float(
                    np.mean(qwen_called)
                ),
                "avg_estimated_cost_ms": float(
                    np.mean(cost)
                ),
                "selected_candidate_order_label": certificate[
                    "selected_candidate_order_label"
                ],
                "selected_candidate_source_target_fpr": (
                    certificate[
                        "selected_candidate_source_target_fpr"
                    ]
                ),
                "risk_control_observed_fpr": certificate[
                    "risk_observed_fpr"
                ],
                "risk_control_one_sided95_upper": certificate[
                    "one_sided95_upper"
                ],
                "risk_control_ltt_p_value": certificate[
                    "ltt_p_value"
                ],
            }
            outer_metric_rows.append(metric_row)

            certificate.update(
                {
                    "outer_recall": outer_metrics["recall"],
                    "outer_fpr": outer_metrics["fpr"],
                    "outer_fpr_one_sided95_upper": (
                        outer_metrics[
                            "fpr_one_sided95_upper"
                        ]
                    ),
                    "outer_qwen_call_rate": float(
                        np.mean(qwen_called)
                    ),
                    "outer_avg_estimated_cost_ms": float(
                        np.mean(cost)
                    ),
                }
            )
            certificate_rows.append(certificate)

            outer_prediction_frames.append(
                pd.DataFrame(
                    {
                        "example_id": outer_test[
                            "example_id"
                        ].astype(str),
                        "scheme": scheme,
                        "outer_fold": fold_name,
                        "method": method,
                        "y": outer_test["y"].astype(int),
                        "intercept_pred": pred.astype(int),
                        "qwen_called": qwen_called.astype(int),
                        "estimated_cost_ms": cost,
                        "selected_candidate_order_label": (
                            certificate[
                                "selected_candidate_order_label"
                            ]
                        ),
                        "risk_control_certificate_pass": True,
                    }
                )
            )

            print(
                method,
                {
                    "selected": certificate[
                        "selected_candidate_order_label"
                    ],
                    "risk_fp": certificate[
                        "risk_false_positive_n"
                    ],
                    "risk_n0": certificate[
                        "risk_negative_n"
                    ],
                    "risk_upper": round(
                        certificate[
                            "one_sided95_upper"
                        ],
                        6,
                    ),
                    "outer_fpr": (
                        None
                        if not np.isfinite(
                            outer_metrics["fpr"]
                        )
                        else round(
                            outer_metrics["fpr"],
                            6,
                        )
                    ),
                    "outer_recall": (
                        None
                        if not np.isfinite(
                            outer_metrics["recall"]
                        )
                        else round(
                            outer_metrics["recall"],
                            6,
                        )
                    ),
                },
            )

    fold_manifest = pd.DataFrame(fold_rows)
    candidate_tests = pd.DataFrame(candidate_test_rows)
    certificates = pd.DataFrame(certificate_rows)
    outer_metrics = pd.DataFrame(outer_metric_rows)
    outer_predictions = pd.concat(
        outer_prediction_frames,
        ignore_index=True,
    )

    # Each source-fold example must appear once per method.
    source_predictions = outer_predictions[
        outer_predictions["scheme"].eq("leave_source_out")
    ]
    for method in [
        "full_information_always_on",
        "selective_acquisition",
    ]:
        subset = source_predictions[
            source_predictions["method"].eq(method)
        ]
        counts = subset["example_id"].value_counts()
        if len(counts) != 2159 or not (counts == 1).all():
            raise RuntimeError(
                f"Invalid source OOF coverage for {method}"
            )

    # Each JBB family-labelled example must appear once per method.
    family_predictions = outer_predictions[
        outer_predictions["scheme"].eq("leave_family_out")
    ]
    for method in [
        "full_information_always_on",
        "selective_acquisition",
    ]:
        subset = family_predictions[
            family_predictions["method"].eq(method)
        ]
        counts = subset["example_id"].value_counts()
        if len(counts) != 200 or not (counts == 1).all():
            raise RuntimeError(
                f"Invalid family OOF coverage for {method}"
            )

    pooled_rows = []
    for (scheme, method), frame in outer_predictions.groupby(
        ["scheme", "method"],
        sort=True,
    ):
        result = policy_module.finite_sample_metrics(
            frame["y"].astype(int).to_numpy(),
            frame["intercept_pred"]
            .astype(bool)
            .to_numpy(),
        )
        pooled_rows.append(
            {
                "scheme": scheme,
                "method": method,
                **result,
                "qwen_call_rate": float(
                    frame["qwen_called"].mean()
                ),
                "avg_estimated_cost_ms": float(
                    frame["estimated_cost_ms"].mean()
                ),
            }
        )
    pooled_metrics = pd.DataFrame(pooled_rows)

    output_paths = {
        "fold_manifest": REPORT_DIR / "fold_manifest.csv",
        "candidate_tests": REPORT_DIR
        / "fixed_sequence_candidate_tests.csv",
        "certificates": REPORT_DIR / "certificates.csv",
        "outer_metrics": REPORT_DIR
        / "outer_metrics_by_fold.csv",
        "outer_predictions": REPORT_DIR
        / "outer_predictions.csv",
        "pooled_metrics": REPORT_DIR
        / "pooled_outer_metrics.csv",
    }

    fold_manifest.to_csv(
        output_paths["fold_manifest"],
        index=False,
    )
    candidate_tests.to_csv(
        output_paths["candidate_tests"],
        index=False,
    )
    certificates.to_csv(
        output_paths["certificates"],
        index=False,
    )
    outer_metrics.to_csv(
        output_paths["outer_metrics"],
        index=False,
    )
    outer_predictions.to_csv(
        output_paths["outer_predictions"],
        index=False,
    )
    pooled_metrics.to_csv(
        output_paths["pooled_metrics"],
        index=False,
    )

    summary = f"""# Nested Learn-then-Test risk control v1

## Procedure

For each outer leave-source-out and leave-family-out fold:

1. exclude the complete outer fold;
2. split the remaining development data into 40% model training, 20% policy
   selection, and 40% untouched risk control;
3. fit and serialize the calibrated cheap-router and full-fusion pipelines on
   the training partition only;
4. learn an ordered candidate sequence on the selection partition only;
5. test the sequence on the untouched risk-control negatives using the exact
   binomial lower-tail test of `H0: FPR >= {RISK_LIMIT}`;
6. use fixed-sequence testing at `delta={CONFIDENCE_DELTA}`, stopping at the
   first non-rejection;
7. evaluate the last certified candidate on the excluded outer fold.

The candidate order is prespecified from conservative to aggressive:
a selection-zero-intercept fallback followed by selection target-FPR levels
`{TARGET_LADDER}`.

## Certificate scope

Each certificate is a 95% fixed-sequence Learn-then-Test certificate for its
own fold and method on the risk-control distribution. It does not certify
performance under the excluded source or attack-family shift. Outer-fold
results diagnose certificate transfer and distribution shift.

## Certificates

{certificates.to_string(index=False)}

## Outer-fold evaluation

{outer_metrics.to_string(index=False)}

## Pooled out-of-fold diagnostics

{pooled_metrics.to_string(index=False)}

## Interpretation limits

- These are per-fold, per-method certificates, not one simultaneous global
  certificate across all twelve sequences.
- The outer XSTest source fold has no positive examples, so recall is undefined.
- The risk guarantee applies to FPR only.
- A certified candidate can be operationally weak; certification does not
  imply high recall or a useful cost-recall tradeoff.
"""
    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    output_paths["summary"] = summary_path

    manifest = {
        "artifact": "nested_ltt_risk_control_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_per_fold_ltt_certification",
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
        "risk_control": {
            "risk_metric": "false positive rate",
            "risk_limit": RISK_LIMIT,
            "confidence_delta": CONFIDENCE_DELTA,
            "confidence_level": 1.0 - CONFIDENCE_DELTA,
            "null_hypothesis": "FPR >= 0.05",
            "p_value": (
                "exact Binomial(n, 0.05) lower-tail CDF at observed "
                "false-positive count"
            ),
            "multiple_testing": (
                "fixed-sequence testing within each fold and method; "
                "stop at first non-rejection"
            ),
            "candidate_order": [
                "selection_zero_intercept_fallback",
                *[
                    f"selection_target_fpr_{value:.3f}"
                    for value in TARGET_LADDER
                ],
            ],
            "per_sequence_not_global": True,
        },
        "partitioning": {
            "risk_control_fraction": RISK_CONTROL_FRACTION,
            "selection_fraction_of_remaining": (
                SELECTION_FRACTION_OF_REMAINDER
            ),
            "effective_train_fraction": 0.40,
            "effective_selection_fraction": 0.20,
            "effective_risk_control_fraction": 0.40,
            "existing_split_column_used": False,
            "risk_control_used_for_fit_or_candidate_learning": False,
            "outer_test_used_for_fit_selection_or_certification": False,
        },
        "schemes": {
            "leave_source_out": SOURCE_FOLDS,
            "leave_family_out": FAMILY_FOLDS,
        },
        "base_seed": BASE_SEED,
        "timing": timing_provenance,
        "fold_artifacts": artifact_manifests,
        "certificate_count": int(len(certificates)),
        "all_sequences_returned_a_certificate": bool(
            certificates["certificate_pass"].all()
        ),
        "outputs": {
            str(path): sha256(path)
            for path in output_paths.values()
        },
        "scope_statement": (
            "Certificates apply to each untouched risk-control partition "
            "under its sampling distribution; excluded outer groups are "
            "shift diagnostics and are not covered by the certificate."
        ),
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== CERTIFICATES ===")
    print(
        certificates[
            [
                "scheme",
                "outer_fold",
                "method",
                "selected_candidate_order_label",
                "risk_negative_n",
                "risk_false_positive_n",
                "risk_observed_fpr",
                "one_sided95_upper",
                "ltt_p_value",
                "outer_recall",
                "outer_fpr",
                "outer_qwen_call_rate",
            ]
        ].to_string(index=False)
    )

    print("\n=== POOLED OUTER DIAGNOSTICS ===")
    print(pooled_metrics.to_string(index=False))

    print("\nmanifest:", MANIFEST)


if __name__ == "__main__":
    main()
