#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evaluation_measurement_pilot_v1.json"
GROUPING_SPEC = (
    ROOT
    / "data/metadata/evaluation_measurement_pilot_v1/dependency_grouping_spec.json"
)
GROUPS_PATH = (
    ROOT
    / "data/metadata/evaluation_measurement_pilot_v1/development_dependency_groups.csv"
)
DEV_MANIFEST = ROOT / "data/processed/v2_development_view/manifest.json"
DEV_DATASET = (
    ROOT
    / "data/processed/v2_development_view/"
    / "unified_dataset_label_audited_v1.development.parquet"
)
DEV_CACHE = (
    ROOT
    / "data/processed/v2_development_view/"
    / "monitor_score_cache_v3.development.parquet"
)

FINAL_DIR = ROOT / "reports/evaluation_measurement_pilot_v1/cpu"
DOWNLOADS = Path("/mnt/c/Users/NOAH/Downloads")

FROZEN_COMMIT = "b9759bf1610f726396183bd7c79c11983bd8956b"
EXPECTED_PROTOCOL_SHA = (
    "da4aff3a298e959d2e4fd5a1fa3f70a4ca69e078afc27225b25549aeeff4f4da"
)
EXPECTED_GROUPING_SPEC_SHA = (
    "c937fe221208f9ddb2b178baf421b2a7cbf6ddcc221608ec7f927eeef3cf999f"
)
EXPECTED_GROUPS_SHA = (
    "ab7abc4c9a6569597987419b5a9fd96dc05c1ed982efd06df146a1baf05f4ef1"
)
EXPECTED_ROWS = 1687
ALLOWED_SPLITS = {"policy_train", "policy_selection", "calibration"}
PROTECTED_SPLITS = {"final_test", "held_out_shift"}

PRIMARY_GROUP = "primary_dependency_group"
GROUPING_CONDITIONS = {
    "dependency_primary": "primary_dependency_group",
    "singleton_weak": "example_id",
    "semantic_0_87": "sensitivity_semantic_0_87_group",
    "semantic_0_92": "sensitivity_semantic_0_92_group",
}
LABEL_CONDITIONS = {
    "audited": "y",
    "original": "y_original",
}

SCORE_FIELDS = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]
DATASET_FIELDS_REQUIRED = [
    "example_id",
    "split",
    "source_dataset",
    "prompt",
    "response",
    "y",
    "y_original",
]
CACHE_FIELDS_REQUIRED = [
    "example_id",
    "split",
    *SCORE_FIELDS,
]

POLICY_KINDS_DEPLOYABLE = [
    "cheap_only",
    "full_information",
    "selective_deployable",
]
POLICY_KINDS_ALL = [
    "cheap_only",
    "full_information",
    "selective_deployable",
    "offline_ranked_diagnostic",
]


def progress(message: str) -> None:
    print(message, flush=True)


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stdout}"
        )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_upper(k: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0:
        return math.nan
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if k == n:
        return 1.0
    alpha = 1.0 - confidence
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=bool)

    pos = y == 1
    neg = y == 0
    tp = int(np.sum(pred & pos))
    fn = int(np.sum((~pred) & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    p = int(pos.sum())
    n = int(neg.sum())

    return {
        "n": int(len(y)),
        "positive_n": p,
        "negative_n": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / p if p else math.nan,
        "fpr": fp / n if n else math.nan,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "intercept_rate": float(np.mean(pred)) if len(pred) else math.nan,
    }


def group_false_positive_risk(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=bool)
    groups = np.asarray(groups, dtype=object)

    negative = y == 0
    if not bool(np.any(negative)):
        return {
            "negative_dependency_groups": 0,
            "failed_dependency_groups": 0,
            "dependency_group_event_rate": math.nan,
            "dependency_group_event_upper95": math.nan,
        }

    frame = pd.DataFrame(
        {
            "group": groups[negative].astype(str),
            "false_positive": pred[negative],
        }
    )
    failures = frame.groupby("group", sort=False)["false_positive"].any()
    n_groups = int(len(failures))
    k_groups = int(failures.sum())

    return {
        "negative_dependency_groups": n_groups,
        "failed_dependency_groups": k_groups,
        "dependency_group_event_rate": (
            k_groups / n_groups if n_groups else math.nan
        ),
        "dependency_group_event_upper95": exact_upper(
            k_groups,
            n_groups,
            confidence,
        ),
    }


def finite_risk_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    primary_groups: np.ndarray,
    *,
    confidence: float,
) -> dict[str, float | int | bool]:
    base = binary_metrics(y, pred)
    row_upper = exact_upper(
        int(base["fp"]),
        int(base["negative_n"]),
        confidence,
    )
    grouped = group_false_positive_risk(
        y,
        pred,
        primary_groups,
        confidence=confidence,
    )
    return {
        **base,
        "row_fpr_upper95": row_upper,
        **grouped,
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a nonempty 1D array")
    if not bool(np.all(np.isfinite(scores))):
        raise ValueError("scores contain non-finite values")

    unique = np.unique(scores)
    above = np.nextafter(float(unique[-1]), np.inf)
    return np.concatenate([unique, np.asarray([above])])


def select_classification_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    maximum_fpr: float,
) -> tuple[float, dict[str, float | int]]:
    best_threshold: float | None = None
    best_metrics: dict[str, float | int] | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in threshold_grid(scores):
        metrics = binary_metrics(y, scores >= threshold)
        fpr = float(metrics["fpr"])
        recall = float(metrics["recall"])

        if not np.isfinite(fpr) or fpr > maximum_fpr + 1e-12:
            continue

        recall_key = recall if np.isfinite(recall) else -np.inf
        key = (
            recall_key,
            -fpr,
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("No feasible classification threshold found.")

    return best_threshold, best_metrics


def select_distance_threshold(
    cheap_probability: np.ndarray,
    target_rate: float,
) -> float:
    distance = np.abs(np.asarray(cheap_probability, dtype=float) - 0.5)
    candidates = np.unique(distance)
    for threshold in candidates:
        if float(np.mean(distance <= threshold)) + 1e-15 >= target_rate:
            return float(threshold)
    return float(candidates[-1])


def ranked_mask(
    cheap_probability: np.ndarray,
    example_id: np.ndarray,
    target_rate: float,
) -> np.ndarray:
    probability = np.asarray(cheap_probability, dtype=float)
    ids = np.asarray(example_id, dtype=str)
    n = len(probability)
    k = int(math.ceil(target_rate * n))
    k = max(0, min(k, n))

    distance = np.abs(probability - 0.5)
    # np.lexsort uses the last key as primary: distance first, then ID tie-break.
    order = np.lexsort((ids, distance))

    mask = np.zeros(n, dtype=bool)
    if k:
        mask[order[:k]] = True
    return mask


def fit_model(
    frame: pd.DataFrame,
    features: list[str],
    label_column: str,
    parameters: dict,
) -> LogisticRegression:
    y = frame[label_column].to_numpy(dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise RuntimeError(
            f"Training data for {label_column} does not contain both classes."
        )

    model = LogisticRegression(
        C=float(parameters["C"]),
        solver=str(parameters["solver"]),
        max_iter=int(parameters["max_iter"]),
        class_weight=parameters.get("class_weight"),
        random_state=int(parameters["random_state"]),
    )
    model.fit(frame[features].to_numpy(dtype=float), y)
    return model


def probabilities(
    model: LogisticRegression,
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    return model.predict_proba(
        frame[features].to_numpy(dtype=float)
    )[:, 1]


def model_definition(
    model: LogisticRegression,
    features: list[str],
) -> dict:
    return {
        "features": list(features),
        "coef": [float(x) for x in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "classes": [int(x) for x in model.classes_.tolist()],
        "probability": "sigmoid(intercept + dot(coef, features))",
    }


def fold_assignment(
    frame: pd.DataFrame,
    *,
    label_column: str,
    group_column: str,
    seed: int,
    n_splits: int,
    fold_roles: dict[str, str],
) -> np.ndarray:
    groups = frame[group_column].astype(str).to_numpy()
    strata = (
        frame["source_dataset"].astype(str)
        + "|"
        + frame[label_column].astype(int).astype(str)
    ).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(seed),
    )

    assignment = np.full(len(frame), -1, dtype=np.int64)

    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            np.zeros((len(frame), 1)),
            strata,
            groups,
        )
    ):
        if np.any(assignment[validation_indices] != -1):
            raise RuntimeError("Duplicate fold assignment.")
        assignment[validation_indices] = fold

    if np.any(assignment < 0):
        raise RuntimeError("Incomplete fold assignment.")

    # Check group-disjoint role assignment.
    role = np.asarray(
        [fold_roles[str(int(fold))] for fold in assignment],
        dtype=object,
    )
    tmp = pd.DataFrame({"group": groups, "role": role})
    role_counts = tmp.groupby("group")["role"].nunique()
    if int(role_counts.max()) != 1:
        raise RuntimeError("Dependency group crosses fold roles.")

    return assignment


def add_policy_predictions(
    output_rows: list[pd.DataFrame],
    *,
    holdout: pd.DataFrame,
    y_hold: np.ndarray,
    analysis_groups: np.ndarray,
    primary_groups: np.ndarray,
    cheap_probability: np.ndarray,
    full_probability: np.ndarray,
    acquire_mask: np.ndarray,
    decision_threshold: float,
    acquisition_threshold: float | None,
    label_condition: str,
    grouping_condition: str,
    seed: int,
    stack: str,
    policy_kind: str,
    target_rate: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.where(
        acquire_mask,
        full_probability,
        cheap_probability,
    )
    pred = combined >= decision_threshold

    part = pd.DataFrame(
        {
            "label_condition": label_condition,
            "grouping_condition": grouping_condition,
            "seed": int(seed),
            "stack": stack,
            "policy_kind": policy_kind,
            "target_rate": (
                math.nan if target_rate is None else float(target_rate)
            ),
            "example_id": holdout["example_id"].astype(str).to_numpy(),
            "source_dataset": holdout["source_dataset"].astype(str).to_numpy(),
            "y_eval": y_hold,
            "y_audited": holdout["y"].to_numpy(dtype=np.int64),
            "y_original": holdout["y_original"].to_numpy(dtype=np.int64),
            "analysis_group": analysis_groups.astype(str),
            "primary_dependency_group": primary_groups.astype(str),
            "cheap_probability": cheap_probability,
            "full_probability": full_probability,
            "acquired": acquire_mask.astype(np.int8),
            "combined_probability": combined,
            "prediction": pred.astype(np.int8),
            "decision_threshold": float(decision_threshold),
            "acquisition_threshold": (
                math.nan
                if acquisition_threshold is None
                else float(acquisition_threshold)
            ),
        }
    )
    output_rows.append(part)
    return combined, pred


def source_metric_rows(
    *,
    holdout: pd.DataFrame,
    y_hold: np.ndarray,
    pred: np.ndarray,
    primary_groups: np.ndarray,
    confidence: float,
    metadata: dict,
) -> list[dict]:
    rows: list[dict] = []
    sources = holdout["source_dataset"].astype(str).to_numpy()

    for source in sorted(set(sources.tolist())):
        mask = sources == source
        metrics = finite_risk_metrics(
            y_hold[mask],
            pred[mask],
            primary_groups[mask],
            confidence=confidence,
        )
        rows.append(
            {
                **metadata,
                "source_dataset": source,
                **metrics,
            }
        )
    return rows


def deterministic_timing_sample(
    frame: pd.DataFrame,
    *,
    rows_per_source: int,
) -> pd.DataFrame:
    records = frame[
        [
            "example_id",
            "source_dataset",
            "prompt",
            "response",
            PRIMARY_GROUP,
        ]
    ].copy()

    records["example_id"] = records["example_id"].astype(str)
    records["source_dataset"] = records["source_dataset"].astype(str)
    records["_hash"] = records["example_id"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )

    base_ids: set[str] = set()
    for source, part in records.groupby("source_dataset", sort=True):
        ordered = part.sort_values(
            ["_hash", "example_id"],
            kind="mergesort",
        )
        take = min(int(rows_per_source), len(ordered))
        base_ids.update(ordered.iloc[:take]["example_id"].tolist())

    base_mask = records["example_id"].isin(base_ids)
    selected_groups = set(
        records.loc[base_mask, PRIMARY_GROUP].astype(str).tolist()
    )
    final_mask = records[PRIMARY_GROUP].astype(str).isin(selected_groups)

    sample = records.loc[final_mask].copy()
    sample["base_selected"] = sample["example_id"].isin(base_ids)
    sample = sample.sort_values(
        ["source_dataset", "_hash", "example_id"],
        kind="mergesort",
    ).drop(columns=["_hash"]).reset_index(drop=True)

    return sample


def package_results(
    final_dir: Path,
    *,
    generator_path: Path,
    protocol_path: Path,
    grouping_spec_path: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = (
        DOWNLOADS
        / f"evaluation_measurement_pilot_v1_cpu_results_{timestamp}.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(final_dir.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    Path("cpu_results") / path.relative_to(final_dir),
                )
        archive.write(
            generator_path,
            Path("reproduction") / generator_path.name,
        )
        archive.write(
            protocol_path,
            Path("reproduction") / protocol_path.name,
        )
        archive.write(
            grouping_spec_path,
            Path("reproduction") / grouping_spec_path.name,
        )

    return zip_path


def main() -> None:
    progress("[1/12] Verifying frozen commit, protocol, and clean data boundary...")

    if not (ROOT / ".git").is_dir():
        raise SystemExit(f"Repository not found: {ROOT}")

    current_branch = run_git(["branch", "--show-current"])
    if current_branch != "evaluation-measurement-aug17-repair":
        raise SystemExit(
            f"Expected evaluation-measurement-aug17-repair, found {current_branch!r}"
        )

    # The implementation may be uncommitted, but the frozen protocol commit
    # must remain in the branch ancestry.
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit(
            "Frozen pre-outcome commit is not an ancestor of current HEAD."
        )

    for path in (PROTOCOL, GROUPING_SPEC, GROUPS_PATH):
        if not path.is_file():
            raise SystemExit(f"Missing frozen artifact: {path}")

    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise SystemExit("Frozen pilot protocol hash mismatch.")
    if sha256(GROUPING_SPEC) != EXPECTED_GROUPING_SPEC_SHA:
        raise SystemExit("Frozen grouping spec hash mismatch.")
    if sha256(GROUPS_PATH) != EXPECTED_GROUPS_SHA:
        raise SystemExit("Frozen grouping CSV hash mismatch.")

    if FINAL_DIR.exists():
        raise SystemExit(
            f"CPU pilot output already exists; refusing overwrite: {FINAL_DIR}"
        )

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    group_spec = json.loads(GROUPING_SPEC.read_text(encoding="utf-8"))

    if protocol["status"] != "frozen_before_pilot_outcomes":
        raise SystemExit("Pilot protocol is not frozen.")
    if protocol["protocol_version"] != "1.1.0":
        raise SystemExit("Unexpected protocol version.")

    progress("[2/12] Verifying development-only materialization before reading rows...")

    for path in (DEV_MANIFEST, DEV_DATASET, DEV_CACHE):
        if not path.is_file():
            raise SystemExit(
                f"Missing development-only input: {path}. "
                "Do not substitute the mixed full-data container."
            )

    dev_manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if dev_manifest.get("protected_rows_materialized") is not False:
        raise SystemExit("Development-view manifest is not fail-closed.")

    manifest_outputs = {
        item["path"]: item for item in dev_manifest["outputs"]
    }
    expected_paths = {
        str(DEV_DATASET.relative_to(ROOT)),
        str(DEV_CACHE.relative_to(ROOT)),
    }
    if set(manifest_outputs) != expected_paths:
        raise SystemExit(
            "Development-view manifest does not describe exactly the two "
            "authorized development artifacts."
        )

    for path in (DEV_DATASET, DEV_CACHE):
        rel = str(path.relative_to(ROOT))
        record = manifest_outputs[rel]
        if int(record["row_count"]) != EXPECTED_ROWS:
            raise SystemExit(f"Unexpected development row count for {rel}.")
        if sha256(path) != record["sha256"]:
            raise SystemExit(f"Development-view hash mismatch for {rel}.")

    development_expected_hash = group_spec["inputs"]["development"]["sha256"]
    if sha256(DEV_DATASET) != development_expected_hash:
        raise SystemExit(
            "Development dataset hash differs from the frozen grouping input."
        )

    progress("[3/12] Loading only development dataset, cached scores, and frozen groups...")

    dataset = pd.read_parquet(DEV_DATASET)
    cache = pd.read_parquet(DEV_CACHE)
    groups = pd.read_csv(GROUPS_PATH)

    for name, frame, required in [
        ("dataset", dataset, DATASET_FIELDS_REQUIRED),
        ("cache", cache, CACHE_FIELDS_REQUIRED),
    ]:
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise SystemExit(f"{name} missing required columns: {missing}")

    for name, frame in [("dataset", dataset), ("cache", cache)]:
        if len(frame) != EXPECTED_ROWS:
            raise SystemExit(f"{name} row count != {EXPECTED_ROWS}")
        if not frame["example_id"].astype(str).is_unique:
            raise SystemExit(f"{name} contains duplicate example_id.")
        observed = set(frame["split"].astype(str).unique().tolist())
        if observed.intersection(PROTECTED_SPLITS):
            raise SystemExit(f"{name} contains a protected split.")
        if observed != ALLOWED_SPLITS:
            raise SystemExit(
                f"{name} does not contain exactly the authorized splits: {observed}"
            )

    if len(groups) != EXPECTED_ROWS:
        raise SystemExit("Frozen grouping row count mismatch.")
    if not groups["example_id"].astype(str).is_unique:
        raise SystemExit("Frozen grouping contains duplicate example IDs.")

    # Keep labels/text/source from dataset; scores only from the cache.
    score_frame = cache[["example_id", *SCORE_FIELDS]].copy()
    frame = dataset.merge(
        score_frame,
        on="example_id",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(
        groups,
        on=["example_id", "source_dataset"],
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != EXPECTED_ROWS:
        raise SystemExit("Merged development frame row count mismatch.")

    frame["example_id"] = frame["example_id"].astype(str)
    frame["source_dataset"] = frame["source_dataset"].astype(str)

    for column in ["y", "y_original"]:
        values = frame[column].to_numpy()
        if not bool(np.all(np.isin(values, [0, 1]))):
            raise SystemExit(f"{column} is not binary.")
        frame[column] = frame[column].astype(np.int64)

    for field in SCORE_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        if not bool(np.all(np.isfinite(values))):
            raise SystemExit(f"{field} contains non-finite values.")
        frame[field] = values

    source_counts = (
        frame.groupby(["source_dataset", "y"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nAUDITED_SOURCE_COUNTS", flush=True)
    print(source_counts.to_string(), flush=True)

    for source, expected in protocol["datasets"].items():
        part = frame[frame["source_dataset"].eq(source)]
        if len(part) != int(expected["expected_rows"]):
            raise SystemExit(
                f"Source count mismatch for {source}: {len(part)}"
            )
        if int((part["y"] == 0).sum()) != int(expected["expected_negatives"]):
            raise SystemExit(f"Negative count mismatch for {source}.")
        if int((part["y"] == 1).sum()) != int(expected["expected_positives"]):
            raise SystemExit(f"Positive count mismatch for {source}.")

    progress("[4/12] Creating deterministic grouped fold assignments...")

    seeds = [int(x) for x in protocol["resampling"]["seeds"]]
    n_splits = int(protocol["resampling"]["n_splits"])
    fold_roles = {
        str(k): str(v)
        for k, v in protocol["resampling"]["fold_roles"].items()
    }

    fold_rows: list[pd.DataFrame] = []
    assignments: dict[tuple[str, str, int], np.ndarray] = {}

    for label_condition, label_column in LABEL_CONDITIONS.items():
        for grouping_condition, group_column in GROUPING_CONDITIONS.items():
            if group_column not in frame.columns:
                raise SystemExit(
                    f"Missing frozen grouping column: {group_column}"
                )
            for seed in seeds:
                assignment = fold_assignment(
                    frame,
                    label_column=label_column,
                    group_column=group_column,
                    seed=seed,
                    n_splits=n_splits,
                    fold_roles=fold_roles,
                )
                assignments[(label_condition, grouping_condition, seed)] = assignment

                roles = np.asarray(
                    [fold_roles[str(int(x))] for x in assignment],
                    dtype=object,
                )
                fold_rows.append(
                    pd.DataFrame(
                        {
                            "label_condition": label_condition,
                            "label_column": label_column,
                            "grouping_condition": grouping_condition,
                            "group_column": group_column,
                            "seed": seed,
                            "example_id": frame["example_id"].to_numpy(),
                            "source_dataset": frame["source_dataset"].to_numpy(),
                            "fold": assignment,
                            "role": roles,
                        }
                    )
                )

    fold_assignments = pd.concat(fold_rows, ignore_index=True)

    progress("[5/12] Fitting frozen logistic models and evaluating policies...")

    params = protocol["predictors"]["parameters"]
    maximum_fpr = float(protocol["decision_operating_point"]["maximum_fpr"])
    confidence = float(protocol["inference"]["confidence_level"])
    target_rates = [
        float(x)
        for x in protocol["routing_comparison"]["target_acquisition_rates"]
    ]

    policy_rows: list[dict] = []
    source_rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []

    primary_policy_definitions = {
        "artifact": "evaluation_measurement_pilot_v1_primary_policy_definitions",
        "protocol_sha256": sha256(PROTOCOL),
        "freeze_commit": FROZEN_COMMIT,
        "label_condition": "audited",
        "grouping_condition": "dependency_primary",
        "seed": int(protocol["resampling"]["primary_inference_seed"]),
        "model_family": "logistic_regression",
        "stacks": {},
    }

    total_fits = (
        len(LABEL_CONDITIONS)
        * len(GROUPING_CONDITIONS)
        * len(seeds)
        * len(protocol["monitor_stacks"])
    )
    fit_number = 0

    for label_condition, label_column in LABEL_CONDITIONS.items():
        for grouping_condition, group_column in GROUPING_CONDITIONS.items():
            for seed in seeds:
                assignment = assignments[
                    (label_condition, grouping_condition, seed)
                ]
                roles = np.asarray(
                    [fold_roles[str(int(x))] for x in assignment],
                    dtype=object,
                )

                train_mask = roles == "policy_train"
                selection_mask = roles == "policy_selection"
                holdout_mask = roles == "internal_risk_holdout"

                train = frame.loc[train_mask].reset_index(drop=True)
                selection = frame.loc[selection_mask].reset_index(drop=True)
                holdout = frame.loc[holdout_mask].reset_index(drop=True)

                y_sel = selection[label_column].to_numpy(dtype=np.int64)
                y_hold = holdout[label_column].to_numpy(dtype=np.int64)
                primary_groups_hold = holdout[PRIMARY_GROUP].astype(str).to_numpy()
                analysis_groups_hold = holdout[group_column].astype(str).to_numpy()

                for stack_name, stack_cfg in protocol["monitor_stacks"].items():
                    fit_number += 1
                    if fit_number == 1 or fit_number % 10 == 0:
                        progress(
                            f"    model fit {fit_number}/{total_fits}: "
                            f"{label_condition}/{grouping_condition}/"
                            f"seed={seed}/{stack_name}"
                        )

                    cheap_features = [str(x) for x in stack_cfg["cheap_features"]]
                    full_features = [str(x) for x in stack_cfg["full_features"]]

                    cheap_model = fit_model(
                        train,
                        cheap_features,
                        label_column,
                        params,
                    )
                    full_model = fit_model(
                        train,
                        full_features,
                        label_column,
                        params,
                    )

                    cheap_sel = probabilities(
                        cheap_model,
                        selection,
                        cheap_features,
                    )
                    full_sel = probabilities(
                        full_model,
                        selection,
                        full_features,
                    )
                    cheap_hold = probabilities(
                        cheap_model,
                        holdout,
                        cheap_features,
                    )
                    full_hold = probabilities(
                        full_model,
                        holdout,
                        full_features,
                    )

                    model_record: dict | None = None
                    if (
                        label_condition == "audited"
                        and grouping_condition == "dependency_primary"
                        and seed
                        == int(protocol["resampling"]["primary_inference_seed"])
                    ):
                        model_record = {
                            "cheap_model": model_definition(
                                cheap_model,
                                cheap_features,
                            ),
                            "full_model": model_definition(
                                full_model,
                                full_features,
                            ),
                            "policies": [],
                        }
                        primary_policy_definitions["stacks"][stack_name] = model_record

                    def evaluate_policy(
                        *,
                        policy_kind: str,
                        target_rate: float | None,
                        acquire_sel: np.ndarray,
                        acquire_hold: np.ndarray,
                        acquisition_threshold: float | None,
                    ) -> None:
                        combined_sel = np.where(
                            acquire_sel,
                            full_sel,
                            cheap_sel,
                        )
                        decision_threshold, selection_metrics = (
                            select_classification_threshold(
                                y_sel,
                                combined_sel,
                                maximum_fpr,
                            )
                        )

                        combined_hold, pred_hold = add_policy_predictions(
                            prediction_parts,
                            holdout=holdout,
                            y_hold=y_hold,
                            analysis_groups=analysis_groups_hold,
                            primary_groups=primary_groups_hold,
                            cheap_probability=cheap_hold,
                            full_probability=full_hold,
                            acquire_mask=acquire_hold,
                            decision_threshold=decision_threshold,
                            acquisition_threshold=acquisition_threshold,
                            label_condition=label_condition,
                            grouping_condition=grouping_condition,
                            seed=seed,
                            stack=stack_name,
                            policy_kind=policy_kind,
                            target_rate=target_rate,
                        )

                        hold_metrics = finite_risk_metrics(
                            y_hold,
                            pred_hold,
                            primary_groups_hold,
                            confidence=confidence,
                        )
                        risk_pass = (
                            float(hold_metrics["row_fpr_upper95"])
                            <= maximum_fpr + 1e-12
                            and float(
                                hold_metrics["dependency_group_event_upper95"]
                            )
                            <= maximum_fpr + 1e-12
                        )

                        metadata = {
                            "label_condition": label_condition,
                            "label_column": label_column,
                            "grouping_condition": grouping_condition,
                            "group_column": group_column,
                            "seed": seed,
                            "stack": stack_name,
                            "policy_kind": policy_kind,
                            "target_rate": (
                                math.nan
                                if target_rate is None
                                else float(target_rate)
                            ),
                            "deployable": (
                                policy_kind != "offline_ranked_diagnostic"
                            ),
                            "acquisition_threshold": (
                                math.nan
                                if acquisition_threshold is None
                                else float(acquisition_threshold)
                            ),
                            "decision_threshold": float(decision_threshold),
                            "selection_acquisition_rate": float(
                                np.mean(acquire_sel)
                            ),
                            "holdout_acquisition_rate": float(
                                np.mean(acquire_hold)
                            ),
                        }

                        policy_rows.append(
                            {
                                **metadata,
                                "selection_n": int(selection_metrics["n"]),
                                "selection_recall": float(
                                    selection_metrics["recall"]
                                ),
                                "selection_fpr": float(
                                    selection_metrics["fpr"]
                                ),
                                **{
                                    f"holdout_{key}": value
                                    for key, value in hold_metrics.items()
                                },
                                "primary_risk_gate_pass": bool(risk_pass),
                            }
                        )

                        source_rows.extend(
                            source_metric_rows(
                                holdout=holdout,
                                y_hold=y_hold,
                                pred=pred_hold,
                                primary_groups=primary_groups_hold,
                                confidence=confidence,
                                metadata=metadata,
                            )
                        )

                        if model_record is not None:
                            model_record["policies"].append(
                                {
                                    "policy_kind": policy_kind,
                                    "target_rate": target_rate,
                                    "deployable": (
                                        policy_kind
                                        != "offline_ranked_diagnostic"
                                    ),
                                    "acquisition_threshold": (
                                        acquisition_threshold
                                    ),
                                    "decision_threshold": (
                                        decision_threshold
                                    ),
                                    "selection_acquisition_rate": float(
                                        np.mean(acquire_sel)
                                    ),
                                    "holdout_acquisition_rate": float(
                                        np.mean(acquire_hold)
                                    ),
                                }
                            )

                    # Cheap-only anchor.
                    evaluate_policy(
                        policy_kind="cheap_only",
                        target_rate=None,
                        acquire_sel=np.zeros(len(selection), dtype=bool),
                        acquire_hold=np.zeros(len(holdout), dtype=bool),
                        acquisition_threshold=None,
                    )

                    # Full-information anchor.
                    evaluate_policy(
                        policy_kind="full_information",
                        target_rate=None,
                        acquire_sel=np.ones(len(selection), dtype=bool),
                        acquire_hold=np.ones(len(holdout), dtype=bool),
                        acquisition_threshold=None,
                    )

                    # Deployable and ranked policies at all frozen acquisition rates.
                    for target_rate in target_rates:
                        distance_threshold = select_distance_threshold(
                            cheap_sel,
                            target_rate,
                        )
                        deploy_sel = (
                            np.abs(cheap_sel - 0.5)
                            <= distance_threshold
                        )
                        deploy_hold = (
                            np.abs(cheap_hold - 0.5)
                            <= distance_threshold
                        )
                        evaluate_policy(
                            policy_kind="selective_deployable",
                            target_rate=target_rate,
                            acquire_sel=deploy_sel,
                            acquire_hold=deploy_hold,
                            acquisition_threshold=distance_threshold,
                        )

                        ranked_sel = ranked_mask(
                            cheap_sel,
                            selection["example_id"].astype(str).to_numpy(),
                            target_rate,
                        )
                        ranked_hold = ranked_mask(
                            cheap_hold,
                            holdout["example_id"].astype(str).to_numpy(),
                            target_rate,
                        )
                        evaluate_policy(
                            policy_kind="offline_ranked_diagnostic",
                            target_rate=target_rate,
                            acquire_sel=ranked_sel,
                            acquire_hold=ranked_hold,
                            acquisition_threshold=None,
                        )

    policy_summary = pd.DataFrame(policy_rows)
    source_metrics = pd.DataFrame(source_rows)
    holdout_predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    progress("[6/12] Building primary-seed paired recall bootstrap inputs...")

    primary_predictions = holdout_predictions[
        holdout_predictions["label_condition"].eq("audited")
        & holdout_predictions["grouping_condition"].eq("dependency_primary")
        & holdout_predictions["seed"].eq(
            int(protocol["resampling"]["primary_inference_seed"])
        )
        & holdout_predictions["policy_kind"].isin(
            [
                "cheap_only",
                "full_information",
                "selective_deployable",
            ]
        )
    ].copy()

    # A compact policy key used by later timing/Pareto analysis.
    primary_predictions["policy_id"] = primary_predictions.apply(
        lambda row: (
            f"{row['stack']}::{row['policy_kind']}"
            if pd.isna(row["target_rate"])
            else (
                f"{row['stack']}::{row['policy_kind']}::"
                f"{float(row['target_rate']):.2f}"
            )
        ),
        axis=1,
    )

    # Save primary prediction matrix rather than claiming a Pareto result
    # before direct E2E timing exists.
    primary_matrix = primary_predictions[
        [
            "policy_id",
            "example_id",
            "source_dataset",
            "y_eval",
            "primary_dependency_group",
            "prediction",
            "acquired",
            "cheap_probability",
            "full_probability",
            "combined_probability",
        ]
    ].copy()

    progress("[7/12] Constructing the label-blind, dependency-closed T4 timing sample...")

    timing_cfg = protocol["cost_measurement"]["timing_sample"]
    timing_sample = deterministic_timing_sample(
        frame,
        rows_per_source=int(timing_cfg["base_target_rows_per_source"]),
    )

    base_counts = (
        timing_sample[timing_sample["base_selected"]]
        .groupby("source_dataset")
        .size()
        .to_dict()
    )
    final_counts = (
        timing_sample.groupby("source_dataset")
        .size()
        .to_dict()
    )
    print("\nTIMING_SAMPLE_COUNTS", flush=True)
    print(
        json.dumps(
            {
                "base_counts": {
                    str(k): int(v) for k, v in sorted(base_counts.items())
                },
                "final_counts_after_group_closure": {
                    str(k): int(v) for k, v in sorted(final_counts.items())
                },
                "final_rows": int(len(timing_sample)),
                "labels_used_for_sampling": False,
            },
            indent=2,
        ),
        flush=True,
    )

    progress("[8/12] Writing CPU pilot outputs atomically...")

    FINAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=".evaluation-measurement-pilot-v1-cpu-",
            dir=FINAL_DIR.parent,
        )
    )

    try:
        fold_assignments.to_csv(
            temp_dir / "fold_assignments.csv",
            index=False,
        )
        policy_summary.to_csv(
            temp_dir / "policy_summary.csv",
            index=False,
        )
        source_metrics.to_csv(
            temp_dir / "source_metrics.csv",
            index=False,
        )
        holdout_predictions.to_parquet(
            temp_dir / "holdout_predictions.parquet",
            index=False,
            compression="zstd",
        )
        primary_matrix.to_parquet(
            temp_dir / "primary_policy_prediction_matrix.parquet",
            index=False,
            compression="zstd",
        )
        timing_sample.to_parquet(
            temp_dir / "timing_sample.parquet",
            index=False,
            compression="zstd",
        )
        (temp_dir / "primary_policy_definitions.json").write_text(
            json.dumps(
                primary_policy_definitions,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        progress("[9/12] Creating environment and provenance manifest...")

        output_names = [
            "fold_assignments.csv",
            "policy_summary.csv",
            "source_metrics.csv",
            "holdout_predictions.parquet",
            "primary_policy_prediction_matrix.parquet",
            "timing_sample.parquet",
            "primary_policy_definitions.json",
        ]

        manifest = {
            "artifact": "evaluation_measurement_pilot_v1_cpu",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "development_only_cpu_outcomes",
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256(PROTOCOL),
            "freeze_commit": FROZEN_COMMIT,
            "current_head": run_git(["rev-parse", "HEAD"]),
            "generator": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "data_boundary": {
                "development_only": True,
                "development_rows": int(len(frame)),
                "authorized_splits": sorted(ALLOWED_SPLITS),
                "protected_splits_opened": False,
                "mixed_full_data_containers_opened": False,
            },
            "inputs": {
                str(DEV_MANIFEST.relative_to(ROOT)): sha256(DEV_MANIFEST),
                str(DEV_DATASET.relative_to(ROOT)): sha256(DEV_DATASET),
                str(DEV_CACHE.relative_to(ROOT)): sha256(DEV_CACHE),
                str(GROUPING_SPEC.relative_to(ROOT)): sha256(GROUPING_SPEC),
                str(GROUPS_PATH.relative_to(ROOT)): sha256(GROUPS_PATH),
            },
            "design": {
                "label_conditions": LABEL_CONDITIONS,
                "grouping_conditions": GROUPING_CONDITIONS,
                "seeds": seeds,
                "monitor_stacks": list(protocol["monitor_stacks"]),
                "target_acquisition_rates": target_rates,
                "primary_inference_seed": int(
                    protocol["resampling"]["primary_inference_seed"]
                ),
                "repeated_seed_pooling": False,
                "primary_cost_estimand_available": False,
                "direct_e2e_timing_pending": True,
                "pareto_claim_available": False,
                "offline_ranked_is_diagnostic_only": True,
            },
            "counts": {
                "policy_summary_rows": int(len(policy_summary)),
                "source_metric_rows": int(len(source_metrics)),
                "holdout_prediction_rows": int(len(holdout_predictions)),
                "primary_prediction_rows": int(len(primary_matrix)),
                "timing_sample_rows": int(len(timing_sample)),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "outputs": {
                name: sha256(temp_dir / name)
                for name in output_names
            },
            "claim_boundary": (
                "CPU outputs characterize predictive/risk/routing measurement "
                "effects only. Direct wall-clock E2E cost is not available until "
                "the frozen T4 timing stage, so no Pareto or iso-cost conclusion "
                "is permitted from this artifact alone."
            ),
        }

        (temp_dir / "cpu_pilot_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        progress("[10/12] Running deterministic output integrity checks...")

        if len(policy_summary) != (
            len(LABEL_CONDITIONS)
            * len(GROUPING_CONDITIONS)
            * len(seeds)
            * len(protocol["monitor_stacks"])
            * (2 + 2 * len(target_rates))
        ):
            raise RuntimeError("Unexpected policy summary row count.")

        if policy_summary["policy_kind"].eq(
            "offline_ranked_diagnostic"
        ).any():
            offline = policy_summary[
                policy_summary["policy_kind"].eq(
                    "offline_ranked_diagnostic"
                )
            ]
            if bool(offline["deployable"].any()):
                raise RuntimeError(
                    "Offline-ranked diagnostic was incorrectly marked deployable."
                )

        deployable = policy_summary[
            policy_summary["policy_kind"].eq("selective_deployable")
        ]
        if bool(
            (
                deployable["selection_acquisition_rate"]
                + 1e-12
                < deployable["target_rate"]
            ).any()
        ):
            raise RuntimeError(
                "A deployable selection threshold missed its frozen target rate."
            )

        # XSTest has no positives: source-specific recall must remain undefined.
        xstest = source_metrics[
            source_metrics["source_dataset"].eq("xstest_safe_gpt4")
        ]
        if not bool(xstest["recall"].isna().all()):
            raise RuntimeError("XSTest recall should be undefined.")

        os.replace(temp_dir, FINAL_DIR)

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    progress("[11/12] Packaging upload-ready CPU results in Windows Downloads...")

    package_path = package_results(
        FINAL_DIR,
        generator_path=Path(__file__).resolve(),
        protocol_path=PROTOCOL,
        grouping_spec_path=GROUPING_SPEC,
    )

    progress("[12/12] Printing compact pilot summary...")

    primary_summary = policy_summary[
        policy_summary["label_condition"].eq("audited")
        & policy_summary["grouping_condition"].eq("dependency_primary")
        & policy_summary["seed"].eq(
            int(protocol["resampling"]["primary_inference_seed"])
        )
    ][
        [
            "stack",
            "policy_kind",
            "target_rate",
            "holdout_acquisition_rate",
            "holdout_recall",
            "holdout_fpr",
            "holdout_row_fpr_upper95",
            "holdout_dependency_group_event_upper95",
            "primary_risk_gate_pass",
        ]
    ].sort_values(
        ["stack", "policy_kind", "target_rate"],
        na_position="first",
    )

    print("\nPRIMARY_SEED_POLICY_SUMMARY", flush=True)
    print(primary_summary.to_string(index=False), flush=True)

    print("\nEVALUATION_MEASUREMENT_PILOT_V1_CPU=PASS", flush=True)
    print(f"results_dir={FINAL_DIR}", flush=True)
    print(f"package_file={package_path}", flush=True)
    print("protected_splits_opened=false", flush=True)
    print("direct_e2e_timing_pending=true", flush=True)
    print("pareto_claim_available=false", flush=True)


if __name__ == "__main__":
    main()
