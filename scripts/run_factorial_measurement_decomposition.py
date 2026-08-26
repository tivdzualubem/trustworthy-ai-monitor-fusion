#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/factorial_measurement_decomposition_v1.json"
OUT_DIR = ROOT / "reports/factorial_measurement_decomposition_v1"

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
GROUPS = (
    ROOT
    / "data/metadata/evaluation_measurement_pilot_v1/"
    / "development_dependency_groups.csv"
)
PILOT_PROTOCOL = ROOT / "configs/evaluation_measurement_pilot_v1.json"
PILOT_PREDICTIONS = (
    ROOT
    / "reports/evaluation_measurement_pilot_v1/cpu/"
    / "holdout_predictions.parquet"
)
PILOT_SUMMARY = (
    ROOT
    / "reports/evaluation_measurement_pilot_v1/cpu/"
    / "policy_summary.csv"
)
PROVENANCE = ROOT / "data/metadata/confirmatory_split_provenance.json"

EXPECTED_BASE_COMMIT = "ffb02e4c1b144d56ce6d8c034212967bf71f63b8"
EXPECTED_BRANCH = "factorial-measurement-decomposition"

AUTHORIZED_SPLITS = {"policy_train", "policy_selection", "calibration"}
PROTECTED_SPLITS = {"final_test", "held_out_shift"}
EXPECTED_ROWS = 1687
PRIMARY_GROUP = "primary_dependency_group"
SCORE_FIELDS = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]
LABEL_COLUMNS = {
    "audited": "y",
    "original": "y_original",
}
GROUP_COLUMNS = {
    "dependency_primary": PRIMARY_GROUP,
    "singleton_weak": "example_id",
}
POLICY_KINDS = {"cheap_only", "full_information", "selective_deployable"}
KEYS = ["seed", "stack", "policy_kind", "target_key"]


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_upper(k: int, n: int, confidence: float) -> float:
    if n <= 0:
        return math.nan
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def risk_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    *,
    confidence: float,
    risk_limit: float,
) -> dict[str, float | int | bool]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=bool)
    groups = np.asarray(groups, dtype=object)

    if len(y) != len(pred) or len(y) != len(groups):
        raise ValueError("Metric arrays must have equal length.")

    pos = y == 1
    neg = y == 0

    tp = int(np.sum(pred & pos))
    fn = int(np.sum((~pred) & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))

    p = int(pos.sum())
    n = int(neg.sum())

    recall = tp / p if p else math.nan
    fpr = fp / n if n else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    row_upper = exact_upper(fp, n, confidence) if n else math.nan

    negative_groups = pd.DataFrame(
        {
            "group": groups[neg].astype(str),
            "false_positive": pred[neg],
        }
    )
    if len(negative_groups):
        group_failures = (
            negative_groups.groupby("group", sort=False)["false_positive"].any()
        )
        negative_group_n = int(len(group_failures))
        failed_group_n = int(group_failures.sum())
        group_rate = (
            failed_group_n / negative_group_n
            if negative_group_n
            else math.nan
        )
        group_upper = exact_upper(
            failed_group_n,
            negative_group_n,
            confidence,
        )
    else:
        negative_group_n = 0
        failed_group_n = 0
        group_rate = math.nan
        group_upper = math.nan

    gate = (
        np.isfinite(row_upper)
        and np.isfinite(group_upper)
        and row_upper <= risk_limit + 1e-12
        and group_upper <= risk_limit + 1e-12
    )

    return {
        "n": int(len(y)),
        "positive_n": p,
        "negative_n": n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "row_fpr_upper95": float(row_upper),
        "negative_dependency_groups": negative_group_n,
        "failed_dependency_groups": failed_group_n,
        "dependency_group_event_rate": float(group_rate),
        "dependency_group_event_upper95": float(group_upper),
        "risk_gate_pass": bool(gate),
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("scores must be a nonempty 1D array")
    if not np.all(np.isfinite(scores)):
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
    best_key: tuple[float, float, float] | None = None
    best_metrics: dict[str, float | int] | None = None

    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)

    for threshold in threshold_grid(scores):
        pred = scores >= threshold
        neg = y == 0
        pos = y == 1

        fp = int(np.sum(pred & neg))
        tp = int(np.sum(pred & pos))
        n = int(neg.sum())
        p = int(pos.sum())

        fpr = fp / n if n else math.nan
        recall = tp / p if p else math.nan

        if not np.isfinite(fpr) or fpr > maximum_fpr + 1e-12:
            continue

        recall_key = recall if np.isfinite(recall) else -np.inf
        key = (recall_key, -fpr, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "n": int(len(y)),
                "recall": float(recall),
                "fpr": float(fpr),
            }

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


def target_key(value) -> str:
    if value is None or pd.isna(value):
        return "anchor"
    return f"{float(value):.2f}"


def deployable_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def scalar_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "pass"}:
        return True
    if normalized in {"0", "false", "no", "fail"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def evaluate_policy(
    *,
    selection: pd.DataFrame,
    outer: pd.DataFrame,
    y_selection: np.ndarray,
    cheap_sel: np.ndarray,
    full_sel: np.ndarray,
    cheap_outer: np.ndarray,
    full_outer: np.ndarray,
    policy_kind: str,
    target_rate: float | None,
    maximum_fpr: float,
) -> dict:
    if policy_kind == "cheap_only":
        acquire_sel = np.zeros(len(selection), dtype=bool)
        acquire_outer = np.zeros(len(outer), dtype=bool)
        acquisition_threshold = math.nan
    elif policy_kind == "full_information":
        acquire_sel = np.ones(len(selection), dtype=bool)
        acquire_outer = np.ones(len(outer), dtype=bool)
        acquisition_threshold = math.nan
    elif policy_kind == "selective_deployable":
        if target_rate is None:
            raise ValueError("Selective policy requires target_rate.")
        acquisition_threshold = select_distance_threshold(
            cheap_sel,
            float(target_rate),
        )
        acquire_sel = (
            np.abs(cheap_sel - 0.5)
            <= float(acquisition_threshold)
        )
        acquire_outer = (
            np.abs(cheap_outer - 0.5)
            <= float(acquisition_threshold)
        )
    else:
        raise ValueError(f"Unsupported policy kind: {policy_kind}")

    combined_sel = np.where(acquire_sel, full_sel, cheap_sel)
    decision_threshold, selection_metrics = (
        select_classification_threshold(
            y_selection,
            combined_sel,
            maximum_fpr,
        )
    )
    combined_outer = np.where(acquire_outer, full_outer, cheap_outer)
    pred_outer = combined_outer >= decision_threshold

    return {
        "prediction": pred_outer.astype(np.int8),
        "combined_probability": combined_outer,
        "acquired": acquire_outer.astype(np.int8),
        "acquisition_threshold": (
            float(acquisition_threshold)
            if np.isfinite(acquisition_threshold)
            else math.nan
        ),
        "decision_threshold": float(decision_threshold),
        "selection_acquisition_rate": float(np.mean(acquire_sel)),
        "outer_acquisition_rate": float(np.mean(acquire_outer)),
        "selection_recall": float(selection_metrics["recall"]),
        "selection_fpr": float(selection_metrics["fpr"]),
    }


def common_outer_ids_from_frozen_predictions(
    predictions: pd.DataFrame,
    seeds: list[int],
) -> dict[int, set[str]]:
    baseline = predictions[
        predictions["label_condition"].eq("audited")
        & predictions["grouping_condition"].eq("dependency_primary")
        & predictions["policy_kind"].isin(POLICY_KINDS)
    ].copy()
    baseline["example_id"] = baseline["example_id"].astype(str)

    result: dict[int, set[str]] = {}
    for seed in seeds:
        part = baseline[baseline["seed"].astype(int).eq(seed)]
        if part.empty:
            raise RuntimeError(f"Missing frozen baseline predictions for seed={seed}")
        ids = set(part["example_id"].tolist())

        counts = (
            part.groupby(
                ["stack", "policy_kind", "target_rate"],
                dropna=False,
                sort=False,
            )["example_id"]
            .nunique()
            .to_numpy()
        )
        if len(counts) != 15:
            raise RuntimeError(
                f"Expected 15 deployable frozen policies for seed={seed}; "
                f"found {len(counts)}"
            )
        if not np.all(counts == len(ids)):
            raise RuntimeError(
                f"Frozen policies do not share one common outer set for seed={seed}."
            )
        result[seed] = ids
    return result


def fixed_policy_cells(
    predictions: pd.DataFrame,
    *,
    confidence: float,
    risk_limit: float,
    seeds: list[int],
) -> pd.DataFrame:
    baseline = predictions[
        predictions["label_condition"].eq("audited")
        & predictions["grouping_condition"].eq("dependency_primary")
        & predictions["policy_kind"].isin(POLICY_KINDS)
    ].copy()

    baseline["seed"] = baseline["seed"].astype(int)
    baseline["target_key"] = baseline["target_rate"].map(target_key)
    baseline["example_id"] = baseline["example_id"].astype(str)
    baseline[PRIMARY_GROUP] = baseline[PRIMARY_GROUP].astype(str)

    records: list[dict] = []

    for key_values, part in baseline.groupby(
        KEYS,
        sort=True,
        dropna=False,
    ):
        seed, stack, policy_kind, target = key_values
        if int(seed) not in seeds:
            continue

        # The policy itself is completely frozen: same rows, predictions,
        # thresholds, models, and routes. Only measurement labels/groups vary.
        pred = part["prediction"].to_numpy(dtype=np.int8).astype(bool)

        for label_condition, y_column in {
            "audited": "y_audited",
            "original": "y_original",
        }.items():
            y = part[y_column].to_numpy(dtype=np.int64)

            for grouping_condition, group_column in {
                "dependency_primary": PRIMARY_GROUP,
                "singleton_weak": "example_id",
            }.items():
                groups = part[group_column].astype(str).to_numpy()
                metrics = risk_metrics(
                    y,
                    pred,
                    groups,
                    confidence=confidence,
                    risk_limit=risk_limit,
                )
                records.append(
                    {
                        "layer": "fixed_policy_measurement",
                        "factor_label_condition": label_condition,
                        "factor_grouping_condition": grouping_condition,
                        "measurement_label_condition": label_condition,
                        "measurement_grouping_condition": grouping_condition,
                        "seed": int(seed),
                        "stack": str(stack),
                        "policy_kind": str(policy_kind),
                        "target_key": str(target),
                        "target_rate": (
                            math.nan
                            if target == "anchor"
                            else float(target)
                        ),
                        "policy_source": (
                            "frozen_audited_dependency_primary_pilot_predictions"
                        ),
                        "outer_rows_source": "frozen_baseline_outer_holdout",
                        **metrics,
                    }
                )

    output = pd.DataFrame(records)
    expected = len(seeds) * 15 * 4
    if len(output) != expected:
        raise RuntimeError(
            f"Fixed-policy cells expected {expected} rows; found {len(output)}."
        )
    return output


def retraining_reselection_cells(
    frame: pd.DataFrame,
    *,
    outer_ids: dict[int, set[str]],
    pilot_protocol: dict,
    confidence: float,
    risk_limit: float,
    seeds: list[int],
) -> pd.DataFrame:
    params = pilot_protocol["predictors"]["parameters"]
    maximum_fpr = float(
        pilot_protocol["decision_operating_point"]["maximum_fpr"]
    )
    stacks = pilot_protocol["monitor_stacks"]
    target_rates = [
        float(x)
        for x in pilot_protocol["routing_comparison"][
            "target_acquisition_rates"
        ]
    ]

    records: list[dict] = []

    for seed in seeds:
        outer_id_set = outer_ids[seed]
        outer = frame[
            frame["example_id"].astype(str).isin(outer_id_set)
        ].copy()
        remaining = frame[
            ~frame["example_id"].astype(str).isin(outer_id_set)
        ].copy()

        if len(outer) != len(outer_id_set):
            raise RuntimeError(
                f"Outer holdout coverage mismatch for seed={seed}."
            )

        # Fail closed against dependency leakage across the fixed outer boundary.
        outer_groups = set(outer[PRIMARY_GROUP].astype(str))
        remaining_groups = set(remaining[PRIMARY_GROUP].astype(str))
        leakage = outer_groups.intersection(remaining_groups)
        if leakage:
            raise RuntimeError(
                f"Primary dependency group leakage across fixed outer boundary "
                f"for seed={seed}: {len(leakage)} groups."
            )

        y_outer_audited = outer["y"].to_numpy(dtype=np.int64)
        outer_primary_groups = outer[PRIMARY_GROUP].astype(str).to_numpy()

        for label_condition, label_column in LABEL_COLUMNS.items():
            for grouping_condition, group_column in GROUP_COLUMNS.items():
                groups = remaining[group_column].astype(str).to_numpy()
                strata = (
                    remaining["source_dataset"].astype(str)
                    + "|"
                    + remaining[label_column].astype(int).astype(str)
                ).to_numpy()

                splitter = StratifiedGroupKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=int(seed),
                )
                assignment = np.full(len(remaining), -1, dtype=np.int64)
                for fold, (_, valid_idx) in enumerate(
                    splitter.split(
                        np.zeros((len(remaining), 1)),
                        strata,
                        groups,
                    )
                ):
                    if np.any(assignment[valid_idx] != -1):
                        raise RuntimeError(
                            "Duplicate retraining/reselection fold assignment."
                        )
                    assignment[valid_idx] = fold

                if np.any(assignment < 0):
                    raise RuntimeError(
                        "Incomplete retraining/reselection fold assignment."
                    )

                selection = remaining.loc[assignment == 0].copy()
                train = remaining.loc[assignment != 0].copy()

                if set(selection["example_id"].astype(str)).intersection(
                    set(train["example_id"].astype(str))
                ):
                    raise RuntimeError("Train/selection row overlap.")

                if grouping_condition == "dependency_primary":
                    sel_groups = set(selection[PRIMARY_GROUP].astype(str))
                    train_groups = set(train[PRIMARY_GROUP].astype(str))
                    if sel_groups.intersection(train_groups):
                        raise RuntimeError(
                            "Dependency group crosses train/selection roles."
                        )

                y_selection = selection[label_column].to_numpy(dtype=np.int64)

                for stack_name, stack_cfg in stacks.items():
                    cheap_features = [
                        str(x) for x in stack_cfg["cheap_features"]
                    ]
                    full_features = [
                        str(x) for x in stack_cfg["full_features"]
                    ]

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
                    cheap_outer = probabilities(
                        cheap_model,
                        outer,
                        cheap_features,
                    )
                    full_outer = probabilities(
                        full_model,
                        outer,
                        full_features,
                    )

                    policy_specs = [
                        ("cheap_only", None),
                        ("full_information", None),
                        *[
                            ("selective_deployable", rate)
                            for rate in target_rates
                        ],
                    ]

                    for policy_kind, target_rate in policy_specs:
                        result = evaluate_policy(
                            selection=selection,
                            outer=outer,
                            y_selection=y_selection,
                            cheap_sel=cheap_sel,
                            full_sel=full_sel,
                            cheap_outer=cheap_outer,
                            full_outer=full_outer,
                            policy_kind=policy_kind,
                            target_rate=target_rate,
                            maximum_fpr=maximum_fpr,
                        )

                        metrics = risk_metrics(
                            y_outer_audited,
                            result["prediction"],
                            outer_primary_groups,
                            confidence=confidence,
                            risk_limit=risk_limit,
                        )

                        records.append(
                            {
                                "layer": "retraining_reselection",
                                "factor_label_condition": label_condition,
                                "factor_grouping_condition": grouping_condition,
                                "measurement_label_condition": "audited",
                                "measurement_grouping_condition": (
                                    "dependency_primary"
                                ),
                                "seed": int(seed),
                                "stack": str(stack_name),
                                "policy_kind": str(policy_kind),
                                "target_key": target_key(target_rate),
                                "target_rate": (
                                    math.nan
                                    if target_rate is None
                                    else float(target_rate)
                                ),
                                "train_n": int(len(train)),
                                "selection_n": int(len(selection)),
                                "outer_n": int(len(outer)),
                                "training_label_column": label_column,
                                "training_group_column": group_column,
                                "acquisition_threshold": result[
                                    "acquisition_threshold"
                                ],
                                "decision_threshold": result[
                                    "decision_threshold"
                                ],
                                "selection_acquisition_rate": result[
                                    "selection_acquisition_rate"
                                ],
                                "outer_acquisition_rate": result[
                                    "outer_acquisition_rate"
                                ],
                                "selection_recall": result[
                                    "selection_recall"
                                ],
                                "selection_fpr": result[
                                    "selection_fpr"
                                ],
                                **metrics,
                            }
                        )

    output = pd.DataFrame(records)
    expected = len(seeds) * 15 * 4
    if len(output) != expected:
        raise RuntimeError(
            f"Retraining/reselection cells expected {expected} rows; "
            f"found {len(output)}."
        )
    return output


def full_protocol_cells(
    policy_summary: pd.DataFrame,
    *,
    seeds: list[int],
) -> pd.DataFrame:
    deployable = policy_summary[
        deployable_mask(policy_summary["deployable"])
        & policy_summary["policy_kind"].isin(POLICY_KINDS)
        & policy_summary["label_condition"].isin(list(LABEL_COLUMNS))
        & policy_summary["grouping_condition"].isin(list(GROUP_COLUMNS))
    ].copy()

    deployable["seed"] = deployable["seed"].astype(int)
    deployable = deployable[deployable["seed"].isin(seeds)].copy()
    deployable["target_key"] = deployable["target_rate"].map(target_key)

    records: list[dict] = []
    for row in deployable.itertuples(index=False):
        records.append(
            {
                "layer": "full_protocol",
                "factor_label_condition": str(row.label_condition),
                "factor_grouping_condition": str(row.grouping_condition),
                "measurement_label_condition": str(row.label_condition),
                # Pilot v1's risk gate always used primary dependency groups,
                # even when grouping_condition changed the fold assignments.
                "measurement_grouping_condition": (
                    "dependency_primary_as_implemented_in_pilot_v1"
                ),
                "seed": int(row.seed),
                "stack": str(row.stack),
                "policy_kind": str(row.policy_kind),
                "target_key": target_key(row.target_rate),
                "target_rate": (
                    math.nan
                    if pd.isna(row.target_rate)
                    else float(row.target_rate)
                ),
                "selection_n": int(row.selection_n),
                "outer_n": int(row.holdout_n),
                "acquisition_threshold": (
                    math.nan
                    if pd.isna(row.acquisition_threshold)
                    else float(row.acquisition_threshold)
                ),
                "decision_threshold": float(row.decision_threshold),
                "selection_acquisition_rate": float(
                    row.selection_acquisition_rate
                ),
                "outer_acquisition_rate": float(
                    row.holdout_acquisition_rate
                ),
                "selection_recall": float(row.selection_recall),
                "selection_fpr": float(row.selection_fpr),
                "n": int(row.holdout_n),
                "positive_n": int(row.holdout_positive_n),
                "negative_n": int(row.holdout_negative_n),
                "tp": int(row.holdout_tp),
                "fn": int(row.holdout_fn),
                "fp": int(row.holdout_fp),
                "tn": int(row.holdout_tn),
                "recall": float(row.holdout_recall),
                "fpr": float(row.holdout_fpr),
                "precision": float(row.holdout_precision),
                "row_fpr_upper95": float(row.holdout_row_fpr_upper95),
                "negative_dependency_groups": int(
                    row.holdout_negative_dependency_groups
                ),
                "failed_dependency_groups": int(
                    row.holdout_failed_dependency_groups
                ),
                "dependency_group_event_rate": float(
                    row.holdout_dependency_group_event_rate
                ),
                "dependency_group_event_upper95": float(
                    row.holdout_dependency_group_event_upper95
                ),
                "risk_gate_pass": scalar_bool(row.primary_risk_gate_pass),
            }
        )

    output = pd.DataFrame(records)
    expected = len(seeds) * 15 * 4
    if len(output) != expected:
        raise RuntimeError(
            f"Full-protocol cells expected {expected} rows; found {len(output)}."
        )
    return output


def pair_contrast(
    cells: pd.DataFrame,
    *,
    layer: str,
    contrast: str,
    left_label: str,
    left_group: str,
    right_label: str,
    right_group: str,
) -> tuple[pd.DataFrame, dict]:
    layer_cells = cells[cells["layer"].eq(layer)].copy()

    left = layer_cells[
        layer_cells["factor_label_condition"].eq(left_label)
        & layer_cells["factor_grouping_condition"].eq(left_group)
    ].copy()
    right = layer_cells[
        layer_cells["factor_label_condition"].eq(right_label)
        & layer_cells["factor_grouping_condition"].eq(right_group)
    ].copy()

    keep_metrics = [
        "risk_gate_pass",
        "recall",
        "fpr",
        "row_fpr_upper95",
        "dependency_group_event_upper95",
        "outer_acquisition_rate",
        "decision_threshold",
        "acquisition_threshold",
    ]

    left = left[KEYS + keep_metrics].rename(
        columns={name: f"left_{name}" for name in keep_metrics}
    )
    right = right[KEYS + keep_metrics].rename(
        columns={name: f"right_{name}" for name in keep_metrics}
    )

    paired = left.merge(
        right,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError(
            f"Contrast pairing mismatch for {layer}/{contrast}: "
            f"left={len(left)}, right={len(right)}, paired={len(paired)}"
        )

    left_pass = paired["left_risk_gate_pass"].astype(bool)
    right_pass = paired["right_risk_gate_pass"].astype(bool)

    paired.insert(0, "layer", layer)
    paired.insert(1, "contrast", contrast)
    paired["pass_flip"] = left_pass != right_pass
    paired["left_pass_right_fail"] = left_pass & (~right_pass)
    paired["left_fail_right_pass"] = (~left_pass) & right_pass

    for metric in [
        "recall",
        "fpr",
        "row_fpr_upper95",
        "dependency_group_event_upper95",
        "outer_acquisition_rate",
        "decision_threshold",
        "acquisition_threshold",
    ]:
        paired[f"delta_{metric}_left_minus_right"] = (
            paired[f"left_{metric}"] - paired[f"right_{metric}"]
        )

    summary = {
        "layer": layer,
        "contrast": contrast,
        "eligible_n": int(len(paired)),
        "left_pass_n": int(left_pass.sum()),
        "right_pass_n": int(right_pass.sum()),
        "pass_flip_n": int((left_pass != right_pass).sum()),
        "left_pass_right_fail_n": int(
            (left_pass & (~right_pass)).sum()
        ),
        "left_fail_right_pass_n": int(
            ((~left_pass) & right_pass).sum()
        ),
        "mean_recall_delta_left_minus_right": float(
            paired["delta_recall_left_minus_right"].mean()
        ),
        "mean_fpr_delta_left_minus_right": float(
            paired["delta_fpr_left_minus_right"].mean()
        ),
        "mean_row_fpr_upper95_delta_left_minus_right": float(
            paired[
                "delta_row_fpr_upper95_left_minus_right"
            ].mean()
        ),
        "mean_dependency_group_upper95_delta_left_minus_right": float(
            paired[
                "delta_dependency_group_event_upper95_left_minus_right"
            ].mean()
        ),
        "descriptive_seed_pooling_only": True,
    }
    return paired, summary


def contrast_outputs(
    cells: pd.DataFrame,
    *,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specifications = [
        (
            "grouping_at_audited",
            "audited",
            "singleton_weak",
            "audited",
            "dependency_primary",
        ),
        (
            "grouping_at_original",
            "original",
            "singleton_weak",
            "original",
            "dependency_primary",
        ),
        (
            "label_at_dependency",
            "original",
            "dependency_primary",
            "audited",
            "dependency_primary",
        ),
        (
            "label_at_singleton",
            "original",
            "singleton_weak",
            "audited",
            "singleton_weak",
        ),
    ]

    pair_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for layer in [
        "fixed_policy_measurement",
        "retraining_reselection",
        "full_protocol",
    ]:
        for (
            name,
            left_label,
            left_group,
            right_label,
            right_group,
        ) in specifications:
            paired, summary = pair_contrast(
                cells,
                layer=layer,
                contrast=name,
                left_label=left_label,
                left_group=left_group,
                right_label=right_label,
                right_group=right_group,
            )
            pair_frames.append(paired)
            summary_rows.append(summary)

    pairs = pd.concat(pair_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    by_seed_rows: list[dict] = []
    for (layer, contrast, seed), part in pairs.groupby(
        ["layer", "contrast", "seed"],
        sort=True,
    ):
        left_pass = part["left_risk_gate_pass"].astype(bool)
        right_pass = part["right_risk_gate_pass"].astype(bool)
        by_seed_rows.append(
            {
                "layer": layer,
                "contrast": contrast,
                "seed": int(seed),
                "eligible_n": int(len(part)),
                "left_pass_n": int(left_pass.sum()),
                "right_pass_n": int(right_pass.sum()),
                "pass_flip_n": int((left_pass != right_pass).sum()),
                "left_pass_right_fail_n": int(
                    (left_pass & (~right_pass)).sum()
                ),
                "left_fail_right_pass_n": int(
                    ((~left_pass) & right_pass).sum()
                ),
            }
        )

    by_seed = pd.DataFrame(by_seed_rows)

    expected_summary_rows = 3 * 4
    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"Expected {expected_summary_rows} contrast summary rows; "
            f"found {len(summary)}."
        )
    if not np.all(summary["eligible_n"].to_numpy(int) == len(seeds) * 15):
        raise RuntimeError("Every aggregate contrast must contain 75 policies.")
    if not np.all(by_seed["eligible_n"].to_numpy(int) == 15):
        raise RuntimeError("Every seed-level contrast must contain 15 policies.")

    return pairs, summary, by_seed


def interaction_summary(cells: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    metrics = [
        "risk_gate_pass",
        "recall",
        "fpr",
        "row_fpr_upper95",
        "dependency_group_event_upper95",
    ]

    for layer in [
        "fixed_policy_measurement",
        "retraining_reselection",
        "full_protocol",
    ]:
        part = cells[cells["layer"].eq(layer)].copy()
        part["cell"] = (
            part["factor_label_condition"].astype(str)
            + "__"
            + part["factor_grouping_condition"].astype(str)
        )

        for metric in metrics:
            tmp = part[KEYS + ["cell", metric]].copy()
            if metric == "risk_gate_pass":
                tmp[metric] = tmp[metric].astype(bool).astype(int)
            pivot = tmp.pivot(
                index=KEYS,
                columns="cell",
                values=metric,
            )

            required = [
                "audited__dependency_primary",
                "audited__singleton_weak",
                "original__dependency_primary",
                "original__singleton_weak",
            ]
            missing = [name for name in required if name not in pivot.columns]
            if missing:
                raise RuntimeError(
                    f"Missing factorial cells for {layer}/{metric}: {missing}"
                )

            did = (
                (
                    pivot["original__singleton_weak"]
                    - pivot["original__dependency_primary"]
                )
                - (
                    pivot["audited__singleton_weak"]
                    - pivot["audited__dependency_primary"]
                )
            )

            finite = did[np.isfinite(did.to_numpy(dtype=float))]
            records.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "eligible_n": int(len(did)),
                    "finite_n": int(len(finite)),
                    "nonzero_interaction_n": int(
                        np.sum(~np.isclose(
                            finite.to_numpy(dtype=float),
                            0.0,
                            atol=1e-15,
                            rtol=0.0,
                        ))
                    ),
                    "mean_difference_in_differences": (
                        float(finite.mean()) if len(finite) else math.nan
                    ),
                    "max_abs_difference_in_differences": (
                        float(np.max(np.abs(finite.to_numpy(dtype=float))))
                        if len(finite)
                        else math.nan
                    ),
                    "descriptive_only": True,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    progress("[1/8] Verifying frozen decomposition protocol and Git state...")

    if run_git(["branch", "--show-current"]) != EXPECTED_BRANCH:
        raise SystemExit(
            f"Expected branch {EXPECTED_BRANCH!r}."
        )
    if run_git(["status", "--short", "-uall"]):
        raise SystemExit(
            "Working tree must be clean before decomposition outcomes are run."
        )

    freeze_commit = run_git(["rev-parse", "HEAD"])
    parent_commit = run_git(["rev-parse", "HEAD^"])
    if parent_commit != EXPECTED_BASE_COMMIT:
        raise SystemExit(
            "Decomposition freeze commit does not descend directly from "
            "the verified provenance correction."
        )

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_decomposition_outcomes":
        raise SystemExit("Decomposition protocol is not frozen.")
    if protocol.get("base_commit") != EXPECTED_BASE_COMMIT:
        raise SystemExit("Unexpected protocol base commit.")

    for relative, expected_hash in protocol["inputs"].items():
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing frozen input: {relative}")
        if sha256(path) != expected_hash:
            raise SystemExit(f"Frozen input hash mismatch: {relative}")

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for split in PROTECTED_SPLITS:
        record = provenance["splits"][split]
        if record["fresh_confirmatory_eligible"] is not False:
            raise SystemExit(
                f"Protected legacy split unexpectedly marked fresh: {split}"
            )

    if OUT_DIR.exists():
        raise SystemExit(
            f"Output directory already exists; refusing overwrite: {OUT_DIR}"
        )

    progress("[2/8] Loading development-only inputs and validating boundaries...")

    dataset = pd.read_parquet(DEV_DATASET)
    cache = pd.read_parquet(DEV_CACHE)
    groups = pd.read_csv(GROUPS)
    predictions = pd.read_parquet(PILOT_PREDICTIONS)
    policy_summary = pd.read_csv(PILOT_SUMMARY)
    pilot_protocol = json.loads(PILOT_PROTOCOL.read_text(encoding="utf-8"))

    if len(predictions) != 403392:
        raise SystemExit(
            f"Frozen pilot prediction row count changed: {len(predictions)}"
        )
    if len(policy_summary) != 960:
        raise SystemExit(
            f"Frozen pilot policy-summary row count changed: {len(policy_summary)}"
        )

    for name, frame in [("dataset", dataset), ("cache", cache)]:
        if len(frame) != EXPECTED_ROWS:
            raise SystemExit(f"{name} row count != {EXPECTED_ROWS}")
        observed = set(frame["split"].astype(str).unique())
        if observed != AUTHORIZED_SPLITS:
            raise SystemExit(
                f"{name} does not contain exactly authorized development splits: "
                f"{sorted(observed)}"
            )
        if observed.intersection(PROTECTED_SPLITS):
            raise SystemExit(f"{name} contains protected legacy splits.")

    if len(groups) != EXPECTED_ROWS:
        raise SystemExit("Dependency-group row count mismatch.")

    required_dataset = {
        "example_id",
        "split",
        "source_dataset",
        "y",
        "y_original",
    }
    required_cache = {"example_id", *SCORE_FIELDS}
    if not required_dataset.issubset(dataset.columns):
        raise SystemExit("Development dataset schema mismatch.")
    if not required_cache.issubset(cache.columns):
        raise SystemExit("Development score cache schema mismatch.")
    if PRIMARY_GROUP not in groups.columns:
        raise SystemExit("Primary dependency grouping column missing.")

    score_frame = cache[["example_id", *SCORE_FIELDS]].copy()
    frame = dataset.merge(
        score_frame,
        on="example_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        groups[["example_id", "source_dataset", PRIMARY_GROUP]],
        on=["example_id", "source_dataset"],
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != EXPECTED_ROWS:
        raise SystemExit("Merged development frame row count mismatch.")

    frame["example_id"] = frame["example_id"].astype(str)
    frame[PRIMARY_GROUP] = frame[PRIMARY_GROUP].astype(str)
    frame["y"] = frame["y"].astype(np.int64)
    frame["y_original"] = frame["y_original"].astype(np.int64)

    for field in SCORE_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise").astype(float)
        if not np.all(np.isfinite(frame[field].to_numpy())):
            raise SystemExit(f"Non-finite score field: {field}")

    seeds = [int(x) for x in protocol["seeds"]]
    confidence = float(protocol["risk"]["confidence"])
    risk_limit = float(protocol["risk"]["maximum_risk"])

    outer_ids = common_outer_ids_from_frozen_predictions(
        predictions,
        seeds,
    )

    progress("[3/8] Computing fixed-policy 2x2 measurement cells...")
    fixed = fixed_policy_cells(
        predictions,
        confidence=confidence,
        risk_limit=risk_limit,
        seeds=seeds,
    )

    progress("[4/8] Computing fixed-outer retraining/reselection 2x2 cells...")
    retraining = retraining_reselection_cells(
        frame,
        outer_ids=outer_ids,
        pilot_protocol=pilot_protocol,
        confidence=confidence,
        risk_limit=risk_limit,
        seeds=seeds,
    )

    progress("[5/8] Loading the existing full-protocol 2x2 cells...")
    full = full_protocol_cells(
        policy_summary,
        seeds=seeds,
    )

    cells = pd.concat(
        [fixed, retraining, full],
        ignore_index=True,
        sort=False,
    )

    progress("[6/8] Computing paired contrasts and factorial interactions...")
    pairs, contrasts, contrasts_by_seed = contrast_outputs(
        cells,
        seeds=seeds,
    )
    interactions = interaction_summary(cells)

    # The full-protocol rows must exactly reproduce the already reported
    # confounded contrasts. These are input-validation checks, not new outcomes.
    full_group = contrasts[
        contrasts["layer"].eq("full_protocol")
        & contrasts["contrast"].eq("grouping_at_audited")
    ]
    full_label = contrasts[
        contrasts["layer"].eq("full_protocol")
        & contrasts["contrast"].eq("label_at_dependency")
    ]
    if len(full_group) != 1 or int(full_group.iloc[0]["pass_flip_n"]) != 19:
        raise SystemExit(
            "Full-protocol grouping contrast no longer reproduces 19/75."
        )
    if len(full_label) != 1 or int(full_label.iloc[0]["pass_flip_n"]) != 14:
        raise SystemExit(
            "Full-protocol label contrast no longer reproduces 14/75."
        )

    # Verify the two new decomposition layers each contain the same policy
    # universe as the original 75-policy descriptive contrast.
    for layer in ["fixed_policy_measurement", "retraining_reselection"]:
        for contrast in ["grouping_at_audited", "label_at_dependency"]:
            row = contrasts[
                contrasts["layer"].eq(layer)
                & contrasts["contrast"].eq(contrast)
            ]
            if len(row) != 1 or int(row.iloc[0]["eligible_n"]) != 75:
                raise SystemExit(
                    f"Unexpected decomposition universe for {layer}/{contrast}."
                )

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    progress("[7/8] Writing decomposition evidence and manifest...")
    files = {
        "factorial_cells.csv": cells,
        "paired_policy_contrasts.csv": pairs,
        "contrast_summary.csv": contrasts,
        "contrast_summary_by_seed.csv": contrasts_by_seed,
        "interaction_summary.csv": interactions,
    }
    for name, table in files.items():
        table.to_csv(OUT_DIR / name, index=False)

    def json_safe_value(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def primary_row(layer: str, contrast: str) -> dict:
        row = contrasts[
            contrasts["layer"].eq(layer)
            & contrasts["contrast"].eq(contrast)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Missing summary row {layer}/{contrast}")
        return {
            key: json_safe_value(value)
            for key, value in row.iloc[0].to_dict().items()
        }

    summary = {
        "artifact": "factorial_measurement_decomposition_v1",
        "status": "completed_development_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_commit": freeze_commit,
        "base_commit": EXPECTED_BASE_COMMIT,
        "development_rows": EXPECTED_ROWS,
        "protected_legacy_splits_used": False,
        "fresh_confirmatory_claim": False,
        "router_superiority_claim": False,
        "inferential_pooling_across_seeds": False,
        "policy_universe_per_primary_contrast": 75,
        "layers": {
            "fixed_policy_measurement": {
                "definition": (
                    "Frozen audited/dependency-primary pilot policies, rows, "
                    "predictions, thresholds, and routes; only evaluation label "
                    "and risk grouping are varied in a 2x2 measurement factorial."
                ),
                "grouping_at_audited": primary_row(
                    "fixed_policy_measurement",
                    "grouping_at_audited",
                ),
                "label_at_dependency": primary_row(
                    "fixed_policy_measurement",
                    "label_at_dependency",
                ),
            },
            "retraining_reselection": {
                "definition": (
                    "A common dependency-closed outer holdout is frozen per seed. "
                    "The remaining development data are retrained/reselected under "
                    "the 2x2 label/grouping factorial, while outer evaluation is "
                    "always audited-label and dependency-primary."
                ),
                "grouping_at_audited": primary_row(
                    "retraining_reselection",
                    "grouping_at_audited",
                ),
                "label_at_dependency": primary_row(
                    "retraining_reselection",
                    "label_at_dependency",
                ),
            },
            "full_protocol": {
                "definition": (
                    "Existing pilot-v1 condition-specific folds, training data, "
                    "models, thresholds, labels, and holdouts. Risk aggregation "
                    "remains primary dependency grouping as implemented in pilot v1."
                ),
                "grouping_at_audited": primary_row(
                    "full_protocol",
                    "grouping_at_audited",
                ),
                "label_at_dependency": primary_row(
                    "full_protocol",
                    "label_at_dependency",
                ),
            },
        },
        "full_protocol_validation": {
            "grouping_flip_n": 19,
            "grouping_eligible_n": 75,
            "label_flip_n": 14,
            "label_eligible_n": 75,
        },
        "next_step": "numerical_route_stability",
    }
    (OUT_DIR / "decomposition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_paths = sorted(
        path for path in OUT_DIR.iterdir() if path.is_file()
    )
    manifest = {
        "artifact": "factorial_measurement_decomposition_v1_manifest",
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_point": "python scripts/run_factorial_measurement_decomposition.py",
        "freeze_commit": freeze_commit,
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "inputs": {
            relative: expected_hash
            for relative, expected_hash in protocol["inputs"].items()
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in output_paths
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Rewrite manifest after adding itself is intentionally avoided; the
    # repository-level reproducibility manifest will hash this file.
    progress("[8/8] Decomposition complete.")

    fixed_group = primary_row(
        "fixed_policy_measurement",
        "grouping_at_audited",
    )
    fixed_label = primary_row(
        "fixed_policy_measurement",
        "label_at_dependency",
    )
    retrain_group = primary_row(
        "retraining_reselection",
        "grouping_at_audited",
    )
    retrain_label = primary_row(
        "retraining_reselection",
        "label_at_dependency",
    )

    print("FACTORIAL_MEASUREMENT_DECOMPOSITION=PASS")
    print(
        "fixed_policy_grouping_flips="
        f"{int(fixed_group['pass_flip_n'])}/75"
    )
    print(
        "fixed_policy_label_flips="
        f"{int(fixed_label['pass_flip_n'])}/75"
    )
    print(
        "retraining_reselection_grouping_flips="
        f"{int(retrain_group['pass_flip_n'])}/75"
    )
    print(
        "retraining_reselection_label_flips="
        f"{int(retrain_label['pass_flip_n'])}/75"
    )
    print("full_protocol_grouping_flips=19/75")
    print("full_protocol_label_flips=14/75")
    print("protected_legacy_splits_used=false")
    print("next_step=numerical_route_stability")


if __name__ == "__main__":
    main()
