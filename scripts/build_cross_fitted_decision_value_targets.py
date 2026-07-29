#!/usr/bin/env python3
"""Build development-only cross-fitted decision-value targets.

For each optional-monitor setup, this script:
1. restricts all work to the frozen development splits;
2. creates five untouched outer development folds;
3. uses four inner folds inside each outer-training partition to generate
   out-of-fold downstream probabilities and select thresholds at FPR <= 0.05;
4. refits the base and augmented downstream models on the complete
   outer-training partition;
5. predicts the untouched outer fold;
6. records realized decision-value targets:
      L(y, base decision) - L(y, augmented decision).

No final_test or held_out_shift row is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)
DATASET_PATH = (
    ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
)
CACHE_PATH = (
    ROOT / "data/processed/monitor_score_cache_v3.parquet"
)
OUTPUT_DIR = ROOT / "reports/decision_value_real_data"


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    fpr: float
    precision: float
    predicted_positive_n: int
    status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(values: pd.Series | np.ndarray) -> str:
    ordered = sorted(str(value) for value in values)
    payload = "\n".join(ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clean_identifier(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.mask(values.isin({"", "nan", "None", "<NA>"}))


def effective_group_key(frame: pd.DataFrame) -> pd.Series:
    group_id = clean_identifier(frame["group_id"])
    pair_id = clean_identifier(frame["pair_id"])
    example_id = clean_identifier(frame["example_id"])

    group_key = group_id.map(
        lambda value: f"group:{value}" if pd.notna(value) else pd.NA
    )
    pair_key = pair_id.map(
        lambda value: f"pair:{value}" if pd.notna(value) else pd.NA
    )
    example_key = example_id.map(
        lambda value: f"example:{value}" if pd.notna(value) else pd.NA
    )
    return group_key.fillna(pair_key).fillna(example_key)


def build_model(random_state: int) -> CalibratedClassifierCV:
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=3,
        ensemble=True,
        n_jobs=1,
    )


def positive_probability(
    model: CalibratedClassifierCV,
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    probability = model.predict_proba(frame[features])
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return probability[:, positive_index].astype(float)


def binary_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)

    positive = y == 1
    negative = y == 0

    tp = int(np.sum(positive & (prediction == 1)))
    fn = int(np.sum(positive & (prediction == 0)))
    fp = int(np.sum(negative & (prediction == 1)))
    tn = int(np.sum(negative & (prediction == 0)))

    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    decision_loss = (fp + fn) / len(y)

    return {
        "n": int(len(y)),
        "positive_n": int(positive.sum()),
        "negative_n": int(negative.sum()),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "decision_loss": float(decision_loss),
    }


def one_sided_clopper_pearson_upper(
    errors: int,
    trials: int,
    alpha: float = 0.05,
) -> float:
    if trials <= 0:
        return 1.0
    if errors >= trials:
        return 1.0
    return float(
        beta.ppf(
            1.0 - alpha,
            errors + 1,
            trials - errors,
        )
    )


def select_threshold_at_fpr(
    y: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> ThresholdResult:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(y) != len(scores):
        raise ValueError("y and scores must have equal length")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")

    unique = np.unique(scores)
    above_max = np.nextafter(float(unique.max()), math.inf)
    candidates = np.concatenate(([above_max], unique[::-1]))

    valid: list[tuple[float, float, float, int]] = []

    for threshold in candidates:
        prediction = (scores >= threshold).astype(int)
        metrics = binary_metrics(y, prediction)

        if metrics["fpr"] <= target_fpr + 1e-15:
            valid.append(
                (
                    float(metrics["recall"]),
                    float(threshold),
                    float(metrics["precision"]),
                    int(prediction.sum()),
                )
            )

    if not valid:
        raise RuntimeError("No threshold satisfied the FPR target")

    # Primary objective: maximum recall.
    # Tie 1: higher threshold.
    # Tie 2: higher precision.
    best = max(valid, key=lambda row: (row[0], row[1], row[2]))
    recall, threshold, precision, predicted_positive_n = best

    prediction = (scores >= threshold).astype(int)
    metrics = binary_metrics(y, prediction)

    status = (
        "zero_positive_fallback"
        if predicted_positive_n == 0
        else "selected_at_empirical_fpr_target"
    )

    return ThresholdResult(
        threshold=float(threshold),
        recall=float(recall),
        fpr=float(metrics["fpr"]),
        precision=float(precision),
        predicted_positive_n=int(predicted_positive_n),
        status=status,
    )


def inner_oof_probabilities(
    outer_train: pd.DataFrame,
    features: list[str],
    inner_folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )

    y = outer_train["y"].to_numpy(dtype=int)
    groups = outer_train["effective_group_key"].to_numpy()

    scores = np.full(len(outer_train), np.nan, dtype=float)
    fold_ids = np.full(len(outer_train), -1, dtype=int)

    for inner_fold, (fit_index, valid_index) in enumerate(
        splitter.split(
            outer_train,
            y,
            groups,
        )
    ):
        fit = outer_train.iloc[fit_index]
        valid = outer_train.iloc[valid_index]

        if set(fit["effective_group_key"]).intersection(
            set(valid["effective_group_key"])
        ):
            raise RuntimeError("Inner group leakage detected")

        model = build_model(
            random_state=random_state + 1000 + inner_fold
        )
        model.fit(
            fit[features],
            fit["y"].to_numpy(dtype=int),
        )
        scores[valid_index] = positive_probability(
            model,
            valid,
            features,
        )
        fold_ids[valid_index] = inner_fold

    if np.isnan(scores).any():
        raise RuntimeError("Missing inner out-of-fold probabilities")
    if (fold_ids < 0).any():
        raise RuntimeError("Missing inner fold assignments")

    return scores, fold_ids


def load_development_frame(
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(DATASET_PATH)
    cache = pd.read_parquet(CACHE_PATH)

    required_dataset = [
        "example_id",
        "split",
        "y",
        "group_id",
        "pair_id",
    ]
    for column in required_dataset:
        if column not in dataset:
            raise KeyError(f"Dataset missing required column: {column}")

    if not dataset["example_id"].is_unique:
        raise RuntimeError("Dataset example_id is not unique")
    if not cache["example_id"].is_unique:
        raise RuntimeError("Cache example_id is not unique")
    if set(dataset["example_id"]) != set(cache["example_id"]):
        raise RuntimeError("Dataset/cache example_id sets differ")

    joined = dataset[required_dataset].merge(
        cache,
        on="example_id",
        how="inner",
        suffixes=("_dataset", "_cache"),
        validate="one_to_one",
    )

    for column in ["split", "y"]:
        left = joined[f"{column}_dataset"]
        right = joined[f"{column}_cache"]
        if not left.astype(str).equals(right.astype(str)):
            raise RuntimeError(f"Dataset/cache mismatch for {column}")

    joined = joined.drop(
        columns=["split_cache", "y_cache"]
    ).rename(
        columns={
            "split_dataset": "split",
            "y_dataset": "y",
        }
    )
    joined["effective_group_key"] = effective_group_key(joined)

    development_splits = set(
        protocol["scope"]["development_splits"]
    )
    excluded_splits = set(protocol["scope"]["excluded_splits"])

    development = joined[
        joined["split"].isin(development_splits)
    ].copy()
    excluded = joined[
        joined["split"].isin(excluded_splits)
    ].copy()

    if len(development) != protocol["scope"][
        "expected_development_rows"
    ]:
        raise RuntimeError("Unexpected development row count")
    if set(excluded["split"]) != excluded_splits:
        raise RuntimeError("Excluded split boundary mismatch")
    if set(development["example_id"]).intersection(
        set(excluded["example_id"])
    ):
        raise RuntimeError("Development/excluded example overlap")

    return development.reset_index(drop=True), excluded.reset_index(
        drop=True
    )


def build_cross_fitted_targets(
    protocol: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    development, excluded = load_development_frame(protocol)

    outer_folds = int(protocol["cross_fitting"]["outer_folds"])
    inner_folds = int(protocol["cross_fitting"]["inner_folds"])
    random_state = int(
        protocol["cross_fitting"]["random_state"]
    )
    target_fpr = float(
        protocol["operating_risk"]["target"]
    )

    y_all = development["y"].to_numpy(dtype=int)
    groups_all = development["effective_group_key"].to_numpy()

    outer_splitter = StratifiedGroupKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_state,
    )

    target_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[dict[str, Any]] = []

    for outer_fold, (train_index, test_index) in enumerate(
        outer_splitter.split(
            development,
            y_all,
            groups_all,
        )
    ):
        outer_train = development.iloc[train_index].copy()
        outer_test = development.iloc[test_index].copy()

        train_groups = set(outer_train["effective_group_key"])
        test_groups = set(outer_test["effective_group_key"])
        if train_groups.intersection(test_groups):
            raise RuntimeError("Outer group leakage detected")

        for example_id in outer_test["example_id"]:
            fold_assignment_rows.append(
                {
                    "example_id": example_id,
                    "outer_fold": outer_fold,
                }
            )

        for setup_number, setup in enumerate(
            protocol["optional_monitor_setups"]
        ):
            setup_id = setup["setup_id"]
            base_features = list(
                setup["base_features"]
            )
            augmented_features = list(
                setup["augmented_features"]
            )

            for feature in base_features + augmented_features:
                if feature not in development:
                    raise KeyError(
                        f"Missing feature {feature} for setup {setup_id}"
                    )

            setup_seed = (
                random_state
                + outer_fold * 100
                + setup_number * 10_000
            )

            base_inner_score, base_inner_fold = (
                inner_oof_probabilities(
                    outer_train=outer_train,
                    features=base_features,
                    inner_folds=inner_folds,
                    random_state=setup_seed + 1,
                )
            )
            augmented_inner_score, augmented_inner_fold = (
                inner_oof_probabilities(
                    outer_train=outer_train,
                    features=augmented_features,
                    inner_folds=inner_folds,
                    random_state=setup_seed + 1,
                )
            )

            if not np.array_equal(
                base_inner_fold,
                augmented_inner_fold,
            ):
                raise RuntimeError(
                    "Base and augmented models used different "
                    "inner fold assignments"
                )

            base_threshold = select_threshold_at_fpr(
                y=outer_train["y"].to_numpy(dtype=int),
                scores=base_inner_score,
                target_fpr=target_fpr,
            )
            augmented_threshold = select_threshold_at_fpr(
                y=outer_train["y"].to_numpy(dtype=int),
                scores=augmented_inner_score,
                target_fpr=target_fpr,
            )

            base_model = build_model(random_state=setup_seed + 3)
            augmented_model = build_model(
                random_state=setup_seed + 4
            )

            base_model.fit(
                outer_train[base_features],
                outer_train["y"].to_numpy(dtype=int),
            )
            augmented_model.fit(
                outer_train[augmented_features],
                outer_train["y"].to_numpy(dtype=int),
            )

            base_score = positive_probability(
                base_model,
                outer_test,
                base_features,
            )
            augmented_score = positive_probability(
                augmented_model,
                outer_test,
                augmented_features,
            )

            base_prediction = (
                base_score >= base_threshold.threshold
            ).astype(int)
            augmented_prediction = (
                augmented_score >= augmented_threshold.threshold
            ).astype(int)
            y_test = outer_test["y"].to_numpy(dtype=int)

            base_loss = (
                base_prediction != y_test
            ).astype(int)
            augmented_loss = (
                augmented_prediction != y_test
            ).astype(int)
            realized_value = base_loss - augmented_loss

            false_positive_change = (
                ((base_prediction == 1) & (y_test == 0)).astype(int)
                - (
                    (augmented_prediction == 1)
                    & (y_test == 0)
                ).astype(int)
            )
            false_negative_change = (
                ((base_prediction == 0) & (y_test == 1)).astype(int)
                - (
                    (augmented_prediction == 0)
                    & (y_test == 1)
                ).astype(int)
            )

            optional_feature = next(
                feature
                for feature in augmented_features
                if feature not in base_features
            )

            target_frame = pd.DataFrame(
                {
                    "example_id": outer_test[
                        "example_id"
                    ].to_numpy(),
                    "setup_id": setup_id,
                    "optional_monitor": setup[
                        "optional_monitor"
                    ],
                    "outer_fold": outer_fold,
                    "y": y_test,
                    "base_score": base_score,
                    "augmented_score": augmented_score,
                    "base_threshold": (
                        base_threshold.threshold
                    ),
                    "augmented_threshold": (
                        augmented_threshold.threshold
                    ),
                    "base_prediction": base_prediction,
                    "augmented_prediction": augmented_prediction,
                    "base_loss": base_loss,
                    "augmented_loss": augmented_loss,
                    "realized_decision_value": realized_value,
                    "decision_changed": (
                        base_prediction
                        != augmented_prediction
                    ).astype(int),
                    "false_positive_reduction": (
                        false_positive_change
                    ),
                    "false_negative_reduction": (
                        false_negative_change
                    ),
                    "base_uncertainty": np.minimum(
                        base_score,
                        1.0 - base_score,
                    ),
                    "optional_monitor_score": outer_test[
                        optional_feature
                    ].to_numpy(dtype=float),
                    "outer_train_n": len(outer_train),
                    "outer_test_n": len(outer_test),
                    "outer_train_id_sha256": sha256_ids(
                        outer_train["example_id"]
                    ),
                    "outer_test_id_sha256": sha256_ids(
                        outer_test["example_id"]
                    ),
                }
            )
            target_rows.append(target_frame)

            base_metrics = binary_metrics(
                y_test,
                base_prediction,
            )
            augmented_metrics = binary_metrics(
                y_test,
                augmented_prediction,
            )

            fold_rows.append(
                {
                    "setup_id": setup_id,
                    "optional_monitor": setup[
                        "optional_monitor"
                    ],
                    "outer_fold": outer_fold,
                    "outer_train_n": len(outer_train),
                    "outer_test_n": len(outer_test),
                    "outer_train_positive_n": int(
                        outer_train["y"].sum()
                    ),
                    "outer_test_positive_n": int(
                        outer_test["y"].sum()
                    ),
                    "base_features": json.dumps(base_features),
                    "augmented_features": json.dumps(
                        augmented_features
                    ),
                    "base_threshold": (
                        base_threshold.threshold
                    ),
                    "base_inner_oof_recall": (
                        base_threshold.recall
                    ),
                    "base_inner_oof_fpr": base_threshold.fpr,
                    "base_inner_oof_precision": (
                        base_threshold.precision
                    ),
                    "base_threshold_status": (
                        base_threshold.status
                    ),
                    "augmented_threshold": (
                        augmented_threshold.threshold
                    ),
                    "augmented_inner_oof_recall": (
                        augmented_threshold.recall
                    ),
                    "augmented_inner_oof_fpr": (
                        augmented_threshold.fpr
                    ),
                    "augmented_inner_oof_precision": (
                        augmented_threshold.precision
                    ),
                    "augmented_threshold_status": (
                        augmented_threshold.status
                    ),
                    "base_outer_recall": base_metrics["recall"],
                    "base_outer_fpr": base_metrics["fpr"],
                    "base_outer_fpr_upper95": (
                        one_sided_clopper_pearson_upper(
                            int(base_metrics["fp"]),
                            int(base_metrics["negative_n"]),
                        )
                    ),
                    "base_outer_precision": (
                        base_metrics["precision"]
                    ),
                    "base_outer_decision_loss": (
                        base_metrics["decision_loss"]
                    ),
                    "augmented_outer_recall": (
                        augmented_metrics["recall"]
                    ),
                    "augmented_outer_fpr": (
                        augmented_metrics["fpr"]
                    ),
                    "augmented_outer_fpr_upper95": (
                        one_sided_clopper_pearson_upper(
                            int(augmented_metrics["fp"]),
                            int(
                                augmented_metrics["negative_n"]
                            ),
                        )
                    ),
                    "augmented_outer_precision": (
                        augmented_metrics["precision"]
                    ),
                    "augmented_outer_decision_loss": (
                        augmented_metrics["decision_loss"]
                    ),
                    "mean_realized_decision_value": float(
                        realized_value.mean()
                    ),
                    "positive_value_rate": float(
                        np.mean(realized_value == 1)
                    ),
                    "zero_value_rate": float(
                        np.mean(realized_value == 0)
                    ),
                    "negative_value_rate": float(
                        np.mean(realized_value == -1)
                    ),
                    "decision_change_rate": float(
                        np.mean(
                            base_prediction
                            != augmented_prediction
                        )
                    ),
                    "outer_train_id_sha256": sha256_ids(
                        outer_train["example_id"]
                    ),
                    "outer_test_id_sha256": sha256_ids(
                        outer_test["example_id"]
                    ),
                    "base_inner_fold_assignment_sha256": (
                        sha256_ids(
                            pd.Series(
                                [
                                    f"{example_id}:{fold_id}"
                                    for example_id, fold_id in zip(
                                        outer_train[
                                            "example_id"
                                        ],
                                        base_inner_fold,
                                    )
                                ]
                            )
                        )
                    ),
                    "augmented_inner_fold_assignment_sha256": (
                        sha256_ids(
                            pd.Series(
                                [
                                    f"{example_id}:{fold_id}"
                                    for example_id, fold_id in zip(
                                        outer_train[
                                            "example_id"
                                        ],
                                        augmented_inner_fold,
                                    )
                                ]
                            )
                        )
                    ),
                }
            )

            print(
                f"completed setup={setup_id} "
                f"outer_fold={outer_fold + 1}/{outer_folds}",
                flush=True,
            )

    targets = pd.concat(target_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_assignments = (
        pd.DataFrame(fold_assignment_rows)
        .drop_duplicates()
        .sort_values("example_id")
        .reset_index(drop=True)
    )

    expected_rows = (
        len(development)
        * len(protocol["optional_monitor_setups"])
    )
    if len(targets) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} target rows, got {len(targets)}"
        )

    duplicated = targets.duplicated(
        ["example_id", "setup_id"]
    )
    if duplicated.any():
        raise RuntimeError(
            "Duplicate example/setup target rows detected"
        )

    if set(targets["example_id"]) != set(
        development["example_id"]
    ):
        raise RuntimeError(
            "Target example IDs do not match development IDs"
        )

    if set(targets["example_id"]).intersection(
        set(excluded["example_id"])
    ):
        raise RuntimeError(
            "Excluded example entered the target artifact"
        )

    if not set(
        targets["realized_decision_value"].unique()
    ).issubset({-1, 0, 1}):
        raise RuntimeError(
            "Realized decision value left {-1,0,1}"
        )

    if not np.array_equal(
        targets["realized_decision_value"].to_numpy(),
        (
            targets["base_loss"]
            - targets["augmented_loss"]
        ).to_numpy(),
    ):
        raise RuntimeError(
            "Decision-value identity check failed"
        )

    setup_summary_rows: list[dict[str, Any]] = []
    for setup_id, group in targets.groupby(
        "setup_id",
        sort=True,
        observed=True,
    ):
        y = group["y"].to_numpy(dtype=int)
        base_prediction = group[
            "base_prediction"
        ].to_numpy(dtype=int)
        augmented_prediction = group[
            "augmented_prediction"
        ].to_numpy(dtype=int)

        base_metrics = binary_metrics(y, base_prediction)
        augmented_metrics = binary_metrics(
            y,
            augmented_prediction,
        )
        value = group[
            "realized_decision_value"
        ].to_numpy(dtype=int)

        setup_summary_rows.append(
            {
                "setup_id": setup_id,
                "n": len(group),
                "positive_n": int(np.sum(y == 1)),
                "negative_n": int(np.sum(y == 0)),
                "base_recall": base_metrics["recall"],
                "base_fpr": base_metrics["fpr"],
                "base_fpr_upper95": (
                    one_sided_clopper_pearson_upper(
                        int(base_metrics["fp"]),
                        int(base_metrics["negative_n"]),
                    )
                ),
                "base_precision": base_metrics["precision"],
                "base_decision_loss": (
                    base_metrics["decision_loss"]
                ),
                "augmented_recall": (
                    augmented_metrics["recall"]
                ),
                "augmented_fpr": augmented_metrics["fpr"],
                "augmented_fpr_upper95": (
                    one_sided_clopper_pearson_upper(
                        int(augmented_metrics["fp"]),
                        int(
                            augmented_metrics["negative_n"]
                        ),
                    )
                ),
                "augmented_precision": (
                    augmented_metrics["precision"]
                ),
                "augmented_decision_loss": (
                    augmented_metrics["decision_loss"]
                ),
                "decision_loss_reduction": (
                    base_metrics["decision_loss"]
                    - augmented_metrics["decision_loss"]
                ),
                "mean_realized_decision_value": float(
                    value.mean()
                ),
                "positive_value_n": int(np.sum(value == 1)),
                "positive_value_rate": float(
                    np.mean(value == 1)
                ),
                "zero_value_n": int(np.sum(value == 0)),
                "zero_value_rate": float(
                    np.mean(value == 0)
                ),
                "negative_value_n": int(
                    np.sum(value == -1)
                ),
                "negative_value_rate": float(
                    np.mean(value == -1)
                ),
                "decision_change_rate": float(
                    group["decision_changed"].mean()
                ),
                "false_positive_reduction_sum": int(
                    group[
                        "false_positive_reduction"
                    ].sum()
                ),
                "false_negative_reduction_sum": int(
                    group[
                        "false_negative_reduction"
                    ].sum()
                ),
            }
        )

    setup_summary = pd.DataFrame(setup_summary_rows)

    manifest = {
        "artifact": (
            "cross_fitted_decision_value_targets_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_cross_fitted_targets_completed",
        "protocol_path": str(
            PROTOCOL_PATH.relative_to(ROOT)
        ),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "cache_path": str(CACHE_PATH.relative_to(ROOT)),
        "cache_sha256": sha256_file(CACHE_PATH),
        "development_splits": sorted(
            protocol["scope"]["development_splits"]
        ),
        "excluded_splits": sorted(
            protocol["scope"]["excluded_splits"]
        ),
        "development_rows": len(development),
        "excluded_rows": len(excluded),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "splitter": (
            protocol["cross_fitting"]["splitter"]
        ),
        "random_state": random_state,
        "target_fpr": target_fpr,
        "model": {
            "family": (
                "StandardScaler + LogisticRegression + "
                "sigmoid CalibratedClassifierCV"
            ),
            "calibration_cv": 3,
            "same_family_for_base_and_augmented": True,
        },
        "setup_ids": sorted(
            setup_summary["setup_id"].tolist()
        ),
        "target_definition": (
            "zero_one_loss(base decision) - "
            "zero_one_loss(augmented decision)"
        ),
        "target_values": [-1, 0, 1],
        "threshold_selection": (
            "inner-OOF maximize recall subject to empirical "
            "FPR <= 0.05; ties prefer higher threshold then "
            "higher precision"
        ),
        "excluded_rows_used": False,
        "final_test_used": False,
        "held_out_shift_used": False,
        "outputs": {},
    }

    return targets, fold_metrics, fold_assignments, setup_summary, manifest


def write_summary(
    setup_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Cross-Fitted Decision-Value Targets",
        "",
        "This artifact uses only the frozen development pool. "
        "`final_test` and `held_out_shift` were not used.",
        "",
        "For each optional monitor, the realized target is:",
        "",
        "`base zero-one loss - augmented zero-one loss`.",
        "",
        "Positive values indicate that acquiring the optional "
        "monitor corrected the downstream decision. Negative "
        "values indicate that it worsened the decision.",
        "",
        "## Development-only pooled diagnostics",
        "",
    ]

    for row in setup_summary.itertuples(index=False):
        lines.extend(
            [
                f"### {row.setup_id}",
                "",
                f"- Rows: {row.n}",
                f"- Base recall: {row.base_recall:.6f}",
                f"- Base FPR: {row.base_fpr:.6f}",
                (
                    "- Base one-sided 95% FPR upper bound: "
                    f"{row.base_fpr_upper95:.6f}"
                ),
                (
                    "- Base decision loss: "
                    f"{row.base_decision_loss:.6f}"
                ),
                (
                    "- Augmented recall: "
                    f"{row.augmented_recall:.6f}"
                ),
                (
                    "- Augmented FPR: "
                    f"{row.augmented_fpr:.6f}"
                ),
                (
                    "- Augmented one-sided 95% FPR upper bound: "
                    f"{row.augmented_fpr_upper95:.6f}"
                ),
                (
                    "- Augmented decision loss: "
                    f"{row.augmented_decision_loss:.6f}"
                ),
                (
                    "- Decision-loss reduction: "
                    f"{row.decision_loss_reduction:.6f}"
                ),
                (
                    "- Positive-value examples: "
                    f"{row.positive_value_n} "
                    f"({row.positive_value_rate:.6f})"
                ),
                (
                    "- Negative-value examples: "
                    f"{row.negative_value_n} "
                    f"({row.negative_value_rate:.6f})"
                ),
                (
                    "- Decision-change rate: "
                    f"{row.decision_change_rate:.6f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "These are target-construction diagnostics only. "
            "They do not yet show that decision value is "
            "predictable from legitimate pre-acquisition features, "
            "and they do not pass the professor's milestone by "
            "themselves.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROTOCOL_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    args = parser.parse_args()

    protocol = json.loads(
        args.protocol.read_text(encoding="utf-8")
    )

    (
        targets,
        fold_metrics,
        fold_assignments,
        setup_summary,
        manifest,
    ) = build_cross_fitted_targets(protocol)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets_path = (
        args.output_dir
        / "cross_fitted_decision_value_targets.parquet"
    )
    targets_csv_path = (
        args.output_dir
        / "cross_fitted_decision_value_targets.csv"
    )
    fold_metrics_path = (
        args.output_dir
        / "cross_fitted_target_fold_metrics.csv"
    )
    fold_assignments_path = (
        args.output_dir
        / "development_outer_fold_assignments.csv"
    )
    setup_summary_path = (
        args.output_dir
        / "cross_fitted_target_setup_summary.csv"
    )
    summary_path = (
        args.output_dir
        / "cross_fitted_target_summary.md"
    )
    manifest_path = (
        args.output_dir
        / "cross_fitted_target_manifest.json"
    )

    targets.to_parquet(targets_path, index=False)
    targets.to_csv(targets_csv_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    fold_assignments.to_csv(
        fold_assignments_path,
        index=False,
    )
    setup_summary.to_csv(setup_summary_path, index=False)
    write_summary(setup_summary, summary_path)

    for path in [
        targets_path,
        targets_csv_path,
        fold_metrics_path,
        fold_assignments_path,
        setup_summary_path,
        summary_path,
    ]:
        manifest["outputs"][
            str(path.relative_to(ROOT))
        ] = sha256_file(path)

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== CROSS-FITTED TARGET SUMMARY ===")
    print(setup_summary.to_string(index=False))

    print()
    print("=== FOLD METRICS ===")
    print(
        fold_metrics[
            [
                "setup_id",
                "outer_fold",
                "base_inner_oof_fpr",
                "augmented_inner_oof_fpr",
                "base_outer_fpr",
                "augmented_outer_fpr",
                "base_outer_recall",
                "augmented_outer_recall",
                "positive_value_rate",
                "negative_value_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("targets:", targets_path.relative_to(ROOT))
    print("manifest:", manifest_path.relative_to(ROOT))
    print("cross-fitted development targets completed")


if __name__ == "__main__":
    main()
