#!/usr/bin/env python3
"""Build repeated-grouped v2 development candidate predictions.

This stage performs development-only model fitting and outer-fold prediction.
It does not calibrate final acquisition thresholds, perform confirmatory
evaluation, or substitute cached latency for measured end-to-end policy cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from monitor_fusion.evaluation.classifier_selection import (
    build_classifier,
    candidates_by_family as classifier_candidates_by_family,
    current_error_candidates_from_protocol,
    direct_fusion_candidates_from_protocol,
    positive_class_probability,
    select_inner_current_error_candidate,
    select_inner_direct_fusion_candidate,
)
from monitor_fusion.evaluation.data_boundary import load_protocol
from monitor_fusion.evaluation.signed_value import (
    cost_aware_signed_value_score,
)
from monitor_fusion.evaluation.signed_value_models import (
    build_signed_value_regressor,
    candidates_by_family as signed_value_candidates_by_family,
)
from monitor_fusion.evaluation.signed_value_selection import (
    select_inner_signed_value_candidate,
    signed_value_candidate_identifier,
)
from monitor_fusion.policies.exact_cost import sha256_uniform


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROTOCOL_SHA256 = (
    "96a0bfdf1a0954d9313ecd7a2ae1272a"
    "07f0df4ba6197eede2f0e0afd9f1c1c7"
)

EXPECTED_SCORE_CACHE_SHA256 = (
    "e567937e5ea8f3c024cfb719e1774a3a"
    "52d9d31e163b6a5deada57b23185d0c3"
)

EXPECTED_DEVELOPMENT_DATA_SHA256 = (
    "f752fe74c7d3cc254ce7864382defeb4"
    "45982438f14195c81823641132d0b336"
)

EXPECTED_TIMING_TARGET_SHA256 = (
    "db112e20c4343b186643958e2fcab11a"
    "1bf47d9e7359538fffb55419c6b7a8d3"
)

EXPECTED_EMBEDDING_SHA256 = (
    "0ad00a173e44a06d1a808fe6d835f550"
    "290e948d7e780307343b2ecc9bdb9959"
)

EXPECTED_ROWS = 1687
EXPECTED_SPLIT_COUNTS = {
    "policy_train": 844,
    "policy_selection": 421,
    "calibration": 422,
}

BASE_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
]

AUGMENTED_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]

NUMERIC_FEATURES = [
    "rule_score",
    "rule_weighted_sum",
    "rule_match_count",
    "rule_latency_ms",
    "compact_unsafe_score",
    "compact_input_tokens",
    "compact_output_tokens",
    "compact_latency_ms",
    "abs_rule_compact_difference",
    "rule_compact_product",
    "rule_compact_mean",
    "rule_compact_max",
    "rule_compact_min",
    "prompt_char_count",
    "response_char_count",
    "prompt_whitespace_token_count",
    "response_whitespace_token_count",
]

SIGNED_SLUGS = {
    "Ridge": "ridge",
    "HistGradientBoostingRegressor": "hgbr",
    "RandomForestRegressor": "rfr",
}

CLASSIFIER_SLUGS = {
    "LogisticRegression": "lr",
    "HistGradientBoostingClassifier": "hgbc",
    "RandomForestClassifier": "rfc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def exact_sha_file(
    root: Path,
    expected_sha256: str,
) -> Path:
    matches: list[Path] = []

    if not root.exists():
        raise FileNotFoundError(root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if sha256_file(path) == expected_sha256:
            matches.append(path)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file with SHA256 "
            f"{expected_sha256} under {root}; found {len(matches)}"
        )

    return matches[0]


def require_unique_ids(
    frame: pd.DataFrame,
    name: str,
) -> None:
    if "example_id" not in frame:
        raise RuntimeError(
            f"{name} has no example_id column"
        )

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{name} row count {len(frame)} != {EXPECTED_ROWS}"
        )

    if frame["example_id"].isna().any():
        raise RuntimeError(
            f"{name} contains missing example_id"
        )

    if frame["example_id"].astype(str).duplicated().any():
        raise RuntimeError(
            f"{name} contains duplicate example_id"
        )


def align_to_ids(
    frame: pd.DataFrame,
    ids: pd.Series,
    name: str,
) -> pd.DataFrame:
    require_unique_ids(frame, name)

    keyed = frame.copy()
    keyed["example_id"] = keyed["example_id"].astype(str)
    keyed = keyed.set_index("example_id", drop=False)

    expected = ids.astype(str).tolist()

    if set(keyed.index) != set(expected):
        raise RuntimeError(
            f"{name} example_id set differs from development cache"
        )

    return keyed.loc[expected].reset_index(drop=True)


def nonempty_identifier(value: Any) -> bool:
    if value is None:
        return False

    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass

    return bool(str(value).strip())


def effective_group_for_row(
    row: pd.Series,
    priority: list[str],
) -> str:
    for key in priority:
        if key in row and nonempty_identifier(row[key]):
            return f"{key}:{str(row[key]).strip()}"

    raise RuntimeError(
        f"No effective group identifier for {row['example_id']}"
    )


def finite_numeric(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(
            f"Required feature missing: {column}"
        )

    values = pd.to_numeric(
        frame[column],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Non-finite values in {column}"
        )

    return values


@dataclass
class DevelopmentData:
    frame: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    numeric: np.ndarray
    embeddings: np.ndarray
    direct: np.ndarray
    optional_latency_ms: np.ndarray
    source_paths: dict[str, Path]


def load_development(
    protocol: dict[str, Any],
) -> DevelopmentData:
    development_root = (
        ROOT
        / "data/processed/v2_development_view"
    )

    score_path = exact_sha_file(
        development_root,
        EXPECTED_SCORE_CACHE_SHA256,
    )

    development_path = exact_sha_file(
        development_root,
        EXPECTED_DEVELOPMENT_DATA_SHA256,
    )

    timing_path = exact_sha_file(
        ROOT
        / "data/processed/"
        "v2_development_optional_monitor_timing/artifact",
        EXPECTED_TIMING_TARGET_SHA256,
    )

    embedding_path = (
        ROOT
        / "reports/decision_value_real_data/"
        "frozen_prompt_response_embeddings.parquet"
    )

    if (
        not embedding_path.is_file()
        or sha256_file(embedding_path)
        != EXPECTED_EMBEDDING_SHA256
    ):
        raise RuntimeError(
            "Frozen development embedding artifact SHA256 mismatch"
        )

    score = pd.read_parquet(score_path)
    development = pd.read_parquet(
        development_path
    )
    timing = pd.read_parquet(timing_path)
    embeddings = pd.read_parquet(
        embedding_path
    )

    require_unique_ids(
        score,
        "development score cache",
    )

    score = score.copy()
    score["example_id"] = (
        score["example_id"].astype(str)
    )

    ids = score["example_id"]

    development = align_to_ids(
        development,
        ids,
        "development label-audited view",
    )

    timing = align_to_ids(
        timing,
        ids,
        "controlled optional-monitor timing target",
    )

    embeddings = align_to_ids(
        embeddings,
        ids,
        "frozen embedding artifact",
    )

    frame = score.copy()

    for column in (
        "prompt",
        "response",
        "split",
        "group_id",
        "pair_id",
    ):
        if column not in frame:
            if column not in development:
                if column in (
                    "group_id",
                    "pair_id",
                ):
                    continue
                raise RuntimeError(
                    f"Required development column missing: {column}"
                )
            frame[column] = development[column]
            continue

        if column in development:
            left = frame[column]
            right = development[column]

            fill = left.isna()
            if left.dtype == object:
                fill = fill | left.astype(
                    str
                ).str.strip().eq("")

            frame.loc[fill, column] = (
                right.loc[fill]
            )

    if "y" not in frame:
        raise RuntimeError(
            "Frozen score cache has no y column"
        )

    y_series = pd.to_numeric(
        frame["y"],
        errors="raise",
    )

    if y_series.isna().any():
        raise RuntimeError(
            "Frozen stratification target y is incomplete"
        )

    y = y_series.to_numpy(dtype=np.int64)

    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError(
            f"Unexpected y values: {np.unique(y)!r}"
        )

    split_counts = {
        str(k): int(v)
        for k, v in (
            frame["split"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            f"Development split counts differ: {split_counts!r}"
        )

    priority = list(
        protocol[
            "development_resampling"
        ]["group_key_priority"]
    )

    groups = frame.apply(
        effective_group_for_row,
        axis=1,
        priority=priority,
    ).to_numpy(dtype=object)

    if len(set(groups.tolist())) != EXPECTED_ROWS:
        raise RuntimeError(
            "Current development view does not have "
            "the previously verified 1687 effective groups"
        )

    prompt = frame["prompt"].fillna("").astype(str)
    response = (
        frame["response"].fillna("").astype(str)
    )

    feature_values: dict[str, np.ndarray] = {}

    for column in (
        "rule_score",
        "rule_weighted_sum",
        "rule_match_count",
        "rule_latency_ms",
        "compact_unsafe_score",
        "compact_input_tokens",
        "compact_output_tokens",
        "compact_latency_ms",
    ):
        feature_values[column] = finite_numeric(
            frame,
            column,
        )

    rule = feature_values["rule_score"]
    compact = feature_values[
        "compact_unsafe_score"
    ]

    feature_values[
        "abs_rule_compact_difference"
    ] = np.abs(rule - compact)

    feature_values[
        "rule_compact_product"
    ] = rule * compact

    feature_values[
        "rule_compact_mean"
    ] = (rule + compact) / 2.0

    feature_values[
        "rule_compact_max"
    ] = np.maximum(rule, compact)

    feature_values[
        "rule_compact_min"
    ] = np.minimum(rule, compact)

    feature_values[
        "prompt_char_count"
    ] = prompt.str.len().to_numpy(
        dtype=np.float64
    )

    feature_values[
        "response_char_count"
    ] = response.str.len().to_numpy(
        dtype=np.float64
    )

    feature_values[
        "prompt_whitespace_token_count"
    ] = prompt.map(
        lambda text: len(text.split())
    ).to_numpy(dtype=np.float64)

    feature_values[
        "response_whitespace_token_count"
    ] = response.map(
        lambda text: len(text.split())
    ).to_numpy(dtype=np.float64)

    frozen_order = (
        protocol[
            "model_input_contract"
        ][
            "exact_pre_acquisition_feature_matrix"
        ][
            "numeric_feature_order"
        ]
    )

    if list(frozen_order) != NUMERIC_FEATURES:
        raise RuntimeError(
            "Frozen numeric feature order differs from runner"
        )

    numeric = np.column_stack(
        [
            feature_values[column]
            for column in NUMERIC_FEATURES
        ]
    )

    if numeric.shape != (
        EXPECTED_ROWS,
        17,
    ):
        raise RuntimeError(
            f"Unexpected numeric matrix shape: {numeric.shape}"
        )

    if not np.isfinite(numeric).all():
        raise RuntimeError(
            "Pre-acquisition numeric matrix contains non-finite values"
        )

    embedding_columns = []

    for column in embeddings.columns:
        if not column.startswith(
            "embedding_"
        ):
            continue
        suffix = column.split(
            "_",
            1,
        )[1]
        if suffix.isdigit():
            embedding_columns.append(
                (
                    int(suffix),
                    column,
                )
            )

    embedding_columns.sort(
        key=lambda item: item[0]
    )

    names = [
        column
        for _, column in embedding_columns
    ]

    if len(names) != 384:
        raise RuntimeError(
            f"Expected 384 embedding columns; found {len(names)}"
        )

    if [
        index
        for index, _ in embedding_columns
    ] != list(range(384)):
        raise RuntimeError(
            "Embedding numeric suffixes are not exactly 0..383"
        )

    embedding_matrix = embeddings[
        names
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(
        embedding_matrix
    ).all():
        raise RuntimeError(
            "Frozen embeddings contain non-finite values"
        )

    direct = np.column_stack(
        [
            finite_numeric(
                frame,
                column,
            )
            for column in AUGMENTED_FEATURES
        ]
    )

    if (
        "optional_monitor_stage_latency_ms"
        not in timing
    ):
        raise RuntimeError(
            "Controlled timing target column missing"
        )

    optional_latency = pd.to_numeric(
        timing[
            "optional_monitor_stage_latency_ms"
        ],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if (
        not np.isfinite(
            optional_latency
        ).all()
        or bool(
            np.any(
                optional_latency < 0.0
            )
        )
    ):
        raise RuntimeError(
            "Invalid optional-monitor latency target"
        )

    return DevelopmentData(
        frame=frame,
        y=y,
        groups=groups,
        numeric=numeric,
        embeddings=embedding_matrix,
        direct=direct,
        optional_latency_ms=optional_latency,
        source_paths={
            "score_cache": score_path,
            "development_view": development_path,
            "timing_target": timing_path,
            "embeddings": embedding_path,
        },
    )


def build_downstream_model(
    random_state: int,
) -> CalibratedClassifierCV:
    base = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=int(
                        random_state
                    ),
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


def probability_one(
    model: Any,
    matrix: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(
        model.predict_proba(
            matrix
        ),
        dtype=np.float64,
    )

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        raise RuntimeError(
            "Classifier has no positive class"
        )

    result = probability[
        :,
        classes.index(1),
    ]

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Classifier produced non-finite probabilities"
        )

    return result


def binary_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    positive = y == 1
    negative = y == 0

    tp = int(
        np.sum(
            positive
            & (prediction == 1)
        )
    )

    fn = int(
        np.sum(
            positive
            & (prediction == 0)
        )
    )

    fp = int(
        np.sum(
            negative
            & (prediction == 1)
        )
    )

    tn = int(
        np.sum(
            negative
            & (prediction == 0)
        )
    )

    return {
        "recall":
            tp / (tp + fn),
        "fpr":
            fp / (fp + tn),
        "precision":
            tp / (tp + fp)
            if tp + fp
            else 0.0,
    }


def historical_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    maximum_fpr: float = 0.05,
) -> float:
    unique = np.unique(
        scores
    )

    above = np.nextafter(
        float(
            unique.max()
        ),
        math.inf,
    )

    candidates = np.concatenate(
        (
            [above],
            unique[::-1],
        )
    )

    feasible: list[
        tuple[
            float,
            float,
            float,
        ]
    ] = []

    for threshold in candidates:
        prediction = (
            scores
            >= threshold
        ).astype(np.int64)

        metric = binary_metrics(
            y,
            prediction,
        )

        if (
            metric["fpr"]
            <= maximum_fpr
            + 1e-15
        ):
            feasible.append(
                (
                    metric["recall"],
                    float(threshold),
                    metric["precision"],
                )
            )

    if not feasible:
        raise RuntimeError(
            "No historical downstream threshold satisfies FPR target"
        )

    # Historical target-generation rule:
    # recall, then higher threshold, then precision.
    return max(
        feasible,
        key=lambda value: (
            value[0],
            value[1],
            value[2],
        ),
    )[1]


@dataclass
class CrossFittedTargets:
    base_scores: np.ndarray
    augmented_scores: np.ndarray
    base_threshold: float
    augmented_threshold: float
    base_prediction: np.ndarray
    augmented_prediction: np.ndarray
    value: np.ndarray
    base_error: np.ndarray
    fold_ids: np.ndarray


def cross_fitted_downstream_targets(
    data: DevelopmentData,
    indices: np.ndarray,
    *,
    split_seed: int,
    inner_folds: int,
) -> CrossFittedTargets:
    local_y = data.y[
        indices
    ]

    local_groups = data.groups[
        indices
    ]

    splitter = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=int(
            split_seed
        ),
    )

    n = len(indices)

    base_scores = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    augmented_scores = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    fold_ids = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for fold_id, (
        fit_local,
        valid_local,
    ) in enumerate(
        splitter.split(
            np.zeros(
                n,
                dtype=np.float64,
            ),
            local_y,
            local_groups,
        )
    ):
        if set(
            local_groups[
                fit_local
            ]
        ).intersection(
            set(
                local_groups[
                    valid_local
                ]
            )
        ):
            raise RuntimeError(
                "Downstream inner group leakage"
            )

        fit_global = indices[
            fit_local
        ]

        valid_global = indices[
            valid_local
        ]

        model_seed = (
            int(split_seed)
            + 1000
            + int(fold_id)
        )

        base = build_downstream_model(
            model_seed
        )

        augmented = (
            build_downstream_model(
                model_seed
            )
        )

        base.fit(
            data.direct[
                fit_global,
                :2,
            ],
            data.y[
                fit_global
            ],
        )

        augmented.fit(
            data.direct[
                fit_global
            ],
            data.y[
                fit_global
            ],
        )

        base_scores[
            valid_local
        ] = probability_one(
            base,
            data.direct[
                valid_global,
                :2,
            ],
        )

        augmented_scores[
            valid_local
        ] = probability_one(
            augmented,
            data.direct[
                valid_global
            ],
        )

        fold_ids[
            valid_local
        ] = int(
            fold_id
        )

    if (
        np.isnan(
            base_scores
        ).any()
        or np.isnan(
            augmented_scores
        ).any()
        or np.any(
            fold_ids < 0
        )
    ):
        raise RuntimeError(
            "Incomplete downstream cross-fitting"
        )

    base_threshold = historical_threshold(
        local_y,
        base_scores,
    )

    augmented_threshold = historical_threshold(
        local_y,
        augmented_scores,
    )

    base_prediction = (
        base_scores
        >= base_threshold
    ).astype(np.int64)

    augmented_prediction = (
        augmented_scores
        >= augmented_threshold
    ).astype(np.int64)

    value = (
        (
            base_prediction
            != local_y
        ).astype(np.int64)
        -
        (
            augmented_prediction
            != local_y
        ).astype(np.int64)
    )

    if not bool(
        np.all(
            np.isin(
                value,
                (-1, 0, 1),
            )
        )
    ):
        raise RuntimeError(
            "Signed-value target left {-1,0,1}"
        )

    base_error = (
        base_prediction
        != local_y
    ).astype(np.int64)

    return CrossFittedTargets(
        base_scores=base_scores,
        augmented_scores=augmented_scores,
        base_threshold=float(
            base_threshold
        ),
        augmented_threshold=float(
            augmented_threshold
        ),
        base_prediction=base_prediction,
        augmented_prediction=augmented_prediction,
        value=value,
        base_error=base_error,
        fold_ids=fold_ids,
    )


def fit_pca_matrices(
    data: DevelopmentData,
    fit_indices: np.ndarray,
    prediction_indices: np.ndarray,
    *,
    fold_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    PCA,
]:
    pca = PCA(
        n_components=32,
        svd_solver="randomized",
        random_state=int(
            fold_seed
        ),
    )

    pca.fit(
        data.embeddings[
            fit_indices
        ]
    )

    fit_matrix = np.column_stack(
        [
            data.numeric[
                fit_indices
            ],
            pca.transform(
                data.embeddings[
                    fit_indices
                ]
            ),
        ]
    )

    prediction_matrix = np.column_stack(
        [
            data.numeric[
                prediction_indices
            ],
            pca.transform(
                data.embeddings[
                    prediction_indices
                ]
            ),
        ]
    )

    if (
        fit_matrix.shape[1]
        != 49
        or prediction_matrix.shape[1]
        != 49
    ):
        raise RuntimeError(
            "Frozen pre-acquisition matrix is not 49-dimensional"
        )

    if (
        not np.isfinite(
            fit_matrix
        ).all()
        or not np.isfinite(
            prediction_matrix
        ).all()
    ):
        raise RuntimeError(
            "Non-finite transformed pre-acquisition features"
        )

    return (
        fit_matrix,
        prediction_matrix,
        pca,
    )


def cost_predictor_entry(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    entries = [
        value
        for value in (
            protocol[
                "model_families"
            ][
                "cost_predictors"
            ]
        )
        if value["family"]
        == "Ridge_on_log_latency"
    ]

    if len(entries) != 1:
        raise RuntimeError(
            "Frozen Ridge cost-predictor family missing"
        )

    return entries[0]


def build_ridge_cost_model(
    entry: dict[str, Any],
    params: dict[str, Any],
) -> Ridge:
    return Ridge(
        alpha=float(
            params["alpha"]
        ),
        fit_intercept=bool(
            entry[
                "fit_intercept"
            ]
        ),
        solver=str(
            entry["solver"]
        ),
        tol=float(
            entry["tol"]
        ),
    )


def select_cost_candidate(
    identifiers: list[str],
    predictions: dict[
        str,
        np.ndarray,
    ],
    target: np.ndarray,
    latency: dict[
        str,
        float,
    ],
) -> str:
    mse = {
        identifier:
            float(
                np.mean(
                    np.square(
                        predictions[
                            identifier
                        ]
                        - target
                    )
                )
            )
        for identifier
        in identifiers
    }

    best = min(
        mse.values()
    )

    ties = [
        identifier
        for identifier in identifiers
        if np.isclose(
            mse[
                identifier
            ],
            best,
            rtol=0.0,
            atol=1e-15,
        )
    ]

    if len(ties) == 1:
        return ties[0]

    best_latency = min(
        latency[
            identifier
        ]
        for identifier in ties
    )

    latency_ties = [
        identifier
        for identifier in ties
        if np.isclose(
            latency[
                identifier
            ],
            best_latency,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if len(
        latency_ties
    ) != 1:
        raise RuntimeError(
            "Frozen Ridge cost-predictor hyperparameter rule "
            "does not resolve exact MSE-and-latency tie"
        )

    return latency_ties[0]


def one_se_min(
    values: dict[
        str,
        list[float],
    ],
) -> dict[str, Any]:
    evidence = []

    for family, numbers in values.items():
        array = np.asarray(
            numbers,
            dtype=np.float64,
        )

        if len(array) != 5:
            raise RuntimeError(
                f"{family} does not have five seed metrics"
            )

        mean = float(
            np.mean(
                array
            )
        )

        sd = float(
            np.std(
                array,
                ddof=1,
            )
        )

        se = float(
            sd
            / math.sqrt(
                len(array)
            )
        )

        evidence.append(
            {
                "family":
                    family,
                "seed_metrics":
                    array.tolist(),
                "mean":
                    mean,
                "sd":
                    sd,
                "se":
                    se,
            }
        )

    best = min(
        evidence,
        key=lambda row:
            row["mean"],
    )

    limit = (
        best["mean"]
        + best["se"]
    )

    eligible = [
        row["family"]
        for row in evidence
        if row["mean"]
        <= limit
        + 1e-15
    ]

    return {
        "direction":
            "minimize",
        "best_family":
            best["family"],
        "best_mean":
            best["mean"],
        "best_standard_error":
            best["se"],
        "one_standard_error_limit":
            limit,
        "eligible_families":
            eligible,
        "selection_status":
            (
                "unique_one_se_family"
                if len(
                    eligible
                ) == 1
                else
                "requires_controlled_online_latency_tiebreak"
            ),
        "selected_family":
            (
                eligible[0]
                if len(
                    eligible
                ) == 1
                else None
            ),
        "families":
            evidence,
    }


def one_se_max(
    values: dict[
        str,
        list[float],
    ],
) -> dict[str, Any]:
    evidence = []

    for family, numbers in values.items():
        array = np.asarray(
            numbers,
            dtype=np.float64,
        )

        if len(array) != 5:
            raise RuntimeError(
                f"{family} does not have five seed metrics"
            )

        mean = float(
            np.mean(
                array
            )
        )

        sd = float(
            np.std(
                array,
                ddof=1,
            )
        )

        se = float(
            sd
            / math.sqrt(
                len(array)
            )
        )

        evidence.append(
            {
                "family":
                    family,
                "seed_metrics":
                    array.tolist(),
                "mean":
                    mean,
                "sd":
                    sd,
                "se":
                    se,
            }
        )

    best = max(
        evidence,
        key=lambda row:
            row["mean"],
    )

    limit = (
        best["mean"]
        - best["se"]
    )

    eligible = [
        row["family"]
        for row in evidence
        if row["mean"]
        >= limit
        - 1e-15
    ]

    return {
        "direction":
            "maximize",
        "best_family":
            best["family"],
        "best_mean":
            best["mean"],
        "best_standard_error":
            best["se"],
        "one_standard_error_limit":
            limit,
        "eligible_families":
            eligible,
        "selection_status":
            (
                "unique_one_se_family"
                if len(
                    eligible
                ) == 1
                else
                "requires_controlled_online_latency_tiebreak"
            ),
        "selected_family":
            (
                eligible[0]
                if len(
                    eligible
                ) == 1
                else None
            ),
        "families":
            evidence,
    }


def self_test() -> None:
    y = np.array(
        [1, 1, 1, 0, 0, 0, 0, 0],
        dtype=np.int64,
    )

    scores = np.array(
        [
            0.95,
            0.85,
            0.75,
            0.90,
            0.70,
            0.20,
            0.10,
            0.05,
        ],
        dtype=np.float64,
    )

    threshold = historical_threshold(
        y,
        scores,
        maximum_fpr=0.20,
    )

    prediction = (
        scores
        >= threshold
    ).astype(np.int64)

    metric = binary_metrics(
        y,
        prediction,
    )

    assert metric["recall"] == 1.0
    assert (
        metric["fpr"]
        <= 0.20
        + 1e-15
    )

    evidence = one_se_min(
        {
            "A": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "B": [
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
        }
    )

    assert (
        evidence[
            "selected_family"
        ]
        == "A"
    )

    print(
        "V2_CANDIDATE_BUNDLE_SELF_TEST=PASS"
    )


def run(
    output_dir: Path,
) -> None:
    protocol_path = (
        ROOT
        / "configs/"
        "exact_cost_risk_cascade_protocol_v2.json"
    )

    if (
        sha256_file(
            protocol_path
        )
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "Frozen v2.1 protocol SHA256 mismatch"
        )

    protocol = load_protocol()

    resampling = protocol[
        "development_resampling"
    ]

    seeds = [
        int(value)
        for value in resampling[
            "fold_seeds"
        ]
    ]

    if seeds != [
        1729,
        2718,
        3141,
        5772,
        8111,
    ]:
        raise RuntimeError(
            "Frozen fold seed list differs"
        )

    if (
        int(
            resampling[
                "outer_folds"
            ]
        )
        != 5
        or int(
            resampling[
                "inner_folds"
            ]
        )
        != 4
    ):
        raise RuntimeError(
            "Frozen fold counts differ"
        )

    if (
        resampling[
            "stratification_target"
        ]
        != "y"
    ):
        raise RuntimeError(
            "Frozen stratification target is not y"
        )

    data = load_development(
        protocol
    )

    cost_manifest_path = (
        ROOT
        / "data/processed/"
        "v2_development_cost_predictor_nested_cv/"
        "run_manifest.json"
    )

    if not cost_manifest_path.is_file():
        raise RuntimeError(
            "Step-79 cost-predictor manifest missing"
        )

    cost_manifest = json.loads(
        cost_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    selected_cost_family = (
        cost_manifest[
            "family_selection"
        ][
            "selected_family_after_one_se_latency_linear_rule"
        ]
    )

    if (
        selected_cost_family
        != "Ridge_on_log_latency"
    ):
        raise RuntimeError(
            f"Unexpected selected cost family: "
            f"{selected_cost_family}"
        )

    signed_groups = (
        signed_value_candidates_by_family(
            protocol
        )
    )

    current_error_groups = (
        classifier_candidates_by_family(
            current_error_candidates_from_protocol(
                protocol
            )
        )
    )

    direct_groups = (
        classifier_candidates_by_family(
            direct_fusion_candidates_from_protocol(
                protocol
            )
        )
    )

    if set(
        signed_groups
    ) != set(
        SIGNED_SLUGS
    ):
        raise RuntimeError(
            "Unexpected signed-value families"
        )

    if set(
        current_error_groups
    ) != set(
        CLASSIFIER_SLUGS
    ):
        raise RuntimeError(
            "Unexpected current-error families"
        )

    if set(
        direct_groups
    ) != set(
        CLASSIFIER_SLUGS
    ):
        raise RuntimeError(
            "Unexpected direct-fusion families"
        )

    cost_entry = (
        cost_predictor_entry(
            protocol
        )
    )

    cost_grid = list(
        cost_entry[
            "candidate_grid"
        ]
    )

    if len(
        cost_grid
    ) != 3:
        raise RuntimeError(
            "Frozen Ridge cost grid must have three candidates"
        )

    random_policy_seeds = None

    for item in (
        protocol[
            "policies"
        ][
            "required_baselines"
        ]
    ):
        if (
            item[
                "policy_id"
            ]
            == "random_acquisition"
        ):
            random_policy_seeds = [
                int(value)
                for value in item[
                    "policy_seeds"
                ]
            ]

    if random_policy_seeds != [
        104729,
        130363,
        155921,
        181081,
        205759,
    ]:
        raise RuntimeError(
            "Frozen random policy seeds differ"
        )

    inner_folds = int(
        resampling[
            "inner_folds"
        ]
    )

    all_rows: list[
        pd.DataFrame
    ] = []

    selection_rows: list[
        dict[str, Any]
    ] = []

    started = time.time()

    print(
        "=== REAL FIVE-SEED DEVELOPMENT CANDIDATE RUN ===",
        flush=True,
    )

    for seed_index, seed in enumerate(
        seeds,
        start=1,
    ):
        outer_splitter = (
            StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed,
            )
        )

        print(
            f"\n--- seed {seed} "
            f"({seed_index}/5) ---",
            flush=True,
        )

        for outer_zero, (
            train_idx,
            eval_idx,
        ) in enumerate(
            outer_splitter.split(
                np.zeros(
                    EXPECTED_ROWS
                ),
                data.y,
                data.groups,
            )
        ):
            outer_fold = (
                outer_zero
                + 1
            )

            if set(
                data.groups[
                    train_idx
                ]
            ).intersection(
                set(
                    data.groups[
                        eval_idx
                    ]
                )
            ):
                raise RuntimeError(
                    "Outer group leakage"
                )

            setup_seed = (
                int(seed)
                + outer_fold
                * 100
            )

            top_targets = (
                cross_fitted_downstream_targets(
                    data,
                    train_idx,
                    split_seed=(
                        setup_seed
                        + 1
                    ),
                    inner_folds=inner_folds,
                )
            )

            base_model = (
                build_downstream_model(
                    setup_seed
                    + 3
                )
            )

            augmented_model = (
                build_downstream_model(
                    setup_seed
                    + 4
                )
            )

            base_model.fit(
                data.direct[
                    train_idx,
                    :2,
                ],
                data.y[
                    train_idx
                ],
            )

            augmented_model.fit(
                data.direct[
                    train_idx
                ],
                data.y[
                    train_idx
                ],
            )

            base_eval_score = (
                probability_one(
                    base_model,
                    data.direct[
                        eval_idx,
                        :2,
                    ],
                )
            )

            augmented_eval_score = (
                probability_one(
                    augmented_model,
                    data.direct[
                        eval_idx
                    ],
                )
            )

            base_eval_prediction = (
                base_eval_score
                >= top_targets.base_threshold
            ).astype(
                np.int64
            )

            augmented_eval_prediction = (
                augmented_eval_score
                >= top_targets.augmented_threshold
            ).astype(
                np.int64
            )

            y_eval = data.y[
                eval_idx
            ]

            realized_eval_value = (
                (
                    base_eval_prediction
                    != y_eval
                ).astype(
                    np.int64
                )
                -
                (
                    augmented_eval_prediction
                    != y_eval
                ).astype(
                    np.int64
                )
            )

            base_eval_error = (
                base_eval_prediction
                != y_eval
            ).astype(
                np.int64
            )

            n_train = len(
                train_idx
            )

            sv_oof: dict[
                str,
                np.ndarray,
            ] = {}

            ce_oof: dict[
                str,
                np.ndarray,
            ] = {}

            df_oof: dict[
                str,
                np.ndarray,
            ] = {}

            cost_oof: dict[
                str,
                np.ndarray,
            ] = {}

            ce_predict_ns: dict[
                str,
                int,
            ] = {}

            ce_predict_n: dict[
                str,
                int,
            ] = {}

            cost_predict_ns: dict[
                str,
                int,
            ] = {}

            cost_predict_n: dict[
                str,
                int,
            ] = {}

            for candidates in (
                signed_groups.values()
            ):
                for candidate in candidates:
                    identifier = (
                        signed_value_candidate_identifier(
                            candidate
                        )
                    )
                    sv_oof[
                        identifier
                    ] = np.full(
                        n_train,
                        np.nan,
                        dtype=np.float64,
                    )

            for candidates in (
                current_error_groups.values()
            ):
                for candidate in candidates:
                    identifier = (
                        candidate.identifier
                    )
                    ce_oof[
                        identifier
                    ] = np.full(
                        n_train,
                        np.nan,
                        dtype=np.float64,
                    )
                    ce_predict_ns[
                        identifier
                    ] = 0
                    ce_predict_n[
                        identifier
                    ] = 0

            for candidates in (
                direct_groups.values()
            ):
                for candidate in candidates:
                    identifier = (
                        candidate.identifier
                    )
                    df_oof[
                        identifier
                    ] = np.full(
                        n_train,
                        np.nan,
                        dtype=np.float64,
                    )

            for index in range(
                len(
                    cost_grid
                )
            ):
                identifier = (
                    "Ridge_on_log_latency:"
                    f"{index:03d}"
                )

                cost_oof[
                    identifier
                ] = np.full(
                    n_train,
                    np.nan,
                    dtype=np.float64,
                )

                cost_predict_ns[
                    identifier
                ] = 0

                cost_predict_n[
                    identifier
                ] = 0

            log_latency_train = (
                np.log1p(
                    data.optional_latency_ms[
                        train_idx
                    ]
                )
            )

            for inner_fold in range(
                inner_folds
            ):
                valid_local = np.flatnonzero(
                    top_targets.fold_ids
                    == inner_fold
                )

                fit_local = np.flatnonzero(
                    top_targets.fold_ids
                    != inner_fold
                )

                if (
                    not len(
                        fit_local
                    )
                    or not len(
                        valid_local
                    )
                ):
                    raise RuntimeError(
                        "Empty estimator inner fold"
                    )

                fit_global = (
                    train_idx[
                        fit_local
                    ]
                )

                valid_global = (
                    train_idx[
                        valid_local
                    ]
                )

                # Crucial leakage guard:
                # regenerate training targets using only this
                # estimator-inner training partition.
                nested_fit_targets = (
                    cross_fitted_downstream_targets(
                        data,
                        fit_global,
                        split_seed=(
                            setup_seed
                            + 1
                        ),
                        inner_folds=inner_folds,
                    )
                )

                X_fit, X_valid, _ = (
                    fit_pca_matrices(
                        data,
                        fit_global,
                        valid_global,
                        fold_seed=seed,
                    )
                )

                for family, candidates in (
                    signed_groups.items()
                ):
                    for candidate in candidates:
                        identifier = (
                            signed_value_candidate_identifier(
                                candidate
                            )
                        )

                        model = (
                            build_signed_value_regressor(
                                candidate,
                                random_state=seed,
                            )
                        )

                        model.fit(
                            X_fit,
                            nested_fit_targets.value,
                        )

                        prediction = np.asarray(
                            model.predict(
                                X_valid
                            ),
                            dtype=np.float64,
                        )

                        sv_oof[
                            identifier
                        ][
                            valid_local
                        ] = prediction

                for family, candidates in (
                    current_error_groups.items()
                ):
                    for candidate in candidates:
                        identifier = (
                            candidate.identifier
                        )

                        model = (
                            build_classifier(
                                candidate,
                                random_state=seed,
                            )
                        )

                        model.fit(
                            X_fit,
                            nested_fit_targets.base_error,
                        )

                        t0 = (
                            time.perf_counter_ns()
                        )

                        probability = (
                            positive_class_probability(
                                model,
                                X_valid,
                            )
                        )

                        t1 = (
                            time.perf_counter_ns()
                        )

                        ce_predict_ns[
                            identifier
                        ] += (
                            t1
                            - t0
                        )

                        ce_predict_n[
                            identifier
                        ] += len(
                            valid_local
                        )

                        ce_oof[
                            identifier
                        ][
                            valid_local
                        ] = probability

                for cost_index, params in enumerate(
                    cost_grid
                ):
                    identifier = (
                        "Ridge_on_log_latency:"
                        f"{cost_index:03d}"
                    )

                    model = (
                        build_ridge_cost_model(
                            cost_entry,
                            params,
                        )
                    )

                    model.fit(
                        X_fit,
                        np.log1p(
                            data.optional_latency_ms[
                                fit_global
                            ]
                        ),
                    )

                    t0 = (
                        time.perf_counter_ns()
                    )

                    prediction = np.asarray(
                        model.predict(
                            X_valid
                        ),
                        dtype=np.float64,
                    )

                    t1 = (
                        time.perf_counter_ns()
                    )

                    cost_predict_ns[
                        identifier
                    ] += (
                        t1
                        - t0
                    )

                    cost_predict_n[
                        identifier
                    ] += len(
                        valid_local
                    )

                    cost_oof[
                        identifier
                    ][
                        valid_local
                    ] = prediction

                direct_fit = (
                    data.direct[
                        fit_global
                    ]
                )

                direct_valid = (
                    data.direct[
                        valid_global
                    ]
                )

                for family, candidates in (
                    direct_groups.items()
                ):
                    for candidate in candidates:
                        identifier = (
                            candidate.identifier
                        )

                        model = (
                            build_classifier(
                                candidate,
                                random_state=seed,
                            )
                        )

                        model.fit(
                            direct_fit,
                            data.y[
                                fit_global
                            ],
                        )

                        df_oof[
                            identifier
                        ][
                            valid_local
                        ] = (
                            positive_class_probability(
                                model,
                                direct_valid,
                            )
                        )

            for collection_name, collection in (
                (
                    "signed-value",
                    sv_oof,
                ),
                (
                    "current-error",
                    ce_oof,
                ),
                (
                    "cost",
                    cost_oof,
                ),
                (
                    "direct-fusion",
                    df_oof,
                ),
            ):
                for identifier, values in (
                    collection.items()
                ):
                    if (
                        not np.isfinite(
                            values
                        ).all()
                    ):
                        raise RuntimeError(
                            f"Incomplete {collection_name} "
                            f"OOF predictions for {identifier}"
                        )

            selected_sv = {}

            for family, candidates in (
                signed_groups.items()
            ):
                result = (
                    select_inner_signed_value_candidate(
                        candidates,
                        realized_signed_value=(
                            top_targets.value
                        ),
                        candidate_predictions={
                            signed_value_candidate_identifier(
                                candidate
                            ):
                            sv_oof[
                                signed_value_candidate_identifier(
                                    candidate
                                )
                            ]
                            for candidate
                            in candidates
                        },
                    )
                )

                selected_sv[
                    family
                ] = (
                    result.selected_candidate
                )

                metric = next(
                    item
                    for item in result.candidate_metrics
                    if (
                        item.candidate_identifier
                        == signed_value_candidate_identifier(
                            result.selected_candidate
                        )
                    )
                )

                selection_rows.append(
                    {
                        "role":
                            "signed_value",
                        "seed":
                            seed,
                        "outer_fold":
                            outer_fold,
                        "family":
                            family,
                        "selected_candidate":
                            signed_value_candidate_identifier(
                                result.selected_candidate
                            ),
                        "inner_metric":
                            (
                                metric
                                .pooled_inner_validation_mean_squared_error
                            ),
                        "inner_metric_name":
                            "pooled_inner_validation_mean_squared_error",
                        "decision_threshold":
                            np.nan,
                    }
                )

            selected_ce = {}

            for family, candidates in (
                current_error_groups.items()
            ):
                latency = {
                    candidate.identifier:
                        (
                            ce_predict_ns[
                                candidate.identifier
                            ]
                            / 1e6
                            / ce_predict_n[
                                candidate.identifier
                            ]
                        )
                    for candidate in candidates
                }

                result = (
                    select_inner_current_error_candidate(
                        candidates,
                        labels=(
                            top_targets.base_error
                        ),
                        candidate_probabilities={
                            candidate.identifier:
                                ce_oof[
                                    candidate.identifier
                                ]
                            for candidate
                            in candidates
                        },
                        candidate_inference_latency_ms_per_example=latency,
                    )
                )

                selected_ce[
                    family
                ] = (
                    result.selected_candidate
                )

                metric = next(
                    item
                    for item in result.candidate_metrics
                    if (
                        item.candidate_identifier
                        == result.selected_candidate.identifier
                    )
                )

                selection_rows.append(
                    {
                        "role":
                            "current_error",
                        "seed":
                            seed,
                        "outer_fold":
                            outer_fold,
                        "family":
                            family,
                        "selected_candidate":
                            result.selected_candidate.identifier,
                        "inner_metric":
                            metric.pooled_binary_log_loss,
                        "inner_metric_name":
                            "pooled_binary_log_loss",
                        "decision_threshold":
                            np.nan,
                    }
                )

            cost_ids = [
                "Ridge_on_log_latency:"
                f"{index:03d}"
                for index in range(
                    len(
                        cost_grid
                    )
                )
            ]

            cost_latency = {
                identifier:
                    (
                        cost_predict_ns[
                            identifier
                        ]
                        / 1e6
                        / cost_predict_n[
                            identifier
                        ]
                    )
                for identifier
                in cost_ids
            }

            selected_cost_id = (
                select_cost_candidate(
                    cost_ids,
                    cost_oof,
                    log_latency_train,
                    cost_latency,
                )
            )

            selected_cost_index = (
                int(
                    selected_cost_id.rsplit(
                        ":",
                        1,
                    )[1]
                )
            )

            selected_cost_params = (
                cost_grid[
                    selected_cost_index
                ]
            )

            selection_rows.append(
                {
                    "role":
                        "incremental_cost",
                    "seed":
                        seed,
                    "outer_fold":
                        outer_fold,
                    "family":
                        "Ridge_on_log_latency",
                    "selected_candidate":
                        selected_cost_id,
                    "inner_metric":
                        float(
                            np.mean(
                                np.square(
                                    cost_oof[
                                        selected_cost_id
                                    ]
                                    - log_latency_train
                                )
                            )
                        ),
                    "inner_metric_name":
                        "pooled_mean_squared_error_on_log1p_optional_monitor_latency",
                    "decision_threshold":
                        np.nan,
                }
            )

            selected_df = {}

            for family, candidates in (
                direct_groups.items()
            ):
                result = (
                    select_inner_direct_fusion_candidate(
                        candidates,
                        labels=(
                            data.y[
                                train_idx
                            ]
                        ),
                        candidate_probabilities={
                            candidate.identifier:
                                df_oof[
                                    candidate.identifier
                                ]
                            for candidate
                            in candidates
                        },
                        maximum_fpr=0.05,
                    )
                )

                selected_df[
                    family
                ] = (
                    result.selected_candidate,
                    float(
                        result.selected_threshold
                    ),
                )

                metric = next(
                    item
                    for item in result.candidate_metrics
                    if (
                        item.candidate_identifier
                        == result.selected_candidate.identifier
                    )
                )

                selection_rows.append(
                    {
                        "role":
                            "direct_fusion",
                        "seed":
                            seed,
                        "outer_fold":
                            outer_fold,
                        "family":
                            family,
                        "selected_candidate":
                            result.selected_candidate.identifier,
                        "inner_metric":
                            metric.recall,
                        "inner_metric_name":
                            "recall_at_inner_validation_FPR_at_most_0.05",
                        "decision_threshold":
                            float(
                                result.selected_threshold
                            ),
                    }
                )

            X_train, X_eval, _ = (
                fit_pca_matrices(
                    data,
                    train_idx,
                    eval_idx,
                    fold_seed=seed,
                )
            )

            cost_model = (
                build_ridge_cost_model(
                    cost_entry,
                    selected_cost_params,
                )
            )

            cost_model.fit(
                X_train,
                np.log1p(
                    data.optional_latency_ms[
                        train_idx
                    ]
                ),
            )

            predicted_log_cost = (
                np.asarray(
                    cost_model.predict(
                        X_eval
                    ),
                    dtype=np.float64,
                )
            )

            estimated_cost_ms = (
                np.maximum(
                    0.0,
                    np.expm1(
                        predicted_log_cost
                    ),
                )
            )

            if not np.isfinite(
                estimated_cost_ms
            ).all():
                raise RuntimeError(
                    "Non-finite outer cost predictions"
                )

            result_frame = pd.DataFrame(
                {
                    "seed":
                        seed,
                    "outer_fold":
                        outer_fold,
                    "example_id":
                        data.frame.iloc[
                            eval_idx
                        ][
                            "example_id"
                        ].astype(str).to_numpy(),
                    "split":
                        data.frame.iloc[
                            eval_idx
                        ][
                            "split"
                        ].astype(str).to_numpy(),
                    "effective_group":
                        data.groups[
                            eval_idx
                        ],
                    "y":
                        y_eval,
                    "base_probability":
                        base_eval_score,
                    "base_decision_threshold":
                        top_targets.base_threshold,
                    "base_prediction":
                        base_eval_prediction,
                    "augmented_probability":
                        augmented_eval_score,
                    "augmented_decision_threshold":
                        top_targets.augmented_threshold,
                    "augmented_prediction":
                        augmented_eval_prediction,
                    "realized_signed_value":
                        realized_eval_value,
                    "base_error":
                        base_eval_error,
                    "threshold_distance_score":
                        -np.abs(
                            base_eval_score
                            - top_targets.base_threshold
                        ),
                    "estimated_incremental_cost_ms":
                        estimated_cost_ms,
                    "cost_predictor_candidate":
                        selected_cost_id,
                    "rule_score":
                        data.direct[
                            eval_idx,
                            0,
                        ],
                    "compact_unsafe_score":
                        data.direct[
                            eval_idx,
                            1,
                        ],
                    "qwen_prompt_response_score":
                        data.direct[
                            eval_idx,
                            2,
                        ],
                }
            )

            for policy_seed in (
                random_policy_seeds
            ):
                result_frame[
                    f"random_acquisition_score_{policy_seed}"
                ] = [
                    sha256_uniform(
                        example_id,
                        policy_id="random_acquisition",
                        hash_seed=policy_seed,
                    )
                    for example_id in (
                        result_frame[
                            "example_id"
                        ].tolist()
                    )
                ]

            for family, candidate in (
                selected_sv.items()
            ):
                slug = SIGNED_SLUGS[
                    family
                ]

                model = (
                    build_signed_value_regressor(
                        candidate,
                        random_state=seed,
                    )
                )

                model.fit(
                    X_train,
                    top_targets.value,
                )

                estimate = np.asarray(
                    model.predict(
                        X_eval
                    ),
                    dtype=np.float64,
                )

                score = (
                    cost_aware_signed_value_score(
                        estimate,
                        estimated_cost_ms,
                        cost_floor_ms=1.0,
                    )
                )

                result_frame[
                    f"signed_value_{slug}_candidate"
                ] = (
                    signed_value_candidate_identifier(
                        candidate
                    )
                )

                result_frame[
                    f"signed_value_{slug}_estimate"
                ] = estimate

                result_frame[
                    f"signed_value_{slug}_cost_score"
                ] = score

            for family, candidate in (
                selected_ce.items()
            ):
                slug = CLASSIFIER_SLUGS[
                    family
                ]

                model = (
                    build_classifier(
                        candidate,
                        random_state=seed,
                    )
                )

                model.fit(
                    X_train,
                    top_targets.base_error,
                )

                probability = (
                    positive_class_probability(
                        model,
                        X_eval,
                    )
                )

                result_frame[
                    f"current_error_{slug}_candidate"
                ] = (
                    candidate.identifier
                )

                result_frame[
                    f"current_error_{slug}_probability"
                ] = probability

            for family, (
                candidate,
                threshold,
            ) in selected_df.items():
                slug = CLASSIFIER_SLUGS[
                    family
                ]

                model = (
                    build_classifier(
                        candidate,
                        random_state=seed,
                    )
                )

                model.fit(
                    data.direct[
                        train_idx
                    ],
                    data.y[
                        train_idx
                    ],
                )

                probability = (
                    positive_class_probability(
                        model,
                        data.direct[
                            eval_idx
                        ],
                    )
                )

                prediction = (
                    probability
                    >= threshold
                ).astype(
                    np.int64
                )

                result_frame[
                    f"direct_fusion_{slug}_candidate"
                ] = (
                    candidate.identifier
                )

                result_frame[
                    f"direct_fusion_{slug}_threshold"
                ] = threshold

                result_frame[
                    f"direct_fusion_{slug}_probability"
                ] = probability

                result_frame[
                    f"direct_fusion_{slug}_prediction"
                ] = prediction

            all_rows.append(
                result_frame
            )

            print(
                f"seed={seed} "
                f"outer_fold={outer_fold}/5 "
                f"eval_n={len(eval_idx)} "
                f"cost={selected_cost_id}",
                flush=True,
            )

    bundle = pd.concat(
        all_rows,
        ignore_index=True,
    )

    selections = pd.DataFrame(
        selection_rows
    )

    if len(bundle) != (
        EXPECTED_ROWS
        * len(seeds)
    ):
        raise RuntimeError(
            f"Unexpected candidate bundle row count: {len(bundle)}"
        )

    if (
        bundle[
            [
                "seed",
                "example_id",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "An example appears more than once inside a seed"
        )

    expected_ids = set(
        data.frame[
            "example_id"
        ].astype(str)
    )

    for seed in seeds:
        seed_rows = bundle[
            bundle[
                "seed"
            ].eq(
                seed
            )
        ]

        if (
            len(
                seed_rows
            )
            != EXPECTED_ROWS
        ):
            raise RuntimeError(
                f"Seed {seed} does not contain {EXPECTED_ROWS} rows"
            )

        if set(
            seed_rows[
                "example_id"
            ].astype(str)
        ) != expected_ids:
            raise RuntimeError(
                f"Seed {seed} does not cover every development example once"
            )

    signed_seed_metrics: dict[
        str,
        list[float],
    ] = {
        family: []
        for family in (
            SIGNED_SLUGS
        )
    }

    current_seed_metrics: dict[
        str,
        list[float],
    ] = {
        family: []
        for family in (
            CLASSIFIER_SLUGS
        )
    }

    direct_seed_metrics: dict[
        str,
        list[float],
    ] = {
        family: []
        for family in (
            CLASSIFIER_SLUGS
        )
    }

    direct_fpr_metrics: dict[
        str,
        list[float],
    ] = {
        family: []
        for family in (
            CLASSIFIER_SLUGS
        )
    }

    for seed in seeds:
        seed_frame = (
            bundle[
                bundle[
                    "seed"
                ].eq(
                    seed
                )
            ]
            .sort_values(
                "example_id"
            )
            .reset_index(
                drop=True
            )
        )

        target_value = (
            seed_frame[
                "realized_signed_value"
            ].to_numpy(
                dtype=np.float64
            )
        )

        base_error = (
            seed_frame[
                "base_error"
            ].to_numpy(
                dtype=np.int64
            )
        )

        labels = (
            seed_frame[
                "y"
            ].to_numpy(
                dtype=np.int64
            )
        )

        for family, slug in (
            SIGNED_SLUGS.items()
        ):
            prediction = (
                seed_frame[
                    f"signed_value_{slug}_estimate"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            signed_seed_metrics[
                family
            ].append(
                float(
                    np.mean(
                        np.square(
                            prediction
                            - target_value
                        )
                    )
                )
            )

        for family, slug in (
            CLASSIFIER_SLUGS.items()
        ):
            probability = (
                seed_frame[
                    f"current_error_{slug}_probability"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            current_seed_metrics[
                family
            ].append(
                float(
                    log_loss(
                        base_error,
                        probability,
                        labels=[
                            0,
                            1,
                        ],
                    )
                )
            )

            prediction = (
                seed_frame[
                    f"direct_fusion_{slug}_prediction"
                ].to_numpy(
                    dtype=np.int64
                )
            )

            metric = binary_metrics(
                labels,
                prediction,
            )

            direct_seed_metrics[
                family
            ].append(
                float(
                    metric[
                        "recall"
                    ]
                )
            )

            direct_fpr_metrics[
                family
            ].append(
                float(
                    metric[
                        "fpr"
                    ]
                )
            )

    signed_evidence = (
        one_se_min(
            signed_seed_metrics
        )
    )

    current_evidence = (
        one_se_min(
            current_seed_metrics
        )
    )

    direct_evidence = (
        one_se_max(
            direct_seed_metrics
        )
    )

    for row in (
        direct_evidence[
            "families"
        ]
    ):
        row[
            "seed_outer_fpr"
        ] = (
            direct_fpr_metrics[
                row[
                    "family"
                ]
            ]
        )
        row[
            "mean_outer_fpr"
        ] = float(
            np.mean(
                direct_fpr_metrics[
                    row[
                        "family"
                    ]
                ]
            )
        )

    family_evidence = {
        "signed_value": {
            "metric":
                "seed_level_pooled_outer_mean_squared_error",
            **signed_evidence,
        },
        "current_error": {
            "metric":
                "seed_level_pooled_outer_binary_log_loss",
            **current_evidence,
        },
        "direct_fusion": {
            "metric":
                "seed_level_pooled_outer_recall_at_frozen_inner_selected_thresholds",
            "outer_fpr_is_diagnostic_not_a_fresh_risk_certificate":
                True,
            **direct_evidence,
        },
        "incremental_cost": {
            "selected_family":
                "Ridge_on_log_latency",
            "source_manifest":
                str(
                    cost_manifest_path.relative_to(
                        ROOT
                    )
                ),
            "source_manifest_sha256":
                sha256_file(
                    cost_manifest_path
                ),
        },
    }

    temp_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "v2_candidate_bundle_"
            ),
            dir=output_dir.parent,
        )
    )

    try:
        bundle_path = (
            temp_root
            / "candidate_bundle.parquet"
        )

        selection_path = (
            temp_root
            / "outer_fold_model_selection.parquet"
        )

        evidence_path = (
            temp_root
            / "family_evidence.json"
        )

        bundle.to_parquet(
            bundle_path,
            index=False,
        )

        selections.to_parquet(
            selection_path,
            index=False,
        )

        evidence_path.write_text(
            json.dumps(
                family_evidence,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        manifest = {
            "status":
                "completed_repeated_grouped_development_candidate_bundle",
            "scope":
                "development_only_model_and_score_generation",
            "protocol_sha256":
                sha256_file(
                    protocol_path
                ),
            "fold_seeds":
                seeds,
            "outer_folds":
                5,
            "inner_folds":
                4,
            "rows_per_seed":
                EXPECTED_ROWS,
            "candidate_bundle_rows":
                int(
                    len(bundle)
                ),
            "signed_value_families":
                list(
                    SIGNED_SLUGS
                ),
            "current_error_families":
                list(
                    CLASSIFIER_SLUGS
                ),
            "direct_fusion_families":
                list(
                    CLASSIFIER_SLUGS
                ),
            "required_selective_baselines_present":
                [
                    "threshold_distance",
                    "current_error_prediction",
                    "random_acquisition",
                ],
            "required_full_cost_baseline_present":
                "direct_fusion",
            "endpoints_present":
                [
                    "never_acquire_via_base_prediction",
                    "always_acquire_via_augmented_prediction",
                ],
            "random_policy_seeds":
                random_policy_seeds,
            "selected_cost_predictor_family":
                "Ridge_on_log_latency",
            "pre_acquisition_feature_dimension":
                49,
            "pca_components":
                32,
            "pca_fit_scope":
                "current estimator training partition only",
            "signed_value_target_generation":
                (
                    "nested downstream cross-fitting; current "
                    "outer evaluation excluded; estimator-inner "
                    "validation excluded from training-target generation"
                ),
            "exact_cost_policy_selection_performed":
                False,
            "reason_exact_cost_not_yet_performed":
                (
                    "frozen protocol requires measured total "
                    "end-to-end policy latency; controlled E2E "
                    "measurement is the next stage"
                ),
            "fresh_calibration_used":
                False,
            "fresh_confirmatory_used":
                False,
            "protected_legacy_used":
                False,
            "legacy_final_test_used":
                False,
            "legacy_held_out_shift_used":
                False,
            "family_evidence":
                family_evidence,
            "input_files": {
                name: {
                    "path":
                        str(
                            path.relative_to(
                                ROOT
                            )
                        ),
                    "sha256":
                        sha256_file(
                            path
                        ),
                }
                for name, path in (
                    data.source_paths.items()
                )
            },
            "files": {
                "candidate_bundle.parquet":
                    sha256_file(
                        bundle_path
                    ),
                "outer_fold_model_selection.parquet":
                    sha256_file(
                        selection_path
                    ),
                "family_evidence.json":
                    sha256_file(
                        evidence_path
                    ),
            },
            "elapsed_seconds":
                float(
                    time.time()
                    - started
                ),
        }

        manifest_path = (
            temp_root
            / "run_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        if output_dir.exists():
            raise RuntimeError(
                f"Output directory appeared during run: {output_dir}"
            )

        os.replace(
            temp_root,
            output_dir,
        )

    finally:
        if temp_root.exists():
            shutil.rmtree(
                temp_root,
                ignore_errors=True,
            )

    print(
        "\n=== FAMILY EVIDENCE ==="
    )

    print(
        json.dumps(
            family_evidence,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "\n=== RESULT VALIDATION ==="
    )

    print(
        f"candidate_bundle_rows={len(bundle)}"
    )

    print(
        f"selection_rows={len(selections)}"
    )

    print(
        f"output={output_dir}"
    )

    print(
        "protected_legacy_used=False"
    )

    print(
        "fresh_data_used=False"
    )

    print(
        "exact_cost_selection_performed=False"
    )

    print(
        "REAL_V2_REPEATED_GROUPED_CANDIDATE_BUNDLE=PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "data/processed/"
            "v2_repeated_grouped_candidate_bundle"
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run(
        args.output_dir.resolve()
    )


if __name__ == "__main__":
    main()
