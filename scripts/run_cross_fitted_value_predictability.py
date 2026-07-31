#!/usr/bin/env python3
"""Run the frozen six-family value-predictability diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)
AUDITED_DATASET_PATH = (
    ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
)
SCORE_CACHE_PATH = (
    ROOT / "data/processed/monitor_score_cache_v3.parquet"
)
EMBEDDING_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "frozen_prompt_response_embeddings.parquet"
)
EMBEDDING_MANIFEST_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "frozen_prompt_response_embedding_manifest.json"
)
NESTED_TARGET_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "nested_value_training_targets.parquet"
)
OUTER_TARGET_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "cross_fitted_decision_value_targets.parquet"
)
ASSIGNMENT_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "development_outer_fold_assignments.csv"
)
OUTPUT_DIR = ROOT / "reports/decision_value_real_data"

OOF_PATH = OUTPUT_DIR / "value_estimator_oof_predictions.parquet"
FOLD_METRIC_PATH = OUTPUT_DIR / "value_estimator_fold_metrics.csv"
CURVE_PATH = OUTPUT_DIR / "matched_budget_value_curves.csv"
RANDOM_PATH = OUTPUT_DIR / "matched_budget_random_repetitions.csv"
SUMMARY_PATH = OUTPUT_DIR / "value_predictability_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "value_predictability_manifest.json"
REPORT_PATH = OUTPUT_DIR / "value_predictability_summary.md"

FORBIDDEN_PREDICTOR_COLUMNS = {
    "y",
    "realized_decision_value",
    "optional_monitor",
    "optional_monitor_score",
    "qwen_prompt_response_score",
    "source_dataset",
    "source_record_id",
    "attack_family",
    "split",
    "group_id",
    "pair_id",
    "prompt",
    "response",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    params: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def stable_tie_value(example_id: str) -> str:
    return hashlib.sha256(
        str(example_id).encode("utf-8")
    ).hexdigest()


def load_protocol() -> dict[str, Any]:
    return json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def validate_frozen_diagnostic(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = protocol[
        "value_predictability_diagnostic"
    ]
    expected_families = protocol[
        "predictor_families"
    ]["comparisons"]

    if diagnostic["status"] != (
        "frozen_before_value_estimator_fit"
    ):
        raise RuntimeError(
            "Value-predictability diagnostic is not frozen"
        )
    if diagnostic["primary_setup"] != (
        "qwen_after_rule_compact"
    ):
        raise RuntimeError("Unexpected primary setup")
    if diagnostic["primary_feature_family"] != (
        "all_features"
    ):
        raise RuntimeError(
            "Unexpected primary feature family"
        )

    all_named = [
        diagnostic["primary_feature_family"],
        *diagnostic["secondary_feature_families"],
    ]
    if set(all_named) != set(expected_families):
        raise RuntimeError(
            "Diagnostic families differ from frozen comparisons"
        )
    if diagnostic["overall_milestone_claim_allowed"]:
        raise RuntimeError(
            "Overall milestone claim must remain disabled"
        )
    return diagnostic


def load_feature_frame(
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    dataset = pd.read_parquet(AUDITED_DATASET_PATH)
    cache = pd.read_parquet(SCORE_CACHE_PATH)
    embeddings = pd.read_parquet(EMBEDDING_PATH)
    assignments = pd.read_csv(ASSIGNMENT_PATH)

    development_splits = set(
        protocol["scope"]["development_splits"]
    )
    excluded_splits = set(
        protocol["scope"]["excluded_splits"]
    )

    dataset_required = {
        "example_id",
        "prompt",
        "response",
        "split",
    }
    cache_required = {
        "example_id",
        "rule_score",
        "rule_weighted_sum",
        "rule_match_count",
        "rule_latency_ms",
        "compact_unsafe_score",
        "compact_input_tokens",
        "compact_output_tokens",
        "compact_latency_ms",
        "qwen_prompt_response_score",
    }

    if dataset_required - set(dataset.columns):
        raise RuntimeError("Audited dataset schema mismatch")
    if cache_required - set(cache.columns):
        raise RuntimeError("Score-cache schema mismatch")

    development = dataset.loc[
        dataset["split"].isin(development_splits),
        ["example_id", "prompt", "response"],
    ].copy()

    excluded_ids = set(
        dataset.loc[
            dataset["split"].isin(excluded_splits),
            "example_id",
        ]
    )

    frame = (
        development
        .merge(
            cache[sorted(cache_required)],
            on="example_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            assignments,
            on="example_id",
            how="inner",
            validate="one_to_one",
        )
    )

    embedding_columns = [
        column
        for column in embeddings.columns
        if column.startswith("embedding_")
    ]
    if len(embedding_columns) != 384:
        raise RuntimeError(
            "Expected 384 frozen embedding columns"
        )

    frame = frame.merge(
        embeddings[
            ["example_id", *embedding_columns]
        ],
        on="example_id",
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != 1687:
        raise RuntimeError(
            f"Expected 1687 development rows, got {len(frame)}"
        )
    if not frame["example_id"].is_unique:
        raise RuntimeError("Feature rows are not unique")
    if set(frame["example_id"]).intersection(excluded_ids):
        raise RuntimeError(
            "Excluded evaluation row entered feature frame"
        )

    frame["prompt_char_count"] = (
        frame["prompt"].astype(str).str.len()
    )
    frame["response_char_count"] = (
        frame["response"].astype(str).str.len()
    )
    frame["prompt_whitespace_token_count"] = (
        frame["prompt"].astype(str).str.split().str.len()
    )
    frame["response_whitespace_token_count"] = (
        frame["response"].astype(str).str.split().str.len()
    )

    rule = frame["rule_score"].to_numpy(dtype=float)
    compact = frame[
        "compact_unsafe_score"
    ].to_numpy(dtype=float)

    frame["abs_rule_compact_difference"] = np.abs(
        rule - compact
    )
    frame["rule_compact_product"] = rule * compact
    frame["rule_compact_mean"] = (
        rule + compact
    ) / 2.0
    frame["rule_compact_max"] = np.maximum(
        rule,
        compact,
    )
    frame["rule_compact_min"] = np.minimum(
        rule,
        compact,
    )

    return frame, embedding_columns


def family_feature_names(
    protocol: dict[str, Any],
    setup_id: str,
    family: str,
) -> tuple[list[str], bool]:
    predictors = protocol["predictor_families"]
    cheap = list(
        predictors["cheap_features"][setup_id]
    )
    metadata = list(
        predictors["runtime_metadata"][setup_id]
    )

    def dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    if family == "cheap_features":
        return cheap, False
    if family == "runtime_metadata":
        return metadata, False
    if family == "cheap_plus_metadata":
        return dedupe([*cheap, *metadata]), False
    if family == "frozen_embedding":
        return [], True
    if family == "cheap_plus_embedding":
        return cheap, True
    if family == "all_features":
        return dedupe([*cheap, *metadata]), True
    raise KeyError(f"Unknown feature family: {family}")


def build_estimator(
    candidate: Candidate,
    random_state: int,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=random_state,
        **candidate.params,
    )


def join_features(
    rows: pd.DataFrame,
    feature_lookup: pd.DataFrame,
) -> pd.DataFrame:
    joined = rows[
        ["example_id"]
    ].merge(
        feature_lookup,
        on="example_id",
        how="left",
        validate="many_to_one",
    )
    if len(joined) != len(rows):
        raise RuntimeError("Feature join changed row count")
    return joined


def matrix_for_rows(
    rows: pd.DataFrame,
    feature_lookup: pd.DataFrame,
    numeric_features: list[str],
    embedding_columns: list[str],
    pca: PCA | None,
) -> np.ndarray:
    joined = join_features(rows, feature_lookup)
    parts: list[np.ndarray] = []

    if numeric_features:
        parts.append(
            joined[numeric_features].to_numpy(
                dtype=np.float64
            )
        )
    if pca is not None:
        embedded = joined[
            embedding_columns
        ].to_numpy(dtype=np.float64)
        parts.append(pca.transform(embedded))
    if not parts:
        raise RuntimeError("Empty feature matrix")

    matrix = np.concatenate(parts, axis=1)
    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Feature matrix contains non-finite values"
        )
    return matrix


def fit_pca(
    rows: pd.DataFrame,
    feature_lookup: pd.DataFrame,
    embedding_columns: list[str],
    components: int,
    random_state: int,
) -> PCA:
    joined = join_features(
        rows,
        feature_lookup[
            ["example_id", *embedding_columns]
        ],
    )
    matrix = joined[
        embedding_columns
    ].to_numpy(dtype=np.float64)

    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    )
    pca.fit(matrix)
    return pca


def evaluate_candidate(
    candidate: Candidate,
    train_rows: pd.DataFrame,
    feature_lookup: pd.DataFrame,
    numeric_features: list[str],
    use_embedding: bool,
    embedding_columns: list[str],
    pca_components: int,
    random_state: int,
) -> tuple[float, list[float]]:
    squared_errors: list[np.ndarray] = []
    fold_mse: list[float] = []

    inner_folds = sorted(
        train_rows["downstream_inner_fold"]
        .astype(int)
        .unique()
        .tolist()
    )
    if inner_folds != [0, 1, 2, 3]:
        raise RuntimeError(
            f"Unexpected estimator inner folds: {inner_folds}"
        )

    for inner_fold in inner_folds:
        fit_rows = train_rows.loc[
            train_rows["downstream_inner_fold"]
            != inner_fold
        ].copy()
        valid_rows = train_rows.loc[
            train_rows["downstream_inner_fold"]
            == inner_fold
        ].copy()

        if set(fit_rows["example_id"]).intersection(
            set(valid_rows["example_id"])
        ):
            raise RuntimeError(
                "Estimator inner-fold example leakage"
            )

        pca = None
        if use_embedding:
            pca = fit_pca(
                rows=fit_rows,
                feature_lookup=feature_lookup,
                embedding_columns=embedding_columns,
                components=pca_components,
                random_state=(
                    random_state
                    + candidate.candidate_id * 101
                    + inner_fold
                ),
            )

        x_fit = matrix_for_rows(
            rows=fit_rows,
            feature_lookup=feature_lookup,
            numeric_features=numeric_features,
            embedding_columns=embedding_columns,
            pca=pca,
        )
        x_valid = matrix_for_rows(
            rows=valid_rows,
            feature_lookup=feature_lookup,
            numeric_features=numeric_features,
            embedding_columns=embedding_columns,
            pca=pca,
        )

        y_fit = fit_rows[
            "realized_decision_value"
        ].to_numpy(dtype=float)
        y_valid = valid_rows[
            "realized_decision_value"
        ].to_numpy(dtype=float)

        model = build_estimator(
            candidate=candidate,
            random_state=(
                random_state
                + candidate.candidate_id * 1009
                + inner_fold
            ),
        )
        model.fit(x_fit, y_fit)
        prediction = model.predict(x_valid)

        error = np.square(prediction - y_valid)
        squared_errors.append(error)
        fold_mse.append(float(error.mean()))

    pooled = np.concatenate(squared_errors)
    return float(pooled.mean()), fold_mse


def regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    rho = spearmanr(
        y,
        prediction,
    ).statistic
    if not np.isfinite(rho):
        rho = 0.0

    positive = (y > 0).astype(int)
    ap = (
        float(
            average_precision_score(
                positive,
                prediction,
            )
        )
        if positive.min() != positive.max()
        else float("nan")
    )

    return {
        "mse": float(
            mean_squared_error(y, prediction)
        ),
        "mae": float(
            mean_absolute_error(y, prediction)
        ),
        "spearman": float(rho),
        "positive_value_average_precision": ap,
    }


def select_exact_budget_mask(
    scores: np.ndarray,
    example_ids: np.ndarray,
    outer_folds: np.ndarray,
    budget: float,
) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)

    for outer_fold in sorted(
        np.unique(outer_folds).tolist()
    ):
        index = np.flatnonzero(
            outer_folds == outer_fold
        )
        n = len(index)
        if budget >= 1.0:
            k = n
        elif budget <= 0.0:
            k = 0
        else:
            k = int(math.floor(budget * n))

        tie = np.asarray(
            [
                stable_tie_value(example_ids[i])
                for i in index
            ],
            dtype=object,
        )
        order = np.lexsort(
            (tie, -scores[index])
        )
        if k > 0:
            mask[index[order[:k]]] = True

    return mask


def select_random_budget_mask(
    n_rows: int,
    outer_folds: np.ndarray,
    budget: float,
    seed: int,
) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    rng = np.random.default_rng(seed)

    for outer_fold in sorted(
        np.unique(outer_folds).tolist()
    ):
        index = np.flatnonzero(
            outer_folds == outer_fold
        )
        n = len(index)
        if budget >= 1.0:
            k = n
        elif budget <= 0.0:
            k = 0
        else:
            k = int(math.floor(budget * n))
        if k > 0:
            mask[rng.permutation(index)[:k]] = True

    return mask


def binary_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(
        prediction,
        dtype=int,
    )
    positive = y == 1
    negative = y == 0

    tp = int(
        np.sum(positive & (prediction == 1))
    )
    fn = int(
        np.sum(positive & (prediction == 0))
    )
    fp = int(
        np.sum(negative & (prediction == 1))
    )
    tn = int(
        np.sum(negative & (prediction == 0))
    )

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": (
            float(tp / (tp + fn))
            if (tp + fn)
            else 0.0
        ),
        "fpr": (
            float(fp / (fp + tn))
            if (fp + tn)
            else 0.0
        ),
        "precision": (
            float(tp / (tp + fp))
            if (tp + fp)
            else 0.0
        ),
        "decision_loss": float(
            np.mean(prediction != y)
        ),
    }


def policy_row(
    frame: pd.DataFrame,
    mask: np.ndarray,
    setup_id: str,
    family: str,
    policy: str,
    budget: float,
    repetition: int | None,
) -> dict[str, Any]:
    y = frame["y"].to_numpy(dtype=int)
    base_prediction = frame[
        "base_prediction"
    ].to_numpy(dtype=int)
    augmented_prediction = frame[
        "augmented_prediction"
    ].to_numpy(dtype=int)
    realized_value = frame[
        "realized_decision_value"
    ].to_numpy(dtype=float)

    mixed_prediction = np.where(
        mask,
        augmented_prediction,
        base_prediction,
    )
    metrics = binary_metrics(
        y,
        mixed_prediction,
    )
    improvement = float(
        np.mean(mask.astype(float) * realized_value)
    )

    return {
        "setup_id": setup_id,
        "feature_family": family,
        "policy": policy,
        "budget": float(budget),
        "repetition": repetition,
        "n": len(frame),
        "acquired_n": int(mask.sum()),
        "actual_acquisition_rate": float(mask.mean()),
        "mean_realized_value_captured": improvement,
        "decision_loss_reduction_vs_never": improvement,
        **metrics,
    }


def paired_integrated_advantage(
    frame: pd.DataFrame,
    budgets: list[float],
    bootstrap_repetitions: int,
    random_state: int,
) -> dict[str, float | bool]:
    example_ids = frame["example_id"].to_numpy()
    outer_folds = frame["outer_fold"].to_numpy(
        dtype=int
    )
    values = frame[
        "realized_decision_value"
    ].to_numpy(dtype=float)
    learned_scores = frame[
        "predicted_decision_value"
    ].to_numpy(dtype=float)
    uncertainty_scores = frame[
        "base_uncertainty"
    ].to_numpy(dtype=float)

    contributions: list[np.ndarray] = []
    for budget in budgets:
        learned_mask = select_exact_budget_mask(
            learned_scores,
            example_ids,
            outer_folds,
            budget,
        )
        uncertainty_mask = select_exact_budget_mask(
            uncertainty_scores,
            example_ids,
            outer_folds,
            budget,
        )
        contributions.append(
            (
                learned_mask.astype(float)
                - uncertainty_mask.astype(float)
            )
            * values
        )

    contribution_matrix = np.column_stack(
        contributions
    )
    per_example_integrated = np.trapezoid(
        contribution_matrix,
        np.asarray(budgets, dtype=float),
        axis=1,
    )
    estimate = float(
        per_example_integrated.mean()
    )

    rng = np.random.default_rng(random_state)
    n = len(per_example_integrated)
    bootstrap = np.empty(
        bootstrap_repetitions,
        dtype=float,
    )
    for repetition in range(
        bootstrap_repetitions
    ):
        index = rng.integers(
            0,
            n,
            size=n,
        )
        bootstrap[repetition] = float(
            per_example_integrated[index].mean()
        )

    lower, upper = np.quantile(
        bootstrap,
        [0.025, 0.975],
    )

    return {
        "integrated_advantage": estimate,
        "paired_bootstrap_lower95": float(lower),
        "paired_bootstrap_upper95": float(upper),
        "predictability_criterion_pass": bool(
            lower > 0.0
        ),
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    primary: dict[str, Any],
) -> None:
    lines = [
        "# Cross-Fitted Value-Predictability Diagnostic",
        "",
        "This report is development-only. It does not use `final_test` or "
        "`held_out_shift`, and it does not change the overall project "
        "`no-go` status.",
        "",
        "## Primary prespecified comparison",
        "",
        f"- Setup: `{primary['setup_id']}`",
        f"- Feature family: `{primary['feature_family']}`",
        (
            "- Integrated learned-minus-uncertainty advantage: "
            f"{primary['integrated_advantage']:.8f}"
        ),
        (
            "- Paired bootstrap 95% CI: "
            f"[{primary['paired_bootstrap_lower95']:.8f}, "
            f"{primary['paired_bootstrap_upper95']:.8f}]"
        ),
        (
            "- Value-predictability criterion: "
            f"{'PASS' if primary['predictability_criterion_pass'] else 'NO-GO'}"
        ),
        "",
        "## All prespecified comparisons",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "The total safety-cost frontier and common-risk selective point "
        "remain required before the professor's overall milestone can pass.",
        "",
    ]
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    protocol = load_protocol()
    diagnostic = validate_frozen_diagnostic(
        protocol
    )
    feature_lookup, embedding_columns = (
        load_feature_frame(protocol)
    )
    nested = pd.read_parquet(NESTED_TARGET_PATH)
    outer = pd.read_parquet(OUTER_TARGET_PATH)

    families = list(
        protocol["predictor_families"]["comparisons"]
    )
    setups = [
        item["setup_id"]
        for item in protocol["optional_monitor_setups"]
    ]
    outer_folds = list(
        range(
            int(
                protocol["cross_fitting"][
                    "outer_folds"
                ]
            )
        )
    )
    pca_components = int(
        protocol["predictor_families"][
            "frozen_embedding"
        ]["pca_components"]
    )
    random_state = int(
        protocol["cross_fitting"]["random_state"]
    )
    candidates = [
        Candidate(
            candidate_id=index,
            params=dict(params),
        )
        for index, params in enumerate(
            diagnostic["model"]["candidate_grid"]
        )
    ]

    prediction_rows: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for setup_index, setup_id in enumerate(setups):
        for outer_fold in outer_folds:
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
                    "Current outer-evaluation example "
                    "entered estimator training"
                )

            for family_index, family in enumerate(
                families
            ):
                numeric_features, use_embedding = (
                    family_feature_names(
                        protocol,
                        setup_id,
                        family,
                    )
                )
                forbidden = set(
                    numeric_features
                ).intersection(
                    FORBIDDEN_PREDICTOR_COLUMNS
                )
                if forbidden:
                    raise RuntimeError(
                        f"Forbidden predictor columns: "
                        f"{sorted(forbidden)}"
                    )

                candidate_metrics: list[
                    tuple[Candidate, float, list[float]]
                ] = []
                fit_started = time.perf_counter()

                for candidate in candidates:
                    cv_mse, fold_mse = (
                        evaluate_candidate(
                            candidate,
                            train_rows,
                            feature_lookup,
                            numeric_features,
                            use_embedding,
                            embedding_columns,
                            pca_components,
                            (
                                random_state
                                + setup_index * 100000
                                + outer_fold * 10000
                                + family_index * 1000
                            ),
                        )
                    )
                    candidate_metrics.append(
                        (
                            candidate,
                            cv_mse,
                            fold_mse,
                        )
                    )

                selected, selected_cv_mse, selected_fold_mse = min(
                    candidate_metrics,
                    key=lambda item: (
                        item[1],
                        item[0].candidate_id,
                    ),
                )

                final_pca = None
                explained_variance = float("nan")
                if use_embedding:
                    final_pca = fit_pca(
                        train_rows,
                        feature_lookup,
                        embedding_columns,
                        pca_components,
                        (
                            random_state
                            + setup_index * 100000
                            + outer_fold * 10000
                            + family_index * 1000
                            + 701
                        ),
                    )
                    explained_variance = float(
                        final_pca.explained_variance_ratio_.sum()
                    )

                x_train = matrix_for_rows(
                    train_rows,
                    feature_lookup,
                    numeric_features,
                    embedding_columns,
                    final_pca,
                )
                x_eval = matrix_for_rows(
                    eval_rows,
                    feature_lookup,
                    numeric_features,
                    embedding_columns,
                    final_pca,
                )

                model = build_estimator(
                    selected,
                    (
                        random_state
                        + setup_index * 100000
                        + outer_fold * 10000
                        + family_index * 1000
                        + 911
                    ),
                )
                model.fit(
                    x_train,
                    train_rows[
                        "realized_decision_value"
                    ].to_numpy(dtype=float),
                )
                prediction = model.predict(x_eval)
                elapsed = (
                    time.perf_counter() - fit_started
                )

                y_eval = eval_rows[
                    "realized_decision_value"
                ].to_numpy(dtype=float)
                metrics = regression_metrics(
                    y_eval,
                    prediction,
                )

                output = eval_rows[
                    [
                        "example_id",
                        "setup_id",
                        "optional_monitor",
                        "outer_fold",
                        "y",
                        "base_prediction",
                        "augmented_prediction",
                        "base_loss",
                        "augmented_loss",
                        "realized_decision_value",
                        "decision_changed",
                        "false_positive_reduction",
                        "false_negative_reduction",
                        "base_uncertainty",
                    ]
                ].copy()
                output["feature_family"] = family
                output[
                    "predicted_decision_value"
                ] = prediction
                output[
                    "selected_candidate_id"
                ] = selected.candidate_id
                prediction_rows.append(output)

                fold_metric_rows.append(
                    {
                        "setup_id": setup_id,
                        "outer_fold": outer_fold,
                        "feature_family": family,
                        "train_n": len(train_rows),
                        "eval_n": len(eval_rows),
                        "numeric_feature_n": len(
                            numeric_features
                        ),
                        "embedding_used": use_embedding,
                        "pca_components": (
                            pca_components
                            if use_embedding
                            else 0
                        ),
                        "pca_explained_variance_ratio_sum": (
                            explained_variance
                        ),
                        "selected_candidate_id": (
                            selected.candidate_id
                        ),
                        "selected_params_json": json.dumps(
                            selected.params,
                            sort_keys=True,
                        ),
                        "selected_inner_cv_mse": (
                            selected_cv_mse
                        ),
                        "selected_inner_fold_mse_json": (
                            json.dumps(
                                selected_fold_mse
                            )
                        ),
                        **metrics,
                        "fit_and_predict_seconds": elapsed,
                    }
                )

                print(
                    "completed "
                    f"setup={setup_id} "
                    f"outer_fold={outer_fold + 1}/5 "
                    f"family={family} "
                    f"candidate={selected.candidate_id} "
                    f"cv_mse={selected_cv_mse:.6f}",
                    flush=True,
                )

    oof = pd.concat(
        prediction_rows,
        ignore_index=True,
    )
    fold_metrics = pd.DataFrame(
        fold_metric_rows
    )

    expected_oof_rows = (
        1687 * len(setups) * len(families)
    )
    if len(oof) != expected_oof_rows:
        raise RuntimeError(
            f"Expected {expected_oof_rows} OOF rows, "
            f"got {len(oof)}"
        )
    if oof.duplicated(
        ["example_id", "setup_id", "feature_family"]
    ).any():
        raise RuntimeError(
            "Duplicate outer-OOF value prediction"
        )

    budgets = [
        float(value)
        for value in protocol["matched_budgets"]
    ]
    random_repetitions = int(
        diagnostic["random_repetitions"]
    )

    deterministic_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for setup_index, setup_id in enumerate(setups):
        for family_index, family in enumerate(families):
            frame = oof.loc[
                (oof["setup_id"] == setup_id)
                & (oof["feature_family"] == family)
            ].sort_values(
                ["outer_fold", "example_id"]
            ).reset_index(drop=True)

            example_ids = frame[
                "example_id"
            ].to_numpy()
            fold_ids = frame[
                "outer_fold"
            ].to_numpy(dtype=int)

            policy_scores = {
                "ordinary_uncertainty": frame[
                    "base_uncertainty"
                ].to_numpy(dtype=float),
                "learned_decision_value": frame[
                    "predicted_decision_value"
                ].to_numpy(dtype=float),
                "oracle_realized_value_diagnostic": frame[
                    "realized_decision_value"
                ].to_numpy(dtype=float),
            }

            for budget in budgets:
                for policy, scores in policy_scores.items():
                    mask = select_exact_budget_mask(
                        scores,
                        example_ids,
                        fold_ids,
                        budget,
                    )
                    deterministic_rows.append(
                        policy_row(
                            frame,
                            mask,
                            setup_id,
                            family,
                            policy,
                            budget,
                            None,
                        )
                    )

                for repetition in range(
                    random_repetitions
                ):
                    mask = select_random_budget_mask(
                        len(frame),
                        fold_ids,
                        budget,
                        (
                            random_state
                            + setup_index * 1000000
                            + family_index * 100000
                            + int(round(budget * 1000))
                            * 100
                            + repetition
                        ),
                    )
                    random_rows.append(
                        policy_row(
                            frame,
                            mask,
                            setup_id,
                            family,
                            "random",
                            budget,
                            repetition,
                        )
                    )

            advantage = paired_integrated_advantage(
                frame,
                budgets,
                int(
                    diagnostic[
                        "paired_uncertainty_comparison"
                    ]["bootstrap_repetitions"]
                ),
                (
                    int(
                        diagnostic[
                            "paired_uncertainty_comparison"
                        ]["random_state"]
                    )
                    + setup_index * 100
                    + family_index
                ),
            )
            pooled_regression = regression_metrics(
                frame[
                    "realized_decision_value"
                ].to_numpy(dtype=float),
                frame[
                    "predicted_decision_value"
                ].to_numpy(dtype=float),
            )
            summary_rows.append(
                {
                    "setup_id": setup_id,
                    "feature_family": family,
                    "primary_comparison": bool(
                        setup_id
                        == diagnostic["primary_setup"]
                        and family
                        == diagnostic[
                            "primary_feature_family"
                        ]
                    ),
                    **pooled_regression,
                    **advantage,
                }
            )

    curves = pd.DataFrame(
        deterministic_rows
    )
    random_repetitions_frame = pd.DataFrame(
        random_rows
    )
    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["setup_id", "feature_family"]
    ).reset_index(drop=True)

    primary_rows = summary.loc[
        summary["primary_comparison"]
    ]
    if len(primary_rows) != 1:
        raise RuntimeError(
            "Expected exactly one primary comparison"
        )
    primary = primary_rows.iloc[0].to_dict()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    oof.to_parquet(
        OOF_PATH,
        index=False,
    )
    fold_metrics.to_csv(
        FOLD_METRIC_PATH,
        index=False,
    )
    curves.to_csv(
        CURVE_PATH,
        index=False,
    )
    random_repetitions_frame.to_csv(
        RANDOM_PATH,
        index=False,
    )
    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )
    write_report(
        REPORT_PATH,
        summary,
        primary,
    )

    embedding_manifest = json.loads(
        EMBEDDING_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "artifact": (
            "cross_fitted_value_predictability_diagnostic_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "development_only_value_predictability_completed"
        ),
        "overall_project_status": "no-go",
        "overall_milestone_claim_made": False,
        "primary_comparison": {
            "setup_id": primary["setup_id"],
            "feature_family": (
                primary["feature_family"]
            ),
            "integrated_advantage": (
                primary["integrated_advantage"]
            ),
            "paired_bootstrap_lower95": (
                primary[
                    "paired_bootstrap_lower95"
                ]
            ),
            "paired_bootstrap_upper95": (
                primary[
                    "paired_bootstrap_upper95"
                ]
            ),
            "predictability_criterion_pass": bool(
                primary[
                    "predictability_criterion_pass"
                ]
            ),
        },
        "scope": {
            "development_rows": 1687,
            "final_test_used": False,
            "held_out_shift_used": False,
            "global_outer_target_rows_used_for_training": False,
            "current_outer_evaluation_fold_excluded": True,
        },
        "model": diagnostic["model"],
        "embedding_transform": (
            diagnostic["embedding_transform"]
        ),
        "ranking": diagnostic["ranking"],
        "matched_budgets": budgets,
        "random_repetitions": random_repetitions,
        "paired_comparison": (
            diagnostic[
                "paired_uncertainty_comparison"
            ]
        ),
        "feature_families": families,
        "setups": setups,
        "embedding_runtime_ms_per_example": (
            embedding_manifest["runtime"][
                "end_to_end_ms_per_example"
            ]
        ),
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                CONFIG_PATH,
                AUDITED_DATASET_PATH,
                SCORE_CACHE_PATH,
                EMBEDDING_PATH,
                NESTED_TARGET_PATH,
                OUTER_TARGET_PATH,
                ASSIGNMENT_PATH,
            ]
        },
        "outputs": {},
    }

    for path in [
        OOF_PATH,
        FOLD_METRIC_PATH,
        CURVE_PATH,
        RANDOM_PATH,
        SUMMARY_PATH,
        REPORT_PATH,
    ]:
        manifest["outputs"][
            str(path.relative_to(ROOT))
        ] = sha256_file(path)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== VALUE-PREDICTABILITY SUMMARY ===")
    print(summary.to_string(index=False))
    print()
    print("primary setup:", primary["setup_id"])
    print(
        "primary family:",
        primary["feature_family"],
    )
    print(
        "integrated advantage:",
        f"{primary['integrated_advantage']:.8f}",
    )
    print(
        "paired 95% CI:",
        (
            f"[{primary['paired_bootstrap_lower95']:.8f}, "
            f"{primary['paired_bootstrap_upper95']:.8f}]"
        ),
    )
    print(
        "predictability criterion:",
        (
            "PASS"
            if primary[
                "predictability_criterion_pass"
            ]
            else "NO-GO"
        ),
    )
    print("overall project status: no-go")
    print(
        "remaining requirement: total safety-cost frontier "
        "and common-risk selective point"
    )


if __name__ == "__main__":
    main()
