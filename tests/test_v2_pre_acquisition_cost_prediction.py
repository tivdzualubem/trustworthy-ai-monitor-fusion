from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitor_fusion.evaluation.heterogeneous_cost_prediction import (
    COST_PREDICTION_FLOOR_MS,
    inverse_optional_monitor_latency_prediction,
    pooled_log_latency_mse,
    transform_optional_monitor_latency_ms,
)
from monitor_fusion.evaluation.pre_acquisition_features import (
    EMBEDDING_DIMENSION,
    NUMERIC_FEATURE_NAMES,
    PCA_COMPONENTS,
    TOTAL_FEATURE_DIMENSION,
    build_numeric_pre_acquisition_features,
    fit_fold_local_pre_acquisition_transform,
)


def synthetic_frame(
    rows: int = 40,
) -> pd.DataFrame:
    index = np.arange(
        rows,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "prompt": [
                f"prompt words {i}"
                for i in range(rows)
            ],
            "response": [
                f"response text words {i}"
                for i in range(rows)
            ],
            "rule_score":
                (index % 10) / 10.0,
            "rule_weighted_sum":
                index + 0.5,
            "rule_match_count":
                index % 4,
            "rule_latency_ms":
                1.0 + index / 100.0,
            "compact_unsafe_score":
                ((index + 3) % 10) / 10.0,
            "compact_input_tokens":
                20.0 + index,
            "compact_output_tokens":
                np.ones(rows),
            "compact_latency_ms":
                10.0 + index / 10.0,
        }
    )


def synthetic_embeddings(
    rows: int = 40,
) -> np.ndarray:
    rng = np.random.default_rng(
        1729
    )

    embeddings = rng.normal(
        size=(
            rows,
            EMBEDDING_DIMENSION,
        )
    )

    norm = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / norm


def test_frozen_feature_dimensions() -> None:
    assert len(
        NUMERIC_FEATURE_NAMES
    ) == 17

    assert PCA_COMPONENTS == 32
    assert TOTAL_FEATURE_DIMENSION == 49


def test_numeric_feature_order_is_frozen() -> None:
    assert NUMERIC_FEATURE_NAMES == (
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
    )


def test_numeric_derived_features() -> None:
    frame = synthetic_frame(
        40
    ).iloc[[0]].copy()

    matrix = (
        build_numeric_pre_acquisition_features(
            frame
        )
    )

    row = dict(
        zip(
            NUMERIC_FEATURE_NAMES,
            matrix[0],
            strict=True,
        )
    )

    rule = float(
        frame.iloc[0]["rule_score"]
    )

    compact = float(
        frame.iloc[0][
            "compact_unsafe_score"
        ]
    )

    assert row[
        "abs_rule_compact_difference"
    ] == pytest.approx(
        abs(rule - compact)
    )

    assert row[
        "rule_compact_product"
    ] == pytest.approx(
        rule * compact
    )

    assert row[
        "rule_compact_mean"
    ] == pytest.approx(
        (rule + compact) / 2.0
    )

    assert row[
        "rule_compact_max"
    ] == pytest.approx(
        max(rule, compact)
    )

    assert row[
        "rule_compact_min"
    ] == pytest.approx(
        min(rule, compact)
    )


def test_fold_local_matrix_has_49_columns() -> None:
    frame = synthetic_frame()
    embeddings = synthetic_embeddings()

    transform, training = (
        fit_fold_local_pre_acquisition_transform(
            frame,
            embeddings,
            random_state=1729,
        )
    )

    assert training.shape == (
        40,
        49,
    )

    validation = transform.transform(
        frame.iloc[:5].copy(),
        embeddings[:5],
    )

    assert validation.shape == (
        5,
        49,
    )


def test_pca_is_deterministic_for_seed() -> None:
    frame = synthetic_frame()
    embeddings = synthetic_embeddings()

    _, first = (
        fit_fold_local_pre_acquisition_transform(
            frame,
            embeddings,
            random_state=2718,
        )
    )

    _, second = (
        fit_fold_local_pre_acquisition_transform(
            frame,
            embeddings,
            random_state=2718,
        )
    )

    np.testing.assert_allclose(
        first,
        second,
    )


def test_transform_does_not_refit_pca() -> None:
    frame = synthetic_frame()
    embeddings = synthetic_embeddings()

    transform, _ = (
        fit_fold_local_pre_acquisition_transform(
            frame,
            embeddings,
            random_state=3141,
        )
    )

    components_before = (
        transform.pca.components_.copy()
    )

    transform.transform(
        frame.iloc[:8],
        embeddings[:8],
    )

    np.testing.assert_array_equal(
        transform.pca.components_,
        components_before,
    )


def test_log_latency_round_trip() -> None:
    latency = np.array(
        [
            0.0,
            1.0,
            100.0,
            3959.0,
            30000.0,
        ]
    )

    transformed = (
        transform_optional_monitor_latency_ms(
            latency
        )
    )

    recovered = (
        inverse_optional_monitor_latency_prediction(
            transformed
        )
    )

    np.testing.assert_allclose(
        recovered,
        latency,
    )


def test_online_prediction_floor() -> None:
    predicted = np.array(
        [
            -100.0,
            0.0,
            np.log1p(10.0),
        ]
    )

    result = (
        inverse_optional_monitor_latency_prediction(
            predicted,
            apply_online_floor=True,
        )
    )

    assert np.all(
        result
        >= COST_PREDICTION_FLOOR_MS
    )

    assert result[-1] == pytest.approx(
        10.0
    )


def test_pooled_mse() -> None:
    truth = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    prediction = np.array(
        [
            1.0,
            3.0,
            2.0,
        ]
    )

    assert pooled_log_latency_mse(
        truth,
        prediction,
    ) == pytest.approx(
        2.0 / 3.0
    )


def test_invalid_cost_targets_fail_closed() -> None:
    with pytest.raises(
        ValueError
    ):
        transform_optional_monitor_latency_ms(
            [
                1.0,
                -1.0,
            ]
        )

    with pytest.raises(
        ValueError
    ):
        pooled_log_latency_mse(
            [1.0],
            [1.0, 2.0],
        )
