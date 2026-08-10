from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge

from monitor_fusion.evaluation.cost_predictor_selection import (
    HGBR_FAMILY,
    HGBR_GRID,
    RIDGE_FAMILY,
    RIDGE_GRID,
    InnerCostPredictorEvaluation,
    build_cost_predictor,
    evaluate_inner_candidate_predictions,
    fit_cost_predictor_candidate,
    frozen_cost_predictor_candidates,
    predict_log_latency,
    select_inner_cost_predictor_candidate,
)


def test_frozen_grid_contains_exactly_six_candidates() -> None:
    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=1729
        )
    )

    assert len(candidates) == 6

    assert [
        candidate.family
        for candidate in candidates
    ] == [
        RIDGE_FAMILY,
        RIDGE_FAMILY,
        RIDGE_FAMILY,
        HGBR_FAMILY,
        HGBR_FAMILY,
        HGBR_FAMILY,
    ]


def test_ridge_grid_exactly_matches_protocol() -> None:
    assert RIDGE_GRID == (
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
    )

    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=2718
        )
    )

    for candidate in candidates[:3]:
        assert candidate.parameters[
            "fit_intercept"
        ] is True

        assert candidate.parameters[
            "solver"
        ] == "auto"

        assert candidate.parameters[
            "tol"
        ] == pytest.approx(
            0.0001
        )


def test_hgbr_grid_exactly_matches_protocol() -> None:
    assert HGBR_GRID == (
        {
            "l2_regularization": 1.0,
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 30,
        },
        {
            "l2_regularization": 1.0,
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
        },
        {
            "l2_regularization": 5.0,
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
    )

    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=3141
        )
    )

    for candidate in candidates[3:]:
        assert candidate.parameters[
            "loss"
        ] == "squared_error"

        assert candidate.parameters[
            "early_stopping"
        ] is False

        assert candidate.parameters[
            "random_state"
        ] == 3141


def test_factory_builds_exact_estimator_classes() -> None:
    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=5772
        )
    )

    assert isinstance(
        build_cost_predictor(
            candidates[0]
        ),
        Ridge,
    )

    assert isinstance(
        build_cost_predictor(
            candidates[3]
        ),
        HistGradientBoostingRegressor,
    )


def test_candidate_fit_and_prediction_are_finite() -> None:
    rng = np.random.default_rng(
        8111
    )

    x = rng.normal(
        size=(80, 49)
    )

    y = rng.normal(
        loc=8.0,
        scale=0.2,
        size=80,
    )

    for candidate in (
        frozen_cost_predictor_candidates(
            fold_seed=8111
        )
    ):
        estimator = (
            fit_cost_predictor_candidate(
                candidate,
                training_features=x,
                training_log_latency=y,
            )
        )

        prediction = predict_log_latency(
            estimator,
            features=x[:7],
        )

        assert prediction.shape == (
            7,
        )

        assert np.all(
            np.isfinite(prediction)
        )








def test_prediction_evaluation_uses_pooled_mse() -> None:
    candidate = (
        frozen_cost_predictor_candidates(
            fold_seed=1729
        )[0]
    )

    result = (
        evaluate_inner_candidate_predictions(
            candidate,
            true_log_latency=np.array(
                [1.0, 2.0, 3.0]
            ),
            predicted_log_latency=np.array(
                [1.0, 3.0, 2.0]
            ),
            online_inference_latency_ms=0.5,
        )
    )

    assert (
        result.pooled_log_latency_mse
        == pytest.approx(
            2.0 / 3.0
        )
    )


def test_invalid_training_data_fails_closed() -> None:
    candidate = (
        frozen_cost_predictor_candidates(
            fold_seed=1729
        )[0]
    )

    with pytest.raises(
        ValueError
    ):
        fit_cost_predictor_candidate(
            candidate,
            training_features=np.array(
                [[1.0, np.nan]]
            ),
            training_log_latency=np.array(
                [1.0]
            ),
        )


def test_duplicate_candidate_evaluation_fails() -> None:
    candidate = (
        frozen_cost_predictor_candidates(
            fold_seed=1729
        )[0]
    )

    item = InnerCostPredictorEvaluation(
        candidate=candidate,
        pooled_log_latency_mse=0.1,
        online_inference_latency_ms=1.0,
    )

    with pytest.raises(
        ValueError
    ):
        select_inner_cost_predictor_candidate(
            (item, item)
        )


def test_inner_selection_rejects_mixed_families() -> None:
    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=1729
        )
    )

    evaluations = (
        InnerCostPredictorEvaluation(
            candidate=candidates[0],
            pooled_log_latency_mse=0.10,
            online_inference_latency_ms=1.0,
        ),
        InnerCostPredictorEvaluation(
            candidate=candidates[3],
            pooled_log_latency_mse=0.09,
            online_inference_latency_ms=1.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="exactly one model family",
    ):
        select_inner_cost_predictor_candidate(
            evaluations
        )


def test_inner_selection_operates_within_ridge_family() -> None:
    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=1729
        )
    )

    evaluations = (
        InnerCostPredictorEvaluation(
            candidate=candidates[0],
            pooled_log_latency_mse=0.20,
            online_inference_latency_ms=0.1,
        ),
        InnerCostPredictorEvaluation(
            candidate=candidates[1],
            pooled_log_latency_mse=0.10,
            online_inference_latency_ms=0.2,
        ),
        InnerCostPredictorEvaluation(
            candidate=candidates[2],
            pooled_log_latency_mse=0.15,
            online_inference_latency_ms=0.1,
        ),
    )

    selected = (
        select_inner_cost_predictor_candidate(
            evaluations
        )
    )

    assert selected.candidate == candidates[1]


def test_inner_selection_operates_within_hgbr_family() -> None:
    candidates = (
        frozen_cost_predictor_candidates(
            fold_seed=5772
        )
    )

    evaluations = (
        InnerCostPredictorEvaluation(
            candidate=candidates[3],
            pooled_log_latency_mse=0.20,
            online_inference_latency_ms=0.5,
        ),
        InnerCostPredictorEvaluation(
            candidate=candidates[4],
            pooled_log_latency_mse=0.20,
            online_inference_latency_ms=0.3,
        ),
        InnerCostPredictorEvaluation(
            candidate=candidates[5],
            pooled_log_latency_mse=0.25,
            online_inference_latency_ms=0.2,
        ),
    )

    selected = (
        select_inner_cost_predictor_candidate(
            evaluations
        )
    )

    assert selected.candidate == candidates[4]
