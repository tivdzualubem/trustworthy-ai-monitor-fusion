from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.data_boundary import (
    load_protocol,
)
from monitor_fusion.evaluation.signed_value_models import (
    candidates_by_family,
)
from monitor_fusion.evaluation.signed_value_selection import (
    select_inner_signed_value_candidate,
    signed_value_candidate_identifier,
)


def grouped():
    return candidates_by_family(
        load_protocol()
    )


def test_selection_requires_one_family() -> None:
    families = grouped()

    mixed = (
        families["Ridge"][0],
        families["HistGradientBoostingRegressor"][0],
    )

    with pytest.raises(
        ValueError,
        match="exactly one signed-value model family",
    ):
        select_inner_signed_value_candidate(
            mixed,
            realized_signed_value=[
                -1,
                0,
                1,
                0,
            ],
            candidate_predictions={
                signed_value_candidate_identifier(mixed[0]): [
                    -0.8,
                    0.1,
                    0.8,
                    0.0,
                ],
                signed_value_candidate_identifier(mixed[1]): [
                    -0.8,
                    0.1,
                    0.8,
                    0.0,
                ],
            },
        )


def test_selection_uses_pooled_inner_validation_mse() -> None:
    candidates = grouped()[
        "HistGradientBoostingRegressor"
    ]

    target = np.array(
        [-1, 0, 1, 0, 1, -1],
        dtype=float,
    )

    predictions = {
        signed_value_candidate_identifier(candidates[0]): [
            -0.5, 0.2, 0.6, 0.2, 0.7, -0.5
        ],
        signed_value_candidate_identifier(candidates[1]): [
            -0.9, 0.0, 0.9, 0.0, 0.9, -0.9
        ],
        signed_value_candidate_identifier(candidates[2]): [
            -0.2, 0.5, 0.3, 0.4, 0.4, -0.2
        ],
    }

    result = select_inner_signed_value_candidate(
        candidates,
        realized_signed_value=target,
        candidate_predictions=predictions,
    )

    assert (
        signed_value_candidate_identifier(result.selected_candidate)
        == signed_value_candidate_identifier(candidates[1])
    )


def test_metric_is_pooled_not_average_of_fold_metrics() -> None:
    candidates = grouped()["Ridge"][:2]

    target = np.array(
        [-1, 0, 1, 0],
        dtype=float,
    )

    predictions = {
        signed_value_candidate_identifier(candidates[0]): [
            -1.0,
            0.0,
            1.0,
            0.0,
        ],
        signed_value_candidate_identifier(candidates[1]): [
            -0.5,
            0.0,
            0.5,
            0.0,
        ],
    }

    result = select_inner_signed_value_candidate(
        candidates,
        realized_signed_value=target,
        candidate_predictions=predictions,
    )

    metrics = {
        x.candidate_identifier:
        x.pooled_inner_validation_mean_squared_error
        for x in result.candidate_metrics
    }

    assert metrics[
        signed_value_candidate_identifier(candidates[0])
    ] == pytest.approx(0.0)

    assert metrics[
        signed_value_candidate_identifier(candidates[1])
    ] == pytest.approx(0.125)


def test_exact_metric_tie_fails_closed() -> None:
    candidates = grouped()["RandomForestRegressor"][:2]

    prediction = [
        -0.5,
        0.0,
        0.5,
        0.0,
    ]

    with pytest.raises(
        RuntimeError,
        match="does not resolve",
    ):
        select_inner_signed_value_candidate(
            candidates,
            realized_signed_value=[
                -1,
                0,
                1,
                0,
            ],
            candidate_predictions={
                signed_value_candidate_identifier(candidates[0]):
                    prediction,
                signed_value_candidate_identifier(candidates[1]):
                    prediction,
            },
        )


@pytest.mark.parametrize(
    "target",
    [
        [0, 2, 1],
        [0, np.nan, 1],
        [],
    ],
)
def test_invalid_targets_fail_closed(
    target,
) -> None:
    candidate = grouped()["Ridge"][0]

    with pytest.raises(ValueError):
        select_inner_signed_value_candidate(
            [candidate],
            realized_signed_value=target,
            candidate_predictions={
                signed_value_candidate_identifier(candidate):
                    [0.0] * len(target)
            },
        )


def test_missing_candidate_prediction_fails_closed() -> None:
    candidate = grouped()["Ridge"][0]

    with pytest.raises(
        KeyError,
        match="Missing predictions",
    ):
        select_inner_signed_value_candidate(
            [candidate],
            realized_signed_value=[
                -1,
                0,
                1,
            ],
            candidate_predictions={},
        )
