from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from monitor_fusion.evaluation.data_boundary import (
    load_protocol,
)
from monitor_fusion.evaluation.signed_value_models import (
    SIGNED_VALUE_FAMILY_ORDER,
    SignedValueModelCandidate,
    build_signed_value_regressor,
    candidates_by_family,
    signed_value_candidates_from_protocol,
)


def test_frozen_family_order_is_exact() -> None:
    assert SIGNED_VALUE_FAMILY_ORDER == (
        "Ridge",
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
    )


def test_protocol_contains_three_candidates_per_family() -> None:
    grouped = candidates_by_family(
        load_protocol()
    )

    assert tuple(grouped) == (
        SIGNED_VALUE_FAMILY_ORDER
    )

    assert {
        family: len(candidates)
        for family, candidates in grouped.items()
    } == {
        "Ridge": 3,
        "HistGradientBoostingRegressor": 3,
        "RandomForestRegressor": 3,
    }


def test_candidate_identifiers_are_stable_and_unique() -> None:
    candidates = signed_value_candidates_from_protocol(
        load_protocol()
    )

    assert len(candidates) == 9

    identifiers = [
        candidate.candidate_id
        for candidate in candidates
    ]

    assert len(set(identifiers)) == 9

    assert identifiers == [
        "Ridge:000",
        "Ridge:001",
        "Ridge:002",
        "HistGradientBoostingRegressor:000",
        "HistGradientBoostingRegressor:001",
        "HistGradientBoostingRegressor:002",
        "RandomForestRegressor:000",
        "RandomForestRegressor:001",
        "RandomForestRegressor:002",
    ]


def test_protocol_is_not_mutated_when_candidates_load() -> None:
    protocol = load_protocol()
    before = copy.deepcopy(protocol)

    signed_value_candidates_from_protocol(
        protocol
    )

    assert protocol == before


def test_ridge_candidates_use_exact_frozen_alphas() -> None:
    grouped = candidates_by_family(
        load_protocol()
    )

    assert [
        candidate.params
        for candidate in grouped["Ridge"]
    ] == [
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
    ]

    models = [
        build_signed_value_regressor(
            candidate,
            random_state=1729,
        )
        for candidate in grouped["Ridge"]
    ]

    assert all(
        isinstance(model, Ridge)
        for model in models
    )

    assert [
        model.alpha
        for model in models
    ] == [0.1, 1.0, 10.0]


def test_hgb_constructor_preserves_existing_repository_settings() -> None:
    candidate = candidates_by_family(
        load_protocol()
    )["HistGradientBoostingRegressor"][0]

    model = build_signed_value_regressor(
        candidate,
        random_state=2718,
    )

    assert isinstance(
        model,
        HistGradientBoostingRegressor,
    )
    assert model.loss == "squared_error"
    assert model.early_stopping is False
    assert model.random_state == 2718

    for name, value in candidate.params.items():
        assert model.get_params()[name] == value


def test_random_forest_uses_frozen_grid_and_deterministic_seed() -> None:
    candidate = candidates_by_family(
        load_protocol()
    )["RandomForestRegressor"][1]

    model = build_signed_value_regressor(
        candidate,
        random_state=3141,
    )

    assert isinstance(
        model,
        RandomForestRegressor,
    )
    assert model.random_state == 3141
    assert model.n_jobs == 1

    for name, value in candidate.params.items():
        assert model.get_params()[name] == value


@pytest.mark.parametrize(
    "family",
    SIGNED_VALUE_FAMILY_ORDER,
)
def test_each_family_fits_and_predicts_signed_targets(
    family: str,
) -> None:
    rng = np.random.default_rng(8111)

    x = rng.normal(size=(120, 6))
    y = (
        0.8 * x[:, 0]
        - 0.5 * x[:, 1]
        + rng.normal(scale=0.2, size=120)
    )

    candidate = candidates_by_family(
        load_protocol()
    )[family][0]

    model = build_signed_value_regressor(
        candidate,
        random_state=5772,
    )

    model.fit(x, y)
    prediction = model.predict(x[:12])

    assert prediction.shape == (12,)
    assert np.all(np.isfinite(prediction))


def test_seeded_nonlinear_constructor_is_reproducible() -> None:
    rng = np.random.default_rng(1729)

    x = rng.normal(size=(80, 4))
    y = x[:, 0] - x[:, 1]

    candidate = candidates_by_family(
        load_protocol()
    )["RandomForestRegressor"][0]

    first = build_signed_value_regressor(
        candidate,
        random_state=2718,
    )
    second = build_signed_value_regressor(
        candidate,
        random_state=2718,
    )

    first.fit(x, y)
    second.fit(x, y)

    np.testing.assert_allclose(
        first.predict(x),
        second.predict(x),
        rtol=0.0,
        atol=0.0,
    )


def test_unknown_family_fails_closed() -> None:
    candidate = SignedValueModelCandidate(
        family="UnsupportedRegressor",
        candidate_index=0,
        params={"alpha": 1.0},
    )

    with pytest.raises(
        ValueError,
        match="Unsupported signed-value family",
    ):
        build_signed_value_regressor(
            candidate,
            random_state=1729,
        )
