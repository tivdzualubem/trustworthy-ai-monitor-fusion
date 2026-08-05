"""Frozen signed-value model-family constructors for protocol v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge


SIGNED_VALUE_FAMILY_ORDER = (
    "Ridge",
    "HistGradientBoostingRegressor",
    "RandomForestRegressor",
)


@dataclass(frozen=True)
class SignedValueModelCandidate:
    """One prespecified signed-value model candidate."""

    family: str
    candidate_index: int
    params: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.family}:"
            f"{self.candidate_index:03d}"
        )


def signed_value_candidates_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[SignedValueModelCandidate, ...]:
    """Load the exact frozen candidate grids."""

    raw_families = protocol["model_families"][
        "signed_value_regressors"
    ]

    observed_order = tuple(
        str(item["family"])
        for item in raw_families
    )

    if observed_order != SIGNED_VALUE_FAMILY_ORDER:
        raise ValueError(
            "Signed-value model-family order differs "
            "from the frozen protocol"
        )

    candidates: list[
        SignedValueModelCandidate
    ] = []

    for family_specification in raw_families:
        family = str(
            family_specification["family"]
        )

        grid = family_specification.get(
            "candidate_grid"
        )

        if not isinstance(grid, list) or not grid:
            raise ValueError(
                f"{family} requires a nonempty candidate grid"
            )

        for candidate_index, params in enumerate(grid):
            if not isinstance(params, dict) or not params:
                raise ValueError(
                    f"{family} candidate {candidate_index} "
                    "must contain parameters"
                )

            candidates.append(
                SignedValueModelCandidate(
                    family=family,
                    candidate_index=candidate_index,
                    params=dict(params),
                )
            )

    identifiers = [
        candidate.candidate_id
        for candidate in candidates
    ]

    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "Signed-value candidate identifiers are not unique"
        )

    return tuple(candidates)


def candidates_by_family(
    protocol: Mapping[str, Any],
) -> dict[
    str,
    tuple[SignedValueModelCandidate, ...],
]:
    """Return candidates grouped in frozen family order."""

    candidates = signed_value_candidates_from_protocol(
        protocol
    )

    return {
        family: tuple(
            candidate
            for candidate in candidates
            if candidate.family == family
        )
        for family in SIGNED_VALUE_FAMILY_ORDER
    }


def build_signed_value_regressor(
    candidate: SignedValueModelCandidate,
    *,
    random_state: int,
) -> (
    Ridge
    | HistGradientBoostingRegressor
    | RandomForestRegressor
):
    """Construct exactly one frozen signed-value candidate."""

    if candidate.family == "Ridge":
        return Ridge(
            **candidate.params,
        )

    if (
        candidate.family
        == "HistGradientBoostingRegressor"
    ):
        return HistGradientBoostingRegressor(
            loss="squared_error",
            early_stopping=False,
            random_state=int(random_state),
            **candidate.params,
        )

    if candidate.family == "RandomForestRegressor":
        return RandomForestRegressor(
            random_state=int(random_state),
            n_jobs=1,
            **candidate.params,
        )

    raise ValueError(
        f"Unsupported signed-value family: "
        f"{candidate.family}"
    )
