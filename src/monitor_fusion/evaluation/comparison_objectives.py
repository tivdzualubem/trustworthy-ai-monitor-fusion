"""Separate iso-cost equivalence from Pareto-dominance claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HISTORICAL_V2_MARGIN_STATUS = (
    "prespecified_engineering_margin_not_externally_justified"
)


@dataclass(frozen=True)
class EquivalenceMarginEvidence:
    margin_ms: float
    justification: str
    externally_justified: bool

    def require_confirmatory_use(self) -> None:
        if (
            not np.isfinite(self.margin_ms)
            or self.margin_ms <= 0.0
        ):
            raise ValueError(
                "equivalence margin must be finite and positive"
            )

        if not self.justification.strip():
            raise ValueError(
                "equivalence-margin justification is required"
            )

        if not self.externally_justified:
            raise ValueError(
                "confirmatory iso-cost inference requires an "
                "externally justified equivalence margin"
            )


@dataclass(frozen=True)
class ParetoEvidence:
    recall_difference_lower_bound: float
    cost_difference_upper_bound_ms: float
    fpr_upper_bound: float
    maximum_fpr: float

    @property
    def passes(self) -> bool:
        return bool(
            self.recall_difference_lower_bound > 0.0
            and self.cost_difference_upper_bound_ms <= 0.0
            and self.fpr_upper_bound <= self.maximum_fpr
        )


def historical_v2_margin_evidence(
    margin_ms: float,
) -> EquivalenceMarginEvidence:
    """Label the frozen historical margin without upgrading its status."""

    if not np.isfinite(margin_ms) or margin_ms <= 0.0:
        raise ValueError(
            "historical margin must be finite and positive"
        )

    return EquivalenceMarginEvidence(
        margin_ms=float(margin_ms),
        justification=HISTORICAL_V2_MARGIN_STATUS,
        externally_justified=False,
    )


def evaluate_pareto_dominance(
    *,
    recall_difference_lower_bound: float,
    cost_difference_upper_bound_ms: float,
    fpr_upper_bound: float,
    maximum_fpr: float = 0.05,
) -> ParetoEvidence:
    """Evaluate the professor-specified Pareto claim without equivalence."""

    values = (
        recall_difference_lower_bound,
        cost_difference_upper_bound_ms,
        fpr_upper_bound,
        maximum_fpr,
    )

    if not all(np.isfinite(value) for value in values):
        raise ValueError("Pareto evidence must be finite")

    if not 0.0 <= fpr_upper_bound <= 1.0:
        raise ValueError(
            "fpr_upper_bound must lie in [0, 1]"
        )

    if not 0.0 < maximum_fpr < 1.0:
        raise ValueError(
            "maximum_fpr must lie strictly between zero and one"
        )

    return ParetoEvidence(
        recall_difference_lower_bound=float(
            recall_difference_lower_bound
        ),
        cost_difference_upper_bound_ms=float(
            cost_difference_upper_bound_ms
        ),
        fpr_upper_bound=float(fpr_upper_bound),
        maximum_fpr=float(maximum_fpr),
    )
