from __future__ import annotations

import pytest

from monitor_fusion.evaluation.comparison_objectives import (
    HISTORICAL_V2_MARGIN_STATUS,
    EquivalenceMarginEvidence,
    evaluate_pareto_dominance,
    historical_v2_margin_evidence,
)
from monitor_fusion.evaluation.cost_equivalence import (
    primary_equivalence_margin_ms,
)


def test_historical_one_percent_rule_is_preserved() -> None:
    assert primary_equivalence_margin_ms(50.0) == 1.0
    assert primary_equivalence_margin_ms(1000.0) == 10.0


def test_historical_margin_is_not_upgraded_to_external_justification() -> None:
    evidence = historical_v2_margin_evidence(
        primary_equivalence_margin_ms(1000.0)
    )

    assert evidence.margin_ms == 10.0
    assert evidence.justification == HISTORICAL_V2_MARGIN_STATUS
    assert evidence.externally_justified is False

    with pytest.raises(
        ValueError,
        match="externally justified",
    ):
        evidence.require_confirmatory_use()


def test_externally_justified_iso_cost_margin_can_be_used() -> None:
    evidence = EquivalenceMarginEvidence(
        margin_ms=5.0,
        justification="prospectively documented domain criterion",
        externally_justified=True,
    )

    evidence.require_confirmatory_use()


def test_pareto_gain_does_not_require_cost_equivalence() -> None:
    evidence = evaluate_pareto_dominance(
        recall_difference_lower_bound=0.03,
        cost_difference_upper_bound_ms=-25.0,
        fpr_upper_bound=0.04,
        maximum_fpr=0.05,
    )

    assert evidence.passes is True


def test_higher_recall_but_higher_cost_does_not_pass_pareto() -> None:
    evidence = evaluate_pareto_dominance(
        recall_difference_lower_bound=0.03,
        cost_difference_upper_bound_ms=5.0,
        fpr_upper_bound=0.04,
    )

    assert evidence.passes is False


def test_fpr_failure_blocks_pareto_claim() -> None:
    evidence = evaluate_pareto_dominance(
        recall_difference_lower_bound=0.03,
        cost_difference_upper_bound_ms=-10.0,
        fpr_upper_bound=0.06,
        maximum_fpr=0.05,
    )

    assert evidence.passes is False
