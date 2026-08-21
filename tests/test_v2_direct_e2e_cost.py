from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.direct_e2e_cost import (
    COMPONENT_SUM_DIAGNOSTIC,
    DIRECT_E2E_ESTIMAND,
    PolicyCostEvidence,
    calibrate_direct_e2e_threshold_mixture,
    realized_direct_e2e_cost,
)


def test_direct_e2e_evidence_is_explicit():
    costs = PolicyCostEvidence.direct_e2e(
        [10.0, 11.0],
        [110.0, 211.0],
    )

    assert costs.estimand == DIRECT_E2E_ESTIMAND


def test_component_sum_is_diagnostic_only():
    costs = PolicyCostEvidence.component_sum_diagnostic(
        [10.0, 11.0],
        [110.0, 211.0],
    )

    assert costs.estimand == COMPONENT_SUM_DIAGNOSTIC

    with pytest.raises(
        ValueError,
        match="summed component latency is diagnostic only",
    ):
        calibrate_direct_e2e_threshold_mixture(
            [0.9, 0.1],
            costs,
            absolute_cost_budget_ms=60.0,
            policy_id="threshold_distance",
            hash_seed=1729,
        )


def test_direct_e2e_calibration_accepts_measured_e2e():
    costs = PolicyCostEvidence.direct_e2e(
        [10.0, 10.0, 10.0, 10.0],
        [110.0, 210.0, 310.0, 410.0],
    )

    policy = calibrate_direct_e2e_threshold_mixture(
        [0.9, 0.8, 0.3, 0.1],
        costs,
        absolute_cost_budget_ms=60.0,
        policy_id="signed_value_cost_aware",
        hash_seed=1729,
    )

    assert (
        policy.calibration_expected_total_cost_ms
        == pytest.approx(60.0)
    )


def test_realized_cost_rejects_component_sum():
    costs = PolicyCostEvidence.component_sum_diagnostic(
        [10.0, 11.0],
        [110.0, 211.0],
    )

    with pytest.raises(
        ValueError,
        match="directly measured wall-clock",
    ):
        realized_direct_e2e_cost(
            [False, True],
            costs,
        )


def test_realized_direct_e2e_cost_uses_selected_path():
    costs = PolicyCostEvidence.direct_e2e(
        [10.0, 11.0, 12.0, 13.0],
        [110.0, 211.0, 312.0, 413.0],
    )

    result = realized_direct_e2e_cost(
        [False, True, False, True],
        costs,
    )

    np.testing.assert_allclose(
        result,
        [10.0, 211.0, 12.0, 413.0],
    )
