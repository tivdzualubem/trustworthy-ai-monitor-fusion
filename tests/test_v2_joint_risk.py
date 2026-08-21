from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.risk_control import (
    RuntimeBoundEvidence,
    bounded_mean_cost_p_value,
    certify_joint_fpr_and_cost,
    exact_binomial_fpr_p_value,
    hoeffding_bentkus_p_value,
)

def enforced_bound(
    bound_ms: float,
) -> RuntimeBoundEvidence:
    return RuntimeBoundEvidence(
        bound_ms=bound_ms,
        enforcement_mechanism="synthetic_test_watchdog",
        operation_terminated_at_bound=True,
        posthoc_clipping_only=False,
    )


def test_exact_binomial_zero_false_positives() -> None:
    actual = exact_binomial_fpr_p_value(
        0,
        100,
        maximum_fpr=0.05,
    )

    assert actual == pytest.approx(0.95**100)


def test_hoeffding_bentkus_is_one_at_or_above_limit() -> None:
    assert hoeffding_bentkus_p_value(
        np.full(100, 0.2),
        risk_limit=0.2,
    ) == pytest.approx(1.0)

    assert hoeffding_bentkus_p_value(
        np.full(100, 0.3),
        risk_limit=0.2,
    ) == pytest.approx(1.0)


def test_joint_candidate_passes_only_when_both_pass() -> None:
    labels = np.concatenate(
        [
            np.zeros(500, dtype=int),
            np.ones(100, dtype=int),
        ]
    )

    decisions = labels.copy()
    costs = np.full(len(labels), 100.0)

    result = certify_joint_fpr_and_cost(
        labels,
        decisions,
        costs,
        candidate_id="safe_low_cost",
        candidate_count=2,
        absolute_cost_budget_ms=200.0,
        normalization_bound_ms=1000.0,
        runtime_bound_evidence=enforced_bound(1000.0),
    )

    assert result.empirical_fpr == 0.0
    assert result.mean_total_cost_ms == 100.0
    assert result.joint_p_value == max(
        result.fpr_p_value,
        result.mean_cost_p_value,
    )
    assert result.bonferroni_threshold == 0.025
    assert result.certified is True


def test_point_estimates_alone_do_not_certify() -> None:
    labels = np.array([0] * 10 + [1] * 2)
    decisions = labels.copy()
    costs = np.full(len(labels), 100.0)

    result = certify_joint_fpr_and_cost(
        labels,
        decisions,
        costs,
        candidate_id="too_small",
        candidate_count=1,
        absolute_cost_budget_ms=200.0,
        normalization_bound_ms=1000.0,
        runtime_bound_evidence=enforced_bound(1000.0),
    )

    assert result.empirical_fpr <= 0.05
    assert result.mean_total_cost_ms <= 200.0
    assert result.certified is False


def test_failed_fpr_blocks_joint_certificate() -> None:
    labels = np.concatenate(
        [
            np.zeros(500, dtype=int),
            np.ones(100, dtype=int),
        ]
    )

    decisions = np.concatenate(
        [
            np.ones(100, dtype=int),
            np.zeros(400, dtype=int),
            np.ones(100, dtype=int),
        ]
    )

    costs = np.full(len(labels), 100.0)

    result = certify_joint_fpr_and_cost(
        labels,
        decisions,
        costs,
        candidate_id="high_fpr",
        candidate_count=1,
        absolute_cost_budget_ms=200.0,
        normalization_bound_ms=1000.0,
        runtime_bound_evidence=enforced_bound(1000.0),
    )

    assert result.empirical_fpr == pytest.approx(0.2)
    assert result.certified is False


def test_failed_cost_blocks_joint_certificate() -> None:
    labels = np.concatenate(
        [
            np.zeros(500, dtype=int),
            np.ones(100, dtype=int),
        ]
    )

    decisions = labels.copy()
    costs = np.full(len(labels), 300.0)

    result = certify_joint_fpr_and_cost(
        labels,
        decisions,
        costs,
        candidate_id="high_cost",
        candidate_count=1,
        absolute_cost_budget_ms=200.0,
        normalization_bound_ms=1000.0,
        runtime_bound_evidence=enforced_bound(1000.0),
    )

    assert result.fpr_p_value < 0.05
    assert result.mean_cost_p_value == 1.0
    assert result.certified is False


def test_cost_bound_violation_is_rejected() -> None:
    with pytest.raises(ValueError):
        bounded_mean_cost_p_value(
            [100.0, 1001.0],
            absolute_cost_budget_ms=200.0,
            normalization_bound_ms=1000.0,
        )


def test_candidate_count_sets_bonferroni_threshold() -> None:
    labels = np.concatenate(
        [
            np.zeros(500, dtype=int),
            np.ones(100, dtype=int),
        ]
    )

    result = certify_joint_fpr_and_cost(
        labels,
        labels.copy(),
        np.full(len(labels), 100.0),
        candidate_id="candidate",
        candidate_count=10,
        absolute_cost_budget_ms=200.0,
        normalization_bound_ms=1000.0,
        runtime_bound_evidence=enforced_bound(1000.0),
    )

    assert result.bonferroni_threshold == pytest.approx(0.005)


def test_joint_certificate_rejects_missing_runtime_enforcement() -> None:
    labels = np.array([0] * 100 + [1] * 20)

    with pytest.raises(
        ValueError,
        match="mechanically enforced runtime bound",
    ):
        certify_joint_fpr_and_cost(
            labels,
            labels.copy(),
            np.full(len(labels), 100.0),
            candidate_id="missing_bound",
            candidate_count=1,
            absolute_cost_budget_ms=200.0,
            normalization_bound_ms=1000.0,
        )


def test_joint_certificate_rejects_posthoc_clipping() -> None:
    labels = np.array([0] * 100 + [1] * 20)

    evidence = RuntimeBoundEvidence(
        bound_ms=1000.0,
        enforcement_mechanism="posthoc_cap",
        operation_terminated_at_bound=False,
        posthoc_clipping_only=True,
    )

    with pytest.raises(
        ValueError,
        match="post-hoc latency clipping",
    ):
        certify_joint_fpr_and_cost(
            labels,
            labels.copy(),
            np.full(len(labels), 100.0),
            candidate_id="posthoc_only",
            candidate_count=1,
            absolute_cost_budget_ms=200.0,
            normalization_bound_ms=1000.0,
            runtime_bound_evidence=evidence,
        )
