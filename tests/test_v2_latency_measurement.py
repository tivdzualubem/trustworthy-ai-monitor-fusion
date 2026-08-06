from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from monitor_fusion.evaluation.latency_measurement import (
    END_TO_END_TIMEOUT_MS,
    OPTIONAL_MONITOR_TIMEOUT_MS,
    REQUIRED_LATENCY_STAGES,
    REQUIRED_LATENCY_STATISTICS,
    TAIL_BOOTSTRAP_REPETITIONS,
    WARMUP_REQUESTS,
    apply_timeout_cap,
    build_per_example_latency_record,
    group_bootstrap_tail_intervals,
    latency_contract_from_protocol,
    measure_monotonic_stage,
    prespecified_sha256_order,
    summarize_latency,
)


def load_protocol() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/exact_cost_risk_cascade_protocol_v2.json"
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_contract_matches_frozen_protocol() -> None:
    contract = latency_contract_from_protocol(
        load_protocol()
    )

    assert contract.clock == "monotonic_high_resolution"
    assert contract.warmup_requests == WARMUP_REQUESTS
    assert contract.required_stages == (
        REQUIRED_LATENCY_STAGES
    )
    assert contract.required_statistics == (
        REQUIRED_LATENCY_STATISTICS
    )
    assert contract.optional_monitor_timeout_ms == (
        OPTIONAL_MONITOR_TIMEOUT_MS
    )
    assert contract.end_to_end_timeout_ms == (
        END_TO_END_TIMEOUT_MS
    )
    assert contract.synchronize_accelerator_stages
    assert contract.cold_start_reported_separately
    assert contract.steady_state_primary
    assert contract.raw_per_example_timings_preserved


def test_sha256_stream_order_is_deterministic() -> None:
    example_ids = [
        "example-c",
        "example-a",
        "example-b",
    ]

    first = prespecified_sha256_order(example_ids)
    second = prespecified_sha256_order(example_ids)

    expected = sorted(
        range(len(example_ids)),
        key=lambda index: (
            hashlib.sha256(
                example_ids[index].encode("utf-8")
            ).digest(),
            example_ids[index],
        ),
    )

    assert first.tolist() == expected
    np.testing.assert_array_equal(first, second)


def test_sha256_stream_order_rejects_duplicates() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        prespecified_sha256_order(
            ["same", "same"]
        )


def test_accelerator_stage_synchronizes_before_and_after() -> None:
    synchronization_calls: list[str] = []
    ticks = iter(
        [
            1_000_000_000,
            1_003_500_000,
        ]
    )

    result, elapsed_ms = measure_monotonic_stage(
        lambda: "result",
        accelerator_stage=True,
        synchronize=lambda: synchronization_calls.append(
            "sync"
        ),
        clock_ns=lambda: next(ticks),
    )

    assert result == "result"
    assert elapsed_ms == pytest.approx(3.5)
    assert synchronization_calls == ["sync", "sync"]


def test_accelerator_stage_requires_synchronization() -> None:
    with pytest.raises(
        ValueError,
        match="synchronization",
    ):
        measure_monotonic_stage(
            lambda: None,
            accelerator_stage=True,
        )


def test_timeout_cap_records_frozen_cap() -> None:
    optional = apply_timeout_cap(
        31_000.0,
        timeout_cap_ms=OPTIONAL_MONITOR_TIMEOUT_MS,
    )
    end_to_end = apply_timeout_cap(
        20_000.0,
        timeout_cap_ms=END_TO_END_TIMEOUT_MS,
        timeout_reported=True,
    )

    assert optional.timed_out
    assert optional.recorded_latency_ms == (
        OPTIONAL_MONITOR_TIMEOUT_MS
    )

    assert end_to_end.timed_out
    assert end_to_end.recorded_latency_ms == (
        END_TO_END_TIMEOUT_MS
    )


def test_raw_record_preserves_all_required_stages() -> None:
    timings = {
        stage: float(index + 1)
        for index, stage in enumerate(
            REQUIRED_LATENCY_STAGES
        )
    }

    timings["optional_monitor"] = 31_000.0
    timings["end_to_end"] = 36_000.0

    record = build_per_example_latency_record(
        example_id="example-1",
        effective_group="group-1",
        stage_latency_ms=timings,
    )

    assert tuple(
        stage
        for stage, _ in record.stage_latency_ms
    ) == REQUIRED_LATENCY_STAGES

    assert record.stage_timings()[
        "optional_monitor"
    ] == OPTIONAL_MONITOR_TIMEOUT_MS

    assert record.primary_total_cost_ms == (
        END_TO_END_TIMEOUT_MS
    )

    assert record.timed_out_stages == (
        "optional_monitor",
        "end_to_end",
    )


def test_record_requires_exact_stage_set() -> None:
    with pytest.raises(
        ValueError,
        match="required stages",
    ):
        build_per_example_latency_record(
            example_id="example-1",
            effective_group="group-1",
            stage_latency_ms={
                "end_to_end": 10.0,
            },
        )


def test_summary_contains_every_required_statistic() -> None:
    values = np.array(
        [10.0, 20.0, 30.0, 40.0],
        dtype=float,
    )

    summary = summarize_latency(
        values,
        [False, True, False, False],
    )

    assert tuple(summary.as_dict()) == (
        REQUIRED_LATENCY_STATISTICS
    )
    assert summary.count == 4
    assert summary.mean == pytest.approx(25.0)
    assert summary.standard_deviation == pytest.approx(
        np.std(values, ddof=0)
    )
    assert summary.median == pytest.approx(25.0)
    assert summary.p90 == pytest.approx(
        np.quantile(values, 0.90)
    )
    assert summary.p95 == pytest.approx(
        np.quantile(values, 0.95)
    )
    assert summary.p99 == pytest.approx(
        np.quantile(values, 0.99)
    )
    assert summary.maximum == 40.0
    assert summary.timeout_rate == pytest.approx(0.25)


def test_group_bootstrap_tail_intervals_are_reproducible() -> None:
    values = np.arange(1, 41, dtype=float)
    groups = [
        f"group-{index // 2}"
        for index in range(len(values))
    ]

    first = group_bootstrap_tail_intervals(
        values,
        groups,
        random_seed=1729,
    )
    second = group_bootstrap_tail_intervals(
        values,
        groups,
        random_seed=1729,
    )

    assert first == second
    assert [item.statistic for item in first] == [
        "p90",
        "p95",
        "p99",
    ]

    for interval in first:
        assert interval.bootstrap_repetitions == (
            TAIL_BOOTSTRAP_REPETITIONS
        )
        assert np.isfinite(interval.estimate_ms)
        assert np.isfinite(interval.lower_95_ms)
        assert np.isfinite(interval.upper_95_ms)
        assert interval.lower_95_ms <= (
            interval.upper_95_ms
        )


def test_group_bootstrap_requires_2000_repetitions() -> None:
    with pytest.raises(
        ValueError,
        match="exactly 2000",
    ):
        group_bootstrap_tail_intervals(
            [1.0, 2.0],
            ["group-a", "group-b"],
            random_seed=1729,
            repetitions=100,
        )


def test_invalid_latency_values_fail_closed() -> None:
    with pytest.raises(ValueError):
        summarize_latency(
            [1.0, np.inf],
            [False, False],
        )

    with pytest.raises(ValueError):
        summarize_latency(
            [1.0, -1.0],
            [False, False],
        )

    with pytest.raises(
        ValueError,
        match="lengths differ",
    ):
        summarize_latency(
            [1.0, 2.0],
            [False],
        )
