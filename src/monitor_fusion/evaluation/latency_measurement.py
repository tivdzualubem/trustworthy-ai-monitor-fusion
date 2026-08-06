"""Protocol-v2 end-to-end and tail-latency measurement primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import time
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
ResultT = TypeVar("ResultT")


REQUIRED_LATENCY_STAGES = (
    "input_preparation",
    "base_monitors",
    "router_features",
    "router_inference",
    "optional_monitor",
    "final_fusion",
    "end_to_end",
)

REQUIRED_LATENCY_STATISTICS = (
    "count",
    "mean",
    "standard_deviation",
    "median",
    "p90",
    "p95",
    "p99",
    "maximum",
    "timeout_rate",
)

REQUIRED_TAIL_QUANTILES = (
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)

WARMUP_REQUESTS = 20
OPTIONAL_MONITOR_TIMEOUT_MS = 30_000.0
END_TO_END_TIMEOUT_MS = 35_000.0
TAIL_BOOTSTRAP_REPETITIONS = 2_000


@dataclass(frozen=True)
class LatencyMeasurementContract:
    """Frozen latency requirements loaded from protocol v2."""

    hardware_requirement: str
    clock: str
    synchronize_accelerator_stages: bool
    warmup_requests: int
    cold_start_reported_separately: bool
    steady_state_primary: bool
    stream_order: str
    required_stages: tuple[str, ...]
    required_statistics: tuple[str, ...]
    tail_confidence_intervals: str
    raw_per_example_timings_preserved: bool
    optional_monitor_timeout_ms: float
    end_to_end_timeout_ms: float


@dataclass(frozen=True)
class BoundedLatency:
    """One measured latency after applying its frozen timeout cap."""

    recorded_latency_ms: float
    timed_out: bool
    timeout_cap_ms: float


@dataclass(frozen=True)
class PerExampleLatencyRecord:
    """Raw stage timings retained for one prompt-response pair."""

    example_id: str
    effective_group: str
    stage_latency_ms: tuple[tuple[str, float], ...]
    timed_out_stages: tuple[str, ...]
    cold_start: bool
    warmup: bool

    @property
    def primary_total_cost_ms(self) -> float:
        """Return the frozen primary end-to-end cost."""

        return dict(self.stage_latency_ms)["end_to_end"]

    def stage_timings(self) -> dict[str, float]:
        """Return stage timings in their frozen stage order."""

        return dict(self.stage_latency_ms)


@dataclass(frozen=True)
class LatencySummary:
    """Required descriptive latency statistics."""

    count: int
    mean: float
    standard_deviation: float
    median: float
    p90: float
    p95: float
    p99: float
    maximum: float
    timeout_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """Return fields with protocol-exact statistic names."""

        return {
            "count": self.count,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "maximum": self.maximum,
            "timeout_rate": self.timeout_rate,
        }


@dataclass(frozen=True)
class TailLatencyInterval:
    """Grouped-bootstrap confidence interval for one tail statistic."""

    statistic: str
    quantile: float
    estimate_ms: float
    lower_95_ms: float
    upper_95_ms: float
    bootstrap_repetitions: int
    random_seed: int


def latency_contract_from_protocol(
    protocol: Mapping[str, object],
) -> LatencyMeasurementContract:
    """Load and fail closed on deviations from the frozen contract."""

    latency = protocol.get("latency_measurement")
    cost = protocol.get("heterogeneous_cost")

    if not isinstance(latency, Mapping):
        raise ValueError(
            "protocol latency_measurement section is required"
        )

    if not isinstance(cost, Mapping):
        raise ValueError(
            "protocol heterogeneous_cost section is required"
        )

    contract = LatencyMeasurementContract(
        hardware_requirement=str(latency["hardware"]),
        clock=str(latency["clock"]),
        synchronize_accelerator_stages=bool(
            latency[
                "accelerator_synchronization_before_and_after_"
                "each_timed_accelerator_stage"
            ]
        ),
        warmup_requests=int(latency["warmup_requests"]),
        cold_start_reported_separately=bool(
            latency["cold_start_reported_separately"]
        ),
        steady_state_primary=bool(
            latency["steady_state_primary"]
        ),
        stream_order=str(latency["stream_order"]),
        required_stages=tuple(
            str(value)
            for value in latency["required_stages"]
        ),
        required_statistics=tuple(
            str(value)
            for value in latency["required_statistics"]
        ),
        tail_confidence_intervals=str(
            latency["tail_confidence_intervals"]
        ),
        raw_per_example_timings_preserved=bool(
            latency["raw_per_example_timings_preserved"]
        ),
        optional_monitor_timeout_ms=float(
            cost["optional_monitor_timeout_ms"]
        ),
        end_to_end_timeout_ms=float(
            cost["end_to_end_timeout_ms"]
        ),
    )

    expected = {
        "clock": "monotonic_high_resolution",
        "warmup_requests": WARMUP_REQUESTS,
        "stream_order": (
            "prespecified_sha256_order_"
            "independent_of_labels_and_scores"
        ),
        "required_stages": REQUIRED_LATENCY_STAGES,
        "required_statistics": REQUIRED_LATENCY_STATISTICS,
        "tail_confidence_intervals": (
            "group_bootstrap_with_2000_repetitions"
        ),
        "optional_monitor_timeout_ms": (
            OPTIONAL_MONITOR_TIMEOUT_MS
        ),
        "end_to_end_timeout_ms": END_TO_END_TIMEOUT_MS,
    }

    observed = {
        "clock": contract.clock,
        "warmup_requests": contract.warmup_requests,
        "stream_order": contract.stream_order,
        "required_stages": contract.required_stages,
        "required_statistics": contract.required_statistics,
        "tail_confidence_intervals": (
            contract.tail_confidence_intervals
        ),
        "optional_monitor_timeout_ms": (
            contract.optional_monitor_timeout_ms
        ),
        "end_to_end_timeout_ms": (
            contract.end_to_end_timeout_ms
        ),
    }

    if observed != expected:
        raise ValueError(
            "latency contract differs from frozen protocol v2"
        )

    if not contract.hardware_requirement.strip():
        raise ValueError(
            "hardware requirement must not be empty"
        )

    required_true = (
        contract.synchronize_accelerator_stages,
        contract.cold_start_reported_separately,
        contract.steady_state_primary,
        contract.raw_per_example_timings_preserved,
    )

    if not all(required_true):
        raise ValueError(
            "required latency safeguards must all be enabled"
        )

    return contract


def _identifier_vector(
    values: Iterable[object],
    *,
    name: str,
) -> tuple[str, ...]:
    identifiers = tuple(str(value) for value in values)

    if not identifiers:
        raise ValueError(f"{name} must not be empty")

    if any(not value.strip() for value in identifiers):
        raise ValueError(
            f"{name} contains an empty identifier"
        )

    return identifiers


def _finite_nonnegative_vector(
    values: Iterable[object],
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    if bool(np.any(array < 0.0)):
        raise ValueError(
            f"{name} must be nonnegative"
        )

    return array


def prespecified_sha256_order(
    example_ids: Iterable[object],
) -> IntArray:
    """Return the label- and score-independent stream order."""

    identifiers = _identifier_vector(
        example_ids,
        name="example_ids",
    )

    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "example_ids must be unique"
        )

    ordered = sorted(
        range(len(identifiers)),
        key=lambda index: (
            hashlib.sha256(
                identifiers[index].encode("utf-8")
            ).digest(),
            identifiers[index],
        ),
    )

    return np.asarray(ordered, dtype=np.int64)


def measure_monotonic_stage(
    operation: Callable[[], ResultT],
    *,
    accelerator_stage: bool,
    synchronize: Callable[[], None] | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> tuple[ResultT, float]:
    """Measure one successful stage using a monotonic clock.

    Accelerator stages require synchronization immediately before
    and after the timed operation.
    """

    if accelerator_stage and synchronize is None:
        raise ValueError(
            "accelerator stages require a synchronization callable"
        )

    clock = clock_ns or time.perf_counter_ns

    if accelerator_stage:
        assert synchronize is not None
        synchronize()

    start_ns = int(clock())
    result = operation()

    if accelerator_stage:
        assert synchronize is not None
        synchronize()

    end_ns = int(clock())

    if end_ns < start_ns:
        raise RuntimeError(
            "monotonic clock moved backwards"
        )

    return result, (end_ns - start_ns) / 1_000_000.0


def apply_timeout_cap(
    elapsed_latency_ms: float,
    *,
    timeout_cap_ms: float,
    timeout_reported: bool = False,
) -> BoundedLatency:
    """Record the timeout cap whenever the stage times out."""

    elapsed = float(elapsed_latency_ms)
    cap = float(timeout_cap_ms)

    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError(
            "elapsed_latency_ms must be finite and nonnegative"
        )

    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError(
            "timeout_cap_ms must be finite and positive"
        )

    timed_out = bool(timeout_reported) or elapsed >= cap

    return BoundedLatency(
        recorded_latency_ms=cap if timed_out else elapsed,
        timed_out=timed_out,
        timeout_cap_ms=cap,
    )


def build_per_example_latency_record(
    *,
    example_id: object,
    effective_group: object,
    stage_latency_ms: Mapping[str, object],
    timed_out_stages: Iterable[str] = (),
    cold_start: bool = False,
    warmup: bool = False,
) -> PerExampleLatencyRecord:
    """Validate and preserve all frozen per-example stage timings."""

    identifier = str(example_id)
    group = str(effective_group)

    if not identifier.strip():
        raise ValueError(
            "example_id must not be empty"
        )

    if not group.strip():
        raise ValueError(
            "effective_group must not be empty"
        )

    observed_stages = set(stage_latency_ms)
    expected_stages = set(REQUIRED_LATENCY_STAGES)

    if observed_stages != expected_stages:
        missing = sorted(expected_stages - observed_stages)
        unexpected = sorted(observed_stages - expected_stages)

        raise ValueError(
            "stage timings must contain exactly the frozen "
            f"required stages; missing={missing}, "
            f"unexpected={unexpected}"
        )

    timeout_set = {str(stage) for stage in timed_out_stages}

    if not timeout_set.issubset(
        {"optional_monitor", "end_to_end"}
    ):
        raise ValueError(
            "only optional_monitor and end_to_end have "
            "frozen timeout caps"
        )

    recorded: list[tuple[str, float]] = []
    recorded_timeouts: list[str] = []

    for stage in REQUIRED_LATENCY_STAGES:
        value = float(stage_latency_ms[stage])

        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{stage} latency must be finite and nonnegative"
            )

        if stage == "optional_monitor":
            bounded = apply_timeout_cap(
                value,
                timeout_cap_ms=OPTIONAL_MONITOR_TIMEOUT_MS,
                timeout_reported=stage in timeout_set,
            )
        elif stage == "end_to_end":
            bounded = apply_timeout_cap(
                value,
                timeout_cap_ms=END_TO_END_TIMEOUT_MS,
                timeout_reported=stage in timeout_set,
            )
        else:
            bounded = BoundedLatency(
                recorded_latency_ms=value,
                timed_out=False,
                timeout_cap_ms=float("inf"),
            )

        recorded.append(
            (
                stage,
                bounded.recorded_latency_ms,
            )
        )

        if bounded.timed_out:
            recorded_timeouts.append(stage)

    return PerExampleLatencyRecord(
        example_id=identifier,
        effective_group=group,
        stage_latency_ms=tuple(recorded),
        timed_out_stages=tuple(recorded_timeouts),
        cold_start=bool(cold_start),
        warmup=bool(warmup),
    )


def summarize_latency(
    latency_ms: Iterable[object],
    timed_out: Iterable[object],
) -> LatencySummary:
    """Compute every frozen descriptive latency statistic."""

    values = _finite_nonnegative_vector(
        latency_ms,
        name="latency_ms",
    )

    timeout_flags = np.asarray(
        [bool(value) for value in timed_out],
        dtype=np.bool_,
    )

    if timeout_flags.ndim != 1:
        raise ValueError(
            "timed_out must be one-dimensional"
        )

    if len(timeout_flags) != len(values):
        raise ValueError(
            "latency_ms and timed_out lengths differ"
        )

    return LatencySummary(
        count=int(len(values)),
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values, ddof=0)),
        median=float(np.quantile(values, 0.50)),
        p90=float(np.quantile(values, 0.90)),
        p95=float(np.quantile(values, 0.95)),
        p99=float(np.quantile(values, 0.99)),
        maximum=float(np.max(values)),
        timeout_rate=float(np.mean(timeout_flags)),
    )


def group_bootstrap_tail_intervals(
    latency_ms: Iterable[object],
    effective_groups: Iterable[object],
    *,
    random_seed: int,
    repetitions: int = TAIL_BOOTSTRAP_REPETITIONS,
) -> tuple[TailLatencyInterval, ...]:
    """Compute grouped-bootstrap p90, p95, and p99 intervals."""

    values = _finite_nonnegative_vector(
        latency_ms,
        name="latency_ms",
    )

    groups = _identifier_vector(
        effective_groups,
        name="effective_groups",
    )

    if len(groups) != len(values):
        raise ValueError(
            "latency_ms and effective_groups lengths differ"
        )

    if isinstance(random_seed, bool) or not isinstance(
        random_seed,
        int,
    ):
        raise ValueError(
            "random_seed must be an integer"
        )

    if random_seed < 0:
        raise ValueError(
            "random_seed must be nonnegative"
        )

    if repetitions != TAIL_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            "protocol v2 requires exactly 2000 repetitions"
        )

    unique_groups = tuple(sorted(set(groups)))

    if len(unique_groups) < 2:
        raise ValueError(
            "group bootstrap requires at least two groups"
        )

    group_indices = {
        group: np.flatnonzero(
            np.asarray(groups, dtype=object) == group
        )
        for group in unique_groups
    }

    quantiles = np.asarray(
        [
            quantile
            for _, quantile in REQUIRED_TAIL_QUANTILES
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(
        (
            repetitions,
            len(REQUIRED_TAIL_QUANTILES),
        ),
        dtype=np.float64,
    )

    for repetition in range(repetitions):
        sampled_group_positions = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )

        sampled_indices = np.concatenate(
            [
                group_indices[
                    unique_groups[int(position)]
                ]
                for position in sampled_group_positions
            ]
        )

        bootstrap[repetition] = np.quantile(
            values[sampled_indices],
            quantiles,
        )

    estimates = np.quantile(values, quantiles)
    intervals: list[TailLatencyInterval] = []

    for column, (name, quantile) in enumerate(
        REQUIRED_TAIL_QUANTILES
    ):
        lower, upper = np.quantile(
            bootstrap[:, column],
            [0.025, 0.975],
        )

        intervals.append(
            TailLatencyInterval(
                statistic=name,
                quantile=quantile,
                estimate_ms=float(estimates[column]),
                lower_95_ms=float(lower),
                upper_95_ms=float(upper),
                bootstrap_repetitions=repetitions,
                random_seed=random_seed,
            )
        )

    return tuple(intervals)
