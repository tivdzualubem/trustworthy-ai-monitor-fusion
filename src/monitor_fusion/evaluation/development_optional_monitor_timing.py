"""Frozen v2 development optional-monitor timing contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from monitor_fusion.evaluation.grouped_resampling import (
    resolve_effective_groups,
)
from monitor_fusion.evaluation.latency_measurement import (
    apply_timeout_cap,
    latency_contract_from_protocol,
    prespecified_sha256_order,
)


QWEN_MODEL_ID = "Qwen/Qwen3Guard-Gen-4B"
QWEN_MODEL_REVISION = (
    "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
)
QWEN_DTYPE = "torch.bfloat16"
QWEN_BATCH_SIZE = 1
QWEN_MAX_NEW_TOKENS = 128

TARGET_DEVICE = "cuda:0"
TARGET_GPU_SUBSTRING = "T4"

EXPECTED_DEVELOPMENT_ROWS = 1687

EXPECTED_DEVELOPMENT_DATASET_SHA256 = (
    "f752fe74c7d3cc254ce7864382defeb4"
    "45982438f14195c81823641132d0b336"
)

ORIGINAL_QWEN_SCORING_RUN_MANIFEST_SHA256 = (
    "cde6d6f890fabf7311349fc4e51eb6113780d0f"
    "efdc0776118ac030c14283816"
)

EXPECTED_DEVELOPMENT_SPLIT_COUNTS = {
    "policy_train": 844,
    "policy_selection": 421,
    "calibration": 422,
}

REQUIRED_INPUT_COLUMNS = (
    "example_id",
    "prompt",
    "response",
    "split",
    "group_id",
    "pair_id",
)


@dataclass(frozen=True)
class DevelopmentOptionalMonitorContract:
    allowed_splits: tuple[str, ...]
    warmup_requests: int
    timeout_ms: float
    measurement_stage: str
    stream_order: str


def development_optional_monitor_contract(
    protocol: Mapping[str, object],
) -> DevelopmentOptionalMonitorContract:
    """Load and fail closed on the frozen development timing contract."""

    latency_contract = latency_contract_from_protocol(protocol)

    heterogeneous = protocol.get("heterogeneous_cost")

    if not isinstance(heterogeneous, Mapping):
        raise ValueError(
            "heterogeneous_cost section is required"
        )

    measurement = heterogeneous.get(
        "development_cost_target_measurement"
    )

    if not isinstance(measurement, Mapping):
        raise ValueError(
            "development_cost_target_measurement is required"
        )

    required_exact = {
        "allowed_examples":
            "materialized_v2_development_view_only",
        "measurement_stage":
            "optional_monitor",
        "fresh_rows_allowed":
            False,
        "protected_legacy_rows_allowed":
            False,
        "labels_required":
            False,
        "labels_may_not_control_measurement_order":
            True,
        "raw_per_example_timing_required":
            True,
        "required_before_cost_predictor_fit":
            True,
        "warmup_excluded_from_training_targets":
            True,
        "historical_score_generation_latency_may_replace_measurement":
            False,
    }

    for key, expected in required_exact.items():
        if measurement.get(key) != expected:
            raise ValueError(
                f"development timing contract mismatch: {key}"
            )

    allowed_splits = tuple(
        str(value)
        for value in measurement["allowed_splits"]
    )

    expected_splits = (
        "policy_train",
        "policy_selection",
        "calibration",
    )

    if allowed_splits != expected_splits:
        raise ValueError(
            "development timing splits differ from frozen protocol"
        )

    timeout_ms = float(measurement["timeout_ms"])

    if timeout_ms != latency_contract.optional_monitor_timeout_ms:
        raise ValueError(
            "development timeout differs from latency contract"
        )

    warmup_requests = int(
        measurement["warmup_requests"]
    )

    if warmup_requests != latency_contract.warmup_requests:
        raise ValueError(
            "development warmups differ from latency contract"
        )

    stream_order = str(measurement["stream_order"])

    if stream_order != latency_contract.stream_order:
        raise ValueError(
            "development stream order differs from latency contract"
        )

    return DevelopmentOptionalMonitorContract(
        allowed_splits=allowed_splits,
        warmup_requests=warmup_requests,
        timeout_ms=timeout_ms,
        measurement_stage=str(
            measurement["measurement_stage"]
        ),
        stream_order=stream_order,
    )


def prepare_development_timing_frame(
    frame: pd.DataFrame,
    *,
    protocol: Mapping[str, object],
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
    expected_split_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Validate and deterministically order the development stream."""

    contract = development_optional_monitor_contract(
        protocol
    )

    if expected_split_counts is None:
        expected_split_counts = (
            EXPECTED_DEVELOPMENT_SPLIT_COUNTS
        )

    missing = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"development timing input missing columns: {missing}"
        )

    working = frame.loc[
        :,
        list(REQUIRED_INPUT_COLUMNS),
    ].copy()

    if len(working) != expected_rows:
        raise ValueError(
            "unexpected development timing row count: "
            f"{len(working)} != {expected_rows}"
        )

    if working["example_id"].isna().any():
        raise ValueError(
            "example_id contains missing values"
        )

    working["example_id"] = (
        working["example_id"].astype(str)
    )

    if working["example_id"].duplicated().any():
        raise ValueError(
            "example_id must be unique"
        )

    for column in ("prompt", "response"):
        if working[column].isna().any():
            raise ValueError(
                f"{column} contains missing values"
            )

    working["split"] = (
        working["split"].astype(str)
    )

    observed_counts = {
        str(key): int(value)
        for key, value in (
            working["split"]
            .value_counts()
            .to_dict()
            .items()
        )
    }

    expected_counts = {
        str(key): int(value)
        for key, value
        in expected_split_counts.items()
    }

    if observed_counts != expected_counts:
        raise ValueError(
            "development split counts differ from frozen view: "
            f"{observed_counts}"
        )

    if set(observed_counts) != set(
        contract.allowed_splits
    ):
        raise ValueError(
            "development timing input contains "
            "an unexpected split"
        )

    effective_groups = resolve_effective_groups(
        working["group_id"].tolist(),
        working["pair_id"].tolist(),
        working["example_id"].tolist(),
    )

    order = prespecified_sha256_order(
        working["example_id"].tolist()
    )

    ordered = working.iloc[
        order.tolist()
    ].reset_index(drop=True)

    ordered["effective_group"] = np.asarray(
        effective_groups,
        dtype=object,
    )[order]

    ordered["stream_position"] = np.arange(
        len(ordered),
        dtype=np.int64,
    )

    return ordered


def cap_optional_monitor_latency(
    observed_latency_ms: float,
    *,
    protocol: Mapping[str, object],
) -> tuple[float, bool]:
    """Apply the protocol-exact optional-monitor timeout target."""

    contract = development_optional_monitor_contract(
        protocol
    )

    capped = apply_timeout_cap(
        observed_latency_ms,
        timeout_cap_ms=contract.timeout_ms,
    )

    return (
        float(capped.recorded_latency_ms),
        bool(capped.timed_out),
    )
