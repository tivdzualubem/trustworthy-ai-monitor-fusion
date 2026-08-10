from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitor_fusion.evaluation.development_optional_monitor_timing import (
    EXPECTED_DEVELOPMENT_DATASET_SHA256,
    EXPECTED_DEVELOPMENT_SPLIT_COUNTS,
    ORIGINAL_QWEN_SCORING_RUN_MANIFEST_SHA256,
    QWEN_BATCH_SIZE,
    QWEN_DTYPE,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    TARGET_DEVICE,
    cap_optional_monitor_latency,
    development_optional_monitor_contract,
    prepare_development_timing_frame,
)


def protocol() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "exact_cost_risk_cascade_protocol_v2.json"
        ).read_text(encoding="utf-8")
    )


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
            ],
            "prompt": [
                "p-a",
                "p-b",
                "p-c",
                "p-d",
                "p-e",
                "p-f",
            ],
            "response": [
                "r-a",
                "r-b",
                "r-c",
                "r-d",
                "r-e",
                "r-f",
            ],
            "split": [
                "policy_train",
                "policy_train",
                "policy_selection",
                "policy_selection",
                "calibration",
                "calibration",
            ],
            "group_id": [
                "g1",
                None,
                None,
                "g4",
                None,
                None,
            ],
            "pair_id": [
                "p1",
                "p2",
                None,
                "p4",
                "p5",
                None,
            ],
        }
    )


def test_frozen_qwen_runtime_identity() -> None:
    assert QWEN_MODEL_ID == (
        "Qwen/Qwen3Guard-Gen-4B"
    )
    assert QWEN_MODEL_REVISION == (
        "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
    )
    assert QWEN_DTYPE == "torch.bfloat16"
    assert QWEN_BATCH_SIZE == 1
    assert QWEN_MAX_NEW_TOKENS == 128
    assert TARGET_DEVICE == "cuda:0"


def test_development_contract_matches_protocol() -> None:
    contract = development_optional_monitor_contract(
        protocol()
    )

    assert contract.allowed_splits == (
        "policy_train",
        "policy_selection",
        "calibration",
    )
    assert contract.warmup_requests == 20
    assert contract.timeout_ms == 30000.0
    assert contract.measurement_stage == (
        "optional_monitor"
    )


def test_preparation_uses_label_independent_fields() -> None:
    frame = synthetic_frame()

    frame["y"] = [
        1,
        0,
        1,
        0,
        1,
        0,
    ]

    first = prepare_development_timing_frame(
        frame,
        protocol=protocol(),
        expected_rows=6,
        expected_split_counts={
            "policy_train": 2,
            "policy_selection": 2,
            "calibration": 2,
        },
    )

    frame["y"] = 1 - frame["y"]

    second = prepare_development_timing_frame(
        frame,
        protocol=protocol(),
        expected_rows=6,
        expected_split_counts={
            "policy_train": 2,
            "policy_selection": 2,
            "calibration": 2,
        },
    )

    assert (
        first["example_id"].tolist()
        == second["example_id"].tolist()
    )

    assert (
        first["effective_group"].tolist()
        == second["effective_group"].tolist()
    )

    assert "y" not in first.columns


def test_preparation_resolves_group_fallback() -> None:
    ordered = prepare_development_timing_frame(
        synthetic_frame(),
        protocol=protocol(),
        expected_rows=6,
        expected_split_counts={
            "policy_train": 2,
            "policy_selection": 2,
            "calibration": 2,
        },
    )

    by_id = ordered.set_index(
        "example_id"
    )

    assert (
        by_id.loc["a", "effective_group"]
        == "group_id:g1"
    )
    assert (
        by_id.loc["b", "effective_group"]
        == "pair_id:p2"
    )
    assert (
        by_id.loc["c", "effective_group"]
        == "example_id:c"
    )


def test_preparation_rejects_protected_split() -> None:
    frame = synthetic_frame()

    frame.loc[
        frame.index[0],
        "split",
    ] = "final_test"

    with pytest.raises(
        ValueError,
        match="split",
    ):
        prepare_development_timing_frame(
            frame,
            protocol=protocol(),
            expected_rows=6,
        expected_split_counts={
            "policy_train": 2,
            "policy_selection": 2,
            "calibration": 2,
        },
        )


def test_preparation_rejects_duplicate_id() -> None:
    frame = synthetic_frame()

    frame.loc[
        frame.index[1],
        "example_id",
    ] = frame.loc[
        frame.index[0],
        "example_id",
    ]

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        prepare_development_timing_frame(
            frame,
            protocol=protocol(),
            expected_rows=6,
        expected_split_counts={
            "policy_train": 2,
            "policy_selection": 2,
            "calibration": 2,
        },
        )


def test_optional_monitor_timeout_cap() -> None:
    recorded, timed_out = (
        cap_optional_monitor_latency(
            31_250.0,
            protocol=protocol(),
        )
    )

    assert recorded == 30000.0
    assert timed_out

    recorded, timed_out = (
        cap_optional_monitor_latency(
            1250.0,
            protocol=protocol(),
        )
    )

    assert recorded == 1250.0
    assert not timed_out


def test_frozen_development_artifact_identity() -> None:
    assert EXPECTED_DEVELOPMENT_DATASET_SHA256 == (
        "f752fe74c7d3cc254ce7864382defeb4"
        "45982438f14195c81823641132d0b336"
    )

    assert EXPECTED_DEVELOPMENT_SPLIT_COUNTS == {
        "policy_train": 844,
        "policy_selection": 421,
        "calibration": 422,
    }

    assert ORIGINAL_QWEN_SCORING_RUN_MANIFEST_SHA256 == (
        "cde6d6f890fabf7311349fc4e51eb6113780d0f"
        "efdc0776118ac030c14283816"
    )


def test_preparation_rejects_wrong_split_counts() -> None:
    frame = synthetic_frame()

    frame.loc[
        frame.index[0],
        "split",
    ] = "policy_selection"

    with pytest.raises(
        ValueError,
        match="split counts",
    ):
        prepare_development_timing_frame(
            frame,
            protocol=protocol(),
            expected_rows=6,
            expected_split_counts={
                "policy_train": 2,
                "policy_selection": 2,
                "calibration": 2,
            },
        )


def test_gpu_runner_reuses_exact_scoring_semantics() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "measure_v2_development_optional_monitor_latency_gpu.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "score_qwen3guard_official_colab.py"
        in runner
    )

    assert (
        "scoring_reference.build_messages"
        in runner
    )

    assert (
        "scoring_reference.parse_guard_output"
        in runner
    )

    assert (
        "EXPECTED_DEVELOPMENT_DATASET_SHA256"
        in runner
    )

    assert (
        "pip_freeze.txt"
        in runner
    )

    # The v2 runner must not introduce a second
    # independently maintained safety-output parser.
    assert "SAFETY_RE =" not in runner
