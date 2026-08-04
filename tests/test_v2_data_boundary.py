from __future__ import annotations

from pathlib import Path

import pytest

from monitor_fusion.evaluation.data_boundary import (
    DataBoundaryError,
    assert_observed_splits,
    load_protocol,
    validate_input_paths,
    validate_split_request,
)


def test_protected_legacy_splits_are_denied_before_read() -> None:
    for split in ("final_test", "held_out_shift"):
        with pytest.raises(DataBoundaryError, match="Protected legacy"):
            validate_split_request([split], phase="development")


def test_development_request_accepts_only_the_three_authorized_splits() -> None:
    accepted = validate_split_request(
        ["policy_train", "policy_selection", "calibration"],
        phase="development",
    )
    assert set(accepted) == {"policy_train", "policy_selection", "calibration"}

    with pytest.raises(DataBoundaryError, match="not authorized"):
        validate_split_request(["fresh_confirmatory"], phase="development")


def test_implementation_phase_cannot_request_real_splits() -> None:
    with pytest.raises(DataBoundaryError, match="not authorized"):
        validate_split_request(["policy_train"], phase="implementation")


def test_development_view_must_contain_exactly_authorized_splits() -> None:
    with pytest.raises(DataBoundaryError, match="exactly"):
        assert_observed_splits(["policy_train"], phase="development")


def test_path_guard_denies_restricted_results() -> None:
    protocol = load_protocol()
    root = Path(__file__).resolve().parents[1]
    restricted = root / "reports/final_evaluation/summary.md"
    with pytest.raises(DataBoundaryError, match="Restricted"):
        validate_input_paths(
            [restricted],
            purpose="development_analysis",
            protocol=protocol,
            root=root,
        )


def test_mixed_containers_have_one_authorized_purpose() -> None:
    protocol = load_protocol()
    root = Path(__file__).resolve().parents[1]
    mixed = root / "data/processed/monitor_score_cache_v3.parquet"

    accepted = validate_input_paths(
        [mixed],
        purpose="development_view_materialization",
        protocol=protocol,
        root=root,
    )
    assert accepted == ("data/processed/monitor_score_cache_v3.parquet",)

    with pytest.raises(DataBoundaryError, match="not authorized"):
        validate_input_paths(
            [mixed],
            purpose="development_analysis",
            protocol=protocol,
            root=root,
        )
