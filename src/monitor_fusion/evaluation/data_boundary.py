"""Data-access boundary for the v2 cascade evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class DataBoundaryError(RuntimeError):
    """Raised before an out-of-protocol data access is attempted."""


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = (
    ROOT / "configs/exact_cost_risk_cascade_protocol_v2.json"
)

DEVELOPMENT_VIEW_DIRECTORY = (
    "data/processed/v2_development_view"
)
DEVELOPMENT_VIEW_DATASET = (
    DEVELOPMENT_VIEW_DIRECTORY
    + "/unified_dataset_label_audited_v1.development.parquet"
)
DEVELOPMENT_VIEW_CACHE = (
    DEVELOPMENT_VIEW_DIRECTORY
    + "/monitor_score_cache_v3.development.parquet"
)
DEVELOPMENT_VIEW_MANIFEST = (
    DEVELOPMENT_VIEW_DIRECTORY + "/manifest.json"
)
DEVELOPMENT_VIEW_ARTIFACTS = frozenset(
    {
        DEVELOPMENT_VIEW_DATASET,
        DEVELOPMENT_VIEW_CACHE,
        DEVELOPMENT_VIEW_MANIFEST,
    }
)


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict:
    """Load the frozen protocol without opening any data artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_relative_path(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise DataBoundaryError(
            f"Input is outside the repository boundary: {resolved_path}"
        ) from exc
    return relative.as_posix()


def validate_split_request(
    requested_splits: Iterable[str],
    *,
    phase: str,
    protocol: Mapping | None = None,
) -> tuple[str, ...]:
    """Validate requested split names before a reader is constructed."""
    config = dict(protocol or load_protocol())
    scope = config["scope"]
    requested = tuple(dict.fromkeys(str(item) for item in requested_splits))
    protected = set(scope["protected_legacy_splits"])

    forbidden = sorted(set(requested).intersection(protected))
    if forbidden:
        raise DataBoundaryError(
            "Protected legacy split access denied: " + ", ".join(forbidden)
        )

    phase_allowed = {
        "implementation": set(),
        "development": set(scope["legacy_development_splits"]),
        "fresh_calibration": {
            "fresh_calibration_optimization",
            "fresh_calibration_risk",
        },
        "fresh_confirmatory": {"fresh_confirmatory"},
    }
    if phase not in phase_allowed:
        raise DataBoundaryError(f"Unknown analysis phase: {phase}")

    unexpected = sorted(set(requested).difference(phase_allowed[phase]))
    if unexpected:
        raise DataBoundaryError(
            f"Splits are not authorized for phase {phase}: "
            + ", ".join(unexpected)
        )
    return requested


def validate_input_paths(
    paths: Sequence[Path],
    *,
    purpose: str,
    protocol: Mapping | None = None,
    root: Path = ROOT,
) -> tuple[str, ...]:
    """Authorize paths before callers open them."""
    config = dict(protocol or load_protocol())
    boundary = config["data_boundary"]
    relative_paths = tuple(
        _normalized_relative_path(Path(path), root) for path in paths
    )

    restricted = tuple(
        str(prefix).rstrip("/")
        for prefix in boundary["restricted_path_prefixes"]
    )
    for relative in relative_paths:
        if any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in restricted
        ):
            raise DataBoundaryError(f"Restricted input path denied: {relative}")

    development_allowed = set(
        boundary["permitted_existing_development_artifacts"]
    )
    mixed = set(boundary["sealed_mixed_split_containers"])

    if purpose == "development_analysis":
        authorized = development_allowed.union(
            DEVELOPMENT_VIEW_ARTIFACTS
        )
        unauthorized = sorted(
            set(relative_paths).difference(authorized)
        )
    elif purpose == "development_view_materialization":
        unauthorized = sorted(set(relative_paths).difference(mixed))
    elif purpose == "synthetic_test":
        unauthorized = []
    else:
        raise DataBoundaryError(f"Unknown data-access purpose: {purpose}")

    if unauthorized:
        raise DataBoundaryError(
            f"Inputs are not authorized for {purpose}: "
            + ", ".join(unauthorized)
        )
    return relative_paths


def assert_observed_splits(
    observed_splits: Iterable[str],
    *,
    phase: str,
    protocol: Mapping | None = None,
) -> tuple[str, ...]:
    """Fail closed if an already materialized frame has unexpected splits."""
    observed = validate_split_request(
        observed_splits,
        phase=phase,
        protocol=protocol,
    )
    if phase == "development":
        expected = set((protocol or load_protocol())["scope"]["legacy_development_splits"])
        if set(observed) != expected:
            raise DataBoundaryError(
                "Development view does not contain exactly the authorized splits"
            )
    return observed
