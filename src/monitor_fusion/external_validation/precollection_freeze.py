from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


MANIFEST_ID = "external_validation_precollection_manifest_v1"
PROTOCOL_ID = "safety_monitor_external_validation_preregistration_v1"


class PrecollectionFreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise PrecollectionFreezeError(f"missing protected file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_precollection_manifest(
    *,
    root: Path,
    protected_paths: list[str],
    component_freeze_commit: str,
    blockers: list[str],
) -> dict[str, Any]:
    root = Path(root).resolve()

    if not component_freeze_commit.strip():
        raise PrecollectionFreezeError(
            "component_freeze_commit must be non-empty"
        )

    if blockers:
        raise PrecollectionFreezeError(
            "cannot create a complete precollection manifest while blockers remain"
        )

    if not protected_paths:
        raise PrecollectionFreezeError(
            "protected_paths must be non-empty"
        )

    if len(set(protected_paths)) != len(protected_paths):
        raise PrecollectionFreezeError(
            "protected_paths must not contain duplicates"
        )

    files: dict[str, str] = {}

    for relative in sorted(protected_paths):
        relative_path = Path(relative)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PrecollectionFreezeError(
                f"protected path must be repository-relative: {relative}"
            )

        files[relative] = sha256_file(root / relative_path)

    return {
        "manifest_id": MANIFEST_ID,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "blockers": [],
        "component_freeze_commit": component_freeze_commit,
        "execution_boundary_at_freeze": {
            "W0_collection_started": False,
            "fresh_monitor_scoring_started": False,
            "confirmation_domain_retuning_allowed": False,
            "existing_data_discovery_closed": True,
        },
        "protected_files": files,
    }


def verify_precollection_manifest(
    *,
    root: Path,
    manifest: dict[str, Any],
) -> None:
    root = Path(root).resolve()

    if manifest.get("manifest_id") != MANIFEST_ID:
        raise PrecollectionFreezeError("unexpected manifest_id")

    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise PrecollectionFreezeError("unexpected protocol_id")

    if manifest.get("complete") is not True:
        raise PrecollectionFreezeError(
            "precollection manifest is not complete"
        )

    blockers = manifest.get("blockers")
    if blockers != []:
        raise PrecollectionFreezeError(
            "precollection manifest contains unresolved blockers"
        )

    freeze_commit = manifest.get("component_freeze_commit")
    if not isinstance(freeze_commit, str) or not freeze_commit.strip():
        raise PrecollectionFreezeError(
            "component_freeze_commit is missing"
        )

    boundary = manifest.get("execution_boundary_at_freeze")
    if not isinstance(boundary, dict):
        raise PrecollectionFreezeError(
            "execution_boundary_at_freeze is missing"
        )

    expected_boundary = {
        "W0_collection_started": False,
        "fresh_monitor_scoring_started": False,
        "confirmation_domain_retuning_allowed": False,
        "existing_data_discovery_closed": True,
    }

    if boundary != expected_boundary:
        raise PrecollectionFreezeError(
            "execution boundary does not match the required pre-W0 state"
        )

    protected = manifest.get("protected_files")
    if not isinstance(protected, dict) or not protected:
        raise PrecollectionFreezeError(
            "protected_files must be a non-empty mapping"
        )

    for relative, expected_hash in sorted(protected.items()):
        if not isinstance(relative, str) or not relative:
            raise PrecollectionFreezeError(
                "protected file paths must be non-empty strings"
            )

        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PrecollectionFreezeError(
                f"invalid protected file path: {relative}"
            )

        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_hash)
        ):
            raise PrecollectionFreezeError(
                f"invalid SHA256 for protected file: {relative}"
            )

        actual_hash = sha256_file(root / relative_path)

        if actual_hash != expected_hash:
            raise PrecollectionFreezeError(
                f"protected file hash mismatch: {relative}"
            )


def assert_w0_start_allowed(
    *,
    root: Path,
    manifest: dict[str, Any],
    fresh_monitor_scoring_started: bool,
) -> None:
    if fresh_monitor_scoring_started is not False:
        raise PrecollectionFreezeError(
            "W0 cannot start after fresh monitor scoring has started"
        )

    verify_precollection_manifest(
        root=root,
        manifest=manifest,
    )
