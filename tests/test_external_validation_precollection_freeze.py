from copy import deepcopy
from pathlib import Path

import pytest

from monitor_fusion.external_validation.precollection_freeze import (
    MANIFEST_ID,
    PROTOCOL_ID,
    PrecollectionFreezeError,
    assert_w0_start_allowed,
    build_precollection_manifest,
    sha256_file,
    verify_precollection_manifest,
)


def _make_files(root: Path) -> list[str]:
    (root / "configs").mkdir()
    (root / "src").mkdir()
    (root / "configs/a.json").write_text('{"a": 1}\n', encoding="utf-8")
    (root / "src/b.py").write_text("VALUE = 2\n", encoding="utf-8")
    return ["configs/a.json", "src/b.py"]


def _manifest(tmp_path: Path):
    paths = _make_files(tmp_path)
    return build_precollection_manifest(
        root=tmp_path,
        protected_paths=paths,
        component_freeze_commit="abc123",
        blockers=[],
    )


def test_manifest_identity_is_frozen():
    assert MANIFEST_ID == "external_validation_precollection_manifest_v1"
    assert PROTOCOL_ID == "safety_monitor_external_validation_preregistration_v1"


def test_sha256_file_is_deterministic(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("abc", encoding="utf-8")
    assert sha256_file(p) == sha256_file(p)


def test_build_manifest_records_sorted_hashes(tmp_path):
    paths = _make_files(tmp_path)
    manifest = build_precollection_manifest(
        root=tmp_path,
        protected_paths=list(reversed(paths)),
        component_freeze_commit="abc123",
        blockers=[],
    )

    assert list(manifest["protected_files"]) == sorted(paths)
    assert manifest["complete"] is True
    assert manifest["blockers"] == []
    assert manifest["execution_boundary_at_freeze"] == {
        "W0_collection_started": False,
        "fresh_monitor_scoring_started": False,
        "confirmation_domain_retuning_allowed": False,
        "existing_data_discovery_closed": True,
    }


def test_manifest_cannot_be_built_with_blockers(tmp_path):
    paths = _make_files(tmp_path)

    with pytest.raises(PrecollectionFreezeError):
        build_precollection_manifest(
            root=tmp_path,
            protected_paths=paths,
            component_freeze_commit="abc123",
            blockers=["monitor_contract_pending"],
        )


def test_manifest_rejects_missing_protected_file(tmp_path):
    with pytest.raises(PrecollectionFreezeError):
        build_precollection_manifest(
            root=tmp_path,
            protected_paths=["missing.json"],
            component_freeze_commit="abc123",
            blockers=[],
        )


def test_manifest_rejects_unsafe_relative_path(tmp_path):
    p = tmp_path / "outside.txt"
    p.write_text("x", encoding="utf-8")

    with pytest.raises(PrecollectionFreezeError):
        build_precollection_manifest(
            root=tmp_path,
            protected_paths=["../outside.txt"],
            component_freeze_commit="abc123",
            blockers=[],
        )


def test_verification_passes_before_change(tmp_path):
    manifest = _manifest(tmp_path)
    verify_precollection_manifest(root=tmp_path, manifest=manifest)


def test_verification_fails_after_protected_file_change(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "src/b.py").write_text("VALUE = 999\n", encoding="utf-8")

    with pytest.raises(PrecollectionFreezeError):
        verify_precollection_manifest(root=tmp_path, manifest=manifest)


def test_verification_fails_if_boundary_is_changed(tmp_path):
    manifest = _manifest(tmp_path)
    changed = deepcopy(manifest)
    changed["execution_boundary_at_freeze"]["W0_collection_started"] = True

    with pytest.raises(PrecollectionFreezeError):
        verify_precollection_manifest(root=tmp_path, manifest=changed)


def test_verification_fails_if_existing_data_discovery_is_reopened(tmp_path):
    manifest = _manifest(tmp_path)
    changed = deepcopy(manifest)
    changed["execution_boundary_at_freeze"]["existing_data_discovery_closed"] = False

    with pytest.raises(PrecollectionFreezeError):
        verify_precollection_manifest(root=tmp_path, manifest=changed)


def test_w0_guard_passes_only_with_valid_manifest_and_no_scoring(tmp_path):
    manifest = _manifest(tmp_path)

    assert_w0_start_allowed(
        root=tmp_path,
        manifest=manifest,
        fresh_monitor_scoring_started=False,
    )

    with pytest.raises(PrecollectionFreezeError):
        assert_w0_start_allowed(
            root=tmp_path,
            manifest=manifest,
            fresh_monitor_scoring_started=True,
        )


def test_duplicate_protected_paths_are_rejected(tmp_path):
    _make_files(tmp_path)

    with pytest.raises(PrecollectionFreezeError):
        build_precollection_manifest(
            root=tmp_path,
            protected_paths=["configs/a.json", "configs/a.json"],
            component_freeze_commit="abc123",
            blockers=[],
        )
