"""Predicate-filtered legacy development-view materialization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from monitor_fusion.evaluation.data_boundary import (
    DEVELOPMENT_VIEW_CACHE,
    DEVELOPMENT_VIEW_DATASET,
    DEVELOPMENT_VIEW_DIRECTORY,
    DEVELOPMENT_VIEW_MANIFEST,
    DataBoundaryError,
    assert_observed_splits,
    validate_input_paths,
    validate_split_request,
)


@dataclass(frozen=True)
class DevelopmentViewArtifact:
    """Verified development-only output artifact."""

    relative_path: str
    row_count: int
    split_counts: dict[str, int]
    sha256: str


@dataclass(frozen=True)
class DevelopmentViewResult:
    """Completed development-view materialization."""

    output_directory: Path
    manifest_path: Path
    artifacts: tuple[DevelopmentViewArtifact, ...]


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _string_column(
    table: pa.Table,
    column_name: str,
) -> tuple[str, ...]:
    if column_name not in table.schema.names:
        raise DataBoundaryError(
            f"Required column is missing: {column_name}"
        )

    raw_values = table[column_name].to_pylist()
    values: list[str] = []

    for value in raw_values:
        if value is None:
            raise DataBoundaryError(
                f"{column_name} contains null values"
            )

        if isinstance(value, bytes):
            normalized = value.decode("utf-8")
        else:
            normalized = str(value)

        if not normalized:
            raise DataBoundaryError(
                f"{column_name} contains an empty value"
            )

        values.append(normalized)

    return tuple(values)


def _validate_development_table(
    table: pa.Table,
    *,
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    example_ids = _string_column(
        table,
        "example_id",
    )
    splits = _string_column(
        table,
        "split",
    )

    if len(set(example_ids)) != len(example_ids):
        raise DataBoundaryError(
            "Development output contains duplicate example_id values"
        )

    assert_observed_splits(
        sorted(set(splits)),
        phase="development",
        protocol=protocol,
    )

    return example_ids, splits


def read_predicate_filtered_development_table(
    path: Path,
    *,
    development_splits: tuple[str, ...],
    protocol: Mapping[str, Any],
) -> pa.Table:
    """Read only authorized splits through the parquet reader predicate."""

    validate_split_request(
        development_splits,
        phase="development",
        protocol=protocol,
    )

    try:
        table = pq.read_table(
            path,
            filters=[
                (
                    "split",
                    "in",
                    list(development_splits),
                )
            ],
        )
    except Exception as exc:
        raise DataBoundaryError(
            f"Predicate-filtered parquet read failed: {path}"
        ) from exc

    _validate_development_table(
        table,
        protocol=protocol,
    )

    return table


def _artifact_record(
    table: pa.Table,
    *,
    temporary_path: Path,
    final_relative_path: str,
    protocol: Mapping[str, Any],
) -> DevelopmentViewArtifact:
    pq.write_table(
        table,
        temporary_path,
        compression="zstd",
    )

    verified = pq.read_table(temporary_path)
    _, verified_splits = _validate_development_table(
        verified,
        protocol=protocol,
    )

    if verified.num_rows != table.num_rows:
        raise DataBoundaryError(
            "Written development artifact row count changed"
        )

    if verified.schema.names != table.schema.names:
        raise DataBoundaryError(
            "Written development artifact schema changed"
        )

    return DevelopmentViewArtifact(
        relative_path=final_relative_path,
        row_count=int(verified.num_rows),
        split_counts=dict(
            sorted(
                Counter(verified_splits).items()
            )
        ),
        sha256=sha256_file(temporary_path),
    )


def materialize_development_view(
    dataset_path: Path,
    cache_path: Path,
    output_directory: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    root: Path,
) -> DevelopmentViewResult:
    """Materialize the three authorized legacy development splits."""

    expected_output = (
        root / DEVELOPMENT_VIEW_DIRECTORY
    ).resolve()

    if output_directory.resolve() != expected_output:
        raise DataBoundaryError(
            "Development view must use the frozen output directory"
        )

    if output_directory.exists():
        raise FileExistsError(
            f"Development view already exists: {output_directory}"
        )

    if (
        len(protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in protocol_sha256.lower()
        )
    ):
        raise ValueError(
            "protocol_sha256 must be a 64-character hexadecimal digest"
        )

    relative_sources = validate_input_paths(
        [dataset_path, cache_path],
        purpose="development_view_materialization",
        protocol=protocol,
        root=root,
    )

    for source_path in (dataset_path, cache_path):
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Required mixed container is missing: {source_path}"
            )

    development_splits = tuple(
        str(split)
        for split in protocol["scope"][
            "legacy_development_splits"
        ]
    )

    validate_split_request(
        development_splits,
        phase="development",
        protocol=protocol,
    )

    output_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=".v2-development-view-",
            dir=output_directory.parent,
        )
    )

    try:
        dataset_table = (
            read_predicate_filtered_development_table(
                dataset_path,
                development_splits=development_splits,
                protocol=protocol,
            )
        )

        cache_table = (
            read_predicate_filtered_development_table(
                cache_path,
                development_splits=development_splits,
                protocol=protocol,
            )
        )

        dataset_ids, dataset_splits = (
            _validate_development_table(
                dataset_table,
                protocol=protocol,
            )
        )
        cache_ids, cache_splits = (
            _validate_development_table(
                cache_table,
                protocol=protocol,
            )
        )

        dataset_assignment = dict(
            zip(
                dataset_ids,
                dataset_splits,
                strict=True,
            )
        )
        cache_assignment = dict(
            zip(
                cache_ids,
                cache_splits,
                strict=True,
            )
        )

        if dataset_assignment != cache_assignment:
            raise DataBoundaryError(
                "Filtered dataset and score cache do not contain "
                "the same example_id-to-split assignments"
            )

        dataset_temporary_path = (
            temporary_directory
            / Path(DEVELOPMENT_VIEW_DATASET).name
        )
        cache_temporary_path = (
            temporary_directory
            / Path(DEVELOPMENT_VIEW_CACHE).name
        )

        dataset_record = _artifact_record(
            dataset_table,
            temporary_path=dataset_temporary_path,
            final_relative_path=DEVELOPMENT_VIEW_DATASET,
            protocol=protocol,
        )
        cache_record = _artifact_record(
            cache_table,
            temporary_path=cache_temporary_path,
            final_relative_path=DEVELOPMENT_VIEW_CACHE,
            protocol=protocol,
        )

        artifacts = (
            dataset_record,
            cache_record,
        )

        manifest = {
            "artifact": "v2_legacy_development_view",
            "protocol_sha256": protocol_sha256.lower(),
            "materializer": (
                "scripts/materialize_v2_development_view.py"
            ),
            "authorized_splits": list(
                development_splits
            ),
            "source_paths": list(
                relative_sources
            ),
            "source_file_hashes_recorded": False,
            "source_row_counts_recorded": False,
            "protected_rows_materialized": False,
            "outputs": [
                {
                    "path": artifact.relative_path,
                    "row_count": artifact.row_count,
                    "split_counts": artifact.split_counts,
                    "sha256": artifact.sha256,
                }
                for artifact in artifacts
            ],
        }

        manifest_temporary_path = (
            temporary_directory
            / Path(DEVELOPMENT_VIEW_MANIFEST).name
        )

        manifest_temporary_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            temporary_directory,
            output_directory,
        )

    except Exception:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )
        raise

    return DevelopmentViewResult(
        output_directory=output_directory,
        manifest_path=(
            root / DEVELOPMENT_VIEW_MANIFEST
        ),
        artifacts=artifacts,
    )
