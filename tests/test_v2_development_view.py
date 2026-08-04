from __future__ import annotations

import inspect
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from monitor_fusion.evaluation import development_view
from monitor_fusion.evaluation.data_boundary import (
    DEVELOPMENT_VIEW_CACHE,
    DEVELOPMENT_VIEW_DATASET,
    DEVELOPMENT_VIEW_DIRECTORY,
    DEVELOPMENT_VIEW_MANIFEST,
    DataBoundaryError,
    load_protocol,
)
from monitor_fusion.evaluation.development_view import (
    materialize_development_view,
)


DEVELOPMENT_SPLITS = (
    "policy_train",
    "policy_selection",
    "calibration",
)

PROTECTED_SPLITS = (
    "final_test",
    "held_out_shift",
)


def write_mixed_sources(
    root: Path,
    *,
    include_calibration: bool = True,
    mismatched_cache: bool = False,
) -> tuple[Path, Path]:
    dataset_path = (
        root
        / "data/processed"
        / "unified_dataset_label_audited_v1.parquet"
    )
    cache_path = (
        root
        / "data/processed"
        / "monitor_score_cache_v3.parquet"
    )

    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = [
        "policy_train",
        "policy_selection",
    ]

    if include_calibration:
        splits.append("calibration")

    splits.extend(PROTECTED_SPLITS)

    dataset_rows = [
        {
            "example_id": f"example-{split}",
            "split": split,
            "y": index % 2,
            "prompt": f"prompt-{index}",
        }
        for index, split in enumerate(splits)
    ]

    cache_rows = [
        {
            "example_id": f"example-{split}",
            "split": split,
            "rule_score": float(index) / 10.0,
            "rule_latency_ms": float(index + 1),
        }
        for index, split in enumerate(splits)
    ]

    if mismatched_cache:
        cache_rows = [
            {
                **row,
                "example_id": (
                    "mismatched-policy-selection"
                ),
            }
            if row["split"] == "policy_selection"
            else row
            for row in cache_rows
        ]

    pq.write_table(
        pa.Table.from_pylist(dataset_rows),
        dataset_path,
    )
    pq.write_table(
        pa.Table.from_pylist(cache_rows),
        cache_path,
    )

    return dataset_path, cache_path


def run_materialization(
    root: Path,
    *,
    include_calibration: bool = True,
    mismatched_cache: bool = False,
):
    dataset_path, cache_path = write_mixed_sources(
        root,
        include_calibration=include_calibration,
        mismatched_cache=mismatched_cache,
    )

    return materialize_development_view(
        dataset_path,
        cache_path,
        root / DEVELOPMENT_VIEW_DIRECTORY,
        protocol=load_protocol(),
        protocol_sha256="0" * 64,
        root=root,
    )


def test_materializer_excludes_all_protected_rows(
    tmp_path: Path,
) -> None:
    result = run_materialization(tmp_path)

    dataset = pq.read_table(
        tmp_path / DEVELOPMENT_VIEW_DATASET
    ).to_pydict()

    cache = pq.read_table(
        tmp_path / DEVELOPMENT_VIEW_CACHE
    ).to_pydict()

    assert set(dataset["split"]) == set(
        DEVELOPMENT_SPLITS
    )
    assert set(cache["split"]) == set(
        DEVELOPMENT_SPLITS
    )

    for protected_split in PROTECTED_SPLITS:
        assert protected_split not in dataset["split"]
        assert protected_split not in cache["split"]
        assert (
            f"example-{protected_split}"
            not in dataset["example_id"]
        )
        assert (
            f"example-{protected_split}"
            not in cache["example_id"]
        )

    assert result.manifest_path == (
        tmp_path / DEVELOPMENT_VIEW_MANIFEST
    )


def test_manifest_records_only_development_outputs(
    tmp_path: Path,
) -> None:
    run_materialization(tmp_path)

    manifest = json.loads(
        (
            tmp_path / DEVELOPMENT_VIEW_MANIFEST
        ).read_text(encoding="utf-8")
    )

    assert manifest["authorized_splits"] == list(
        DEVELOPMENT_SPLITS
    )
    assert (
        manifest["source_file_hashes_recorded"]
        is False
    )
    assert (
        manifest["source_row_counts_recorded"]
        is False
    )
    assert (
        manifest["protected_rows_materialized"]
        is False
    )

    serialized = json.dumps(manifest)

    for protected_split in PROTECTED_SPLITS:
        assert protected_split not in serialized

    assert {
        output["path"]
        for output in manifest["outputs"]
    } == {
        DEVELOPMENT_VIEW_DATASET,
        DEVELOPMENT_VIEW_CACHE,
    }

    for output in manifest["outputs"]:
        assert output["row_count"] == 3
        assert output["split_counts"] == {
            "calibration": 1,
            "policy_selection": 1,
            "policy_train": 1,
        }
        assert len(output["sha256"]) == 64


def test_source_reads_receive_reader_level_split_predicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, cache_path = write_mixed_sources(
        tmp_path
    )

    real_read_table = pq.read_table
    calls: list[tuple[Path, object]] = []

    def spy_read_table(
        where: object,
        *args: object,
        **kwargs: object,
    ):
        calls.append(
            (
                Path(where),
                kwargs.get("filters"),
            )
        )
        return real_read_table(
            where,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        development_view.pq,
        "read_table",
        spy_read_table,
    )

    materialize_development_view(
        dataset_path,
        cache_path,
        tmp_path / DEVELOPMENT_VIEW_DIRECTORY,
        protocol=load_protocol(),
        protocol_sha256="0" * 64,
        root=tmp_path,
    )

    source_calls = {
        path: filters
        for path, filters in calls
        if path in {dataset_path, cache_path}
    }

    assert set(source_calls) == {
        dataset_path,
        cache_path,
    }

    expected_filter = [
        (
            "split",
            "in",
            list(DEVELOPMENT_SPLITS),
        )
    ]

    assert source_calls[dataset_path] == (
        expected_filter
    )
    assert source_calls[cache_path] == (
        expected_filter
    )


def test_missing_development_split_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DataBoundaryError,
        match="exactly",
    ):
        run_materialization(
            tmp_path,
            include_calibration=False,
        )

    assert not (
        tmp_path / DEVELOPMENT_VIEW_DIRECTORY
    ).exists()


def test_mismatched_source_assignments_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DataBoundaryError,
        match="same example_id-to-split",
    ):
        run_materialization(
            tmp_path,
            mismatched_cache=True,
        )

    assert not (
        tmp_path / DEVELOPMENT_VIEW_DIRECTORY
    ).exists()


def test_existing_output_is_refused_before_source_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, cache_path = write_mixed_sources(
        tmp_path
    )

    output = tmp_path / DEVELOPMENT_VIEW_DIRECTORY
    output.mkdir(parents=True)

    def forbidden_read(*args: object, **kwargs: object):
        raise AssertionError(
            "Source reader should not run"
        )

    monkeypatch.setattr(
        development_view.pq,
        "read_table",
        forbidden_read,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        materialize_development_view(
            dataset_path,
            cache_path,
            output,
            protocol=load_protocol(),
            protocol_sha256="0" * 64,
            root=tmp_path,
        )


def test_output_directory_cannot_be_changed(
    tmp_path: Path,
) -> None:
    dataset_path, cache_path = write_mixed_sources(
        tmp_path
    )

    with pytest.raises(
        DataBoundaryError,
        match="frozen output directory",
    ):
        materialize_development_view(
            dataset_path,
            cache_path,
            tmp_path / "different-output",
            protocol=load_protocol(),
            protocol_sha256="0" * 64,
            root=tmp_path,
        )


def test_materializer_never_converts_mixed_sources_to_pandas() -> None:
    source = inspect.getsource(
        development_view
    )

    assert "to_pandas" not in source
    assert "read_parquet" not in source
    assert "import pandas" not in source
    assert "filters=" in source
