from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports/decision_value_real_data"
ARTIFACT_PATH = (
    OUTPUT_DIR
    / "frozen_prompt_response_embeddings.parquet"
)
MANIFEST_PATH = (
    OUTPUT_DIR
    / "frozen_prompt_response_embedding_manifest.json"
)
RUNTIME_PATH = (
    OUTPUT_DIR
    / "frozen_prompt_response_embedding_batch_runtime.csv"
)


def test_artifact_is_development_only() -> None:
    artifact = pd.read_parquet(ARTIFACT_PATH)
    dataset = pd.read_parquet(
        ROOT
        / "data/processed/unified_dataset_label_audited_v1.parquet"
    )

    development_ids = set(
        dataset.loc[
            dataset["split"].isin(
                [
                    "policy_train",
                    "policy_selection",
                    "calibration",
                ]
            ),
            "example_id",
        ]
    )
    excluded_ids = set(
        dataset.loc[
            dataset["split"].isin(
                ["final_test", "held_out_shift"]
            ),
            "example_id",
        ]
    )

    assert len(artifact) == 1687
    assert artifact["example_id"].is_unique
    assert set(artifact["example_id"]) == development_ids
    assert not set(artifact["example_id"]).intersection(
        excluded_ids
    )


def test_embedding_shape_and_normalization() -> None:
    artifact = pd.read_parquet(ARTIFACT_PATH)

    embedding_columns = [
        column
        for column in artifact.columns
        if column.startswith("embedding_")
    ]
    matrix = artifact[
        embedding_columns
    ].to_numpy(dtype=np.float32)

    assert len(embedding_columns) == 384
    assert matrix.shape == (1687, 384)
    assert np.isfinite(matrix).all()
    assert np.allclose(
        np.linalg.norm(matrix, axis=1),
        1.0,
        atol=2e-5,
        rtol=2e-5,
    )

    forbidden = {
        "y",
        "split",
        "source_dataset",
        "attack_family",
        "group_id",
        "pair_id",
        "prompt",
        "response",
    }
    assert forbidden.isdisjoint(artifact.columns)


def test_complete_token_coverage() -> None:
    artifact = pd.read_parquet(ARTIFACT_PATH)

    assert (
        artifact["complete_content_token_count"]
        > 0
    ).all()
    assert (
        artifact["complete_content_token_count"]
        == artifact["covered_content_token_count"]
    ).all()
    assert (artifact["token_coverage"] == 1.0).all()
    assert artifact["full_text_covered"].all()
    assert not artifact["truncated"].any()
    assert (artifact["chunk_count"] >= 1).all()
    assert (artifact["chunk_count"] > 1).any()


def test_rows_match_frozen_outer_folds() -> None:
    artifact = pd.read_parquet(ARTIFACT_PATH)
    assignments = pd.read_csv(
        OUTPUT_DIR
        / "development_outer_fold_assignments.csv"
    )

    joined = artifact[
        ["example_id", "outer_fold"]
    ].merge(
        assignments,
        on="example_id",
        suffixes=("_artifact", "_frozen"),
        validate="one_to_one",
    )

    assert len(joined) == 1687
    assert (
        joined["outer_fold_artifact"]
        == joined["outer_fold_frozen"]
    ).all()


def test_manifest_and_runtime() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    runtime = pd.read_csv(RUNTIME_PATH)

    assert manifest["status"] == (
        "development_only_complete_text_embeddings_completed"
    )

    assert manifest["scope"]["development_rows"] == 1687
    assert manifest["scope"]["final_test_used"] is False
    assert (
        manifest["scope"]["held_out_shift_used"]
        is False
    )
    assert manifest["scope"]["label_used"] is False

    assert manifest["model"]["resolved_revision"] == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert manifest["model"]["dimension"] == 384

    assert manifest["software"][
        "sentence_transformers"
    ] == "5.6.0"
    assert manifest["software"]["transformers"] == (
        "4.57.6"
    )
    assert manifest["software"]["huggingface_hub"] == (
        "0.36.2"
    )
    assert manifest["software"]["tokenizers"] == (
        "0.22.2"
    )
    assert manifest["software"]["torch"] == (
        "2.13.0+cpu"
    )

    inputs = manifest["inputs"]
    assert inputs["builder_script_path"] == (
        "scripts/build_full_text_prompt_response_embeddings.py"
    )
    assert inputs["builder_script_sha256"]
    assert inputs["embedding_requirements_path"] == (
        "requirements-embedding-cpu.txt"
    )
    assert inputs["embedding_requirements_sha256"]

    method = manifest["long_text_method"]
    assert method["strategy"] == (
        "deterministic_non_overlapping_token_chunks"
    )
    assert method["chunk_overlap_tokens"] == 0
    assert method["truncation_allowed"] is False

    coverage = manifest["full_text_coverage"]
    assert coverage["example_n"] == 1687
    assert coverage["total_chunk_n"] >= 1687
    assert coverage["multi_chunk_example_n"] > 0
    assert (
        coverage["total_content_token_n"]
        == coverage["total_covered_content_token_n"]
    )
    assert (
        coverage["minimum_per_example_token_coverage"]
        == 1.0
    )
    assert (
        coverage["maximum_per_example_token_coverage"]
        == 1.0
    )
    assert coverage["truncated_example_n"] == 0

    assert manifest["runtime"]["device"] == "cpu"
    assert (
        manifest["runtime"]["chunk_encoding_seconds"]
        > 0
    )
    assert (
        manifest["runtime"]["end_to_end_ms_per_example"]
        > 0
    )

    assert runtime["batch_chunk_n"].sum() == (
        coverage["total_chunk_n"]
    )
    assert (runtime["elapsed_seconds"] > 0).all()
