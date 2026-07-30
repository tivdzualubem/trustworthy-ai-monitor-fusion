#!/usr/bin/env python3
"""Build complete-text frozen prompt-response embeddings.

The frozen input text is tokenized without truncation. Long inputs are divided
into deterministic, non-overlapping content-token chunks. Each chunk is
encoded with the same frozen Sentence Transformer revision. Raw chunk vectors
are combined using content-token-count weights, then the final example vector
is L2-normalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs/decision_value_real_data_protocol_v1.json"
)
DATASET_PATH = (
    ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
)
FOLD_ASSIGNMENTS_PATH = (
    ROOT
    / "reports/decision_value_real_data/"
    "development_outer_fold_assignments.csv"
)
OUTPUT_DIR = ROOT / "reports/decision_value_real_data"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REQUESTED_REVISION = "1110a24"
RESOLVED_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
EXPECTED_DIMENSION = 384
INPUT_TEMPLATE = "[PROMPT]\n{prompt}\n[RESPONSE]\n{response}"

EXPECTED_SOFTWARE = {
    "sentence-transformers": "5.6.0",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "tokenizers": "0.22.2",
    "torch": "2.13.0+cpu",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()




def validate_software_environment() -> None:
    installed = {
        package: version(package)
        for package in EXPECTED_SOFTWARE
    }

    mismatches = {
        package: {
            "expected": EXPECTED_SOFTWARE[package],
            "installed": installed[package],
        }
        for package in EXPECTED_SOFTWARE
        if installed[package] != EXPECTED_SOFTWARE[package]
    }

    if mismatches:
        raise RuntimeError(
            "Embedding software versions do not match the "
            f"frozen CPU environment: {mismatches}"
        )

    if torch.cuda.is_available():
        raise RuntimeError(
            "The frozen embedding run must use CPU-only PyTorch"
        )

def load_protocol() -> dict[str, Any]:
    return json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    embedding = protocol[
        "predictor_families"
    ]["frozen_embedding"]

    expected = {
        "model_id": MODEL_ID,
        "requested_revision": REQUESTED_REVISION,
        "dimension": EXPECTED_DIMENSION,
        "full_text_coverage_required": True,
        "truncation_allowed": False,
        "long_text_strategy": (
            "deterministic_non_overlapping_token_chunks"
        ),
        "chunk_size_rule": (
            "model_max_seq_length minus tokenizer special-token count"
        ),
        "chunk_overlap_tokens": 0,
        "chunk_aggregation": (
            "content_token_count_weighted_mean_then_l2_normalize"
        ),
        "minimum_required_token_coverage": 1.0,
    }

    for key, expected_value in expected.items():
        actual = embedding.get(key)
        if actual != expected_value:
            raise RuntimeError(
                f"Protocol mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )


def verify_model_revision() -> dict[str, Any]:
    info = HfApi(token=False).model_info(
        repo_id=MODEL_ID,
        revision=RESOLVED_REVISION,
    )

    if info.sha != RESOLVED_REVISION:
        raise RuntimeError(
            f"Unexpected model revision: {info.sha}"
        )
    if not RESOLVED_REVISION.startswith(
        REQUESTED_REVISION
    ):
        raise RuntimeError(
            "Resolved revision does not match requested revision"
        )

    return {
        "repo_id": info.id,
        "sha": info.sha,
        "last_modified": (
            info.last_modified.isoformat()
            if info.last_modified is not None
            else None
        ),
    }


def load_development_frame(
    protocol: dict[str, Any],
) -> pd.DataFrame:
    dataset = pd.read_parquet(DATASET_PATH)
    assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH)

    required = {
        "example_id",
        "prompt",
        "response",
        "split",
    }
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(
            f"Dataset missing columns: {sorted(missing)}"
        )

    if not dataset["example_id"].is_unique:
        raise RuntimeError("Dataset example_id is not unique")
    if not assignments["example_id"].is_unique:
        raise RuntimeError(
            "Fold assignments are not unique"
        )

    development_splits = set(
        protocol["scope"]["development_splits"]
    )
    excluded_splits = set(
        protocol["scope"]["excluded_splits"]
    )

    development = dataset.loc[
        dataset["split"].isin(development_splits),
        ["example_id", "prompt", "response"],
    ].copy()

    expected_rows = int(
        protocol["scope"]["expected_development_rows"]
    )
    if len(development) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(development)}"
        )

    excluded_ids = set(
        dataset.loc[
            dataset["split"].isin(excluded_splits),
            "example_id",
        ]
    )

    if set(development["example_id"]) != set(
        assignments["example_id"]
    ):
        raise RuntimeError(
            "Development rows do not match frozen fold assignments"
        )

    if set(development["example_id"]).intersection(
        excluded_ids
    ):
        raise RuntimeError(
            "Excluded row entered embedding inputs"
        )

    frame = development.merge(
        assignments,
        on="example_id",
        how="inner",
        validate="one_to_one",
    ).sort_values(
        ["outer_fold", "example_id"]
    ).reset_index(drop=True)

    if frame["prompt"].isna().any():
        raise RuntimeError("Missing prompt text")
    if frame["response"].isna().any():
        raise RuntimeError("Missing response text")

    return frame


def render_texts(frame: pd.DataFrame) -> list[str]:
    texts = [
        INPUT_TEMPLATE.format(
            prompt=str(prompt),
            response=str(response),
        )
        for prompt, response in zip(
            frame["prompt"],
            frame["response"],
        )
    ]

    if len(texts) != len(frame):
        raise RuntimeError("Rendered row count changed")
    if any(not text.strip() for text in texts):
        raise RuntimeError("Empty rendered text detected")

    return texts


def build_chunks(
    model: SentenceTransformer,
    texts: list[str],
) -> tuple[
    list[list[int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    tokenizer = model.tokenizer

    required_methods = [
        "prepare_for_model",
        "num_special_tokens_to_add",
        "pad",
    ]
    for method in required_methods:
        if not hasattr(tokenizer, method):
            raise RuntimeError(
                f"Tokenizer lacks required v4 method: {method}"
            )

    model_max_sequence_length = int(
        model.max_seq_length
    )
    special_token_count = int(
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )
    maximum_content_tokens = (
        model_max_sequence_length
        - special_token_count
    )

    if maximum_content_tokens <= 0:
        raise RuntimeError(
            "No room remains for content tokens"
        )

    chunks: list[list[int]] = []
    chunk_example_indices: list[int] = []

    complete_token_count = np.zeros(
        len(texts),
        dtype=np.int64,
    )
    chunk_count = np.zeros(
        len(texts),
        dtype=np.int32,
    )

    for example_index, text in enumerate(texts):
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        if not token_ids:
            raise RuntimeError(
                f"Example {example_index} has no content tokens"
            )

        complete_token_count[example_index] = len(
            token_ids
        )
        represented = 0

        for start in range(
            0,
            len(token_ids),
            maximum_content_tokens,
        ):
            chunk = token_ids[
                start : start + maximum_content_tokens
            ]
            if not chunk:
                raise RuntimeError("Empty chunk created")

            chunks.append(
                [int(token_id) for token_id in chunk]
            )
            chunk_example_indices.append(example_index)
            chunk_count[example_index] += 1
            represented += len(chunk)

        if represented != len(token_ids):
            raise RuntimeError(
                "Chunking failed to preserve all content tokens"
            )

    if np.any(chunk_count < 1):
        raise RuntimeError(
            "At least one example has no chunk"
        )

    return (
        chunks,
        np.asarray(
            chunk_example_indices,
            dtype=np.int32,
        ),
        complete_token_count,
        chunk_count,
        model_max_sequence_length,
        special_token_count,
        maximum_content_tokens,
    )


def encode_and_aggregate(
    model: SentenceTransformer,
    chunks: list[list[int]],
    chunk_example_indices: np.ndarray,
    example_count: int,
    batch_size: int,
    model_max_sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    tokenizer = model.tokenizer

    weighted_sum = np.zeros(
        (example_count, EXPECTED_DIMENSION),
        dtype=np.float64,
    )
    covered_token_count = np.zeros(
        example_count,
        dtype=np.int64,
    )
    runtime_rows: list[
        dict[str, float | int]
    ] = []

    return_token_type_ids = (
        "token_type_ids"
        in tokenizer.model_input_names
    )

    for batch_index, start in enumerate(
        range(0, len(chunks), batch_size)
    ):
        stop = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:stop]
        batch_example_indices = (
            chunk_example_indices[start:stop]
        )
        content_weights = np.asarray(
            [len(chunk) for chunk in batch_chunks],
            dtype=np.float64,
        )

        started = time.perf_counter()

        prepared: list[dict[str, list[int]]] = []
        for chunk in batch_chunks:
            item = tokenizer.prepare_for_model(
                chunk,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=True,
                return_token_type_ids=(
                    return_token_type_ids
                ),
            )

            input_ids = list(item["input_ids"])
            attention_mask = list(
                item["attention_mask"]
            )

            if len(input_ids) > model_max_sequence_length:
                raise RuntimeError(
                    "Prepared chunk exceeds model limit"
                )
            if len(input_ids) != len(attention_mask):
                raise RuntimeError(
                    "Input IDs and attention mask differ in length"
                )

            clean_item: dict[str, list[int]] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if return_token_type_ids:
                token_type_ids = list(
                    item["token_type_ids"]
                )
                if len(token_type_ids) != len(input_ids):
                    raise RuntimeError(
                        "Token-type IDs have wrong length"
                    )
                clean_item[
                    "token_type_ids"
                ] = token_type_ids

            prepared.append(clean_item)

        features = tokenizer.pad(
            prepared,
            padding=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model(features)
            chunk_embeddings = output[
                "sentence_embedding"
            ]

        batch_embeddings = (
            chunk_embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if batch_embeddings.shape != (
            len(batch_chunks),
            EXPECTED_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected chunk embedding shape"
            )

        for local_index, example_index in enumerate(
            batch_example_indices
        ):
            weight = content_weights[local_index]
            weighted_sum[example_index] += (
                batch_embeddings[local_index] * weight
            )
            covered_token_count[example_index] += int(
                weight
            )

        elapsed_seconds = (
            time.perf_counter() - started
        )
        runtime_rows.append(
            {
                "batch_index": batch_index,
                "start_chunk": start,
                "stop_chunk_exclusive": stop,
                "batch_chunk_n": len(batch_chunks),
                "batch_content_token_n": int(
                    content_weights.sum()
                ),
                "elapsed_seconds": elapsed_seconds,
                "per_chunk_ms": (
                    elapsed_seconds
                    * 1000.0
                    / len(batch_chunks)
                ),
            }
        )

        print(
            f"encoded chunks {start + 1}-{stop}/"
            f"{len(chunks)} in "
            f"{elapsed_seconds:.3f}s",
            flush=True,
        )

    if np.any(covered_token_count <= 0):
        raise RuntimeError(
            "At least one example has no covered tokens"
        )

    weighted_mean = (
        weighted_sum
        / covered_token_count[:, None]
    )
    norms = np.linalg.norm(
        weighted_mean,
        axis=1,
    )

    if np.any(norms <= 0):
        raise RuntimeError(
            "At least one aggregate has zero norm"
        )

    embeddings = (
        weighted_mean
        / norms[:, None]
    ).astype(np.float32)

    return (
        embeddings,
        covered_token_count,
        pd.DataFrame(runtime_rows),
    )


def write_summary(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    coverage = manifest["full_text_coverage"]
    runtime = manifest["runtime"]

    text = f"""# Complete-Text Frozen Prompt-Response Embeddings

This artifact contains frozen embeddings for the 1,687 development examples
only. It excludes `final_test`, `held_out_shift`, labels, source-dataset,
attack-family, group-identifier, and raw-text fields.

## Method

The exact input template is:

```text
[PROMPT]
{{prompt}}
[RESPONSE]
{{response}}
```

The complete text is tokenized without truncation. Content tokens are split
into deterministic, non-overlapping chunks. Raw chunk embeddings are combined
using content-token-count weights, and the final 384-dimensional example
vector is L2-normalized.

- Model: `{manifest["model"]["model_id"]}`
- Resolved revision: `{manifest["model"]["resolved_revision"]}`
- Transformers: `{manifest["software"]["transformers"]}`
- Model sequence length: {coverage["model_max_sequence_length"]}
- Maximum content tokens/chunk:
  {coverage["maximum_content_tokens_per_chunk"]}
- Total chunks: {coverage["total_chunk_n"]}
- Multi-chunk examples: {coverage["multi_chunk_example_n"]}
- Maximum chunks/example: {coverage["maximum_chunks_per_example"]}
- Total content tokens: {coverage["total_content_token_n"]}
- Covered content tokens:
  {coverage["total_covered_content_token_n"]}
- Minimum token coverage:
  {coverage["minimum_per_example_token_coverage"]:.6f}
- Truncated examples: {coverage["truncated_example_n"]}
- End-to-end cost:
  {runtime["end_to_end_ms_per_example"]:.6f} ms/example

This development-only artifact does not itself establish value predictability
and does not itself pass the professor's milestone.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    validate_software_environment()
    protocol = load_protocol()
    validate_protocol(protocol)
    model_info = verify_model_revision()
    frame = load_development_frame(protocol)
    texts = render_texts(frame)

    print("verified revision:", RESOLVED_REVISION)
    print("development rows:", len(frame))
    print("loading frozen model on CPU...", flush=True)

    model_load_started = time.perf_counter()
    model = SentenceTransformer(
        MODEL_ID,
        revision=RESOLVED_REVISION,
        device="cpu",
        trust_remote_code=False,
        token=False,
    )
    model.eval()
    model_load_seconds = (
        time.perf_counter() - model_load_started
    )

    dimension = int(
        model.get_embedding_dimension()
    )
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Unexpected embedding dimension: {dimension}"
        )

    tokenization_started = time.perf_counter()
    (
        chunks,
        chunk_example_indices,
        complete_token_count,
        chunk_count,
        model_max_sequence_length,
        special_token_count,
        maximum_content_tokens,
    ) = build_chunks(
        model=model,
        texts=texts,
    )
    full_text_tokenization_seconds = (
        time.perf_counter() - tokenization_started
    )

    encoding_started = time.perf_counter()
    (
        embeddings,
        covered_token_count,
        runtime_frame,
    ) = encode_and_aggregate(
        model=model,
        chunks=chunks,
        chunk_example_indices=(
            chunk_example_indices
        ),
        example_count=len(frame),
        batch_size=args.chunk_batch_size,
        model_max_sequence_length=(
            model_max_sequence_length
        ),
    )
    chunk_encoding_seconds = (
        time.perf_counter() - encoding_started
    )

    if embeddings.shape != (
        len(frame),
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embedding matrix contains non-finite values"
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )
    if not np.allclose(
        norms,
        1.0,
        atol=2e-5,
        rtol=2e-5,
    ):
        raise RuntimeError(
            "Final embeddings are not L2-normalized"
        )

    if not np.array_equal(
        complete_token_count,
        covered_token_count,
    ):
        raise RuntimeError(
            "Complete and covered token counts differ"
        )

    token_coverage = (
        covered_token_count
        / complete_token_count.astype(np.float64)
    )
    if not np.array_equal(
        token_coverage,
        np.ones_like(token_coverage),
    ):
        raise RuntimeError(
            "Per-example token coverage is not exactly 1.0"
        )

    embedding_columns = [
        f"embedding_{index:03d}"
        for index in range(EXPECTED_DIMENSION)
    ]

    artifact = pd.concat(
        [
            frame[
                ["example_id", "outer_fold"]
            ].reset_index(drop=True),
            pd.DataFrame(
                {
                    "input_text_sha256": [
                        sha256_text(text)
                        for text in texts
                    ],
                    "complete_content_token_count": (
                        complete_token_count
                    ),
                    "covered_content_token_count": (
                        covered_token_count
                    ),
                    "chunk_count": chunk_count,
                    "token_coverage": token_coverage,
                    "full_text_covered": True,
                    "truncated": False,
                }
            ),
            pd.DataFrame(
                embeddings,
                columns=embedding_columns,
            ),
        ],
        axis=1,
    )

    forbidden_columns = {
        "y",
        "split",
        "source_dataset",
        "attack_family",
        "group_id",
        "pair_id",
        "prompt",
        "response",
    }
    if forbidden_columns.intersection(
        artifact.columns
    ):
        raise RuntimeError(
            "Forbidden field entered embedding artifact"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = (
        args.output_dir
        / "frozen_prompt_response_embeddings.parquet"
    )
    runtime_path = (
        args.output_dir
        / "frozen_prompt_response_embedding_batch_runtime.csv"
    )
    manifest_path = (
        args.output_dir
        / "frozen_prompt_response_embedding_manifest.json"
    )
    summary_path = (
        args.output_dir
        / "frozen_prompt_response_embedding_summary.md"
    )

    artifact.to_parquet(
        artifact_path,
        index=False,
    )
    runtime_frame.to_csv(
        runtime_path,
        index=False,
    )

    total_end_to_end_seconds = (
        model_load_seconds
        + full_text_tokenization_seconds
        + chunk_encoding_seconds
    )

    manifest: dict[str, Any] = {
        "artifact": (
            "complete_text_frozen_prompt_response_embeddings_v1"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "development_only_complete_text_embeddings_completed"
        ),
        "scope": {
            "development_splits": sorted(
                protocol["scope"]["development_splits"]
            ),
            "excluded_splits": sorted(
                protocol["scope"]["excluded_splits"]
            ),
            "development_rows": len(frame),
            "excluded_rows_used": False,
            "final_test_used": False,
            "held_out_shift_used": False,
            "label_used": False,
        },
        "model": {
            "model_id": MODEL_ID,
            "requested_revision": REQUESTED_REVISION,
            "resolved_revision": RESOLVED_REVISION,
            "verified_hub_model_info": model_info,
            "dimension": EXPECTED_DIMENSION,
            "input_template": INPUT_TEMPLATE,
            "l2_normalized": True,
            "trust_remote_code": False,
        },
        "long_text_method": {
            "strategy": (
                "deterministic_non_overlapping_token_chunks"
            ),
            "chunk_overlap_tokens": 0,
            "chunk_size_rule": (
                "model_max_seq_length minus tokenizer "
                "special-token count"
            ),
            "aggregation": (
                "content_token_count_weighted_mean_then_l2_normalize"
            ),
            "truncation_allowed": False,
        },
        "full_text_coverage": {
            "example_n": len(frame),
            "model_max_sequence_length": (
                model_max_sequence_length
            ),
            "special_token_count": special_token_count,
            "maximum_content_tokens_per_chunk": (
                maximum_content_tokens
            ),
            "total_chunk_n": len(chunks),
            "multi_chunk_example_n": int(
                np.sum(chunk_count > 1)
            ),
            "maximum_chunks_per_example": int(
                chunk_count.max()
            ),
            "mean_chunks_per_example": float(
                chunk_count.mean()
            ),
            "total_content_token_n": int(
                complete_token_count.sum()
            ),
            "total_covered_content_token_n": int(
                covered_token_count.sum()
            ),
            "minimum_per_example_token_coverage": float(
                token_coverage.min()
            ),
            "maximum_per_example_token_coverage": float(
                token_coverage.max()
            ),
            "truncated_example_n": 0,
            "truncated_example_rate": 0.0,
        },
        "runtime": {
            "device": "cpu",
            "torch_num_threads": (
                torch.get_num_threads()
            ),
            "chunk_batch_size": args.chunk_batch_size,
            "model_load_seconds": model_load_seconds,
            "full_text_tokenization_seconds": (
                full_text_tokenization_seconds
            ),
            "chunk_encoding_seconds": (
                chunk_encoding_seconds
            ),
            "total_end_to_end_seconds": (
                total_end_to_end_seconds
            ),
            "end_to_end_ms_per_example": (
                total_end_to_end_seconds
                * 1000.0
                / len(frame)
            ),
            "chunk_encoding_ms_per_example": (
                chunk_encoding_seconds
                * 1000.0
                / len(frame)
            ),
            "batch_runtime_file": str(
                runtime_path.relative_to(ROOT)
            ),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sentence_transformers": version(
                "sentence-transformers"
            ),
            "transformers": version("transformers"),
            "huggingface_hub": version(
                "huggingface-hub"
            ),
            "tokenizers": version("tokenizers"),
            "torch": torch.__version__,
            "pandas": version("pandas"),
            "numpy": version("numpy"),
        },
        "inputs": {
            "builder_script_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ),
            "builder_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
            "embedding_requirements_path": (
                "requirements-embedding-cpu.txt"
            ),
            "embedding_requirements_sha256": sha256_file(
                ROOT / "requirements-embedding-cpu.txt"
            ),
            "protocol_path": str(
                PROTOCOL_PATH.relative_to(ROOT)
            ),
            "protocol_sha256": sha256_file(
                PROTOCOL_PATH
            ),
            "dataset_path": str(
                DATASET_PATH.relative_to(ROOT)
            ),
            "dataset_sha256": sha256_file(
                DATASET_PATH
            ),
            "fold_assignments_path": str(
                FOLD_ASSIGNMENTS_PATH.relative_to(ROOT)
            ),
            "fold_assignments_sha256": sha256_file(
                FOLD_ASSIGNMENTS_PATH
            ),
            "ordered_example_id_sha256": sha256_text(
                "\n".join(
                    frame["example_id"].astype(str)
                )
            ),
            "ordered_input_text_sha256": sha256_text(
                "\n".join(texts)
            ),
        },
        "outputs": {},
    }

    write_summary(
        summary_path,
        manifest,
    )

    for output_path in [
        artifact_path,
        runtime_path,
        summary_path,
    ]:
        manifest["outputs"][
            str(output_path.relative_to(ROOT))
        ] = sha256_file(output_path)

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== COMPLETE-TEXT EMBEDDING SUMMARY ===")
    print("shape:", embeddings.shape)
    print("total chunks:", len(chunks))
    print(
        "multi-chunk examples:",
        int(np.sum(chunk_count > 1)),
    )
    print(
        "maximum chunks/example:",
        int(chunk_count.max()),
    )
    print(
        "total content tokens:",
        int(complete_token_count.sum()),
    )
    print(
        "covered content tokens:",
        int(covered_token_count.sum()),
    )
    print(
        "minimum token coverage:",
        float(token_coverage.min()),
    )
    print("truncated examples:", 0)
    print(
        "norm range:",
        float(norms.min()),
        float(norms.max()),
    )
    print(
        "chunk encoding seconds:",
        f"{chunk_encoding_seconds:.6f}",
    )
    print(
        "end-to-end ms/example:",
        f"{manifest['runtime']['end_to_end_ms_per_example']:.6f}",
    )
    print(
        "artifact:",
        artifact_path.relative_to(ROOT),
    )
    print(
        "manifest:",
        manifest_path.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
