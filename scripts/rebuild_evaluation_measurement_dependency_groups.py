#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
DEV = (
    ROOT
    / "data/processed/v2_development_view/"
    / "unified_dataset_label_audited_v1.development.parquet"
)
META = ROOT / "data/metadata/evaluation_measurement_pilot_v1"
SPEC = META / "dependency_grouping_spec.json"
EMBED = META / "prompt_only_grouping_embeddings.npz"
FROZEN = META / "development_dependency_groups.csv"
NGRAM_N = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory and require exact equality with the frozen CSV.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional regenerated CSV path; the frozen artifact is never overwritten.",
    )
    return p.parse_args()


def progress(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_prompt(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text.lower()).strip()


def stable_components(n: int, edges: set[tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in sorted(edges):
        union(int(a), int(b))

    raw = np.fromiter((find(i) for i in range(n)), dtype=np.int32, count=n)
    members: dict[int, list[int]] = {}
    for idx, root in enumerate(raw.tolist()):
        members.setdefault(int(root), []).append(idx)

    stable = np.empty(n, dtype=object)
    for number, group in enumerate(
        sorted(members.values(), key=lambda values: values[0]),
        start=1,
    ):
        stable[group] = f"depgrp_{number:04d}"
    return stable


def semantic_edges(E: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    S = E @ E.T
    i, j = np.triu_indices(len(E), k=1)
    mask = S[i, j] >= threshold
    return set(zip(i[mask].tolist(), j[mask].tolist()))


def lexical_edges(prompts: pd.Series, threshold: float) -> set[tuple[int, int]]:
    vectorizer = CountVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(NGRAM_N, NGRAM_N),
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.int32,
    )
    B = vectorizer.fit_transform(prompts.fillna("").astype(str)).tocsr()
    sizes = np.asarray(B.sum(axis=1)).ravel().astype(np.float64)

    coo = (B @ B.T).tocoo()
    upper = coo.row < coo.col
    i = coo.row[upper].astype(np.int32)
    j = coo.col[upper].astype(np.int32)
    inter = coo.data[upper].astype(np.float64)
    union = sizes[i] + sizes[j] - inter

    valid = union > 0
    i, j, inter, union = i[valid], j[valid], inter[valid], union[valid]
    jac = inter / union
    keep = jac >= threshold
    return set(zip(i[keep].tolist(), j[keep].tolist()))


def exact_edges(prompts: pd.Series) -> set[tuple[int, int]]:
    normalized = prompts.map(normalize_prompt)
    edges: set[tuple[int, int]] = set()
    for _, indices in normalized.groupby(normalized).groups.items():
        ids = sorted(int(i) for i in indices)
        if len(ids) <= 1:
            continue
        anchor = ids[0]
        for j in ids[1:]:
            edges.add((anchor, j))
    return edges


def main() -> None:
    a = parse_args()

    progress("[1/5] Verifying frozen grouping inputs...")
    for path in (DEV, SPEC, EMBED, FROZEN):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    inputs = spec["inputs"]

    if sha256(DEV) != inputs["development"]["sha256"]:
        raise SystemExit("Development-view hash mismatch.")
    if sha256(EMBED) != inputs["prompt_only_embeddings"]["sha256"]:
        raise SystemExit("Prompt-only embedding hash mismatch.")

    progress("[2/5] Loading development rows and prompt-only embeddings...")
    dev = pd.read_parquet(
        DEV,
        columns=["example_id", "split", "source_dataset", "prompt"],
    ).sort_values("example_id").reset_index(drop=True)

    if dev["split"].astype(str).isin(["final_test", "held_out_shift"]).any():
        raise SystemExit("Protected legacy split detected.")

    # The committed NPZ is SHA-256 verified before pickle is enabled.
    npz = np.load(EMBED, allow_pickle=True)
    ids = npz["example_id"].astype(str)
    E = np.asarray(npz["embedding"], dtype=np.float64)

    if not np.array_equal(ids, dev["example_id"].astype(str).to_numpy()):
        raise SystemExit("Embedding/example ID order mismatch.")
    if E.shape != (len(dev), 384):
        raise SystemExit(f"Unexpected embedding shape: {E.shape}")

    E = E.copy()
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise SystemExit("Invalid embedding norm.")
    E /= norms

    progress("[3/5] Regenerating exact, lexical, and semantic dependency edges...")
    rule = spec["primary_rule"]
    exact = exact_edges(dev["prompt"])
    lexical = lexical_edges(
        dev["prompt"],
        float(rule["lexical_5gram_jaccard_threshold"]),
    )

    variants = {
        "primary_dependency_group": float(rule["semantic_threshold"]),
        "sensitivity_semantic_0_87_group": 0.87,
        "sensitivity_semantic_0_92_group": 0.92,
    }

    output = pd.DataFrame(
        {
            "example_id": dev["example_id"].astype(str),
            "source_dataset": dev["source_dataset"].astype(str),
        }
    )

    for column, threshold in variants.items():
        output[column] = stable_components(
            len(dev),
            exact | lexical | semantic_edges(E, threshold),
        )

    progress("[4/5] Checking exact equality with the frozen grouping artifact...")
    frozen = pd.read_csv(FROZEN, dtype=str)
    candidate = output.astype(str)

    if list(candidate.columns) != list(frozen.columns):
        raise SystemExit(
            f"Column mismatch: candidate={candidate.columns.tolist()} "
            f"frozen={frozen.columns.tolist()}"
        )

    if not candidate.equals(frozen):
        mismatch = np.flatnonzero(
            np.any(candidate.to_numpy() != frozen.to_numpy(), axis=1)
        )
        raise SystemExit(
            "Regenerated dependency groups differ from frozen artifact; "
            f"mismatch_rows={len(mismatch)}, first={mismatch[:10].tolist()}"
        )

    if sha256(FROZEN) != spec["outputs"]["groups_csv"]["sha256"]:
        raise SystemExit("Frozen grouping CSV hash does not match grouping spec.")

    if a.output is not None:
        destination = a.output.expanduser().resolve()
        if destination == FROZEN.resolve():
            raise SystemExit("Refusing to overwrite the frozen grouping artifact.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(destination, index=False)
        print(f"wrote={destination}", flush=True)

    progress("[5/5] Complete.")
    print("DEPENDENCY_GROUPING_REPRODUCTION=PASS", flush=True)
    print(f"rows={len(output)}", flush=True)
    print(
        f"primary_groups={output['primary_dependency_group'].nunique()}",
        flush=True,
    )
    print("protected_splits_opened=false", flush=True)


if __name__ == "__main__":
    main()
