from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


SEMANTIC_THRESHOLD = 0.90
LEXICAL_THRESHOLD = 0.75
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

PROTOCOL_ID = "safety_monitor_external_validation_preregistration_v1"
REPRESENTATIVE_SALT = "dependency_rep_v1"


def normalize_dependency_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _pair_edges_from_equal_values(values: list[str]) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)

    for i, value in enumerate(values):
        if value:
            groups[value].append(i)

    edges: set[tuple[int, int]] = set()

    for indices in groups.values():
        if len(indices) <= 1:
            continue
        anchor = indices[0]
        for j in indices[1:]:
            edges.add((anchor, j))

    return edges


def base_intent_edges(base_intent_ids: list[str]) -> set[tuple[int, int]]:
    return _pair_edges_from_equal_values(
        [str(x).strip() for x in base_intent_ids]
    )


def exact_text_edges(dependency_texts: list[str]) -> set[tuple[int, int]]:
    normalized = [normalize_dependency_text(x) for x in dependency_texts]
    return _pair_edges_from_equal_values(normalized)


def lexical_edges(
    dependency_texts: list[str],
    threshold: float = LEXICAL_THRESHOLD,
) -> set[tuple[int, int]]:
    if not dependency_texts:
        return set()

    vectorizer = CountVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(5, 5),
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.int32,
    )

    try:
        matrix = vectorizer.fit_transform(dependency_texts).tocsr()
    except ValueError:
        # No valid 5-grams exist.
        return set()

    sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    product = (matrix @ matrix.T).tocoo()

    edges: set[tuple[int, int]] = set()

    for i, j, intersection in zip(
        product.row.tolist(),
        product.col.tolist(),
        product.data.tolist(),
    ):
        if i >= j:
            continue

        union = sizes[i] + sizes[j] - float(intersection)
        if union <= 0:
            continue

        jaccard = float(intersection) / union
        if jaccard >= threshold:
            edges.add((int(i), int(j)))

    return edges


def semantic_edges(
    embeddings: np.ndarray,
    threshold: float = SEMANTIC_THRESHOLD,
) -> set[tuple[int, int]]:
    matrix = np.asarray(embeddings, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if len(matrix) == 0:
        return set()

    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain non-finite values")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms <= 0):
        raise ValueError("embeddings contain zero-norm rows")

    normalized = matrix / norms
    similarity = normalized @ normalized.T

    i, j = np.triu_indices(len(normalized), k=1)
    keep = similarity[i, j] >= threshold

    return set(zip(i[keep].tolist(), j[keep].tolist()))


def _connected_components(
    n: int,
    edges: set[tuple[int, int]],
) -> list[list[int]]:
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)

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
        union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)

    for i in range(n):
        groups[find(i)].append(i)

    return list(groups.values())


def _dependency_group_id(member_example_ids: list[str]) -> str:
    canonical = "\n".join(sorted(member_example_ids))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"depgrp_{digest[:16]}"


def build_dependency_groups(
    *,
    example_ids: list[str],
    base_intent_ids: list[str],
    dependency_texts: list[str],
    embeddings: np.ndarray,
) -> list[str]:
    n = len(example_ids)

    if not (
        len(base_intent_ids) == n
        and len(dependency_texts) == n
        and len(embeddings) == n
    ):
        raise ValueError("all dependency inputs must have equal row counts")

    if len(set(example_ids)) != n:
        raise ValueError("example_id values must be unique")

    edges = set()
    edges |= base_intent_edges(base_intent_ids)
    edges |= exact_text_edges(dependency_texts)
    edges |= lexical_edges(dependency_texts)
    edges |= semantic_edges(embeddings)

    result = [""] * n

    for members in _connected_components(n, edges):
        group_id = _dependency_group_id(
            [example_ids[i] for i in members]
        )
        for i in members:
            result[i] = group_id

    assert all(result)
    return result


def validate_no_cross_cell_dependencies(
    dependency_group_ids: list[str],
    cell_ids: list[str],
) -> None:
    if len(dependency_group_ids) != len(cell_ids):
        raise ValueError("group and cell arrays must have equal length")

    cells_by_group: dict[str, set[str]] = defaultdict(set)

    for group_id, cell_id in zip(dependency_group_ids, cell_ids):
        cells_by_group[group_id].add(cell_id)

    violations = {
        group_id: sorted(cells)
        for group_id, cells in cells_by_group.items()
        if len(cells) > 1
    }

    if violations:
        raise ValueError(
            f"dependency groups cross validation cells: {violations}"
        )


def representative_hash(example_id: str) -> str:
    payload = (
        f"{PROTOCOL_ID}|{REPRESENTATIVE_SALT}|{example_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_primary_representatives(
    example_ids: list[str],
    dependency_group_ids: list[str],
) -> dict[str, str]:
    if len(example_ids) != len(dependency_group_ids):
        raise ValueError("example and group arrays must have equal length")

    members: dict[str, list[str]] = defaultdict(list)

    for example_id, group_id in zip(example_ids, dependency_group_ids):
        members[group_id].append(example_id)

    return {
        group_id: min(ids, key=representative_hash)
        for group_id, ids in members.items()
    }


def provenance_cluster_id(
    *,
    source: str,
    author_id: str | None = None,
    generator_batch_id: str | None = None,
) -> str:
    if source == "human":
        if not author_id:
            raise ValueError("human source requires author_id")
        return f"human:{author_id}"

    if source == "model_generated":
        if not generator_batch_id:
            raise ValueError(
                "model_generated source requires generator_batch_id"
            )
        return f"model:{generator_batch_id}"

    raise ValueError(f"unknown source: {source}")
