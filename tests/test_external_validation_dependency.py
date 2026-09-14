import numpy as np
import pytest

from monitor_fusion.external_validation.dependency import (
    LEXICAL_THRESHOLD,
    SEMANTIC_THRESHOLD,
    base_intent_edges,
    build_dependency_groups,
    exact_text_edges,
    provenance_cluster_id,
    representative_hash,
    select_primary_representatives,
    semantic_edges,
    validate_no_cross_cell_dependencies,
)


def test_frozen_thresholds():
    assert SEMANTIC_THRESHOLD == 0.90
    assert LEXICAL_THRESHOLD == 0.75


def test_same_base_intent_creates_dependency_edge():
    edges = base_intent_edges(["intent_a", "intent_a", "intent_b"])
    assert (0, 1) in edges
    assert (0, 2) not in edges


def test_exact_normalized_text_creates_edge():
    edges = exact_text_edges([
        "Example   Request",
        " example request ",
        "different request",
    ])
    assert (0, 1) in edges


def test_semantic_threshold_is_applied():
    E = np.array([
        [1.0, 0.0],
        [0.95, 0.10],
        [0.0, 1.0],
    ])
    edges = semantic_edges(E)
    assert (0, 1) in edges
    assert (0, 2) not in edges


def test_dependency_groups_are_deterministic():
    ids = ["e1", "e2", "e3"]
    intents = ["i1", "i1", "i3"]
    texts = [
        "short first request",
        "completely different wording",
        "third independent request",
    ]
    E = np.eye(3)

    a = build_dependency_groups(
        example_ids=ids,
        base_intent_ids=intents,
        dependency_texts=texts,
        embeddings=E,
    )
    b = build_dependency_groups(
        example_ids=ids,
        base_intent_ids=intents,
        dependency_texts=texts,
        embeddings=E,
    )

    assert a == b
    assert a[0] == a[1]
    assert a[2] != a[0]


def test_cross_cell_dependency_is_rejected():
    with pytest.raises(ValueError):
        validate_no_cross_cell_dependencies(
            ["g1", "g1"],
            ["A_val", "T"],
        )


def test_same_cell_dependency_is_allowed():
    validate_no_cross_cell_dependencies(
        ["g1", "g1", "g2"],
        ["T", "T", "F"],
    )


def test_representative_selection_is_deterministic():
    ids = ["e1", "e2", "e3"]
    groups = ["g1", "g1", "g2"]

    selected = select_primary_representatives(ids, groups)

    expected = min(["e1", "e2"], key=representative_hash)
    assert selected["g1"] == expected
    assert selected["g2"] == "e3"


def test_human_provenance_cluster():
    assert (
        provenance_cluster_id(source="human", author_id="a17")
        == "human:a17"
    )


def test_model_provenance_cluster():
    assert (
        provenance_cluster_id(
            source="model_generated",
            generator_batch_id="batch04",
        )
        == "model:batch04"
    )


def test_missing_provenance_is_rejected():
    with pytest.raises(ValueError):
        provenance_cluster_id(source="human")

    with pytest.raises(ValueError):
        provenance_cluster_id(source="model_generated")
