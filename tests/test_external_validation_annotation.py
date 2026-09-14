import pytest

from monitor_fusion.external_validation.annotation import (
    AUDIT_FRACTION,
    adjudication_overturn,
    audit_hash,
    audit_overturn,
    is_two_to_one_disagreement,
    is_unanimous,
    majority_label,
    select_unanimous_audit,
)


def test_audit_fraction_is_frozen():
    assert AUDIT_FRACTION == 0.10


def test_majority_vote():
    assert majority_label([1, 1, 0]) == 1
    assert majority_label([0, 0, 1]) == 0
    assert majority_label([1, 1, 1]) == 1


def test_majority_requires_three_binary_ratings():
    with pytest.raises(ValueError):
        majority_label([1, 0])
    with pytest.raises(ValueError):
        majority_label([1, 0, 2])


def test_disagreement_detection():
    assert is_two_to_one_disagreement([1, 1, 0])
    assert not is_two_to_one_disagreement([1, 1, 1])


def test_unanimous_detection():
    assert is_unanimous([0, 0, 0])
    assert not is_unanimous([0, 1, 0])


def test_adjudication_overturn():
    assert adjudication_overturn([1, 1, 0], 0)
    assert not adjudication_overturn([1, 1, 0], 1)


def test_adjudication_rejects_unanimous_case():
    with pytest.raises(ValueError):
        adjudication_overturn([1, 1, 1], 0)


def test_audit_overturn():
    assert audit_overturn([0, 0, 0], 1)
    assert not audit_overturn([0, 0, 0], 0)


def test_audit_hash_is_deterministic():
    assert audit_hash("e1", "T") == audit_hash("e1", "T")
    assert audit_hash("e1", "T") != audit_hash("e1", "F")


def test_unanimous_audit_is_exact_10_percent_rounded_up_per_cell():
    ids = [f"e{i:02d}" for i in range(20)]
    cells = ["T"] * 10 + ["F"] * 10
    unanimous = [True] * 20

    selected = select_unanimous_audit(ids, cells, unanimous)

    assert len(selected) == 2
    assert sum(x in selected for x in ids[:10]) == 1
    assert sum(x in selected for x in ids[10:]) == 1


def test_nonunanimous_cases_are_never_selected_for_audit():
    selected = select_unanimous_audit(
        ["e1", "e2"],
        ["T", "T"],
        [False, True],
    )
    assert "e1" not in selected
    assert "e2" in selected
