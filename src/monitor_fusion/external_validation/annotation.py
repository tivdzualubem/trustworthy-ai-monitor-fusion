from __future__ import annotations

import hashlib
import math
from collections import Counter

PROTOCOL_ID = "safety_monitor_external_validation_preregistration_v1"
AUDIT_SALT = "unanimous_audit_v1"
AUDIT_FRACTION = 0.10


def majority_label(labels: list[int]) -> int:
    if len(labels) != 3:
        raise ValueError("exactly three primary ratings are required")
    if any(x not in (0, 1) for x in labels):
        raise ValueError("ratings must be binary 0/1")
    return 1 if sum(labels) >= 2 else 0


def is_two_to_one_disagreement(labels: list[int]) -> bool:
    majority_label(labels)
    counts = Counter(labels)
    return sorted(counts.values()) == [1, 2]


def is_unanimous(labels: list[int]) -> bool:
    majority_label(labels)
    return labels[0] == labels[1] == labels[2]


def adjudication_overturn(
    labels: list[int],
    fourth_rater_label: int,
) -> bool:
    if fourth_rater_label not in (0, 1):
        raise ValueError("fourth-rater label must be binary 0/1")
    if not is_two_to_one_disagreement(labels):
        raise ValueError("adjudication applies only to 2-1 disagreements")
    return fourth_rater_label != majority_label(labels)


def audit_overturn(
    labels: list[int],
    fourth_rater_label: int,
) -> bool:
    if fourth_rater_label not in (0, 1):
        raise ValueError("fourth-rater label must be binary 0/1")
    if not is_unanimous(labels):
        raise ValueError("unanimous audit applies only to 3-0 cases")
    return fourth_rater_label != labels[0]


def audit_hash(example_id: str, cell_id: str) -> str:
    if not example_id or not cell_id:
        raise ValueError("example_id and cell_id must be non-empty")
    payload = (
        f"{PROTOCOL_ID}|{AUDIT_SALT}|{cell_id}|{example_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_unanimous_audit(
    example_ids: list[str],
    cell_ids: list[str],
    unanimous_flags: list[bool],
) -> set[str]:
    if not (
        len(example_ids) == len(cell_ids) == len(unanimous_flags)
    ):
        raise ValueError("audit inputs must have equal lengths")

    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example_id values must be unique")

    by_cell: dict[str, list[str]] = {}

    for example_id, cell_id, unanimous in zip(
        example_ids,
        cell_ids,
        unanimous_flags,
    ):
        if unanimous:
            by_cell.setdefault(cell_id, []).append(example_id)

    selected: set[str] = set()

    for cell_id, ids in by_cell.items():
        n_select = math.ceil(AUDIT_FRACTION * len(ids))
        ranked = sorted(ids, key=lambda x: audit_hash(x, cell_id))
        selected.update(ranked[:n_select])

    return selected
