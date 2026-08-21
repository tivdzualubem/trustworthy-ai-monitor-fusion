"""Structural adequacy checks for dependency-aware grouped evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GroupingStructure:
    row_count: int
    group_count: int
    singleton_group_count: int
    repeated_group_count: int
    maximum_group_size: int
    one_group_per_example: bool


def summarize_grouping(
    effective_groups: Iterable[object],
) -> GroupingStructure:
    groups = [str(value).strip() for value in effective_groups]

    if not groups:
        raise ValueError("effective_groups must not be empty")

    if any(not value for value in groups):
        raise ValueError(
            "effective_groups must not contain empty identifiers"
        )

    counts = Counter(groups)

    singleton = sum(
        count == 1
        for count in counts.values()
    )

    repeated = sum(
        count > 1
        for count in counts.values()
    )

    return GroupingStructure(
        row_count=len(groups),
        group_count=len(counts),
        singleton_group_count=singleton,
        repeated_group_count=repeated,
        maximum_group_size=max(counts.values()),
        one_group_per_example=(
            len(counts) == len(groups)
        ),
    )


def require_non_degenerate_grouping(
    effective_groups: Iterable[object],
) -> GroupingStructure:
    """Reject grouping that reduces to one identifier per example."""

    summary = summarize_grouping(effective_groups)

    if summary.one_group_per_example:
        raise ValueError(
            "grouping is effectively one group per example; "
            "this does not protect against template or "
            "near-duplicate dependence"
        )

    return summary
