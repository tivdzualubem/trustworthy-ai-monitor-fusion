from __future__ import annotations

import pytest

from monitor_fusion.evaluation.grouping_adequacy import (
    require_non_degenerate_grouping,
    summarize_grouping,
)


def test_one_group_per_example_is_detected() -> None:
    summary = summarize_grouping(
        ["example:a", "example:b", "example:c"]
    )

    assert summary.row_count == 3
    assert summary.group_count == 3
    assert summary.singleton_group_count == 3
    assert summary.repeated_group_count == 0
    assert summary.maximum_group_size == 1
    assert summary.one_group_per_example is True


def test_dependency_group_structure_is_non_degenerate() -> None:
    summary = require_non_degenerate_grouping(
        [
            "template:a",
            "template:a",
            "template:b",
            "template:b",
            "template:c",
        ]
    )

    assert summary.row_count == 5
    assert summary.group_count == 3
    assert summary.repeated_group_count == 2
    assert summary.maximum_group_size == 2
    assert summary.one_group_per_example is False


def test_example_level_grouping_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="one group per example",
    ):
        require_non_degenerate_grouping(
            [
                "example:1",
                "example:2",
                "example:3",
                "example:4",
            ]
        )


def test_empty_groups_are_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_grouping([])

    with pytest.raises(ValueError):
        summarize_grouping(["a", ""])
