from __future__ import annotations

import numpy as np
import pytest

from monitor_fusion.evaluation.data_boundary import (
    load_protocol,
)
from monitor_fusion.evaluation.grouped_resampling import (
    make_grouped_folds,
    make_nested_repeated_grouped_folds,
    outer_fold_assignment_matrix,
    resampling_spec_from_protocol,
    resolve_effective_groups,
)


def synthetic_grouped_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    group_count = 60
    rows_per_group = 2

    groups = np.repeat(
        np.arange(group_count),
        rows_per_group,
    ).astype(object)

    labels_by_group = (
        np.arange(group_count) % 2
    )

    y = np.repeat(
        labels_by_group,
        rows_per_group,
    ).astype(int)

    return y, groups


def test_protocol_resampling_spec_is_exact() -> None:
    spec = resampling_spec_from_protocol(
        load_protocol()
    )

    assert spec.splitter == (
        "StratifiedGroupKFold"
    )
    assert spec.outer_folds == 5
    assert spec.inner_folds == 4
    assert spec.shuffle is True
    assert spec.fold_seeds == (
        1729,
        2718,
        3141,
        5772,
        8111,
    )
    assert spec.group_key_priority == (
        "group_id",
        "pair_id",
        "example_id",
    )
    assert spec.stratification_target == "y"


def test_effective_group_priority_and_namespacing() -> None:
    effective = resolve_effective_groups(
        ["shared", None, "", None],
        ["ignored", "shared", "pair", None],
        ["example-1", "example-2", "example-3", "shared"],
    )

    np.testing.assert_array_equal(
        effective,
        [
            "group_id:shared",
            "pair_id:shared",
            "pair_id:pair",
            "example_id:shared",
        ],
    )


def test_effective_groups_reject_missing_fallback() -> None:
    with pytest.raises(
        ValueError,
        match="Every row requires",
    ):
        resolve_effective_groups(
            [None],
            [None],
            [None],
        )


def test_single_seed_grouped_folds_are_deterministic() -> None:
    y, groups = synthetic_grouped_data()

    first = make_grouped_folds(
        y,
        groups,
        n_splits=5,
        seed=1729,
    )

    second = make_grouped_folds(
        y,
        groups,
        n_splits=5,
        seed=1729,
    )

    assert len(first) == 5

    for left, right in zip(
        first,
        second,
        strict=True,
    ):
        np.testing.assert_array_equal(
            left.train_indices,
            right.train_indices,
        )
        np.testing.assert_array_equal(
            left.validation_indices,
            right.validation_indices,
        )


def test_each_seed_partitions_rows_once_without_group_leakage() -> None:
    y, groups = synthetic_grouped_data()

    folds = make_grouped_folds(
        y,
        groups,
        n_splits=5,
        seed=2718,
    )

    validation_count = np.zeros(
        len(y),
        dtype=int,
    )

    for fold in folds:
        validation_count[
            fold.validation_indices
        ] += 1

        train_groups = set(
            groups[
                fold.train_indices
            ].tolist()
        )
        validation_groups = set(
            groups[
                fold.validation_indices
            ].tolist()
        )

        assert train_groups.isdisjoint(
            validation_groups
        )

    np.testing.assert_array_equal(
        validation_count,
        np.ones(len(y), dtype=int),
    )


def test_nested_repeated_design_has_five_by_five_outer_folds() -> None:
    y, groups = synthetic_grouped_data()

    spec = resampling_spec_from_protocol(
        load_protocol()
    )

    nested = make_nested_repeated_grouped_folds(
        y,
        groups,
        spec=spec,
    )

    assert len(nested) == 25

    for outer in nested:
        assert len(outer.inner_folds) == 4


def test_outer_validation_groups_never_enter_inner_folds() -> None:
    y, groups = synthetic_grouped_data()

    spec = resampling_spec_from_protocol(
        load_protocol()
    )

    nested = make_nested_repeated_grouped_folds(
        y,
        groups,
        spec=spec,
    )

    for outer in nested:
        held_out_groups = set(
            groups[
                outer.outer_validation_indices
            ].tolist()
        )

        for inner in outer.inner_folds:
            inner_train_groups = set(
                groups[
                    inner.train_indices
                ].tolist()
            )
            inner_validation_groups = set(
                groups[
                    inner.validation_indices
                ].tolist()
            )

            assert held_out_groups.isdisjoint(
                inner_train_groups
            )
            assert held_out_groups.isdisjoint(
                inner_validation_groups
            )
            assert inner_train_groups.isdisjoint(
                inner_validation_groups
            )


def test_assignment_matrix_contains_one_fold_per_seed_and_row() -> None:
    y, groups = synthetic_grouped_data()

    spec = resampling_spec_from_protocol(
        load_protocol()
    )

    nested = make_nested_repeated_grouped_folds(
        y,
        groups,
        spec=spec,
    )

    assignments = outer_fold_assignment_matrix(
        nested,
        fold_seeds=spec.fold_seeds,
        n_examples=len(y),
    )

    assert assignments.shape == (
        5,
        len(y),
    )

    assert set(
        np.unique(assignments).tolist()
    ) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    ("y", "groups", "message"),
    [
        (
            [0, 0, 0, 0],
            ["a", "b", "c", "d"],
            "both classes",
        ),
        (
            [0, 1, 0],
            ["a", "b"],
            "lengths differ",
        ),
        (
            [0, 1, 0, 1],
            ["a", "a", "b", "b"],
            "Not enough unique groups",
        ),
    ],
)
def test_invalid_grouped_designs_fail_closed(
    y: list[int],
    groups: list[str],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        make_grouped_folds(
            y,
            groups,
            n_splits=3,
            seed=1729,
        )
