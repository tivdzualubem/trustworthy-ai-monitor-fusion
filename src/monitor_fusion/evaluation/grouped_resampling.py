"""Repeated nested stratified-group resampling for protocol v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import StratifiedGroupKFold


IntArray = NDArray[np.int64]
ObjectArray = NDArray[np.object_]


@dataclass(frozen=True)
class RepeatedGroupedResamplingSpec:
    """Frozen repeated nested resampling parameters."""

    splitter: str
    outer_folds: int
    inner_folds: int
    shuffle: bool
    fold_seeds: tuple[int, ...]
    group_key_priority: tuple[str, ...]
    stratification_target: str


@dataclass(frozen=True)
class GroupedFold:
    """One group-disjoint train/validation fold."""

    seed: int
    fold: int
    train_indices: IntArray
    validation_indices: IntArray


@dataclass(frozen=True)
class NestedGroupedFold:
    """One outer fold and its inner model-selection folds."""

    seed: int
    outer_fold: int
    outer_train_indices: IntArray
    outer_validation_indices: IntArray
    inner_folds: tuple[GroupedFold, ...]


def resampling_spec_from_protocol(
    protocol: Mapping[str, Any],
) -> RepeatedGroupedResamplingSpec:
    """Load and validate the frozen resampling section."""

    section = protocol["development_resampling"]

    spec = RepeatedGroupedResamplingSpec(
        splitter=str(section["splitter"]),
        outer_folds=int(section["outer_folds"]),
        inner_folds=int(section["inner_folds"]),
        shuffle=bool(section["shuffle"]),
        fold_seeds=tuple(
            int(seed) for seed in section["fold_seeds"]
        ),
        group_key_priority=tuple(
            str(key)
            for key in section["group_key_priority"]
        ),
        stratification_target=str(
            section["stratification_target"]
        ),
    )

    if spec.splitter != "StratifiedGroupKFold":
        raise ValueError(
            "Protocol splitter must be StratifiedGroupKFold"
        )

    if not spec.shuffle:
        raise ValueError(
            "Protocol requires shuffled grouped folds"
        )

    if spec.outer_folds < 2 or spec.inner_folds < 2:
        raise ValueError(
            "Outer and inner fold counts must be at least two"
        )

    if len(spec.fold_seeds) < 2:
        raise ValueError(
            "At least two fold seeds are required"
        )

    if len(set(spec.fold_seeds)) != len(spec.fold_seeds):
        raise ValueError(
            "fold_seeds must be unique"
        )

    if spec.group_key_priority != (
        "group_id",
        "pair_id",
        "example_id",
    ):
        raise ValueError(
            "Unexpected frozen group-key priority"
        )

    if spec.stratification_target != "y":
        raise ValueError(
            "Unexpected stratification target"
        )

    return spec


def _object_vector(
    values: Sequence[object] | ArrayLike,
    *,
    name: str,
) -> ObjectArray:
    array = np.asarray(values, dtype=object)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional array"
        )

    return array


def _binary_target(values: ArrayLike) -> IntArray:
    array = np.asarray(values)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "y must be a nonempty one-dimensional array"
        )

    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(
            "y must contain numeric binary values"
        ) from exc

    if not bool(np.all(finite)) or not bool(
        np.all(np.isin(array, (0, 1)))
    ):
        raise ValueError(
            "y must contain only zero and one"
        )

    result = array.astype(np.int64, copy=False)

    if len(np.unique(result)) != 2:
        raise ValueError(
            "y must contain both classes"
        )

    return result


def _normalized_group_value(
    value: object,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None

    if isinstance(value, bytes):
        text = value.decode("utf-8").strip()
    else:
        text = str(value).strip()

    return text or None


def resolve_effective_groups(
    group_id: Sequence[object] | ArrayLike,
    pair_id: Sequence[object] | ArrayLike,
    example_id: Sequence[object] | ArrayLike,
) -> ObjectArray:
    """Resolve group_id, then pair_id, then example_id."""

    groups = _object_vector(
        group_id,
        name="group_id",
    )
    pairs = _object_vector(
        pair_id,
        name="pair_id",
    )
    examples = _object_vector(
        example_id,
        name="example_id",
    )

    if not (
        len(groups) == len(pairs) == len(examples)
    ):
        raise ValueError(
            "group_id, pair_id, and example_id lengths differ"
        )

    resolved: list[str] = []

    for group, pair, example in zip(
        groups,
        pairs,
        examples,
        strict=True,
    ):
        candidates = (
            (
                "group_id",
                _normalized_group_value(group),
            ),
            (
                "pair_id",
                _normalized_group_value(pair),
            ),
            (
                "example_id",
                _normalized_group_value(example),
            ),
        )

        selected = next(
            (
                f"{source}:{value}"
                for source, value in candidates
                if value is not None
            ),
            None,
        )

        if selected is None:
            raise ValueError(
                "Every row requires group_id, pair_id, "
                "or example_id"
            )

        resolved.append(selected)

    return np.asarray(resolved, dtype=object)


def make_grouped_folds(
    y: ArrayLike,
    effective_groups: Sequence[object] | ArrayLike,
    *,
    n_splits: int,
    seed: int,
) -> tuple[GroupedFold, ...]:
    """Create deterministic stratified group-disjoint folds."""

    target = _binary_target(y)
    groups = _object_vector(
        effective_groups,
        name="effective_groups",
    )

    if len(target) != len(groups):
        raise ValueError(
            "y and effective_groups lengths differ"
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least two"
        )

    if len(set(groups.tolist())) < n_splits:
        raise ValueError(
            "Not enough unique groups for n_splits"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(seed),
    )

    folds: list[GroupedFold] = []
    validation_counts = np.zeros(
        len(target),
        dtype=np.int64,
    )

    for fold, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(
            np.zeros((len(target), 1)),
            target,
            groups,
        )
    ):
        train = np.asarray(
            train_indices,
            dtype=np.int64,
        )
        validation = np.asarray(
            validation_indices,
            dtype=np.int64,
        )

        if np.intersect1d(
            groups[train],
            groups[validation],
        ).size:
            raise RuntimeError(
                "Group leakage detected"
            )

        validation_counts[validation] += 1

        folds.append(
            GroupedFold(
                seed=int(seed),
                fold=fold,
                train_indices=train,
                validation_indices=validation,
            )
        )

    if not bool(np.all(validation_counts == 1)):
        raise RuntimeError(
            "Validation folds do not partition all rows exactly once"
        )

    return tuple(folds)


def make_nested_repeated_grouped_folds(
    y: ArrayLike,
    effective_groups: Sequence[object] | ArrayLike,
    *,
    spec: RepeatedGroupedResamplingSpec,
) -> tuple[NestedGroupedFold, ...]:
    """Create every frozen seed's outer and inner grouped folds."""

    target = _binary_target(y)
    groups = _object_vector(
        effective_groups,
        name="effective_groups",
    )

    if len(target) != len(groups):
        raise ValueError(
            "y and effective_groups lengths differ"
        )

    nested: list[NestedGroupedFold] = []

    for seed in spec.fold_seeds:
        outer_folds = make_grouped_folds(
            target,
            groups,
            n_splits=spec.outer_folds,
            seed=seed,
        )

        for outer in outer_folds:
            outer_train = outer.train_indices
            outer_validation = (
                outer.validation_indices
            )

            inner_local = make_grouped_folds(
                target[outer_train],
                groups[outer_train],
                n_splits=spec.inner_folds,
                seed=seed,
            )

            inner_global: list[GroupedFold] = []

            for inner in inner_local:
                global_train = outer_train[
                    inner.train_indices
                ]
                global_validation = outer_train[
                    inner.validation_indices
                ]

                if np.intersect1d(
                    groups[global_train],
                    groups[global_validation],
                ).size:
                    raise RuntimeError(
                        "Inner group leakage detected"
                    )

                if np.intersect1d(
                    groups[global_train],
                    groups[outer_validation],
                ).size:
                    raise RuntimeError(
                        "Outer validation group leaked "
                        "into inner training"
                    )

                if np.intersect1d(
                    groups[global_validation],
                    groups[outer_validation],
                ).size:
                    raise RuntimeError(
                        "Outer validation group leaked "
                        "into inner validation"
                    )

                inner_global.append(
                    GroupedFold(
                        seed=seed,
                        fold=inner.fold,
                        train_indices=global_train,
                        validation_indices=(
                            global_validation
                        ),
                    )
                )

            nested.append(
                NestedGroupedFold(
                    seed=seed,
                    outer_fold=outer.fold,
                    outer_train_indices=(
                        outer_train
                    ),
                    outer_validation_indices=(
                        outer_validation
                    ),
                    inner_folds=tuple(
                        inner_global
                    ),
                )
            )

    expected = (
        len(spec.fold_seeds)
        * spec.outer_folds
    )

    if len(nested) != expected:
        raise RuntimeError(
            "Unexpected nested fold count"
        )

    return tuple(nested)


def outer_fold_assignment_matrix(
    nested_folds: Sequence[NestedGroupedFold],
    *,
    fold_seeds: Sequence[int],
    n_examples: int,
) -> IntArray:
    """Return one outer-fold assignment per seed and example."""

    if n_examples <= 0:
        raise ValueError(
            "n_examples must be positive"
        )

    seeds = tuple(int(seed) for seed in fold_seeds)

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "fold_seeds must be unique"
        )

    seed_position = {
        seed: position
        for position, seed in enumerate(seeds)
    }

    assignments = np.full(
        (len(seeds), n_examples),
        -1,
        dtype=np.int64,
    )

    for nested in nested_folds:
        if nested.seed not in seed_position:
            raise ValueError(
                "Nested fold contains an unknown seed"
            )

        row = seed_position[nested.seed]
        indices = nested.outer_validation_indices

        if bool(
            np.any(assignments[row, indices] != -1)
        ):
            raise RuntimeError(
                "Example assigned to multiple outer folds"
            )

        assignments[
            row,
            indices,
        ] = nested.outer_fold

    if bool(np.any(assignments < 0)):
        raise RuntimeError(
            "Some examples lack an outer-fold assignment"
        )

    return assignments
