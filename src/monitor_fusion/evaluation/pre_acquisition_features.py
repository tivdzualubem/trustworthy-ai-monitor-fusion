"""Frozen v2 pre-acquisition feature construction.

The feature matrix contains the 17 ordered numeric/runtime features
followed by 32 fold-local PCA embedding components. PCA is fit only on
the current training rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


NUMERIC_FEATURE_NAMES = (
    "rule_score",
    "rule_weighted_sum",
    "rule_match_count",
    "rule_latency_ms",
    "compact_unsafe_score",
    "compact_input_tokens",
    "compact_output_tokens",
    "compact_latency_ms",
    "abs_rule_compact_difference",
    "rule_compact_product",
    "rule_compact_mean",
    "rule_compact_max",
    "rule_compact_min",
    "prompt_char_count",
    "response_char_count",
    "prompt_whitespace_token_count",
    "response_whitespace_token_count",
)

EMBEDDING_DIMENSION = 384
PCA_COMPONENTS = 32
TOTAL_FEATURE_DIMENSION = 49


_REQUIRED_FRAME_COLUMNS = (
    "prompt",
    "response",
    "rule_score",
    "rule_weighted_sum",
    "rule_match_count",
    "rule_latency_ms",
    "compact_unsafe_score",
    "compact_input_tokens",
    "compact_output_tokens",
    "compact_latency_ms",
)


def _numeric_vector(
    frame: pd.DataFrame,
    name: str,
) -> np.ndarray:
    values = pd.to_numeric(
        frame[name],
        errors="raise",
    ).to_numpy(
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return values


def build_numeric_pre_acquisition_features(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Return the frozen ordered 17-column numeric feature block."""

    missing = [
        column
        for column in _REQUIRED_FRAME_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "pre-acquisition frame missing columns: "
            f"{missing}"
        )

    for column in (
        "prompt",
        "response",
    ):
        if frame[column].isna().any():
            raise ValueError(
                f"{column} contains missing values"
            )

    rule_score = _numeric_vector(
        frame,
        "rule_score",
    )

    compact_score = _numeric_vector(
        frame,
        "compact_unsafe_score",
    )

    numeric = {
        "rule_score":
            rule_score,

        "rule_weighted_sum":
            _numeric_vector(
                frame,
                "rule_weighted_sum",
            ),

        "rule_match_count":
            _numeric_vector(
                frame,
                "rule_match_count",
            ),

        "rule_latency_ms":
            _numeric_vector(
                frame,
                "rule_latency_ms",
            ),

        "compact_unsafe_score":
            compact_score,

        "compact_input_tokens":
            _numeric_vector(
                frame,
                "compact_input_tokens",
            ),

        "compact_output_tokens":
            _numeric_vector(
                frame,
                "compact_output_tokens",
            ),

        "compact_latency_ms":
            _numeric_vector(
                frame,
                "compact_latency_ms",
            ),

        "abs_rule_compact_difference":
            np.abs(
                rule_score
                - compact_score
            ),

        "rule_compact_product":
            rule_score
            * compact_score,

        "rule_compact_mean":
            (
                rule_score
                + compact_score
            )
            / 2.0,

        "rule_compact_max":
            np.maximum(
                rule_score,
                compact_score,
            ),

        "rule_compact_min":
            np.minimum(
                rule_score,
                compact_score,
            ),

        "prompt_char_count":
            frame["prompt"]
            .astype(str)
            .map(len)
            .to_numpy(
                dtype=np.float64,
            ),

        "response_char_count":
            frame["response"]
            .astype(str)
            .map(len)
            .to_numpy(
                dtype=np.float64,
            ),

        "prompt_whitespace_token_count":
            frame["prompt"]
            .astype(str)
            .map(
                lambda value:
                    len(value.split())
            )
            .to_numpy(
                dtype=np.float64,
            ),

        "response_whitespace_token_count":
            frame["response"]
            .astype(str)
            .map(
                lambda value:
                    len(value.split())
            )
            .to_numpy(
                dtype=np.float64,
            ),
    }

    matrix = np.column_stack(
        [
            numeric[name]
            for name
            in NUMERIC_FEATURE_NAMES
        ]
    ).astype(
        np.float64,
        copy=False,
    )

    if matrix.shape != (
        len(frame),
        len(NUMERIC_FEATURE_NAMES),
    ):
        raise RuntimeError(
            "numeric feature matrix has "
            "unexpected shape"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "numeric feature matrix "
            "contains non-finite values"
        )

    return matrix


def validate_embeddings(
    embeddings: np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    """Validate the frozen 384-dimensional embedding input."""

    array = np.asarray(
        embeddings,
        dtype=np.float64,
    )

    if array.shape != (
        expected_rows,
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            "embeddings must have shape "
            f"({expected_rows}, "
            f"{EMBEDDING_DIMENSION})"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "embeddings contain "
            "non-finite values"
        )

    return array


@dataclass
class FoldLocalPreAcquisitionTransform:
    """PCA fitted only on a current grouped-training fold."""

    pca: PCA

    def transform(
        self,
        frame: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        numeric = (
            build_numeric_pre_acquisition_features(
                frame
            )
        )

        embedding_array = (
            validate_embeddings(
                embeddings,
                expected_rows=len(frame),
            )
        )

        reduced = self.pca.transform(
            embedding_array
        )

        matrix = np.column_stack(
            [
                numeric,
                reduced,
            ]
        ).astype(
            np.float64,
            copy=False,
        )

        if matrix.shape != (
            len(frame),
            TOTAL_FEATURE_DIMENSION,
        ):
            raise RuntimeError(
                "pre-acquisition feature matrix "
                "must have exactly 49 columns"
            )

        if not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                "pre-acquisition feature matrix "
                "contains non-finite values"
            )

        return matrix


def fit_fold_local_pre_acquisition_transform(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    random_state: int,
) -> tuple[
    FoldLocalPreAcquisitionTransform,
    np.ndarray,
]:
    """Fit PCA on training rows and return their 49-D matrix."""

    if len(frame) < PCA_COMPONENTS:
        raise ValueError(
            "training fold must contain at least "
            "32 rows for frozen PCA"
        )

    numeric = (
        build_numeric_pre_acquisition_features(
            frame
        )
    )

    embedding_array = (
        validate_embeddings(
            embeddings,
            expected_rows=len(frame),
        )
    )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=int(
            random_state
        ),
    )

    reduced = pca.fit_transform(
        embedding_array
    )

    matrix = np.column_stack(
        [
            numeric,
            reduced,
        ]
    ).astype(
        np.float64,
        copy=False,
    )

    if matrix.shape != (
        len(frame),
        TOTAL_FEATURE_DIMENSION,
    ):
        raise RuntimeError(
            "training feature matrix "
            "must have exactly 49 columns"
        )

    return (
        FoldLocalPreAcquisitionTransform(
            pca=pca
        ),
        matrix,
    )
