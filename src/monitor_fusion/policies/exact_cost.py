"""Deployable threshold mixtures for heterogeneous exact-cost matching."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        result = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _example_ids(values: Iterable[object], *, expected_length: int) -> list[str]:
    result = [str(value) for value in values]
    if len(result) != expected_length:
        raise ValueError(
            f"example_ids has length {len(result)}; expected {expected_length}"
        )
    if any(not value for value in result):
        raise ValueError("example_ids must not contain empty identifiers")
    return result


def _validate_hash_inputs(policy_id: str, hash_seed: int) -> None:
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy_id must be a nonempty string")
    if isinstance(hash_seed, bool) or not isinstance(hash_seed, int):
        raise ValueError("hash_seed must be an integer")
    if hash_seed < 0:
        raise ValueError("hash_seed must be nonnegative")


def sha256_uniform(
    example_id: object,
    *,
    policy_id: str,
    hash_seed: int,
) -> float:
    """Map an example and frozen policy seed deterministically to ``[0, 1)``."""

    _validate_hash_inputs(policy_id, hash_seed)
    identifier = str(example_id)
    if not identifier:
        raise ValueError("example_id must not be empty")
    payload = json.dumps(
        [identifier, policy_id, hash_seed],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


@dataclass(frozen=True)
class ExactCostThresholdMixture:
    """Two fixed adjacent thresholds mixed by a deterministic hash.

    The lower threshold is the more aggressive policy.  Its probability is
    chosen on calibration data so the randomization expectation equals the
    absolute target cost.  Evaluation applies only these stored constants.
    """

    policy_id: str
    lower_acquisition_threshold: float
    upper_acquisition_threshold: float
    lower_threshold_probability: float
    hash_seed: int
    absolute_cost_budget_ms: float
    calibration_expected_total_cost_ms: float
    calibration_lower_threshold_cost_ms: float
    calibration_upper_threshold_cost_ms: float

    def __post_init__(self) -> None:
        _validate_hash_inputs(self.policy_id, self.hash_seed)
        numeric = {
            "lower_acquisition_threshold": self.lower_acquisition_threshold,
            "upper_acquisition_threshold": self.upper_acquisition_threshold,
            "lower_threshold_probability": self.lower_threshold_probability,
            "absolute_cost_budget_ms": self.absolute_cost_budget_ms,
            "calibration_expected_total_cost_ms": (
                self.calibration_expected_total_cost_ms
            ),
            "calibration_lower_threshold_cost_ms": (
                self.calibration_lower_threshold_cost_ms
            ),
            "calibration_upper_threshold_cost_ms": (
                self.calibration_upper_threshold_cost_ms
            ),
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.lower_acquisition_threshold
            > self.upper_acquisition_threshold
        ):
            raise ValueError(
                "lower_acquisition_threshold must not exceed "
                "upper_acquisition_threshold"
            )
        if not 0.0 <= self.lower_threshold_probability <= 1.0:
            raise ValueError(
                "lower_threshold_probability must lie in [0, 1]"
            )
        if self.absolute_cost_budget_ms < 0.0:
            raise ValueError("absolute_cost_budget_ms must be nonnegative")
        if any(
            value < 0.0
            for value in (
                self.calibration_expected_total_cost_ms,
                self.calibration_lower_threshold_cost_ms,
                self.calibration_upper_threshold_cost_ms,
            )
        ):
            raise ValueError("calibration total costs must be nonnegative")
        if (
            self.calibration_lower_threshold_cost_ms
            < self.calibration_upper_threshold_cost_ms
        ):
            raise ValueError(
                "the lower threshold must not cost less than the upper threshold"
            )
        expected = (
            self.lower_threshold_probability
            * self.calibration_lower_threshold_cost_ms
            + (1.0 - self.lower_threshold_probability)
            * self.calibration_upper_threshold_cost_ms
        )
        tolerance = 1e-10 * max(1.0, abs(expected))
        if not math.isclose(
            expected,
            self.calibration_expected_total_cost_ms,
            rel_tol=1e-10,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "calibration_expected_total_cost_ms does not match the "
                "threshold mixture"
            )
        if not math.isclose(
            expected,
            self.absolute_cost_budget_ms,
            rel_tol=1e-10,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "threshold mixture does not match absolute_cost_budget_ms"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable frozen policy record."""

        return asdict(self)


def apply_threshold_mixture(
    scores: ArrayLike,
    example_ids: Iterable[object],
    policy: ExactCostThresholdMixture,
) -> BoolArray:
    """Apply a frozen online policy without evaluation-set ranking."""

    score = _finite_vector(scores, name="scores")
    identifiers = _example_ids(example_ids, expected_length=len(score))
    if (
        policy.lower_acquisition_threshold
        == policy.upper_acquisition_threshold
    ):
        return score > policy.lower_acquisition_threshold

    uniform = np.fromiter(
        (
            sha256_uniform(
                identifier,
                policy_id=policy.policy_id,
                hash_seed=policy.hash_seed,
            )
            for identifier in identifiers
        ),
        dtype=np.float64,
        count=len(identifiers),
    )
    use_lower_threshold = uniform < policy.lower_threshold_probability
    threshold = np.where(
        use_lower_threshold,
        policy.lower_acquisition_threshold,
        policy.upper_acquisition_threshold,
    )
    return score > threshold


def total_cost_per_example(
    acquisition: ArrayLike,
    no_acquisition_total_cost_ms: ArrayLike,
    acquisition_total_cost_ms: ArrayLike,
) -> FloatArray:
    """Select each example's realized total end-to-end policy cost."""

    acquired = np.asarray(acquisition)
    if acquired.ndim != 1 or acquired.size == 0:
        raise ValueError("acquisition must be a nonempty one-dimensional array")
    if not bool(np.all(np.isin(acquired, (False, True, 0, 1)))):
        raise ValueError("acquisition must be binary")
    without_optional = _finite_vector(
        no_acquisition_total_cost_ms,
        name="no_acquisition_total_cost_ms",
    )
    with_optional = _finite_vector(
        acquisition_total_cost_ms,
        name="acquisition_total_cost_ms",
    )
    if len(acquired) != len(without_optional) or len(acquired) != len(
        with_optional
    ):
        raise ValueError("acquisition and cost arrays must have equal length")
    if bool(np.any(without_optional < 0.0)) or bool(
        np.any(with_optional < 0.0)
    ):
        raise ValueError("total costs must be nonnegative")
    if bool(np.any(with_optional < without_optional)):
        raise ValueError(
            "acquisition_total_cost_ms must include and therefore not be "
            "less than no_acquisition_total_cost_ms"
        )
    return np.where(acquired.astype(bool), with_optional, without_optional)


def _candidate_thresholds(scores: FloatArray) -> FloatArray:
    unique_descending = np.unique(scores)[::-1]
    always_threshold = np.nextafter(unique_descending[-1], -math.inf)
    if not math.isfinite(float(always_threshold)):
        raise ValueError(
            "score range does not permit a finite always-acquire threshold"
        )
    return np.concatenate((unique_descending, [always_threshold]))


def calibrate_exact_cost_threshold_mixture(
    scores: ArrayLike,
    no_acquisition_total_cost_ms: ArrayLike,
    acquisition_total_cost_ms: ArrayLike,
    *,
    absolute_cost_budget_ms: float,
    policy_id: str,
    hash_seed: int,
) -> ExactCostThresholdMixture:
    """Freeze adjacent thresholds whose expected calibration cost is exact.

    This function is for the independent calibration-optimization subset.
    It uses heterogeneous per-example total costs and never uses labels.
    """

    _validate_hash_inputs(policy_id, hash_seed)
    if not math.isfinite(absolute_cost_budget_ms):
        raise ValueError("absolute_cost_budget_ms must be finite")

    score = _finite_vector(scores, name="scores")
    without_optional = _finite_vector(
        no_acquisition_total_cost_ms,
        name="no_acquisition_total_cost_ms",
    )
    with_optional = _finite_vector(
        acquisition_total_cost_ms,
        name="acquisition_total_cost_ms",
    )
    if len(score) != len(without_optional) or len(score) != len(with_optional):
        raise ValueError("scores and cost arrays must have equal length")
    if bool(np.any(without_optional < 0.0)) or bool(
        np.any(with_optional < 0.0)
    ):
        raise ValueError("total costs must be nonnegative")
    if bool(np.any(with_optional < without_optional)):
        raise ValueError(
            "acquisition_total_cost_ms must include and therefore not be "
            "less than no_acquisition_total_cost_ms"
        )

    thresholds = _candidate_thresholds(score)
    candidate_costs = np.asarray(
        [
            float(
                np.mean(
                    total_cost_per_example(
                        score > threshold,
                        without_optional,
                        with_optional,
                    )
                )
            )
            for threshold in thresholds
        ],
        dtype=np.float64,
    )
    if bool(np.any(np.diff(candidate_costs) < -1e-12)):
        raise RuntimeError("nested acquisition candidates have decreasing cost")

    target = float(absolute_cost_budget_ms)
    tolerance = 1e-12 * max(1.0, abs(target))
    minimum = float(candidate_costs[0])
    maximum = float(candidate_costs[-1])
    if target < minimum - tolerance or target > maximum + tolerance:
        raise ValueError(
            "absolute_cost_budget_ms must lie between the never-acquire "
            f"cost {minimum:.12g} and always-acquire cost {maximum:.12g}"
        )
    target = min(max(target, minimum), maximum)

    exact_indices = np.flatnonzero(
        np.isclose(candidate_costs, target, rtol=1e-12, atol=tolerance)
    )
    if len(exact_indices):
        index = int(exact_indices[0])
        threshold = float(thresholds[index])
        matched_cost = float(candidate_costs[index])
        return ExactCostThresholdMixture(
            policy_id=policy_id,
            lower_acquisition_threshold=threshold,
            upper_acquisition_threshold=threshold,
            lower_threshold_probability=1.0,
            hash_seed=hash_seed,
            absolute_cost_budget_ms=float(absolute_cost_budget_ms),
            calibration_expected_total_cost_ms=matched_cost,
            calibration_lower_threshold_cost_ms=matched_cost,
            calibration_upper_threshold_cost_ms=matched_cost,
        )

    aggressive_index = int(np.searchsorted(candidate_costs, target, side="right"))
    conservative_index = aggressive_index - 1
    aggressive_cost = float(candidate_costs[aggressive_index])
    conservative_cost = float(candidate_costs[conservative_index])
    cost_gap = aggressive_cost - conservative_cost
    if cost_gap <= 0.0:
        raise RuntimeError("adjacent calibration candidates do not bracket target")
    aggressive_probability = (target - conservative_cost) / cost_gap
    expected_cost = (
        aggressive_probability * aggressive_cost
        + (1.0 - aggressive_probability) * conservative_cost
    )

    return ExactCostThresholdMixture(
        policy_id=policy_id,
        lower_acquisition_threshold=float(thresholds[aggressive_index]),
        upper_acquisition_threshold=float(thresholds[conservative_index]),
        lower_threshold_probability=float(aggressive_probability),
        hash_seed=hash_seed,
        absolute_cost_budget_ms=float(absolute_cost_budget_ms),
        calibration_expected_total_cost_ms=float(expected_cost),
        calibration_lower_threshold_cost_ms=aggressive_cost,
        calibration_upper_threshold_cost_ms=conservative_cost,
    )
