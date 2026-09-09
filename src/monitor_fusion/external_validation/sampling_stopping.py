from __future__ import annotations

from dataclasses import dataclass


# Retained only for A-opt threshold-selection checkpoints.
BATCH_SIZE = 250
VALIDATION_CELLS = ("A_val", "T", "S", "F", "SF")

CELL_SPECS = {
    "A_opt": {"Y1": 200, "Y0": 250, "cap": 2500, "window": "W0"},
    "A_val": {"Y1": None, "Y0": None, "cap": None, "window": "W0"},
    "T": {"Y1": None, "Y0": None, "cap": None, "window": "W1"},
    "S": {"Y1": None, "Y0": None, "cap": None, "window": "W1"},
    "F": {"Y1": None, "Y0": None, "cap": None, "window": "W1"},
    "SF": {"Y1": None, "Y0": None, "cap": None, "window": "W1"},
}

ALLOWED_TOP_UP_REASONS = frozenset(
    {
        "prespecified_eligibility_failure",
        "label_count_shortfall",
        "duplicate_or_dependency_exclusion",
        "quota_shortfall",
    }
)


class SamplingDesignNotFrozen(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionDecision:
    cell_id: str
    window: str
    candidates_collected: int
    eligible_y1: int
    eligible_y0: int
    required_y1: int
    required_y0: int
    candidate_cap: int
    y1_shortfall: int
    y0_shortfall: int
    status: str
    close_cell: bool
    next_full_batch_allowed: bool


def _spec(cell_id: str) -> dict[str, int | str | None]:
    try:
        return CELL_SPECS[cell_id]
    except KeyError as exc:
        raise ValueError(f"unknown study cell: {cell_id}") from exc


def _require_frozen_collection_spec(
    cell_id: str,
    spec: dict[str, int | str | None],
) -> tuple[int, int, int]:
    y1 = spec["Y1"]
    y0 = spec["Y0"]
    cap = spec["cap"]

    if y1 is None or y0 is None or cap is None:
        raise SamplingDesignNotFrozen(
            f"collection design for {cell_id} is not frozen: "
            "Y1/Y0 quotas, provenance-cluster minima/caps, validation candidate "
            "cap, and terminal-batch rule must be fixed after the full "
            "cluster-aware design grid before collection is authorized"
        )

    return int(y1), int(y0), int(cap)


def _validate_checkpoint(
    *,
    cell_id: str,
    candidates_collected: int,
    eligible_y1: int,
    eligible_y0: int,
) -> tuple[dict[str, int | str | None], int, int, int]:
    spec = _spec(cell_id)
    required_y1, required_y0, cap = _require_frozen_collection_spec(
        cell_id,
        spec,
    )

    for name, value in (
        ("candidates_collected", candidates_collected),
        ("eligible_y1", eligible_y1),
        ("eligible_y0", eligible_y0),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    if candidates_collected > cap:
        raise ValueError("candidates_collected exceeds the frozen candidate cap")

    if candidates_collected % BATCH_SIZE != 0:
        raise ValueError(
            "collection checkpoints must occur after complete 250-candidate batches"
        )

    if eligible_y1 + eligible_y0 > candidates_collected:
        raise ValueError(
            "eligible Y1 + Y0 dependency representatives cannot exceed raw candidates"
        )

    return spec, required_y1, required_y0, cap


def assess_collection_checkpoint(
    *,
    cell_id: str,
    candidates_collected: int,
    eligible_y1: int,
    eligible_y0: int,
    fresh_monitor_scoring_started: bool = False,
) -> CollectionDecision:
    spec, required_y1, required_y0, cap = _validate_checkpoint(
        cell_id=cell_id,
        candidates_collected=candidates_collected,
        eligible_y1=eligible_y1,
        eligible_y0=eligible_y0,
    )

    if not isinstance(fresh_monitor_scoring_started, bool):
        raise ValueError("fresh_monitor_scoring_started must be boolean")

    y1_shortfall = max(0, required_y1 - eligible_y1)
    y0_shortfall = max(0, required_y0 - eligible_y0)
    quotas_met = y1_shortfall == 0 and y0_shortfall == 0
    cap_reached = candidates_collected == cap

    if quotas_met:
        status = "quota_complete"
        close_cell = True
        next_full_batch_allowed = False
    elif cap_reached:
        status = "cap_reached_incomplete"
        close_cell = True
        next_full_batch_allowed = False
    else:
        status = "continue_collection"
        close_cell = False
        next_full_batch_allowed = not fresh_monitor_scoring_started

    if fresh_monitor_scoring_started and not close_cell:
        status = "invalid_post_scoring_shortfall"
        next_full_batch_allowed = False

    return CollectionDecision(
        cell_id=cell_id,
        window=str(spec["window"]),
        candidates_collected=candidates_collected,
        eligible_y1=eligible_y1,
        eligible_y0=eligible_y0,
        required_y1=required_y1,
        required_y0=required_y0,
        candidate_cap=cap,
        y1_shortfall=y1_shortfall,
        y0_shortfall=y0_shortfall,
        status=status,
        close_cell=close_cell,
        next_full_batch_allowed=next_full_batch_allowed,
    )


def authorize_next_batch(
    *,
    cell_id: str,
    candidates_collected: int,
    eligible_y1: int,
    eligible_y0: int,
    top_up_reason: str,
    fresh_monitor_scoring_started: bool = False,
) -> int:
    decision = assess_collection_checkpoint(
        cell_id=cell_id,
        candidates_collected=candidates_collected,
        eligible_y1=eligible_y1,
        eligible_y0=eligible_y0,
        fresh_monitor_scoring_started=fresh_monitor_scoring_started,
    )

    if fresh_monitor_scoring_started:
        raise RuntimeError(
            "collection/top-up is forbidden after fresh monitor scoring starts"
        )

    if decision.close_cell:
        raise RuntimeError(
            f"cell {cell_id} is closed under frozen stopping rule: {decision.status}"
        )

    if top_up_reason not in ALLOWED_TOP_UP_REASONS:
        raise ValueError(
            f"top_up_reason must be one of {sorted(ALLOWED_TOP_UP_REASONS)}"
        )

    next_total = candidates_collected + BATCH_SIZE

    if next_total > decision.candidate_cap:
        raise RuntimeError("next fixed batch would exceed the frozen candidate cap")

    return next_total


def all_cells_collection_closed(
    checkpoints: dict[str, tuple[int, int, int]],
) -> bool:
    if set(checkpoints) != set(CELL_SPECS):
        raise ValueError("checkpoints must contain exactly A_opt,A_val,T,S,F,SF")

    # This intentionally fails closed while any validation spec is pending.
    for cell_id in VALIDATION_CELLS:
        _require_frozen_collection_spec(cell_id, _spec(cell_id))

    decisions = [
        assess_collection_checkpoint(
            cell_id=cell_id,
            candidates_collected=values[0],
            eligible_y1=values[1],
            eligible_y0=values[2],
        )
        for cell_id, values in checkpoints.items()
    ]

    return all(decision.close_cell for decision in decisions)
