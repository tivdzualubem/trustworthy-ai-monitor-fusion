#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/safety_availability_control_plane_kill_study_v1.json"
DEV = ROOT / "data/processed/v2_development_view/unified_dataset_label_audited_v1.development.parquet"
DEFS = ROOT / "reports/evaluation_measurement_pilot_v1/cpu/primary_policy_definitions.json"
LATENCY = ROOT / "reports/evaluation_measurement_pilot_v1/t4/policy_latency_summary.csv"
STABILITY = ROOT / "reports/numerical_route_stability_v1/final/summary.json"
OUT = ROOT / "reports/safety_availability_control_plane_kill_study_v1"

FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def probability(model: dict, frame: pd.DataFrame) -> np.ndarray:
    names = list(model["features"])
    missing = [x for x in names if x not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing model features: {missing}")
    x = frame[names].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite monitor features for model features {names}")
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = float(model["intercept"])
    return sigmoid(intercept + x @ coef)


def selected_policy(stack_cfg: dict, target: float) -> dict:
    matches = [
        p
        for p in stack_cfg["policies"]
        if bool(p.get("deployable", False))
        and p.get("policy_kind") == "selective_deployable"
        and p.get("target_rate") is not None
        and abs(float(p["target_rate"]) - target) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one deployable target={target} policy, found {len(matches)}"
        )
    return matches[0]


def route_probability_stress(
    frame: pd.DataFrame,
    stack: str,
    cfg: dict,
    policy: dict,
    epsilon: float,
    perturbations: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cheap = probability(cfg["cheap_model"], frame)
    full = probability(cfg["full_model"], frame)
    y = frame["y"].to_numpy(dtype=int)

    threshold = float(policy["acquisition_threshold"])
    decision = float(policy["decision_threshold"])
    if not (0.0 < threshold < 0.5):
        raise RuntimeError(f"Unexpected acquisition threshold for {stack}: {threshold}")

    # Step-3 fail-closed interpretation: the numerical ambiguity shell escalates.
    effective_radius = threshold + epsilon
    fail_closed_route = np.abs(cheap - 0.5) <= effective_radius

    cheap_pred = cheap >= decision
    full_pred = full >= decision

    full_caught_unsafe = (y == 1) & full_pred
    cheap_missed_unsafe = (y == 1) & (~cheap_pred)
    witness = full_caught_unsafe & cheap_missed_unsafe

    # To bypass on the cheap-safe side, p' must be strictly below BOTH the
    # cheap decision threshold and the unambiguous non-escalation boundary.
    safe_route_boundary = min(
        decision,
        0.5 - threshold - epsilon,
    )
    strict_safe_target = np.nextafter(
        np.float64(safe_route_boundary),
        np.float64(-np.inf),
    )
    required_down_shift = np.maximum(0.0, cheap - strict_safe_target)

    baseline_bypass = witness & (~fail_closed_route)
    witness_n = int(witness.sum())
    full_caught_n = int(full_caught_unsafe.sum())

    bypass_rows = []
    for delta in perturbations:
        bypassable = witness & (required_down_shift <= float(delta))
        incremental = bypassable & (~baseline_bypass)
        bypass_rows.append(
            {
                "stack": stack,
                "perturbation_budget": float(delta),
                "development_rows": int(len(frame)),
                "unsafe_rows": int((y == 1).sum()),
                "full_caught_unsafe_n": full_caught_n,
                "full_caught_cheap_miss_witness_n": witness_n,
                "baseline_unambiguous_bypass_witness_n": int(baseline_bypass.sum()),
                "bypassable_witness_n": int(bypassable.sum()),
                "incremental_bypassable_witness_n": int(incremental.sum()),
                "incremental_full_caught_unsafe_bypassable_fraction": (
                    float(incremental.sum() / full_caught_n)
                    if full_caught_n
                    else 0.0
                ),
                "route_deadband_epsilon": epsilon,
                "realizability_claim": False,
            }
        )

    distance_to_escalation = np.maximum(
        0.0,
        np.abs(cheap - 0.5) - effective_radius,
    )
    escalation_rows = []
    for delta in perturbations:
        forceable = distance_to_escalation <= float(delta)
        escalation_rows.append(
            {
                "stack": stack,
                "perturbation_budget": float(delta),
                "development_rows": int(len(frame)),
                "baseline_fail_closed_escalation_n": int(fail_closed_route.sum()),
                "force_escalation_n": int(forceable.sum()),
                "force_escalation_fraction": float(forceable.mean()),
                "route_deadband_epsilon": epsilon,
            }
        )

    witness_summary = {
        "stack": stack,
        "development_rows": int(len(frame)),
        "unsafe_rows": int((y == 1).sum()),
        "full_caught_unsafe_n": full_caught_n,
        "cheap_missed_unsafe_n": int(cheap_missed_unsafe.sum()),
        "full_caught_cheap_miss_witness_n": witness_n,
        "baseline_fail_closed_escalation_n": int(fail_closed_route.sum()),
        "baseline_fail_closed_escalation_fraction": float(fail_closed_route.mean()),
        "baseline_unambiguous_bypass_witness_n": int(baseline_bypass.sum()),
        "acquisition_threshold": threshold,
        "decision_threshold": decision,
    }
    return pd.DataFrame(bypass_rows), pd.DataFrame(escalation_rows), witness_summary


def load_capacity_stress(
    stack: str,
    latency: pd.DataFrame,
    escalation: pd.DataFrame,
    nominal_utilization: float,
    load_multipliers: list[float],
) -> tuple[pd.DataFrame, dict]:
    cheap_id = f"{stack}::cheap_only"
    full_id = f"{stack}::full_information"
    selective_id = f"{stack}::selective_deployable::0.25"

    def one(policy_id: str) -> pd.Series:
        row = latency.loc[latency["policy_id"].eq(policy_id)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one latency row for {policy_id}, found {len(row)}")
        return row.iloc[0]

    cheap_row = one(cheap_id)
    full_row = one(full_id)
    selective_row = one(selective_id)

    cheap_ms = float(cheap_row["direct_mean_ms"])
    full_ms = float(full_row["direct_mean_ms"])
    increment_ms = full_ms - cheap_ms
    if increment_ms <= 0:
        raise RuntimeError(f"Non-positive expensive incremental latency for {stack}")

    mu = 1000.0 / increment_ms
    q_baseline = float(selective_row["runtime_acquisition_rate"])
    if not (0.0 < q_baseline <= 1.0):
        raise RuntimeError(f"Invalid baseline acquisition rate for {stack}: {q_baseline}")

    nominal_total_arrival = nominal_utilization * mu / q_baseline

    q_conditions = [
        ("synthetic_route_pressure", float(r.perturbation_budget), float(r.force_escalation_fraction))
        for r in escalation.itertuples(index=False)
    ]
    q_conditions.append(("worst_case_all_escalate", math.nan, 1.0))

    rows = []
    for attack_mode, delta, q in q_conditions:
        for multiplier in load_multipliers:
            total_lambda = nominal_total_arrival * float(multiplier)
            escalated_lambda = q * total_lambda
            rho = escalated_lambda / mu
            overloaded = escalated_lambda > mu + 1e-12
            overflow_rate = max(0.0, escalated_lambda - mu)
            overflow_fraction_total = (
                overflow_rate / total_lambda if total_lambda > 0 else 0.0
            )

            # 1) Fail-open cap: resource use is capped, but any overflow bypasses
            # the expensive safety step and therefore breaks fail-closed semantics.
            rows.append(
                {
                    "stack": stack,
                    "attack_mode": attack_mode,
                    "perturbation_budget": delta,
                    "force_escalation_fraction": q,
                    "load_multiplier": float(multiplier),
                    "total_arrival_rate_per_s": total_lambda,
                    "expensive_service_rate_per_s": mu,
                    "escalated_arrival_rate_per_s": escalated_lambda,
                    "expensive_utilization": rho,
                    "strategy": "fail_open_budget_cap",
                    "queue_stable": True,
                    "resource_bound_preserved": True,
                    "fail_closed_safety_semantics_preserved": not overloaded,
                    "ordinary_response_fraction": 1.0,
                    "defer_reject_fraction": 0.0,
                    "expensive_bypass_fraction": overflow_fraction_total,
                }
            )

            # 2) Fail-closed accept-all: no bypass, but overload creates unbounded
            # queue growth in the fluid-capacity model.
            rows.append(
                {
                    "stack": stack,
                    "attack_mode": attack_mode,
                    "perturbation_budget": delta,
                    "force_escalation_fraction": q,
                    "load_multiplier": float(multiplier),
                    "total_arrival_rate_per_s": total_lambda,
                    "expensive_service_rate_per_s": mu,
                    "escalated_arrival_rate_per_s": escalated_lambda,
                    "expensive_utilization": rho,
                    "strategy": "fail_closed_no_admission",
                    "queue_stable": not overloaded,
                    "resource_bound_preserved": not overloaded,
                    "fail_closed_safety_semantics_preserved": True,
                    "ordinary_response_fraction": 1.0,
                    "defer_reject_fraction": 0.0,
                    "expensive_bypass_fraction": 0.0,
                }
            )

            # 3) Fail-closed with explicit non-service action: keep the expensive
            # stage within capacity and do not silently bypass it.
            rows.append(
                {
                    "stack": stack,
                    "attack_mode": attack_mode,
                    "perturbation_budget": delta,
                    "force_escalation_fraction": q,
                    "load_multiplier": float(multiplier),
                    "total_arrival_rate_per_s": total_lambda,
                    "expensive_service_rate_per_s": mu,
                    "escalated_arrival_rate_per_s": escalated_lambda,
                    "expensive_utilization": min(rho, 1.0),
                    "strategy": "fail_closed_defer_reject",
                    "queue_stable": True,
                    "resource_bound_preserved": True,
                    "fail_closed_safety_semantics_preserved": True,
                    "ordinary_response_fraction": 1.0 - overflow_fraction_total,
                    "defer_reject_fraction": overflow_fraction_total,
                    "expensive_bypass_fraction": 0.0,
                }
            )

    capacity = {
        "stack": stack,
        "cheap_direct_mean_ms": cheap_ms,
        "full_information_direct_mean_ms": full_ms,
        "expensive_incremental_mean_ms": increment_ms,
        "single_t4_expensive_service_rate_per_s": mu,
        "baseline_runtime_acquisition_rate": q_baseline,
        "nominal_expensive_utilization": nominal_utilization,
        "nominal_total_arrival_rate_per_s": nominal_total_arrival,
    }
    return pd.DataFrame(rows), capacity


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DEV)
    defs = json.loads(DEFS.read_text(encoding="utf-8"))
    latency = pd.read_csv(LATENCY)
    stability = json.loads(STABILITY.read_text(encoding="utf-8"))

    if len(frame) != 1687:
        raise RuntimeError(f"Expected 1,687 development rows, found {len(frame)}")
    required = {"example_id", "y", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Development view is missing required columns: {missing}")
    if set(frame["y"].dropna().astype(int).unique()) - {0, 1}:
        raise RuntimeError("Audited label y is not binary.")
    if int(frame["y"].sum()) != 291:
        raise RuntimeError(
            f"Expected 291 audited-positive development rows, found {int(frame['y'].sum())}"
        )

    epsilon = float(stability["deadband"]["reporting_epsilon"])
    frozen_epsilon = float(protocol["numerical_fail_closed_rule"]["route_deadband_epsilon"])
    if epsilon != frozen_epsilon:
        raise RuntimeError(
            f"Step-3 epsilon changed: protocol={frozen_epsilon}, current={epsilon}"
        )

    target = float(protocol["primary_design"]["target_rate"])
    stacks = [
        protocol["primary_design"]["primary_stack"],
        *protocol["primary_design"]["sensitivity_stacks"],
    ]
    perturbations = [
        float(x) for x in protocol["routing_bypass_pressure"]["perturbation_grid"]
    ]
    if perturbations != [
        float(x) for x in protocol["routing_escalation_pressure"]["perturbation_grid"]
    ]:
        raise RuntimeError("Bypass and escalation perturbation grids differ.")

    bypass_parts = []
    escalation_parts = []
    witness_rows = []
    load_parts = []
    capacity_rows = []

    for stack in stacks:
        cfg = defs["stacks"][stack]
        policy = selected_policy(cfg, target)

        bypass, escalation, witness = route_probability_stress(
            frame,
            stack,
            cfg,
            policy,
            epsilon,
            perturbations,
        )
        bypass_parts.append(bypass)
        escalation_parts.append(escalation)
        witness_rows.append(witness)

        load, capacity = load_capacity_stress(
            stack,
            latency,
            escalation,
            float(protocol["load_escalation_pressure"]["nominal_expensive_utilization"]),
            [float(x) for x in protocol["load_escalation_pressure"]["total_arrival_load_multipliers"]],
        )
        load_parts.append(load)
        capacity_rows.append(capacity)

    bypass_all = pd.concat(bypass_parts, ignore_index=True)
    escalation_all = pd.concat(escalation_parts, ignore_index=True)
    witnesses = pd.DataFrame(witness_rows)
    load_all = pd.concat(load_parts, ignore_index=True)
    capacity_all = pd.DataFrame(capacity_rows)

    bypass_all.to_csv(OUT / "routing_bypass_sweep.csv", index=False)
    escalation_all.to_csv(OUT / "routing_escalation_sweep.csv", index=False)
    witnesses.to_csv(OUT / "safety_witness_summary.csv", index=False)
    load_all.to_csv(OUT / "load_capacity_sweep.csv", index=False)
    capacity_all.to_csv(OUT / "capacity_calibration.csv", index=False)

    # Formal necessity audit.
    worst = load_all[
        load_all["attack_mode"].eq("worst_case_all_escalate")
    ].copy()
    no_admission_overload = worst[
        worst["strategy"].eq("fail_closed_no_admission")
        & (~worst["queue_stable"].astype(bool))
    ]
    defer_rows = worst[
        worst["strategy"].eq("fail_closed_defer_reject")
        & (worst["defer_reject_fraction"] > 0)
    ]
    fail_open_rows = worst[
        worst["strategy"].eq("fail_open_budget_cap")
        & (worst["expensive_bypass_fraction"] > 0)
    ]
    if no_admission_overload.empty:
        raise RuntimeError("Worst-case load grid never overloaded fail-closed accept-all.")
    if defer_rows.empty:
        raise RuntimeError("Defer/reject strategy never exercised under overload.")
    if fail_open_rows.empty:
        raise RuntimeError("Fail-open budget-cap strategy never exercised under overload.")

    deadband = float(epsilon)
    at_deadband = bypass_all[np.isclose(
        bypass_all["perturbation_budget"],
        deadband,
        rtol=0.0,
        atol=1e-15,
    )].copy()
    if len(at_deadband) != len(stacks):
        raise RuntimeError("Could not evaluate the frozen dead-band perturbation on all stacks.")

    novelty_threshold = float(
        protocol["internal_novelty_kill_rule"]["deadband_scale_incremental_bypass_threshold"]
    )
    min_stacks = int(
        protocol["internal_novelty_kill_rule"]["minimum_stacks_for_nontrivial_signal"]
    )
    signal_stacks = at_deadband[
        at_deadband["incremental_full_caught_unsafe_bypassable_fraction"]
        >= novelty_threshold
    ]["stack"].tolist()

    continue_signal = len(signal_stacks) >= min_stacks
    direction = (
        "candidate_requires_literature_validation"
        if continue_signal
        else "standalone_security_paper_not_supported_by_internal_kill_study"
    )

    # The necessity statement does not depend on observed labels: if every accepted
    # request can be forced onto the expensive path, finite capacity cannot serve
    # arbitrary load while accepting everything.
    formal = {
        "artifact": "safety_availability_control_plane_kill_study_v1_formal_necessity",
        "finite_budget_condition": "B >= N*c_e for accept-all all-expensive fail-closed service",
        "queue_stability_condition": (
            "q*lambda <= mu has no positive backlog drift in the deterministic "
            "fluid model; q*lambda > mu is overloaded"
        ),
        "worst_case_q": 1.0,
        "defer_reject_or_worst_case_provisioning_necessary_when_budget_insufficient": True,
        "defer_reject_or_worst_case_provisioning_necessary_when_q_lambda_gt_mu": True,
        "interpretation": (
            "The necessity result is a capacity/admission-control condition, not by "
            "itself a standalone security novelty claim."
        ),
    }
    (OUT / "formal_necessity.json").write_text(
        json.dumps(formal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "artifact": "safety_availability_control_plane_kill_study_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_development_only",
        "rows": int(len(frame)),
        "stacks": stacks,
        "target_rate": target,
        "route_deadband_epsilon": epsilon,
        "routing_bypass_pressure_completed": True,
        "routing_escalation_pressure_completed": True,
        "load_escalation_pressure_completed": True,
        "strategies_compared": [
            "fail_open_budget_cap",
            "fail_closed_no_admission",
            "fail_closed_defer_reject",
        ],
        "formal_necessity": formal,
        "deadband_scale_incremental_bypass_signal_stacks": signal_stacks,
        "internal_continue_signal": continue_signal,
        "standalone_security_direction": direction,
        "literature_novelty_claim": False,
        "claim_boundary": {
            "protected_legacy_splits_used": False,
            "fresh_confirmatory_claim": False,
            "router_superiority_claim": False,
            "pareto_claim_available": False,
            "production_claim": False,
        },
        "next_step": "integrate_ordered_studies_and_reassess_paper_direction",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs = [
        OUT / "routing_bypass_sweep.csv",
        OUT / "routing_escalation_sweep.csv",
        OUT / "safety_witness_summary.csv",
        OUT / "load_capacity_sweep.csv",
        OUT / "capacity_calibration.csv",
        OUT / "formal_necessity.json",
        OUT / "summary.json",
    ]
    manifest = {
        "artifact": "safety_availability_control_plane_kill_study_v1_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [PROTOCOL, DEV, DEFS, LATENCY, STABILITY]
        },
        "outputs": {
            str(p.relative_to(OUT)): sha256(p)
            for p in outputs
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("SAFETY_AVAILABILITY_KILL_STUDY=PASS")
    print(f"rows={len(frame)}")
    print(f"route_deadband_epsilon={epsilon:.0e}")
    print(f"signal_stacks_at_deadband={len(signal_stacks)}/{len(stacks)}")
    print(f"standalone_security_direction={direction}")
    print("defer_reject_or_worst_case_provisioning_necessary_under_overload=true")
    print("protected_legacy_splits_used=false")
    print("literature_novelty_claim=false")
    print("next_step=integrate_ordered_studies_and_reassess_paper_direction")


if __name__ == "__main__":
    main()
