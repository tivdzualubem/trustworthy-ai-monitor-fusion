#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/numerical_route_stability_v1.json"
RAW = ROOT / "reports/evaluation_measurement_pilot_v1/t4/policy_latency_raw.parquet"
ROUTE = ROOT / "reports/evaluation_measurement_pilot_v1/t4/route_mismatches.csv"
PRED = ROOT / "reports/evaluation_measurement_pilot_v1/t4/prediction_mismatches.csv"
VALIDATION = ROOT / "reports/evaluation_measurement_pilot_v1/t4/score_and_route_validation.json"
DEFS = ROOT / "reports/evaluation_measurement_pilot_v1/cpu/primary_policy_definitions.json"
OUT = ROOT / "reports/numerical_route_stability_v1/cpu"

FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]
EPS = [0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid_python(z: float) -> float:
    z = float(z)
    if z >= 0.0:
        t = math.exp(-z)
        return 1.0 / (1.0 + t)
    t = math.exp(z)
    return t / (1.0 + t)


def probability_python(model: dict, features: dict[str, float]) -> float:
    z = float(model["intercept"])
    for coef, name in zip(model["coef"], model["features"], strict=True):
        z += float(coef) * float(features[name])
    return sigmoid_python(z)


def probability_numpy(model: dict, features: dict[str, float], dtype) -> float:
    coefs = np.asarray(model["coef"], dtype=dtype)
    vals = np.asarray([features[x] for x in model["features"]], dtype=dtype)
    intercept = np.asarray(model["intercept"], dtype=dtype)
    z = intercept + np.sum(coefs * vals, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)
    p = one / (one + np.exp(-z))
    return float(p)


def probability_torch(model: dict, features: dict[str, float], dtype) -> float:
    coefs = torch.tensor(model["coef"], dtype=dtype, device="cpu")
    vals = torch.tensor(
        [features[x] for x in model["features"]],
        dtype=dtype,
        device="cpu",
    )
    intercept = torch.tensor(model["intercept"], dtype=dtype, device="cpu")
    z = intercept + torch.sum(coefs * vals)
    return float(torch.sigmoid(z).item())


def policy_id(stack: str, policy: dict) -> str:
    kind = str(policy["policy_kind"])
    target = policy.get("target_rate")
    return (
        f"{stack}::{kind}"
        if target is None
        else f"{stack}::{kind}::{float(target):.2f}"
    )


def records(defs: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for stack, cfg in defs["stacks"].items():
        for policy in cfg["policies"]:
            if not bool(policy.get("deployable", False)):
                continue
            pid = policy_id(stack, policy)
            out[pid] = {
                "stack": stack,
                "policy": policy,
                "cheap_model": cfg["cheap_model"],
                "full_model": cfg["full_model"],
            }
    if len(out) != 15:
        raise RuntimeError(f"Expected 15 deployable policies, found {len(out)}")
    return out


def feature_dict(row, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in FEATURES:
        column = f"{prefix}{name}"
        value = getattr(row, column)
        result[name] = float(value) if pd.notna(value) else math.nan
    return result


def ensure_model_features(model: dict, features: dict[str, float]) -> None:
    for name in model["features"]:
        if not math.isfinite(float(features[name])):
            raise RuntimeError(f"Missing required feature {name}")


def evaluate(
    rec: dict,
    features: dict[str, float],
    implementation: str,
    precision: str,
) -> tuple[bool, float, int, float, float]:
    if implementation == "python_scalar":
        if precision != "float64":
            raise RuntimeError("python_scalar is defined only for float64")
        fn = probability_python
    elif implementation == "numpy":
        dtype = np.float64 if precision == "float64" else np.float32
        fn = lambda model, x: probability_numpy(model, x, dtype)
    elif implementation == "torch_cpu":
        dtype = torch.float64 if precision == "float64" else torch.float32
        fn = lambda model, x: probability_torch(model, x, dtype)
    else:
        raise RuntimeError(implementation)

    ensure_model_features(rec["cheap_model"], features)
    cheap = fn(rec["cheap_model"], features)
    policy = rec["policy"]
    kind = str(policy["policy_kind"])

    if kind == "cheap_only":
        acquired = False
        probability = cheap
    elif kind == "full_information":
        acquired = True
        ensure_model_features(rec["full_model"], features)
        probability = fn(rec["full_model"], features)
    elif kind == "selective_deployable":
        threshold = float(policy["acquisition_threshold"])
        acquired = abs(cheap - 0.5) <= threshold
        if acquired:
            ensure_model_features(rec["full_model"], features)
            probability = fn(rec["full_model"], features)
        else:
            probability = cheap
    else:
        raise RuntimeError(kind)

    decision_threshold = float(policy["decision_threshold"])
    prediction = int(probability >= decision_threshold)
    route_margin = (
        float(policy["acquisition_threshold"]) - abs(cheap - 0.5)
        if kind == "selective_deployable"
        else math.nan
    )
    decision_margin = probability - decision_threshold
    return acquired, probability, prediction, route_margin, decision_margin


def exact_taxonomy(route: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    route_keys = set(zip(route["policy_id"], route["example_id"]))
    pred_keys = set(zip(pred["policy_id"], pred["example_id"]))
    if len(route) != 2 or len(route_keys) != 2:
        raise RuntimeError("Expected exactly two unique route mismatches.")
    if len(pred) != 5 or len(pred_keys) != 5:
        raise RuntimeError("Expected exactly five unique prediction mismatches.")
    if len(route_keys & pred_keys) != 2:
        raise RuntimeError("Both route mismatches must overlap prediction mismatches.")
    if len(pred_keys - route_keys) != 3:
        raise RuntimeError("Expected exactly three pure decision-threshold mismatches.")

    rows = []
    merged = pred.copy()
    for row in merged.itertuples(index=False):
        key = (str(row.policy_id), str(row.example_id))
        rows.append(
            {
                "policy_id": str(row.policy_id),
                "example_id": str(row.example_id),
                "stack": str(row.stack),
                "policy_kind": str(row.policy_kind),
                "target_rate": row.target_rate,
                "route_mismatch": key in route_keys,
                "prediction_mismatch": True,
                "prediction_mismatch_type": (
                    "downstream_of_route_mismatch"
                    if key in route_keys
                    else "pure_decision_threshold"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw = pd.read_parquet(RAW)
    route = pd.read_csv(ROUTE)
    pred = pd.read_csv(PRED)
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    defs = json.loads(DEFS.read_text(encoding="utf-8"))
    recs = records(defs)

    if int(validation["route_mismatch_rows"]) != 2:
        raise RuntimeError("Archived validation no longer has two route mismatches.")
    if int(validation["prediction_mismatch_rows"]) != 5:
        raise RuntimeError("Archived validation no longer has five prediction mismatches.")
    if len(raw) != 5445:
        raise RuntimeError(f"Expected 5,445 archived policy calls, found {len(raw)}")
    if set(raw["policy_id"].unique()) != set(recs):
        raise RuntimeError("Raw timing policies do not match frozen definitions.")

    taxonomy = exact_taxonomy(route, pred)
    taxonomy.to_csv(OUT / "mismatch_taxonomy.csv", index=False)

    # Canonical exact margins for all archived calls.
    margin_rows = []
    for row in raw.itertuples(index=False):
        rec = recs[str(row.policy_id)]
        ref_features = feature_dict(row, "reference_")
        runtime_features = feature_dict(row, "")

        # Cheap model uses only features available before routing.
        ensure_model_features(rec["cheap_model"], ref_features)
        ensure_model_features(rec["cheap_model"], runtime_features)
        ref_cheap = probability_python(rec["cheap_model"], ref_features)
        runtime_cheap = probability_python(rec["cheap_model"], runtime_features)
        kind = str(rec["policy"]["policy_kind"])

        ref_route_margin = (
            float(rec["policy"]["acquisition_threshold"]) - abs(ref_cheap - 0.5)
            if kind == "selective_deployable"
            else math.nan
        )
        runtime_route_margin = (
            float(rec["policy"]["acquisition_threshold"]) - abs(runtime_cheap - 0.5)
            if kind == "selective_deployable"
            else math.nan
        )

        decision_threshold = float(rec["policy"]["decision_threshold"])
        margin_rows.append(
            {
                "policy_id": str(row.policy_id),
                "example_id": str(row.example_id),
                "stack": str(row.stack),
                "policy_kind": kind,
                "target_rate": row.target_rate,
                "reference_cheap_probability": ref_cheap,
                "runtime_cheap_probability": runtime_cheap,
                "cheap_probability_delta_runtime_minus_reference": runtime_cheap - ref_cheap,
                "reference_route_margin": ref_route_margin,
                "runtime_route_margin": runtime_route_margin,
                "reference_decision_margin": float(row.reference_probability) - decision_threshold,
                "runtime_decision_margin": float(row.runtime_probability) - decision_threshold,
                "route_mismatch": not bool(row.route_matches_reference),
                "prediction_mismatch": not bool(row.prediction_matches_reference),
            }
        )
    margins = pd.DataFrame(margin_rows)
    margins.to_parquet(OUT / "margin_population.parquet", index=False)

    # Join exact mismatch taxonomy to margins and retain the actual ~1e-9 distances.
    mismatch_margins = margins.merge(
        taxonomy[
            ["policy_id", "example_id", "prediction_mismatch_type"]
        ],
        on=["policy_id", "example_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(mismatch_margins) != 5:
        raise RuntimeError("Mismatch-margin table must contain five rows.")
    mismatch_margins.to_csv(OUT / "mismatch_margins.csv", index=False)

    # Dead-band sweep. No epsilon is selected here; all prespecified values are reported.
    deadband_rows = []
    selective = margins[margins["policy_kind"].eq("selective_deployable")].copy()
    for eps in EPS:
        route_ref_amb = selective["reference_route_margin"].abs() <= eps
        route_run_amb = selective["runtime_route_margin"].abs() <= eps
        route_disagree = selective["route_mismatch"].astype(bool)
        pred_ref_amb = margins["reference_decision_margin"].abs() <= eps
        pred_run_amb = margins["runtime_decision_margin"].abs() <= eps
        pred_disagree = margins["prediction_mismatch"].astype(bool)

        deadband_rows.append(
            {
                "epsilon": eps,
                "route_rows": int(len(selective)),
                "route_reference_ambiguous_n": int(route_ref_amb.sum()),
                "route_runtime_ambiguous_n": int(route_run_amb.sum()),
                "route_union_ambiguous_n": int((route_ref_amb | route_run_amb).sum()),
                "route_mismatch_n": int(route_disagree.sum()),
                "route_mismatches_covered_by_union_deadband_n": int(
                    (route_disagree & (route_ref_amb | route_run_amb)).sum()
                ),
                "decision_rows": int(len(margins)),
                "decision_reference_ambiguous_n": int(pred_ref_amb.sum()),
                "decision_runtime_ambiguous_n": int(pred_run_amb.sum()),
                "decision_union_ambiguous_n": int((pred_ref_amb | pred_run_amb).sum()),
                "prediction_mismatch_n": int(pred_disagree.sum()),
                "prediction_mismatches_covered_by_union_deadband_n": int(
                    (pred_disagree & (pred_ref_amb | pred_run_amb)).sum()
                ),
            }
        )
    deadband = pd.DataFrame(deadband_rows)
    deadband.to_csv(OUT / "deadband_sweep.csv", index=False)

    # Runtime/implementation and precision matrix on identical reference features.
    canonical = {}
    implementation_rows = []
    variants = [
        ("python_scalar", "float64"),
        ("numpy", "float64"),
        ("numpy", "float32"),
        ("torch_cpu", "float64"),
        ("torch_cpu", "float32"),
    ]

    # Compute canonical results once.
    for row in raw.itertuples(index=False):
        key = (str(row.policy_id), str(row.example_id))
        rec = recs[str(row.policy_id)]
        ref_features = feature_dict(row, "reference_")
        canonical[key] = evaluate(
            rec, ref_features, "python_scalar", "float64"
        )

    for implementation, precision in variants:
        max_probability_delta = 0.0
        route_flip_n = 0
        prediction_flip_n = 0
        route_margin_delta_max = 0.0
        decision_margin_delta_max = 0.0

        for row in raw.itertuples(index=False):
            key = (str(row.policy_id), str(row.example_id))
            rec = recs[str(row.policy_id)]
            ref_features = feature_dict(row, "reference_")
            result = evaluate(rec, ref_features, implementation, precision)
            base = canonical[key]

            route_flip_n += int(result[0] != base[0])
            prediction_flip_n += int(result[2] != base[2])
            max_probability_delta = max(
                max_probability_delta, abs(result[1] - base[1])
            )
            if math.isfinite(result[3]) and math.isfinite(base[3]):
                route_margin_delta_max = max(
                    route_margin_delta_max, abs(result[3] - base[3])
                )
            decision_margin_delta_max = max(
                decision_margin_delta_max, abs(result[4] - base[4])
            )

        implementation_rows.append(
            {
                "implementation": implementation,
                "precision": precision,
                "hardware": "cpu",
                "rows": int(len(raw)),
                "max_abs_final_probability_delta_vs_python_float64": max_probability_delta,
                "route_flip_n_vs_python_float64": route_flip_n,
                "prediction_flip_n_vs_python_float64": prediction_flip_n,
                "max_abs_route_margin_delta": route_margin_delta_max,
                "max_abs_decision_margin_delta": decision_margin_delta_max,
            }
        )

    implementation_matrix = pd.DataFrame(implementation_rows)
    implementation_matrix.to_csv(
        OUT / "runtime_precision_matrix.csv", index=False
    )

    # Observed monitor-score envelope from the archived reference-vs-T4 run.
    score_envelopes = {}
    for feature in FEATURES:
        a = pd.to_numeric(raw[feature], errors="coerce")
        b = pd.to_numeric(raw[f"reference_{feature}"], errors="coerce")
        mask = a.notna() & b.notna()
        score_envelopes[feature] = (
            float((a[mask] - b[mask]).abs().max()) if mask.any() else 0.0
        )

    analytic_rows = []
    for stack, cfg in defs["stacks"].items():
        for role in ("cheap_model", "full_model"):
            model = cfg[role]
            logit_bound = 0.0
            for coef, name in zip(model["coef"], model["features"], strict=True):
                logit_bound += abs(float(coef)) * score_envelopes[name]
            probability_bound = 0.25 * logit_bound
            analytic_rows.append(
                {
                    "stack": stack,
                    "model_role": role,
                    "logit_perturbation_bound_from_archived_score_envelope": logit_bound,
                    "probability_perturbation_bound_via_sigmoid_lipschitz": probability_bound,
                    "bound_role": "conservative_archived_envelope_not_final_deadband",
                }
            )
    analytic = pd.DataFrame(analytic_rows)
    analytic.to_csv(OUT / "analytic_perturbation_bounds.csv", index=False)

    route_keys = set(zip(route["policy_id"], route["example_id"]))
    pred_keys = set(zip(pred["policy_id"], pred["example_id"]))
    pure_decision = pred_keys - route_keys

    summary = {
        "artifact": "numerical_route_stability_v1_cpu",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development_only_archived_t4_and_cpu_arithmetic",
        "route_mismatch_rows": 2,
        "prediction_mismatch_rows": 5,
        "pure_decision_threshold_mismatch_rows": 3,
        "route_induced_prediction_mismatch_rows": 2,
        "raw_policy_calls": int(len(raw)),
        "selective_route_rows": int(len(selective)),
        "epsilon_grid": EPS,
        "no_posthoc_epsilon_selection": True,
        "final_deadband_pending_controlled_hardware_phase": True,
        "score_envelopes_archived_runtime_minus_reference": score_envelopes,
        "protected_legacy_splits_used": False,
        "fresh_confirmatory_claim": False,
        "router_superiority_claim": False,
        "cost_or_pareto_claim": False,
        "hardware_phase_required": True,
        "hardware_phase_target": "same compact monitor and same torch policy arithmetic on CPU and NVIDIA T4",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_files = [
        OUT / "mismatch_taxonomy.csv",
        OUT / "mismatch_margins.csv",
        OUT / "margin_population.parquet",
        OUT / "deadband_sweep.csv",
        OUT / "runtime_precision_matrix.csv",
        OUT / "analytic_perturbation_bounds.csv",
        OUT / "summary.json",
    ]
    manifest = {
        "artifact": "numerical_route_stability_v1_cpu_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [PROTOCOL, RAW, ROUTE, PRED, VALIDATION, DEFS]
        },
        "outputs": {
            str(p.relative_to(OUT)): sha256(p)
            for p in output_files
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Exact taxonomy invariants.
    if len(pure_decision) != 3:
        raise RuntimeError("Pure-decision mismatch taxonomy changed.")
    if int(margins["route_mismatch"].sum()) != 2:
        raise RuntimeError("Route mismatch count changed.")
    if int(margins["prediction_mismatch"].sum()) != 5:
        raise RuntimeError("Prediction mismatch count changed.")

    print("NUMERICAL_ROUTE_STABILITY_CPU=PASS")
    print("route_threshold_mismatches=2")
    print("pure_decision_threshold_mismatches=3")
    print("route_induced_prediction_mismatches=2")
    print("hardware_phase_required=true")


if __name__ == "__main__":
    main()
