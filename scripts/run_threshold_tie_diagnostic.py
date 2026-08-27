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
PROTOCOL = ROOT / "configs/threshold_tie_diagnostic_v1.json"
MARGINS = ROOT / "reports/numerical_route_stability_v1/cpu/margin_population.parquet"
RAW = ROOT / "reports/evaluation_measurement_pilot_v1/t4/policy_latency_raw.parquet"
DEFS = ROOT / "reports/evaluation_measurement_pilot_v1/cpu/primary_policy_definitions.json"
PRIOR_SWEEP = ROOT / "reports/numerical_route_stability_v1/cpu/deadband_sweep.csv"
PRIOR_FINAL = ROOT / "reports/numerical_route_stability_v1/final/summary.json"
OUT = ROOT / "reports/threshold_tie_diagnostic_v1"

EXPECTED_ROUTE_ROWS = 3267
EXPECTED_ALL_ROWS = 5445
EXPECTED_PRIOR_EXACT_REFERENCE_TIES = 200
EXPECTED_PRIOR_ROUTE_MISMATCHES = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_id(stack: str, policy: dict) -> str:
    kind = str(policy["policy_kind"])
    target = policy.get("target_rate")
    return (
        f"{stack}::{kind}"
        if target is None
        else f"{stack}::{kind}::{float(target):.2f}"
    )


def exact_float_key(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("Non-finite value in exact float key.")
    return value.hex()


def frozen_maps(defs: dict) -> tuple[dict[str, list[str]], dict[str, float]]:
    cheap_features: dict[str, list[str]] = {}
    thresholds: dict[str, float] = {}

    for stack, cfg in defs["stacks"].items():
        features = [str(x) for x in cfg["cheap_model"]["features"]]
        if not features:
            raise RuntimeError(f"Stack {stack} has no cheap-model features.")
        cheap_features[str(stack)] = features

        for policy in cfg["policies"]:
            if not bool(policy.get("deployable", False)):
                continue
            if str(policy["policy_kind"]) != "selective_deployable":
                continue
            pid = policy_id(str(stack), policy)
            threshold = policy.get("acquisition_threshold")
            if threshold is None:
                raise RuntimeError(f"Missing acquisition threshold for {pid}.")
            thresholds[pid] = float(threshold)

    if len(cheap_features) != 3:
        raise RuntimeError(
            f"Expected 3 monitor stacks, found {len(cheap_features)}."
        )
    if len(thresholds) != 9:
        raise RuntimeError(
            f"Expected 9 deployable selective policies, found {len(thresholds)}."
        )
    return cheap_features, thresholds


def build_stack_example_states(
    raw: pd.DataFrame,
    margins: pd.DataFrame,
    cheap_features: dict[str, list[str]],
) -> pd.DataFrame:
    required = {"policy_id", "example_id", "stack"}
    if not required.issubset(raw.columns):
        raise RuntimeError(
            f"Archived policy calls missing columns: {sorted(required - set(raw.columns))}"
        )

    margin_small = margins[
        ["policy_id", "example_id", "stack", "reference_cheap_probability"]
    ].copy()

    merged = raw.merge(
        margin_small,
        on=["policy_id", "example_id", "stack"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_ALL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ALL_ROWS} joined policy calls, found {len(merged)}."
        )

    rows: list[dict] = []
    for stack, features in cheap_features.items():
        part = merged[merged["stack"].astype(str).eq(stack)].copy()
        if part.empty:
            raise RuntimeError(f"No archived rows for stack {stack}.")

        state_keys: list[str] = []
        for row in part.itertuples(index=False):
            pieces = []
            for feature in features:
                col = f"reference_{feature}"
                if col not in part.columns:
                    raise RuntimeError(f"Missing archived reference feature: {col}")
                value = getattr(row, col)
                if pd.isna(value):
                    raise RuntimeError(
                        f"Missing required reference feature {col} for {stack}."
                    )
                pieces.append(f"{feature}={exact_float_key(float(value))}")
            state_keys.append("|".join(pieces))
        part["raw_state_key"] = state_keys
        part["cheap_probability_key"] = part["reference_cheap_probability"].map(
            exact_float_key
        )
        part["reference_distance"] = (
            part["reference_cheap_probability"].astype(float) - 0.5
        ).abs()
        part["distance_key"] = part["reference_distance"].map(exact_float_key)

        # The same stack/example appears under five deployable policy forms.
        # Cheap inputs/probabilities must be policy-independent.
        grouped = part.groupby("example_id", sort=False)
        for col in [
            "raw_state_key",
            "cheap_probability_key",
            "distance_key",
        ]:
            max_unique = int(grouped[col].nunique(dropna=False).max())
            if max_unique != 1:
                raise RuntimeError(
                    f"{col} is not invariant across policies for stack {stack}."
                )

        one = (
            part.sort_values(["example_id", "policy_id"])
            .drop_duplicates(["example_id"], keep="first")
            [
                [
                    "stack",
                    "example_id",
                    "raw_state_key",
                    "cheap_probability_key",
                    "reference_cheap_probability",
                    "reference_distance",
                    "distance_key",
                ]
            ]
            .copy()
        )

        state_mult = one["raw_state_key"].value_counts()
        prob_mult = one["cheap_probability_key"].value_counts()
        distance_mult = one["distance_key"].value_counts()

        one["raw_state_multiplicity"] = one["raw_state_key"].map(state_mult).astype(int)
        one["cheap_probability_multiplicity"] = (
            one["cheap_probability_key"].map(prob_mult).astype(int)
        )
        one["distance_multiplicity"] = (
            one["distance_key"].map(distance_mult).astype(int)
        )
        rows.extend(one.to_dict(orient="records"))

    states = pd.DataFrame(rows)
    if states.duplicated(["stack", "example_id"]).any():
        raise RuntimeError("Duplicate stack/example states after deduplication.")
    return states


def grouped_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for key, g in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: value for col, value in zip(group_cols, key, strict=True)}
        dead = g["in_reporting_deadband"].astype(bool)
        tie = g["exact_threshold_tie"].astype(bool)
        mismatch = g["route_mismatch"].astype(bool)
        repeated_raw = g["repeated_raw_cheap_state"].astype(bool)
        repeated_distance = g["repeated_route_distance_mass"].astype(bool)

        rec.update(
            {
                "route_rows": int(len(g)),
                "deadband_n": int(dead.sum()),
                "exact_threshold_tie_n": int((dead & tie).sum()),
                "nonexact_deadband_n": int((dead & ~tie).sum()),
                "deadband_repeated_raw_state_n": int((dead & repeated_raw).sum()),
                "deadband_repeated_distance_mass_n": int(
                    (dead & repeated_distance).sum()
                ),
                "route_mismatch_n": int(mismatch.sum()),
                "route_mismatch_at_exact_tie_n": int((mismatch & tie).sum()),
                "route_mismatch_nonexact_n": int((mismatch & ~tie).sum()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    epsilon = float(protocol["frozen_reporting"]["reporting_epsilon"])
    if epsilon != 1e-6:
        raise RuntimeError("Reporting epsilon must remain frozen at 1e-6.")

    margins = pd.read_parquet(MARGINS)
    raw = pd.read_parquet(RAW)
    defs = json.loads(DEFS.read_text(encoding="utf-8"))
    prior_sweep = pd.read_csv(PRIOR_SWEEP)
    prior_final = json.loads(PRIOR_FINAL.read_text(encoding="utf-8"))

    if len(margins) != EXPECTED_ALL_ROWS or len(raw) != EXPECTED_ALL_ROWS:
        raise RuntimeError(
            "Archived diagnostic population changed from 5,445 policy calls."
        )

    selective = margins[
        margins["policy_kind"].astype(str).eq("selective_deployable")
    ].copy()
    if len(selective) != EXPECTED_ROUTE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROUTE_ROWS} selective route rows, found {len(selective)}."
        )

    zero_row = prior_sweep[
        prior_sweep["epsilon"].astype(float).eq(0.0)
    ]
    eps_row = prior_sweep[
        prior_sweep["epsilon"].astype(float).eq(epsilon)
    ]
    if len(zero_row) != 1 or len(eps_row) != 1:
        raise RuntimeError("Prior dead-band sweep is missing frozen reporting rows.")
    zero_row = zero_row.iloc[0]
    eps_row = eps_row.iloc[0]

    if int(zero_row["route_reference_ambiguous_n"]) != EXPECTED_PRIOR_EXACT_REFERENCE_TIES:
        raise RuntimeError("Prior epsilon=0 exact-reference tie count changed.")
    if int(eps_row["route_union_ambiguous_n"]) != EXPECTED_PRIOR_EXACT_REFERENCE_TIES:
        raise RuntimeError("Prior epsilon=1e-6 route dead-band count changed.")
    if int(eps_row["route_mismatch_n"]) != EXPECTED_PRIOR_ROUTE_MISMATCHES:
        raise RuntimeError("Prior route mismatch count changed.")

    prior_fraction = float(prior_final["deadband"]["route_ambiguous_fraction"])
    expected_fraction = EXPECTED_PRIOR_EXACT_REFERENCE_TIES / EXPECTED_ROUTE_ROWS
    if not math.isclose(prior_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Prior reported 6.12% route fraction changed.")

    controlled_route_flips = int(
        prior_final["hardware"]["float32_policy_arithmetic_route_flips_cpu_vs_t4"]
    )
    controlled_prediction_flips = int(
        prior_final["hardware"]["float32_policy_arithmetic_prediction_flips_cpu_vs_t4"]
    )
    if controlled_route_flips != 0 or controlled_prediction_flips != 0:
        raise RuntimeError("Controlled CPU-T4 float32 no-flip result changed.")

    cheap_features, thresholds = frozen_maps(defs)
    states = build_stack_example_states(raw, margins, cheap_features)

    selective = selective.merge(
        states[
            [
                "stack",
                "example_id",
                "raw_state_key",
                "raw_state_multiplicity",
                "cheap_probability_multiplicity",
                "reference_distance",
                "distance_key",
                "distance_multiplicity",
            ]
        ],
        on=["stack", "example_id"],
        how="left",
        validate="many_to_one",
    )
    if selective["raw_state_key"].isna().any():
        raise RuntimeError("Failed to map some selective rows to cheap states.")

    selective["acquisition_threshold"] = selective["policy_id"].map(thresholds)
    if selective["acquisition_threshold"].isna().any():
        missing = sorted(
            selective.loc[
                selective["acquisition_threshold"].isna(), "policy_id"
            ].astype(str).unique()
        )
        raise RuntimeError(f"Missing frozen thresholds for policies: {missing}")

    recomputed = (
        selective["acquisition_threshold"].astype(float)
        - selective["reference_distance"].astype(float)
    )
    max_margin_error = float(
        np.max(
            np.abs(
                recomputed.to_numpy(dtype=float)
                - selective["reference_route_margin"].to_numpy(dtype=float)
            )
        )
    )
    if max_margin_error > 2e-15:
        raise RuntimeError(
            f"Frozen route-margin reconstruction error too large: {max_margin_error}"
        )

    selective["in_reporting_deadband"] = (
        selective["reference_route_margin"].abs() <= epsilon
    ) | (
        selective["runtime_route_margin"].abs() <= epsilon
    )
    selective["exact_threshold_tie"] = (
        selective["reference_route_margin"].astype(float) == 0.0
    )
    selective["runtime_exact_threshold_tie"] = (
        selective["runtime_route_margin"].astype(float) == 0.0
    )
    selective["repeated_raw_cheap_state"] = (
        selective["raw_state_multiplicity"].astype(int) >= 2
    )
    selective["repeated_route_distance_mass"] = (
        selective["distance_multiplicity"].astype(int) >= 2
    )
    selective["route_mismatch"] = selective["route_mismatch"].astype(bool)
    selective["non_tied_route_crossing"] = (
        selective["route_mismatch"] & ~selective["exact_threshold_tie"]
    )

    dead = selective[selective["in_reporting_deadband"]].copy()
    dead.sort_values(["stack", "policy_id", "example_id"], inplace=True)

    computed_deadband_n = int(len(dead))
    exact_tie_n = int(dead["exact_threshold_tie"].sum())
    nonexact_deadband_n = int(
        (dead["in_reporting_deadband"] & ~dead["exact_threshold_tie"]).sum()
    )
    route_mismatch_n = int(selective["route_mismatch"].sum())
    route_mismatch_at_tie_n = int(
        (selective["route_mismatch"] & selective["exact_threshold_tie"]).sum()
    )
    route_mismatch_nonexact_n = int(selective["non_tied_route_crossing"].sum())
    repeated_raw_tie_n = int(
        (dead["exact_threshold_tie"] & dead["repeated_raw_cheap_state"]).sum()
    )
    repeated_distance_tie_n = int(
        (dead["exact_threshold_tie"] & dead["repeated_route_distance_mass"]).sum()
    )
    runtime_exact_tie_n = int(dead["runtime_exact_threshold_tie"].sum())
    reference_ties_shifted_off_exact_runtime_n = int(
        (
            dead["exact_threshold_tie"]
            & ~dead["runtime_exact_threshold_tie"]
        ).sum()
    )

    if computed_deadband_n != int(eps_row["route_union_ambiguous_n"]):
        raise RuntimeError("Recomputed route dead-band count does not match prior study.")
    if exact_tie_n != int(zero_row["route_reference_ambiguous_n"]):
        raise RuntimeError("Exact-tie count does not match prior epsilon=0 evidence.")
    if route_mismatch_n != EXPECTED_PRIOR_ROUTE_MISMATCHES:
        raise RuntimeError("Archived route mismatch count changed.")

    OUT.mkdir(parents=True, exist_ok=True)

    dead_columns = [
        "policy_id",
        "example_id",
        "stack",
        "target_rate",
        "acquisition_threshold",
        "reference_cheap_probability",
        "runtime_cheap_probability",
        "cheap_probability_delta_runtime_minus_reference",
        "reference_distance",
        "reference_route_margin",
        "runtime_route_margin",
        "exact_threshold_tie",
        "runtime_exact_threshold_tie",
        "raw_state_multiplicity",
        "cheap_probability_multiplicity",
        "distance_multiplicity",
        "repeated_raw_cheap_state",
        "repeated_route_distance_mass",
        "route_mismatch",
        "non_tied_route_crossing",
    ]
    dead[dead_columns].to_csv(OUT / "deadband_rows.csv", index=False)

    policy_summary = grouped_summary(
        selective,
        ["stack", "policy_id", "target_rate", "acquisition_threshold"],
    )
    policy_summary.to_csv(OUT / "policy_summary.csv", index=False)

    stack_summary = grouped_summary(selective, ["stack"])
    stack_summary.to_csv(OUT / "stack_summary.csv", index=False)

    state_rows: list[dict] = []
    for stack, g in states.groupby("stack", sort=True):
        state_rows.append(
            {
                "stack": str(stack),
                "cheap_monitor_features": ",".join(cheap_features[str(stack)]),
                "unique_stack_example_states": int(len(g)),
                "raw_state_support_size": int(g["raw_state_key"].nunique()),
                "cheap_probability_support_size": int(
                    g["cheap_probability_key"].nunique()
                ),
                "route_distance_support_size": int(g["distance_key"].nunique()),
                "examples_in_repeated_raw_states": int(
                    (g["raw_state_multiplicity"] >= 2).sum()
                ),
                "examples_in_repeated_probability_mass": int(
                    (g["cheap_probability_multiplicity"] >= 2).sum()
                ),
                "examples_in_repeated_distance_mass": int(
                    (g["distance_multiplicity"] >= 2).sum()
                ),
                "max_raw_state_multiplicity": int(
                    g["raw_state_multiplicity"].max()
                ),
                "max_probability_multiplicity": int(
                    g["cheap_probability_multiplicity"].max()
                ),
                "max_distance_multiplicity": int(
                    g["distance_multiplicity"].max()
                ),
            }
        )
    cheap_state_summary = pd.DataFrame(state_rows)
    cheap_state_summary.to_csv(OUT / "cheap_state_summary.csv", index=False)

    deadband_fraction = computed_deadband_n / EXPECTED_ROUTE_ROWS
    exact_tie_share = (
        exact_tie_n / computed_deadband_n if computed_deadband_n else math.nan
    )
    repeated_raw_share = (
        repeated_raw_tie_n / exact_tie_n if exact_tie_n else math.nan
    )
    repeated_distance_share = (
        repeated_distance_tie_n / exact_tie_n if exact_tie_n else math.nan
    )

    if exact_tie_n == computed_deadband_n and route_mismatch_nonexact_n == 0:
        interpretation = (
            "At the frozen 1e-6 reporting epsilon, every route-deadband row "
            "already has exactly zero canonical reference route margin. The "
            "epsilon adds no non-tied route rows. The observed route flips occur "
            "on exact ties, with no observed non-tied route crossing. Repeated "
            "cheap-score states and routing-distance mass points are reported "
            "separately to quantify discreteness; the result is specific to this "
            "archived development-only population."
        )
    else:
        interpretation = (
            "The route dead-band contains both exact ties and non-tied rows. "
            "Counts are reported without assigning universal causal attribution."
        )

    summary = {
        "artifact": "threshold_tie_diagnostic_v1",
        "status": "completed_development_only_archived_diagnostic",
        "base_commit": str(protocol["base_commit"]),
        "reporting_epsilon": epsilon,
        "route_rows": EXPECTED_ROUTE_ROWS,
        "deadband_ambiguous_n": computed_deadband_n,
        "deadband_ambiguous_fraction": deadband_fraction,
        "exact_threshold_tie_n": exact_tie_n,
        "exact_threshold_tie_share_of_deadband": exact_tie_share,
        "nonexact_deadband_n": nonexact_deadband_n,
        "nonexact_deadband_fraction_of_route_rows": (
            nonexact_deadband_n / EXPECTED_ROUTE_ROWS
        ),
        "exact_ties_with_repeated_raw_cheap_state_n": repeated_raw_tie_n,
        "exact_ties_with_repeated_raw_cheap_state_share": repeated_raw_share,
        "exact_ties_with_repeated_route_distance_mass_n": repeated_distance_tie_n,
        "exact_ties_with_repeated_route_distance_mass_share": repeated_distance_share,
        "route_mismatch_n": route_mismatch_n,
        "route_mismatch_at_exact_tie_n": route_mismatch_at_tie_n,
        "route_mismatch_nonexact_n": route_mismatch_nonexact_n,
        "runtime_exact_threshold_tie_n_in_deadband": runtime_exact_tie_n,
        "reference_ties_shifted_off_exact_runtime_n": (
            reference_ties_shifted_off_exact_runtime_n
        ),
        "controlled_cpu_t4_float32_route_flips": controlled_route_flips,
        "controlled_cpu_t4_float32_prediction_flips": controlled_prediction_flips,
        "max_route_margin_reconstruction_error": max_margin_error,
        "interpretation": interpretation,
        "claim_boundary": {
            "floating_point_general_safety_failure_claim": False,
            "universal_causal_attribution_claim": False,
            "fresh_confirmatory_claim": False,
            "router_superiority_claim": False,
            "protected_legacy_splits_used": False,
        },
        "data_boundary": {
            "new_data_collected": False,
            "labels_used": False,
            "model_fitting": False,
            "threshold_reselection": False,
            "router_retuning": False,
        },
        "workflow_boundary": {
            "threshold_tie_diagnostic_complete": True,
            "existing_data_discovery_closed": True,
            "new_discovery_on_existing_data_allowed": False,
            "fresh_data_collection_started": False,
            "fresh_transport_protocol_run": False,
            "next_step": "prepare_preregistered_fresh_risk_certificate_transport_protocol",
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs = [
        OUT / "deadband_rows.csv",
        OUT / "policy_summary.csv",
        OUT / "stack_summary.csv",
        OUT / "cheap_state_summary.csv",
        OUT / "summary.json",
    ]
    manifest = {
        "artifact": "threshold_tie_diagnostic_v1_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [MARGINS, RAW, DEFS, PRIOR_SWEEP, PRIOR_FINAL]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in outputs
        },
        "protected_legacy_splits_used": False,
        "new_data_collected": False,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("THRESHOLD_TIE_DIAGNOSTIC=PASS")
    print(f"route_rows={EXPECTED_ROUTE_ROWS}")
    print(f"deadband_ambiguous_n={computed_deadband_n}")
    print(f"deadband_ambiguous_fraction={deadband_fraction:.15f}")
    print(f"exact_threshold_tie_n={exact_tie_n}")
    print(f"exact_threshold_tie_share_of_deadband={exact_tie_share:.15f}")
    print(f"nonexact_deadband_n={nonexact_deadband_n}")
    print(f"exact_ties_with_repeated_raw_cheap_state_n={repeated_raw_tie_n}")
    print(f"exact_ties_with_repeated_route_distance_mass_n={repeated_distance_tie_n}")
    print(f"route_mismatch_n={route_mismatch_n}")
    print(f"route_mismatch_at_exact_tie_n={route_mismatch_at_tie_n}")
    print(f"route_mismatch_nonexact_n={route_mismatch_nonexact_n}")
    print(f"runtime_exact_threshold_tie_n_in_deadband={runtime_exact_tie_n}")
    print(
        "reference_ties_shifted_off_exact_runtime_n="
        f"{reference_ties_shifted_off_exact_runtime_n}"
    )
    print(f"controlled_cpu_t4_float32_route_flips={controlled_route_flips}")
    print(
        "controlled_cpu_t4_float32_prediction_flips="
        f"{controlled_prediction_flips}"
    )
    print("existing_data_discovery_closed=true")
    print("next_step=prepare_preregistered_fresh_risk_certificate_transport_protocol")


if __name__ == "__main__":
    main()
