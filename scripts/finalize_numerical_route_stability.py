#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CPU_MODULE_PATH = ROOT / "scripts/run_numerical_route_stability_cpu.py"
PROTOCOL = ROOT / "configs/numerical_route_stability_v1.json"
RAW = ROOT / "reports/evaluation_measurement_pilot_v1/t4/policy_latency_raw.parquet"
DEFS = ROOT / "reports/evaluation_measurement_pilot_v1/cpu/primary_policy_definitions.json"
CPU_DEADBAND = ROOT / "reports/numerical_route_stability_v1/cpu/deadband_sweep.csv"
CPU_RUNTIME = ROOT / "reports/numerical_route_stability_v1/cpu/runtime_precision_matrix.csv"
T4_DIR = ROOT / "reports/numerical_route_stability_v1/t4"
FINAL_DIR = ROOT / "reports/numerical_route_stability_v1/final"

EPS = [0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cpu_module():
    spec = importlib.util.spec_from_file_location("numerical_stability_cpu", CPU_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen CPU stability implementation.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["numerical_stability_cpu"] = module
    spec.loader.exec_module(module)
    return module


def validate_t4_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = T4_DIR / "manifest.json"
    env_path = T4_DIR / "environment.json"
    if not manifest_path.is_file() or not env_path.is_file():
        raise RuntimeError("T4 results are incomplete.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact") != "numerical_route_stability_v1_t4_results":
        raise RuntimeError("Unexpected T4 results artifact.")

    for name, expected in manifest["files"].items():
        path = T4_DIR / name
        if not path.is_file():
            raise RuntimeError(f"Missing T4 result file: {name}")
        if sha256(path) != expected:
            raise RuntimeError(f"T4 result hash mismatch: {name}")

    env = json.loads(env_path.read_text(encoding="utf-8"))
    if "T4" not in str(env.get("gpu_name", "")):
        raise RuntimeError("Controlled hardware result is not from an NVIDIA T4.")
    if env.get("compact_model_id") != "KoalaAI/Text-Moderation":
        raise RuntimeError("Compact monitor identity changed.")
    if env.get("compact_revision") != "838ade0edb66dcffc5532d08ff6ed5c899abfb5c":
        raise RuntimeError("Compact monitor revision changed.")
    if int(env.get("repetitions", -1)) != 10:
        raise RuntimeError("Unexpected hardware repetition count.")
    if env.get("tf32_disabled") is not True:
        raise RuntimeError("TF32 was not disabled in the controlled T4 run.")
    if env.get("cudnn_deterministic") is not True:
        raise RuntimeError("Deterministic cuDNN setting missing.")

    monitor_cmp = pd.read_csv(T4_DIR / "compact_monitor_cpu_t4_comparison.csv")
    monitor_summary = pd.read_csv(T4_DIR / "compact_monitor_summary.csv")
    arithmetic = pd.read_csv(T4_DIR / "policy_arithmetic_hardware_precision.csv")

    if len(monitor_cmp) != 5 or monitor_cmp["example_id"].nunique() != 5:
        raise RuntimeError("Expected five controlled monitor-comparison examples.")
    if len(arithmetic) != 25:
        raise RuntimeError("Expected 25 controlled policy-arithmetic rows.")
    return monitor_cmp, monitor_summary, arithmetic, env


def cross_hardware_summary(arithmetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for precision in ("float64", "float32"):
        cpu = arithmetic[
            (arithmetic["hardware"] == "cpu")
            & (arithmetic["precision"] == precision)
        ].set_index(["policy_id", "example_id"])
        t4 = arithmetic[
            (arithmetic["hardware"] == "t4")
            & (arithmetic["precision"] == precision)
        ].set_index(["policy_id", "example_id"])
        joined = cpu.join(t4, lsuffix="_cpu", rsuffix="_t4", validate="one_to_one")
        if len(joined) != 5:
            raise RuntimeError(f"Hardware arithmetic coverage changed for {precision}.")
        rows.append(
            {
                "precision": precision,
                "cases": int(len(joined)),
                "route_flip_n_cpu_vs_t4": int(
                    (joined["acquired_cpu"] != joined["acquired_t4"]).sum()
                ),
                "prediction_flip_n_cpu_vs_t4": int(
                    (joined["prediction_cpu"] != joined["prediction_t4"]).sum()
                ),
                "max_abs_probability_delta_cpu_vs_t4": float(
                    (joined["probability_cpu"] - joined["probability_t4"]).abs().max()
                ),
                "max_abs_route_margin_delta_cpu_vs_t4": float(
                    (joined["route_margin_cpu"] - joined["route_margin_t4"])
                    .abs()
                    .max(skipna=True)
                ),
                "max_abs_decision_margin_delta_cpu_vs_t4": float(
                    (joined["decision_margin_cpu"] - joined["decision_margin_t4"])
                    .abs()
                    .max()
                ),
            }
        )
    return pd.DataFrame(rows)


def full_population_axis_analysis(cpu) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(RAW)
    defs = json.loads(DEFS.read_text(encoding="utf-8"))
    recs = cpu.records(defs)

    variants = [
        ("numpy_float64", "numpy", "float64"),
        ("torch_cpu_float64", "torch_cpu", "float64"),
        ("numpy_float32", "numpy", "float32"),
        ("torch_cpu_float32", "torch_cpu", "float32"),
    ]

    rows = []
    summary = {}
    for variant_name, implementation, precision in variants:
        route_flips = 0
        pred_flips = 0
        same_route_pred_flips = 0
        max_route_margin_delta = 0.0
        max_same_route_probability_delta = 0.0
        for row in raw.itertuples(index=False):
            rec = recs[str(row.policy_id)]
            features = cpu.feature_dict(row, "reference_")
            base = cpu.evaluate(rec, features, "python_scalar", "float64")
            result = cpu.evaluate(rec, features, implementation, precision)

            route_flip = bool(result[0] != base[0])
            pred_flip = bool(result[2] != base[2])
            route_flips += int(route_flip)
            pred_flips += int(pred_flip)

            if math.isfinite(base[3]) and math.isfinite(result[3]):
                rdelta = abs(result[3] - base[3])
                max_route_margin_delta = max(max_route_margin_delta, rdelta)
            if not route_flip:
                pdelta = abs(result[1] - base[1])
                max_same_route_probability_delta = max(
                    max_same_route_probability_delta, pdelta
                )
                if pred_flip:
                    same_route_pred_flips += 1

            rows.append(
                {
                    "variant": variant_name,
                    "policy_id": str(row.policy_id),
                    "example_id": str(row.example_id),
                    "route_flip": route_flip,
                    "prediction_flip": pred_flip,
                    "same_route_prediction_flip": bool(pred_flip and not route_flip),
                    "canonical_route_margin": base[3],
                    "variant_route_margin": result[3],
                    "canonical_decision_margin": base[4],
                    "variant_decision_margin": result[4],
                    "abs_probability_delta_if_same_route": (
                        abs(result[1] - base[1]) if not route_flip else math.nan
                    ),
                }
            )

        summary[variant_name] = {
            "route_flip_n": route_flips,
            "prediction_flip_n": pred_flips,
            "same_route_prediction_flip_n": same_route_pred_flips,
            "max_abs_route_margin_delta": max_route_margin_delta,
            "max_abs_probability_delta_if_same_route": max_same_route_probability_delta,
        }

    return pd.DataFrame(rows), summary


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    monitor_cmp, monitor_summary, arithmetic, env = validate_t4_results()
    cpu = load_cpu_module()

    hardware = cross_hardware_summary(arithmetic)
    hardware.to_csv(FINAL_DIR / "hardware_axis_summary.csv", index=False)

    axis_rows, axis_summary = full_population_axis_analysis(cpu)
    axis_rows.to_parquet(FINAL_DIR / "full_population_axis_diagnostics.parquet", index=False)

    # Controlled float32 hardware behavior.
    hw32 = hardware.loc[hardware["precision"] == "float32"].iloc[0]
    if int(hw32["route_flip_n_cpu_vs_t4"]) != 0:
        raise RuntimeError("Controlled float32 CPU-vs-T4 arithmetic changed a route.")
    if int(hw32["prediction_flip_n_cpu_vs_t4"]) != 0:
        raise RuntimeError("Controlled float32 CPU-vs-T4 arithmetic changed a prediction.")

    # Repeated compact-monitor evaluation was deterministic within each hardware/precision cell.
    finite_std = pd.to_numeric(monitor_summary["std_score"], errors="coerce").fillna(0.0)
    max_repeat_std = float(finite_std.max())
    if max_repeat_std != 0.0:
        raise RuntimeError("Controlled compact-monitor repetitions were not deterministic.")

    monitor_hw_max = float(monitor_cmp["abs_score_difference"].max())

    # Exploratory float16 is explicitly excluded from the supported path.
    t4_32 = arithmetic[
        (arithmetic["hardware"] == "t4")
        & (arithmetic["precision"] == "float32")
    ].set_index(["policy_id", "example_id"])
    t4_16 = arithmetic[
        (arithmetic["hardware"] == "t4")
        & (arithmetic["precision"] == "float16_exploratory")
    ].set_index(["policy_id", "example_id"])
    exploratory = t4_32.join(t4_16, lsuffix="_32", rsuffix="_16", validate="one_to_one")
    float16_route_flips = int(
        (exploratory["acquired_32"] != exploratory["acquired_16"]).sum()
    )
    float16_prediction_flips = int(
        (exploratory["prediction_32"] != exploratory["prediction_16"]).sum()
    )

    # Non-exploratory full-population precision/runtime envelope.
    route_axis_envelope = max(
        axis_summary["numpy_float32"]["max_abs_route_margin_delta"],
        axis_summary["torch_cpu_float32"]["max_abs_route_margin_delta"],
    )
    decision_axis_envelope_same_route = max(
        axis_summary["numpy_float32"]["max_abs_probability_delta_if_same_route"],
        axis_summary["torch_cpu_float32"]["max_abs_probability_delta_if_same_route"],
    )
    runtime_float64_envelope = max(
        axis_summary["numpy_float64"]["max_abs_route_margin_delta"],
        axis_summary["torch_cpu_float64"]["max_abs_route_margin_delta"],
        axis_summary["numpy_float64"]["max_abs_probability_delta_if_same_route"],
        axis_summary["torch_cpu_float64"]["max_abs_probability_delta_if_same_route"],
    )

    hardware_route_envelope = float(hw32["max_abs_route_margin_delta_cpu_vs_t4"])
    hardware_decision_envelope = float(hw32["max_abs_decision_margin_delta_cpu_vs_t4"])

    # Convert the controlled monitor-score CPU/T4 difference into a conservative
    # downstream probability perturbation using sigmoid's global 1/4 Lipschitz constant.
    defs = json.loads(DEFS.read_text(encoding="utf-8"))
    compact_coefs = []
    for stack_cfg in defs["stacks"].values():
        for role in ("cheap_model", "full_model"):
            model = stack_cfg[role]
            for coef, feature in zip(model["coef"], model["features"], strict=True):
                if feature == "compact_unsafe_score":
                    compact_coefs.append(abs(float(coef)))
    max_compact_coef = max(compact_coefs)
    monitor_probability_bound = 0.25 * max_compact_coef * monitor_hw_max

    # Conservative composition for the supported non-exploratory path:
    # canonical CPU float64 -> CPU float32 implementation -> T4 float32 arithmetic
    # -> hardware monitor-score variation. This is an envelope, not a fitted epsilon.
    conservative_route_envelope = (
        route_axis_envelope + hardware_route_envelope + monitor_probability_bound
    )
    conservative_decision_envelope = (
        decision_axis_envelope_same_route
        + hardware_decision_envelope
        + monitor_probability_bound
    )
    required_envelope = max(
        conservative_route_envelope,
        conservative_decision_envelope,
        runtime_float64_envelope,
    )

    reporting_epsilon = next(
        (eps for eps in EPS if eps >= required_envelope),
        None,
    )
    if reporting_epsilon is None:
        raise RuntimeError(
            "Prespecified dead-band grid does not cover the measured "
            f"non-exploratory envelope ({required_envelope:.17g})."
        )

    # Enforce the precommitted selection rule itself, not an outcome-specific
    # numerical value. The chosen point must be the FIRST frozen grid value
    # that covers the measured envelope.
    reporting_index = EPS.index(reporting_epsilon)
    if reporting_epsilon < required_envelope:
        raise RuntimeError("Reporting epsilon does not cover the measured envelope.")
    if (
        reporting_index > 0
        and EPS[reporting_index - 1] >= required_envelope
    ):
        raise RuntimeError(
            "Reporting epsilon is not the first prespecified grid point "
            "covering the measured envelope."
        )

    dead = pd.read_csv(CPU_DEADBAND)
    row = dead.loc[
        np.isclose(
            dead["epsilon"],
            reporting_epsilon,
            rtol=0.0,
            atol=1e-20,
        )
    ]
    if len(row) != 1:
        raise RuntimeError(
            "Could not locate the envelope-derived dead-band row "
            f"for epsilon={reporting_epsilon:.17g}."
        )
    d = row.iloc[0]
    route_ambiguous_fraction = float(
        d["route_union_ambiguous_n"] / d["route_rows"]
    )
    decision_ambiguous_fraction = float(
        d["decision_union_ambiguous_n"] / d["decision_rows"]
    )

    # Verify that every observed non-exploratory float32 diagnostic flip lies
    # inside the envelope-derived ambiguity region. This is a validation of
    # the dead-band interpretation, not a rule for choosing epsilon.
    diagnostic_coverage = {}
    for name in ("numpy_float32", "torch_cpu_float32"):
        part = axis_rows[axis_rows["variant"].eq(name)]

        route_flips = part[part["route_flip"].astype(bool)]
        route_covered = route_flips[
            route_flips["canonical_route_margin"].abs().le(reporting_epsilon)
            | route_flips["variant_route_margin"].abs().le(reporting_epsilon)
        ]

        decision_flips = part[
            part["same_route_prediction_flip"].astype(bool)
        ]
        decision_covered = decision_flips[
            decision_flips["canonical_decision_margin"].abs().le(reporting_epsilon)
            | decision_flips["variant_decision_margin"].abs().le(reporting_epsilon)
        ]

        if len(route_covered) != len(route_flips):
            raise RuntimeError(
                f"Envelope-derived route dead-band does not cover all {name} "
                "diagnostic route flips."
            )
        if len(decision_covered) != len(decision_flips):
            raise RuntimeError(
                f"Envelope-derived decision dead-band does not cover all {name} "
                "same-route diagnostic prediction flips."
            )

        diagnostic_coverage[name] = {
            "route_flip_n": int(len(route_flips)),
            "route_flip_covered_n": int(len(route_covered)),
            "same_route_prediction_flip_n": int(len(decision_flips)),
            "same_route_prediction_flip_covered_n": int(len(decision_covered)),
        }

    envelope_rows = pd.DataFrame(
        [
            {
                "axis": "runtime_implementation_float64_cpu",
                "supported": True,
                "max_margin_or_same_route_probability_perturbation": runtime_float64_envelope,
            },
            {
                "axis": "precision_float32_cpu",
                "supported": True,
                "max_margin_or_same_route_probability_perturbation": max(
                    route_axis_envelope, decision_axis_envelope_same_route
                ),
            },
            {
                "axis": "hardware_float32_cpu_vs_t4_arithmetic",
                "supported": True,
                "max_margin_or_same_route_probability_perturbation": max(
                    hardware_route_envelope, hardware_decision_envelope
                ),
            },
            {
                "axis": "hardware_float32_compact_monitor_lipschitz_bound",
                "supported": True,
                "max_margin_or_same_route_probability_perturbation": monitor_probability_bound,
            },
            {
                "axis": "t4_float16_exploratory",
                "supported": False,
                "max_margin_or_same_route_probability_perturbation": math.nan,
            },
        ]
    )
    envelope_rows.to_csv(FINAL_DIR / "stability_envelopes.csv", index=False)

    final_deadband = pd.DataFrame(
        [
            {
                "reporting_epsilon": reporting_epsilon,
                "selection_rule": (
                    "first_prespecified_grid_point_covering_conservative_"
                    "nonexploratory_perturbation_envelope"
                ),
                "posthoc_mismatch_elimination_used": False,
                "required_nonexploratory_envelope": required_envelope,
                "previous_prespecified_epsilon": (
                    math.nan
                    if reporting_index == 0
                    else EPS[reporting_index - 1]
                ),
                "runtime_float64_envelope": runtime_float64_envelope,
                "precision_float32_route_envelope": route_axis_envelope,
                "precision_float32_same_route_decision_envelope": decision_axis_envelope_same_route,
                "hardware_float32_route_envelope": hardware_route_envelope,
                "hardware_float32_decision_envelope": hardware_decision_envelope,
                "compact_monitor_probability_lipschitz_bound": monitor_probability_bound,
                "conservative_route_envelope": conservative_route_envelope,
                "conservative_same_route_decision_envelope": conservative_decision_envelope,
                "route_ambiguous_n": int(d["route_union_ambiguous_n"]),
                "route_rows": int(d["route_rows"]),
                "route_ambiguous_fraction": route_ambiguous_fraction,
                "decision_ambiguous_n": int(d["decision_union_ambiguous_n"]),
                "decision_rows": int(d["decision_rows"]),
                "decision_ambiguous_fraction": decision_ambiguous_fraction,
            }
        ]
    )
    final_deadband.to_csv(FINAL_DIR / "final_deadband_summary.csv", index=False)

    summary = {
        "artifact": "numerical_route_stability_v1_final",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_development_only",
        "archived_mismatch_taxonomy": {
            "route_threshold_mismatches": 2,
            "prediction_mismatches": 5,
            "pure_decision_threshold_mismatches": 3,
            "route_induced_prediction_mismatches": 2,
        },
        "runtime_implementation": {
            "float64_cpu_max_perturbation": runtime_float64_envelope,
            "interpretation": "float64 Python/NumPy/PyTorch CPU arithmetic is effectively invariant",
        },
        "precision": {
            "float32_cpu_route_envelope": route_axis_envelope,
            "float32_cpu_same_route_decision_envelope": decision_axis_envelope_same_route,
            "numpy_float32_route_flips": axis_summary["numpy_float32"]["route_flip_n"],
            "numpy_float32_prediction_flips": axis_summary["numpy_float32"]["prediction_flip_n"],
            "torch_float32_route_flips": axis_summary["torch_cpu_float32"]["route_flip_n"],
            "torch_float32_prediction_flips": axis_summary["torch_cpu_float32"]["prediction_flip_n"],
            "t4_float16_exploratory_route_flips_on_five_cases": float16_route_flips,
            "t4_float16_exploratory_prediction_flips_on_five_cases": float16_prediction_flips,
            "float16_supported_path": False,
        },
        "hardware": {
            "gpu": env["gpu_name"],
            "float32_monitor_max_abs_score_difference_cpu_vs_t4": monitor_hw_max,
            "float32_monitor_max_repeat_std": max_repeat_std,
            "float32_policy_arithmetic_route_flips_cpu_vs_t4": int(
                hw32["route_flip_n_cpu_vs_t4"]
            ),
            "float32_policy_arithmetic_prediction_flips_cpu_vs_t4": int(
                hw32["prediction_flip_n_cpu_vs_t4"]
            ),
            "float32_policy_arithmetic_max_margin_delta_cpu_vs_t4": max(
                hardware_route_envelope, hardware_decision_envelope
            ),
        },
        "deadband": {
            "prespecified_grid": EPS,
            "reporting_epsilon": reporting_epsilon,
            "required_nonexploratory_envelope": required_envelope,
            "previous_prespecified_epsilon": (
                None
                if reporting_index == 0
                else EPS[reporting_index - 1]
            ),
            "selection_rule": (
                "first prespecified grid point covering the conservative "
                "non-exploratory perturbation envelope"
            ),
            "selected_to_eliminate_mismatches": False,
            "diagnostic_flip_coverage": diagnostic_coverage,
            "route_ambiguous_fraction": route_ambiguous_fraction,
            "decision_ambiguous_fraction": decision_ambiguous_fraction,
            "interpretation": (
                "Margins inside the dead-band are numerically ambiguous; this study "
                "does not assign a deployment action to ambiguity."
            ),
        },
        "claim_boundary": {
            "protected_legacy_splits_used": False,
            "fresh_confirmatory_claim": False,
            "router_superiority_claim": False,
            "pareto_claim_available": False,
            "universal_hardware_invariance_claim": False,
            "security_control_plane_study_started": False,
        },
        "next_step": "small_safety_availability_control_plane_kill_study",
    }
    (FINAL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs = [
        FINAL_DIR / "hardware_axis_summary.csv",
        FINAL_DIR / "full_population_axis_diagnostics.parquet",
        FINAL_DIR / "stability_envelopes.csv",
        FINAL_DIR / "final_deadband_summary.csv",
        FINAL_DIR / "summary.json",
    ]
    manifest = {
        "artifact": "numerical_route_stability_v1_final_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                PROTOCOL,
                RAW,
                DEFS,
                CPU_DEADBAND,
                CPU_RUNTIME,
                T4_DIR / "manifest.json",
                T4_DIR / "environment.json",
                T4_DIR / "compact_monitor_cpu_t4_comparison.csv",
                T4_DIR / "compact_monitor_summary.csv",
                T4_DIR / "policy_arithmetic_hardware_precision.csv",
            ]
        },
        "outputs": {
            str(p.relative_to(FINAL_DIR)): sha256(p)
            for p in outputs
        },
    }
    (FINAL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("NUMERICAL_ROUTE_STABILITY_FINAL=PASS")
    print(f"runtime_float64_envelope={runtime_float64_envelope:.17g}")
    print(f"precision_float32_route_envelope={route_axis_envelope:.17g}")
    print(
        "precision_float32_same_route_decision_envelope="
        f"{decision_axis_envelope_same_route:.17g}"
    )
    print(f"hardware_float32_route_envelope={hardware_route_envelope:.17g}")
    print(f"hardware_float32_decision_envelope={hardware_decision_envelope:.17g}")
    print(
        "compact_monitor_probability_lipschitz_bound="
        f"{monitor_probability_bound:.17g}"
    )
    print(f"required_nonexploratory_envelope={required_envelope:.17g}")
    print(f"reporting_epsilon={reporting_epsilon:.0e}")
    print(f"route_ambiguous_fraction={route_ambiguous_fraction:.9f}")
    print(f"decision_ambiguous_fraction={decision_ambiguous_fraction:.9f}")
    print(f"float16_exploratory_route_flips={float16_route_flips}/5")
    print(f"float16_exploratory_prediction_flips={float16_prediction_flips}/5")
    print("step3_complete=true")
    print("next_step=small_safety_availability_control_plane_kill_study")


if __name__ == "__main__":
    main()
