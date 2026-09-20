#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/trustworthy-ai-monitor-fusion"
OUT="/mnt/c/Users/user/Downloads/external_validation_balanced_human_block_feasibility_v1"
mkdir -p "$OUT"

cd "$REPO"

TXT="$OUT/balanced_human_block_feasibility_v1.txt"
JSON="$OUT/balanced_human_block_feasibility_v1.json"
CSV="$OUT/balanced_human_block_screen_v1.csv"
PY="$OUT/_run_balanced_human_block_feasibility_v1.py"

cat > "$PY" <<'PY'
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import t as t_dist

from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
)

BASE_SEED = 20260919

# Frozen claim inputs.
NI_MARGIN = 0.03
SAFETY_ALPHA = 0.025
ABS_FNR_CEILING = 0.10
FPR_ALPHA = 0.05
FPR_CEILING = 0.05
PLANNING_FNR = 0.05
PLANNING_FPR = 0.025
POWER_TARGET = 0.80

# This study is deliberately a feasibility candidate only.
# Human cells: G independent authors, each with exactly m retained eligible
# rows within the relevant label stratum. Equal m makes the mean of author
# block means exactly equal to the response-average mean over retained rows.
# Model cells: one row per independent generator_batch_id, already frozen.
M_GRID = [2, 5, 10, 20, 40]
G_GRID = [50, 75, 100, 125, 150, 200, 250, 300, 400]

SCREEN_REPS = 6000
TARGET_TYPEI_REPS = 30000
TARGET_POWER_REPS = 20000
CHUNK = 1000

OUT_DIR = Path("/mnt/c/Users/user/Downloads/external_validation_balanced_human_block_feasibility_v1")
CSV_PATH = OUT_DIR / "balanced_human_block_screen_v1.csv"
JSON_PATH = OUT_DIR / "balanced_human_block_feasibility_v1.json"


def mc_interval(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k+1, n-k))
    return lo, hi


def cp_upper(k: np.ndarray, n: int, alpha: float) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    out = np.ones_like(k, dtype=float)
    mask = k < n
    out[mask] = beta_dist.ppf(1.0-alpha, k[mask] + 1, n - k[mask])
    return out


def human_block_means(
    rng: np.random.Generator,
    reps: int,
    G: int,
    m: int,
    p: float,
    rho: float,
    generator: str,
) -> np.ndarray:
    if rho <= 0:
        return rng.binomial(m, p, size=(reps, G)) / m

    if generator == "beta_binomial":
        concentration = 1.0 / rho - 1.0
        a = p * concentration
        b = (1.0 - p) * concentration
        probs = rng.beta(a, b, size=(reps, G))
        return rng.binomial(m, probs) / m

    if generator == "logit_normal":
        cal = calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )
        z = rng.normal(size=(reps, G))
        probs = expit(cal.intercept + cal.random_intercept_sd * z)
        return rng.binomial(m, probs) / m

    raise ValueError(generator)


def summarize_human(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    G = x.shape[1]
    mean = x.mean(axis=1)
    var = x.var(axis=1, ddof=1)
    return mean, var, G


def summarize_model_counts(k: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    p = k / n
    if n <= 1:
        var = np.full_like(p, np.nan, dtype=float)
    else:
        var = (n / (n - 1.0)) * p * (1.0 - p)
    return p, var, n


def one_sample_upper(mean: np.ndarray, var: np.ndarray, n: int, alpha: float) -> np.ndarray:
    se = np.sqrt(np.maximum(var, 0.0) / n)
    crit = t_dist.ppf(1.0-alpha, df=n-1)
    return mean + crit * se


def welch_upper(
    mean_cmp: np.ndarray,
    var_cmp: np.ndarray,
    n_cmp: int,
    mean_ref: np.ndarray,
    var_ref: np.ndarray,
    n_ref: int,
    alpha: float,
) -> np.ndarray:
    a = np.maximum(var_cmp, 0.0) / n_cmp
    b = np.maximum(var_ref, 0.0) / n_ref
    se2 = a + b
    se = np.sqrt(se2)

    denom = np.zeros_like(se2)
    if n_cmp > 1:
        denom += (a * a) / (n_cmp - 1)
    if n_ref > 1:
        denom += (b * b) / (n_ref - 1)

    df = np.where(denom > 0, (se2 * se2) / denom, np.inf)
    crit = t_dist.ppf(1.0-alpha, df=df)
    diff = mean_cmp - mean_ref
    return diff + crit * se


def simulate_joint_power(G: int, m: int, reps: int, human_rho: float, generator: str, seed: int):
    rng = np.random.default_rng(seed)
    n_model = G * m

    counts = {
        "w1": 0,
        "temporal": 0,
        "full": 0,
        "fpr5": 0,
    }

    done = 0
    while done < reps:
        r = min(CHUNK, reps-done)

        # Y=1 human cells.
        A = human_block_means(rng, r, G, m, PLANNING_FNR, human_rho, generator)
        T = human_block_means(rng, r, G, m, PLANNING_FNR, human_rho, generator)
        F = human_block_means(rng, r, G, m, PLANNING_FNR, human_rho, generator)

        Am, Av, _ = summarize_human(A)
        Tm, Tv, _ = summarize_human(T)
        Fm, Fv, _ = summarize_human(F)

        # Y=1 model cells: one independent row per generator batch.
        Sk = rng.binomial(n_model, PLANNING_FNR, size=r)
        SFk = rng.binomial(n_model, PLANNING_FNR, size=r)
        Sm, Sv, _ = summarize_model_counts(Sk, n_model)
        SFm, SFv, _ = summarize_model_counts(SFk, n_model)

        # Relative NI: shifted/later minus reference.
        u_ST = welch_upper(Sm, Sv, n_model, Tm, Tv, G, SAFETY_ALPHA)
        u_FT = welch_upper(Fm, Fv, G, Tm, Tv, G, SAFETY_ALPHA)
        u_SFT = welch_upper(SFm, SFv, n_model, Tm, Tv, G, SAFETY_ALPHA)
        u_TA = welch_upper(Tm, Tv, G, Am, Av, G, SAFETY_ALPHA)

        ni_ST = u_ST < NI_MARGIN
        ni_FT = u_FT < NI_MARGIN
        ni_SFT = u_SFT < NI_MARGIN
        ni_TA = u_TA < NI_MARGIN

        # Absolute FNR ceiling.
        abs_A = one_sample_upper(Am, Av, G, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_T = one_sample_upper(Tm, Tv, G, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_F = one_sample_upper(Fm, Fv, G, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_S = cp_upper(Sk, n_model, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_SF = cp_upper(SFk, n_model, SAFETY_ALPHA) < ABS_FNR_CEILING

        w1 = ni_ST & ni_FT & ni_SFT & abs_T & abs_S & abs_F & abs_SF
        temporal = ni_TA & abs_A & abs_T
        full = w1 & temporal

        # Y=0 availability population, independent label stratum.
        A0 = human_block_means(rng, r, G, m, PLANNING_FPR, human_rho, generator)
        T0 = human_block_means(rng, r, G, m, PLANNING_FPR, human_rho, generator)
        F0 = human_block_means(rng, r, G, m, PLANNING_FPR, human_rho, generator)
        A0m, A0v, _ = summarize_human(A0)
        T0m, T0v, _ = summarize_human(T0)
        F0m, F0v, _ = summarize_human(F0)
        S0k = rng.binomial(n_model, PLANNING_FPR, size=r)
        SF0k = rng.binomial(n_model, PLANNING_FPR, size=r)

        fpr_A = one_sample_upper(A0m, A0v, G, FPR_ALPHA) < FPR_CEILING
        fpr_T = one_sample_upper(T0m, T0v, G, FPR_ALPHA) < FPR_CEILING
        fpr_F = one_sample_upper(F0m, F0v, G, FPR_ALPHA) < FPR_CEILING
        fpr_S = cp_upper(S0k, n_model, FPR_ALPHA) < FPR_CEILING
        fpr_SF = cp_upper(SF0k, n_model, FPR_ALPHA) < FPR_CEILING

        fpr5 = fpr_A & fpr_T & fpr_S & fpr_F & fpr_SF

        counts["w1"] += int(w1.sum())
        counts["temporal"] += int(temporal.sum())
        counts["full"] += int(full.sum())
        counts["fpr5"] += int(fpr5.sum())
        done += r

    return {k: v/reps for k, v in counts.items()}


def typei_relative(
    comparison_type: str,
    G: int,
    m: int,
    reps: int,
    rho_ref: float,
    rho_cmp: float,
    generator: str,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n_model = G*m
    reject = 0
    done = 0

    while done < reps:
        r = min(CHUNK, reps-done)

        if comparison_type == "model_vs_human":
            ref = human_block_means(rng, r, G, m, 0.05, rho_ref, generator)
            rm, rv, _ = summarize_human(ref)
            ck = rng.binomial(n_model, 0.08, size=r)
            cm, cv, _ = summarize_model_counts(ck, n_model)
            u = welch_upper(cm, cv, n_model, rm, rv, G, SAFETY_ALPHA)

        elif comparison_type == "human_vs_human":
            ref = human_block_means(rng, r, G, m, 0.05, rho_ref, generator)
            cmp = human_block_means(rng, r, G, m, 0.08, rho_cmp, generator)
            rm, rv, _ = summarize_human(ref)
            cm, cv, _ = summarize_human(cmp)
            u = welch_upper(cm, cv, G, rm, rv, G, SAFETY_ALPHA)

        else:
            raise ValueError(comparison_type)

        reject += int((u < NI_MARGIN).sum())
        done += r

    return reject/reps


def typei_one_sample_human(
    boundary: float,
    alpha: float,
    G: int,
    m: int,
    reps: int,
    rho: float,
    generator: str,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    reject = 0
    done = 0
    while done < reps:
        r = min(CHUNK, reps-done)
        x = human_block_means(rng, r, G, m, boundary, rho, generator)
        mean, var, _ = summarize_human(x)
        u = one_sample_upper(mean, var, G, alpha)
        reject += int((u < boundary).sum())
        done += r
    return reject/reps


def row_with_ci(rate: float, reps: int) -> dict:
    k = int(round(rate*reps))
    lo, hi = mc_interval(k, reps)
    return {
        "rate": rate,
        "mc95_lower": lo,
        "mc95_upper": hi,
    }


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status_before = subprocess.check_output(["git", "status", "--short"], text=True)

    print("=== BALANCED HUMAN-BLOCK FEASIBILITY v1 ===")
    print("READ-ONLY candidate redesign; no protocol/quota/W0 freeze.")
    print(f"branch: {branch}")
    print(f"HEAD: {head}")
    print()
    print("Candidate structure:")
    print("  human A-val/T/F: G independent authors, exactly m retained eligible rows")
    print("  model S/SF: one row per independent generator_batch_id; n_model = G*m")
    print("  equal m within human cells => author-mean average equals response-average retained-row mean")
    print("  primary inference candidate: independent-block means + Welch/one-sample t;")
    print("  exact binomial upper bounds for singleton model-cell absolute FNR/FPR.")
    print("  This does NOT yet define how fixed retained m is operationally obtained.")
    print()

    screen_rows = []
    print("=== STAGE 1: JOINT-POWER SCREEN (beta-binomial, human ICC=0.10) ===")
    for m in M_GRID:
        for G in G_GRID:
            p = simulate_joint_power(
                G, m, SCREEN_REPS, 0.10, "beta_binomial",
                BASE_SEED + 100000*m + G,
            )
            row = {"G_authors": G, "m_rows_per_author": m, "model_rows": G*m, **p}
            screen_rows.append(row)
            print(
                f"G={G:3d} m={m:2d} model_n={G*m:5d} | "
                f"W1={p['w1']:.3f} temporal={p['temporal']:.3f} "
                f"full={p['full']:.3f} FPR5={p['fpr5']:.3f}"
            )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(screen_rows[0].keys()))
        writer.writeheader()
        writer.writerows(screen_rows)

    eligible = [
        r for r in screen_rows
        if r["full"] >= POWER_TARGET and r["fpr5"] >= POWER_TARGET
    ]
    if eligible:
        eligible.sort(key=lambda r: (r["G_authors"], r["G_authors"]*r["m_rows_per_author"], r["m_rows_per_author"]))
        screen_candidate = eligible[0]
    else:
        screen_candidate = None

    print()
    print("=== SCREEN CANDIDATE ===")
    print(screen_candidate)

    if screen_candidate is None:
        payload = {
            "study_id": "external_validation_balanced_human_block_feasibility_v1",
            "status": "no_screen_candidate",
            "screen_rows": screen_rows,
            "W0_authorized": False,
            "repo_modified": False,
        }
        JSON_PATH.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
        print("No candidate reached 0.80 point power for both full safety and availability.")
        return

    G0 = int(screen_candidate["G_authors"])
    m0 = int(screen_candidate["m_rows_per_author"])

    # Candidate + nearest lower/higher author count already present in grid for same m.
    same_m = sorted(x for x in G_GRID)
    i = same_m.index(G0)
    target_G = sorted(set(
        [G0]
        + ([same_m[i-1]] if i > 0 else [])
        + ([same_m[i+1]] if i+1 < len(same_m) else [])
    ))

    print()
    print("=== STAGE 2: TARGETED TYPE-I CALIBRATION ===")
    typei_rows = []
    generators = ["beta_binomial", "logit_normal"]

    for G in target_G:
        for gen in generators:
            # Relative model-vs-human boundary, human ICC stress.
            for rho_h in [0.05, 0.10, 0.20]:
                rate = typei_relative(
                    "model_vs_human", G, m0, TARGET_TYPEI_REPS,
                    rho_h, 0.0, gen,
                    BASE_SEED + 1000000 + G*100 + int(rho_h*1000) + (1 if gen=="logit_normal" else 0),
                )
                rec = {
                    "family": "relative_NI_model_vs_human",
                    "G": G, "m": m0, "generator": gen,
                    "rho_ref": rho_h, "rho_cmp": 0.0,
                    "alpha": SAFETY_ALPHA,
                    **row_with_ci(rate, TARGET_TYPEI_REPS),
                }
                rec["clear_anti_conservative"] = rec["mc95_lower"] > SAFETY_ALPHA
                typei_rows.append(rec)
                print(
                    f"{rec['family']} G={G} m={m0} gen={gen} rhoH={rho_h:.2f}: "
                    f"{rate:.4f} MC95=[{rec['mc95_lower']:.4f},{rec['mc95_upper']:.4f}] "
                    f"{'FAIL' if rec['clear_anti_conservative'] else 'OK'}"
                )

            # Relative human-vs-human boundary with unequal ICC.
            for rho_ref, rho_cmp in [(0.05,0.10),(0.10,0.20),(0.20,0.05)]:
                rate = typei_relative(
                    "human_vs_human", G, m0, TARGET_TYPEI_REPS,
                    rho_ref, rho_cmp, gen,
                    BASE_SEED + 2000000 + G*100 + int(rho_ref*1000) + int(rho_cmp*10000) + (1 if gen=="logit_normal" else 0),
                )
                rec = {
                    "family": "relative_NI_human_vs_human",
                    "G": G, "m": m0, "generator": gen,
                    "rho_ref": rho_ref, "rho_cmp": rho_cmp,
                    "alpha": SAFETY_ALPHA,
                    **row_with_ci(rate, TARGET_TYPEI_REPS),
                }
                rec["clear_anti_conservative"] = rec["mc95_lower"] > SAFETY_ALPHA
                typei_rows.append(rec)
                print(
                    f"{rec['family']} G={G} m={m0} gen={gen} "
                    f"rho={rho_ref:.2f}->{rho_cmp:.2f}: "
                    f"{rate:.4f} MC95=[{rec['mc95_lower']:.4f},{rec['mc95_upper']:.4f}] "
                    f"{'FAIL' if rec['clear_anti_conservative'] else 'OK'}"
                )

            # Human one-sample FNR/FPR boundary.
            for fam, boundary, alpha in [
                ("absolute_FNR_human", ABS_FNR_CEILING, SAFETY_ALPHA),
                ("FPR_human", FPR_CEILING, FPR_ALPHA),
            ]:
                for rho in [0.10, 0.20]:
                    rate = typei_one_sample_human(
                        boundary, alpha, G, m0, TARGET_TYPEI_REPS,
                        rho, gen,
                        BASE_SEED + 3000000 + G*100 + int(boundary*10000) + int(rho*1000) + (1 if gen=="logit_normal" else 0),
                    )
                    rec = {
                        "family": fam, "G": G, "m": m0,
                        "generator": gen, "rho_ref": rho, "rho_cmp": None,
                        "alpha": alpha,
                        **row_with_ci(rate, TARGET_TYPEI_REPS),
                    }
                    rec["clear_anti_conservative"] = rec["mc95_lower"] > alpha
                    typei_rows.append(rec)
                    print(
                        f"{fam} G={G} m={m0} gen={gen} rho={rho:.2f}: "
                        f"{rate:.4f} MC95=[{rec['mc95_lower']:.4f},{rec['mc95_upper']:.4f}] "
                        f"{'FAIL' if rec['clear_anti_conservative'] else 'OK'}"
                    )

    print()
    print("=== STAGE 3: TARGETED JOINT POWER ===")
    power_rows = []
    for G in target_G:
        for gen in generators:
            for rho in [0.10, 0.20]:
                p = simulate_joint_power(
                    G, m0, TARGET_POWER_REPS, rho, gen,
                    BASE_SEED + 4000000 + G*100 + int(rho*1000) + (1 if gen=="logit_normal" else 0),
                )
                rec = {"G": G, "m": m0, "generator": gen, "human_rho": rho}
                for key, rate in p.items():
                    lo, hi = mc_interval(int(round(rate*TARGET_POWER_REPS)), TARGET_POWER_REPS)
                    rec[key] = rate
                    rec[f"{key}_mc95_lower"] = lo
                    rec[f"{key}_mc95_upper"] = hi
                power_rows.append(rec)
                print(
                    f"G={G} m={m0} gen={gen} rho={rho:.2f}: "
                    f"W1={p['w1']:.3f} (L={rec['w1_mc95_lower']:.3f}) "
                    f"full={p['full']:.3f} (L={rec['full_mc95_lower']:.3f}) "
                    f"FPR5={p['fpr5']:.3f} (L={rec['fpr5_mc95_lower']:.3f})"
                )

    clear_typei_failures = [r for r in typei_rows if r["clear_anti_conservative"]]

    primary_power = [
        r for r in power_rows
        if r["human_rho"] == 0.10
    ]
    candidate_primary = [
        r for r in primary_power
        if r["G"] == G0
    ]
    primary_full_lower = min(r["full_mc95_lower"] for r in candidate_primary)
    primary_fpr_lower = min(r["fpr5_mc95_lower"] for r in candidate_primary)

    stress_candidate = [
        r for r in power_rows
        if r["G"] == G0 and r["human_rho"] == 0.20
    ]
    stress_full_lower = min(r["full_mc95_lower"] for r in stress_candidate)
    stress_fpr_lower = min(r["fpr5_mc95_lower"] for r in stress_candidate)

    payload = {
        "study_id": "external_validation_balanced_human_block_feasibility_v1",
        "status": "candidate_feasibility_only_not_frozen",
        "repo_head_observed": head,
        "candidate_design": {
            "human_cells": "G independent authors with exactly m retained eligible rows per label stratum",
            "model_cells": "one row per generator_batch_id; model rows set to G*m for feasibility comparison",
            "response_average_identity": "with constant m within each human cell, mean author-block mean equals response-average retained-row mean",
            "analysis_candidate": "Welch independent-block mean NI; one-sample t for human absolute FNR/FPR; exact binomial for singleton model absolute FNR/FPR",
            "sampling_mechanism_for_realizing_fixed_m": "NOT YET FROZEN",
        },
        "frozen_claim_inputs": {
            "NI_margin": NI_MARGIN,
            "absolute_FNR_ceiling": ABS_FNR_CEILING,
            "safety_alpha": SAFETY_ALPHA,
            "FPR_ceiling": FPR_CEILING,
            "FPR_alpha": FPR_ALPHA,
            "planning_FNR": PLANNING_FNR,
            "planning_FPR": PLANNING_FPR,
            "planning_power_target": POWER_TARGET,
        },
        "screen_candidate": screen_candidate,
        "target_G_values": target_G,
        "typeI_rows": typei_rows,
        "power_rows": power_rows,
        "clear_typeI_failure_count": len(clear_typei_failures),
        "candidate_primary_power_lower_bounds": {
            "full_preservation": primary_full_lower,
            "five_cell_availability": primary_fpr_lower,
        },
        "candidate_rho0p20_sensitivity_lower_bounds": {
            "full_preservation": stress_full_lower,
            "five_cell_availability": stress_fpr_lower,
        },
        "interpretation": {
            "statistical_candidate_survives_targeted_screen": (
                len(clear_typei_failures) == 0
                and primary_full_lower >= POWER_TARGET
                and primary_fpr_lower >= POWER_TARGET
            ),
            "fixed_m_operational_sampling_is_not_yet_justified": True,
            "W0_authorized": False,
            "quota_frozen": False,
        },
        "screen_rows": screen_rows,
        "W0_authorized": False,
        "repo_modified": False,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    print()
    print("=== SYNTHESIS ===")
    print(f"screen candidate: G={G0} authors/cell, m={m0} retained rows/author/label-stratum")
    print(f"model singleton rows/cell at candidate: {G0*m0}")
    print(f"clear targeted type-I failures: {len(clear_typei_failures)}")
    print(f"candidate primary rho=.10 full-preservation min lower95: {primary_full_lower:.3f}")
    print(f"candidate primary rho=.10 FPR5 min lower95: {primary_fpr_lower:.3f}")
    print(f"candidate rho=.20 sensitivity full-preservation min lower95: {stress_full_lower:.3f}")
    print(f"candidate rho=.20 sensitivity FPR5 min lower95: {stress_fpr_lower:.3f}")
    print()
    print("BOUNDARY:")
    print("- This is feasibility evidence only.")
    print("- Do not freeze G, m, quotas, sampling, inference, W0, or scoring.")
    print("- Even if statistics look adequate, a legitimate pre-scoring mechanism for obtaining fixed m")
    print("  retained eligible Y1/Y0 responses per author must be defined before this design can be frozen.")
    print("- If calibration fails, retain the failure; do not tune alpha or erase stress cases.")

    status_after = subprocess.check_output(["git", "status", "--short"], text=True)
    print()
    print("=== REPO MODIFICATION CHECK ===")
    print("unchanged:", status_before == status_after)
    if status_before != status_after:
        print("BEFORE:")
        print(status_before)
        print("AFTER:")
        print(status_after)

    print()
    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")


if __name__ == "__main__":
    main()
PY

{
  echo "=== COMMAND ==="
  echo "repo=$REPO"
  echo "output=$OUT"
  echo
  "$REPO/.venv/bin/python" "$PY"
} 2>&1 | tee "$TXT"

echo
echo "=== COMPLETE ==="
echo "Upload this file to ChatGPT:"
echo "$TXT"
