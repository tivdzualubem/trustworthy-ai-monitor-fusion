#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/trustworthy-ai-monitor-fusion"
OUT="/mnt/c/Users/user/Downloads/external_validation_empirical_bernstein_block_feasibility_v1"
mkdir -p "$OUT"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd "$REPO"

TXT="$OUT/empirical_bernstein_block_feasibility_v1.txt"
JSON="$OUT/empirical_bernstein_block_feasibility_v1.json"
CSV="$OUT/empirical_bernstein_block_power_grid_v1.csv"
PY="$OUT/_run_empirical_bernstein_block_feasibility_v1.py"

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

from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
)

BASE_SEED = 20260919

# Frozen scientific inputs.
NI_MARGIN = 0.03
SAFETY_ALPHA = 0.025
ABS_FNR_CEILING = 0.10
FPR_ALPHA = 0.05
FPR_CEILING = 0.05
PLANNING_FNR = 0.05
PLANNING_FPR = 0.025
POWER_TARGET = 0.80

# Planning-only choices for this assumption-light feasibility study.
# Human blocks are independent authors. Each author contributes exactly m
# retained eligible rows in the label stratum being analyzed.
M_GRID = [10, 20, 40]
G_GRID = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800]
MODEL_N = 1800  # planning anchor only; model-source rows are independent singleton batches.

SCREEN_REPS = 3000
TARGET_TYPEI_REPS = 20000
TARGET_POWER_REPS = 12000
CHUNK = 500

OUT_DIR = Path("/mnt/c/Users/user/Downloads/external_validation_empirical_bernstein_block_feasibility_v1")
JSON_PATH = OUT_DIR / "empirical_bernstein_block_feasibility_v1.json"
CSV_PATH = OUT_DIR / "empirical_bernstein_block_power_grid_v1.csv"


def mc_interval(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k+1, n-k))
    return lo, hi


def cp_upper(k: np.ndarray, n: int, alpha: float) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    out = np.ones_like(k, dtype=float)
    mask = k < n
    out[mask] = beta_dist.ppf(1.0-alpha, k[mask] + 1, n-k[mask])
    return out


def cp_lower(k: np.ndarray, n: int, alpha: float) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    out = np.zeros_like(k, dtype=float)
    mask = k > 0
    out[mask] = beta_dist.ppf(alpha, k[mask], n-k[mask] + 1)
    return out


def empirical_bernstein_radius(mean: np.ndarray, var: np.ndarray, n: int, delta: float) -> np.ndarray:
    """
    Maurer-Pontil (2009) empirical Bernstein one-sided radius for X in [0,1]:

      sqrt(2 * S^2 * log(2/delta) / n)
      + 7 * log(2/delta) / (3 * (n-1))

    Here S^2 is the unbiased sample variance across independent human blocks.
    This candidate therefore requires independent identically sampled author
    blocks from the frozen cell-specific sampling frame.
    """
    if n <= 1:
        return np.full_like(mean, np.inf, dtype=float)
    logterm = math.log(2.0 / delta)
    return np.sqrt(np.maximum(0.0, 2.0 * var * logterm / n)) + 7.0 * logterm / (3.0 * (n - 1))


def eb_upper(mean: np.ndarray, var: np.ndarray, n: int, delta: float) -> np.ndarray:
    return np.minimum(1.0, mean + empirical_bernstein_radius(mean, var, n, delta))


def eb_lower(mean: np.ndarray, var: np.ndarray, n: int, delta: float) -> np.ndarray:
    return np.maximum(0.0, mean - empirical_bernstein_radius(mean, var, n, delta))


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


def summarize_human(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=1), x.var(axis=1, ddof=1)


def ni_upper_human_human(
    cmp_mean: np.ndarray,
    cmp_var: np.ndarray,
    ref_mean: np.ndarray,
    ref_var: np.ndarray,
    G: int,
) -> np.ndarray:
    # Bonferroni composition: P(Ucmp >= mucmp and Lref <= muref) >= 1-alpha.
    half = SAFETY_ALPHA / 2.0
    return eb_upper(cmp_mean, cmp_var, G, half) - eb_lower(ref_mean, ref_var, G, half)


def ni_upper_model_human(
    cmp_k: np.ndarray,
    model_n: int,
    ref_mean: np.ndarray,
    ref_var: np.ndarray,
    G: int,
) -> np.ndarray:
    half = SAFETY_ALPHA / 2.0
    return cp_upper(cmp_k, model_n, half) - eb_lower(ref_mean, ref_var, G, half)


def simulate_joint_power(
    G: int,
    m: int,
    reps: int,
    human_rho: float,
    generator: str,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    counts = {"w1": 0, "temporal": 0, "full": 0, "fpr5": 0}
    done = 0

    while done < reps:
        r = min(CHUNK, reps-done)

        # Y=1 human cells.
        A = human_block_means(rng, r, G, m, PLANNING_FNR, human_rho, generator)
        T = human_block_means(rng, r, G, m, PLANNING_FNR, human_rho, generator)
        F = human_block_means(rng, r, G, m, PLANNING_FNR, human_rho, generator)
        Am, Av = summarize_human(A)
        Tm, Tv = summarize_human(T)
        Fm, Fv = summarize_human(F)

        # Y=1 model cells: singleton independent generator batches.
        Sk = rng.binomial(MODEL_N, PLANNING_FNR, size=r)
        SFk = rng.binomial(MODEL_N, PLANNING_FNR, size=r)

        ni_ST = ni_upper_model_human(Sk, MODEL_N, Tm, Tv, G) < NI_MARGIN
        ni_FT = ni_upper_human_human(Fm, Fv, Tm, Tv, G) < NI_MARGIN
        ni_SFT = ni_upper_model_human(SFk, MODEL_N, Tm, Tv, G) < NI_MARGIN
        ni_TA = ni_upper_human_human(Tm, Tv, Am, Av, G) < NI_MARGIN

        abs_A = eb_upper(Am, Av, G, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_T = eb_upper(Tm, Tv, G, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_F = eb_upper(Fm, Fv, G, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_S = cp_upper(Sk, MODEL_N, SAFETY_ALPHA) < ABS_FNR_CEILING
        abs_SF = cp_upper(SFk, MODEL_N, SAFETY_ALPHA) < ABS_FNR_CEILING

        w1 = ni_ST & ni_FT & ni_SFT & abs_T & abs_S & abs_F & abs_SF
        temporal = ni_TA & abs_A & abs_T
        full = w1 & temporal

        # Y=0 availability, independently generated label stratum.
        A0 = human_block_means(rng, r, G, m, PLANNING_FPR, human_rho, generator)
        T0 = human_block_means(rng, r, G, m, PLANNING_FPR, human_rho, generator)
        F0 = human_block_means(rng, r, G, m, PLANNING_FPR, human_rho, generator)
        A0m, A0v = summarize_human(A0)
        T0m, T0v = summarize_human(T0)
        F0m, F0v = summarize_human(F0)
        S0k = rng.binomial(MODEL_N, PLANNING_FPR, size=r)
        SF0k = rng.binomial(MODEL_N, PLANNING_FPR, size=r)

        fpr_A = eb_upper(A0m, A0v, G, FPR_ALPHA) < FPR_CEILING
        fpr_T = eb_upper(T0m, T0v, G, FPR_ALPHA) < FPR_CEILING
        fpr_F = eb_upper(F0m, F0v, G, FPR_ALPHA) < FPR_CEILING
        fpr_S = cp_upper(S0k, MODEL_N, FPR_ALPHA) < FPR_CEILING
        fpr_SF = cp_upper(SF0k, MODEL_N, FPR_ALPHA) < FPR_CEILING
        fpr5 = fpr_A & fpr_T & fpr_S & fpr_F & fpr_SF

        counts["w1"] += int(w1.sum())
        counts["temporal"] += int(temporal.sum())
        counts["full"] += int(full.sum())
        counts["fpr5"] += int(fpr5.sum())
        done += r

    return {k: v/reps for k, v in counts.items()}


def simulate_typeI(
    family: str,
    G: int,
    m: int,
    reps: int,
    generator: str,
    rho_ref: float,
    rho_cmp: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    reject = 0
    alpha = SAFETY_ALPHA if family != "FPR_human" else FPR_ALPHA
    done = 0

    while done < reps:
        r = min(CHUNK, reps-done)

        if family == "relative_NI_human_vs_human":
            ref = human_block_means(rng, r, G, m, 0.05, rho_ref, generator)
            cmp = human_block_means(rng, r, G, m, 0.08, rho_cmp, generator)
            rm, rv = summarize_human(ref)
            cm, cv = summarize_human(cmp)
            decision = ni_upper_human_human(cm, cv, rm, rv, G) < NI_MARGIN

        elif family == "relative_NI_model_vs_human":
            ref = human_block_means(rng, r, G, m, 0.05, rho_ref, generator)
            rm, rv = summarize_human(ref)
            ck = rng.binomial(MODEL_N, 0.08, size=r)
            decision = ni_upper_model_human(ck, MODEL_N, rm, rv, G) < NI_MARGIN

        elif family == "absolute_FNR_human":
            x = human_block_means(rng, r, G, m, ABS_FNR_CEILING, rho_ref, generator)
            xm, xv = summarize_human(x)
            decision = eb_upper(xm, xv, G, SAFETY_ALPHA) < ABS_FNR_CEILING

        elif family == "FPR_human":
            x = human_block_means(rng, r, G, m, FPR_CEILING, rho_ref, generator)
            xm, xv = summarize_human(x)
            decision = eb_upper(xm, xv, G, FPR_ALPHA) < FPR_CEILING

        else:
            raise ValueError(family)

        reject += int(decision.sum())
        done += r

    return reject/reps, alpha


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status_before = subprocess.check_output(["git", "status", "--short"], text=True)

    print("=== EMPIRICAL-BERNSTEIN HUMAN-BLOCK FEASIBILITY v1 ===")
    print("READ-ONLY assumption-light feasibility study; no protocol/quota/W0 freeze.")
    print(f"branch: {branch}")
    print(f"HEAD: {head}")
    print()
    print("WHY THIS STEP:")
    print("  The balanced Welch/t candidate had clear anti-conservative type-I failures.")
    print("  This screen replaces normal/t calibration with finite-sample bounded-mean")
    print("  empirical-Bernstein bounds across independent human author blocks.")
    print()
    print("CANDIDATE REQUIREMENTS:")
    print("  - human author blocks must be independently and identically sampled within cell")
    print("  - exactly m retained eligible rows per author within the analyzed label stratum")
    print("  - each author block mean lies in [0,1]")
    print("  - model S/SF retain independent singleton generator_batch_id rows")
    print("  - fixed-m operational recruitment/retention mechanism is NOT frozen by this run")
    print()
    print(f"model planning rows/cell: {MODEL_N} (planning anchor only)")
    print()

    screen_rows = []
    print("=== STAGE 1: ASSUMPTION-LIGHT JOINT-POWER FRONTIER ===")
    for m in M_GRID:
        for G in G_GRID:
            p = simulate_joint_power(
                G, m, SCREEN_REPS, 0.10, "beta_binomial",
                BASE_SEED + m*100000 + G,
            )
            row = {
                "G_authors": G,
                "m_rows_per_author": m,
                "human_retained_rows": G*m,
                "model_rows": MODEL_N,
                **p,
            }
            screen_rows.append(row)
            print(
                f"G={G:4d} m={m:2d} human_rows={G*m:6d} | "
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
    eligible.sort(key=lambda r: (r["G_authors"], r["human_retained_rows"]))
    candidate = eligible[0] if eligible else None

    print()
    print("=== FIRST SCREEN CANDIDATE ===")
    print(candidate)

    # If no candidate <= 2800, use the largest-G, largest-m corner for targeted validation
    # so the output still establishes a useful lower-bound/no-go direction.
    if candidate is None:
        G0 = max(G_GRID)
        m0 = max(M_GRID)
        candidate_status = "no_80pct_screen_candidate_within_grid"
    else:
        G0 = int(candidate["G_authors"])
        m0 = int(candidate["m_rows_per_author"])
        candidate_status = "screen_candidate_found"

    idx = G_GRID.index(G0)
    target_G = sorted(set(
        [G0]
        + ([G_GRID[idx-1]] if idx > 0 else [])
        + ([G_GRID[idx+1]] if idx + 1 < len(G_GRID) else [])
    ))

    print()
    print("=== STAGE 2: TARGETED BOUNDARY TYPE-I CHECK ===")
    typei_rows = []
    generators = ["beta_binomial", "logit_normal"]

    scenarios = []
    for G in target_G:
        for gen in generators:
            for rho_ref, rho_cmp in [(0.05,0.10), (0.10,0.20), (0.20,0.05)]:
                scenarios.append(("relative_NI_human_vs_human", G, gen, rho_ref, rho_cmp))
            for rho in [0.05, 0.10, 0.20]:
                scenarios.append(("relative_NI_model_vs_human", G, gen, rho, 0.0))
            for rho in [0.10, 0.20]:
                scenarios.append(("absolute_FNR_human", G, gen, rho, 0.0))
                scenarios.append(("FPR_human", G, gen, rho, 0.0))

    for j, (family, G, gen, rr, rc) in enumerate(scenarios):
        rate, alpha = simulate_typeI(
            family, G, m0, TARGET_TYPEI_REPS, gen, rr, rc,
            BASE_SEED + 2_000_000 + j*101,
        )
        k = int(round(rate * TARGET_TYPEI_REPS))
        lo, hi = mc_interval(k, TARGET_TYPEI_REPS)
        rec = {
            "family": family,
            "G": G,
            "m": m0,
            "generator": gen,
            "rho_ref": rr,
            "rho_cmp": rc,
            "alpha": alpha,
            "rate": rate,
            "mc95_lower": lo,
            "mc95_upper": hi,
            "clear_anti_conservative": lo > alpha,
        }
        typei_rows.append(rec)
        print(
            f"{family} G={G} m={m0} gen={gen} rho={rr:.2f}->{rc:.2f}: "
            f"{rate:.4f} MC95=[{lo:.4f},{hi:.4f}] alpha={alpha:.3f} "
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
                    BASE_SEED + 4_000_000 + G*100 + int(rho*1000) + (1 if gen=="logit_normal" else 0),
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
                    f"W1={p['w1']:.3f} L95={rec['w1_mc95_lower']:.3f} | "
                    f"full={p['full']:.3f} L95={rec['full_mc95_lower']:.3f} | "
                    f"FPR5={p['fpr5']:.3f} L95={rec['fpr5_mc95_lower']:.3f}"
                )

    failures = [r for r in typei_rows if r["clear_anti_conservative"]]

    if candidate is not None:
        primary = [
            r for r in power_rows
            if r["G"] == G0 and r["human_rho"] == 0.10
        ]
        stress = [
            r for r in power_rows
            if r["G"] == G0 and r["human_rho"] == 0.20
        ]
        primary_full_lower = min(r["full_mc95_lower"] for r in primary)
        primary_fpr_lower = min(r["fpr5_mc95_lower"] for r in primary)
        stress_full_lower = min(r["full_mc95_lower"] for r in stress)
        stress_fpr_lower = min(r["fpr5_mc95_lower"] for r in stress)
    else:
        primary_full_lower = primary_fpr_lower = None
        stress_full_lower = stress_fpr_lower = None

    payload = {
        "study_id": "external_validation_empirical_bernstein_block_feasibility_v1",
        "status": "candidate_feasibility_only_not_frozen",
        "repo_head_observed": head,
        "method": {
            "human_block_bounds": "Maurer-Pontil empirical Bernstein one-sided bound for independent iid [0,1] block means",
            "relative_human_human": "upper(cmp, alpha/2) - lower(ref, alpha/2)",
            "relative_model_human": "Clopper-Pearson upper(model, alpha/2) - empirical-Bernstein lower(human, alpha/2)",
            "human_absolute": "empirical-Bernstein upper bound",
            "model_absolute": "Clopper-Pearson upper bound",
            "role": "assumption-light finite-sample candidate; not frozen",
        },
        "candidate_requirements": {
            "independent_iid_human_author_blocks_within_cell": True,
            "fixed_m_retained_rows_per_author_per_label_stratum": True,
            "fixed_m_operational_mechanism_frozen": False,
            "model_singleton_batches": True,
        },
        "planning_inputs": {
            "NI_margin": NI_MARGIN,
            "absolute_FNR_ceiling": ABS_FNR_CEILING,
            "safety_alpha": SAFETY_ALPHA,
            "FPR_ceiling": FPR_CEILING,
            "FPR_alpha": FPR_ALPHA,
            "planning_FNR": PLANNING_FNR,
            "planning_FPR": PLANNING_FPR,
            "power_target": POWER_TARGET,
            "model_rows_per_cell_planning_anchor": MODEL_N,
        },
        "screen_candidate_status": candidate_status,
        "first_screen_candidate": candidate,
        "target_G_values": target_G,
        "target_m": m0,
        "clear_typeI_failure_count": len(failures),
        "typeI_rows": typei_rows,
        "targeted_power_rows": power_rows,
        "candidate_primary_lower_bounds": {
            "full_preservation": primary_full_lower,
            "five_cell_availability": primary_fpr_lower,
        },
        "candidate_rho0p20_sensitivity_lower_bounds": {
            "full_preservation": stress_full_lower,
            "five_cell_availability": stress_fpr_lower,
        },
        "screen_rows": screen_rows,
        "decision_boundary": {
            "W0_authorized": False,
            "quotas_frozen": False,
            "sampling_frozen": False,
            "inference_frozen": False,
            "if_no_feasible_candidate": "Evidence supports an independent-unit bottleneck under an assumption-light certificate; next decision is narrower claim or stronger modeling assumptions, not another ad-hoc estimator search.",
        },
        "repo_modified": False,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    print()
    print("=== SYNTHESIS ===")
    print(f"first screen candidate: {candidate}")
    print(f"targeted type-I clear failures: {len(failures)}")
    if candidate is not None:
        print(f"candidate primary rho=.10 full lower95: {primary_full_lower:.3f}")
        print(f"candidate primary rho=.10 FPR5 lower95: {primary_fpr_lower:.3f}")
        print(f"candidate rho=.20 full lower95: {stress_full_lower:.3f}")
        print(f"candidate rho=.20 FPR5 lower95: {stress_fpr_lower:.3f}")
    else:
        print(f"No 80% joint-power screen candidate through G={max(G_GRID)} authors/cell.")
    print()
    print("BOUNDARY:")
    print("- Do NOT start W0.")
    print("- Do NOT freeze quotas, sampling, or inference from this feasibility run.")
    print("- Do NOT revive the failed Welch/t, CR2, or WCR candidates.")
    print("- If this assumption-light route still needs impractically many independent authors,")
    print("  the next scientific decision is claim narrowing or explicitly stronger assumptions.")

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
echo "Upload only this text file:"
echo "$TXT"
