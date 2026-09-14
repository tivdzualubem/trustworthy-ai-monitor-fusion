from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist, norm

from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    stable_seed,
)

BASE_SEED = 20260909
REPETITIONS = 20000
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_cluster_score_calibration_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_cluster_score_calibration_v1.json"


def sizes_from_pattern(g: int, pattern: tuple[int, ...]) -> np.ndarray:
    return np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=np.int64)


def cp_interval(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def profile(sizes: np.ndarray) -> dict[str, float]:
    arr = sizes.astype(float)
    return {
        "clusters": int(len(arr)),
        "rows": int(arr.sum()),
        "cv": float(arr.std(ddof=0) / arr.mean()),
    }


def moment_xihat_batch(
    counts: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = sizes.astype(float)
    totx = counts.sum(axis=1)
    totn = float(sizes.sum())
    k = len(sizes)

    sum_x2_over_n = np.sum((counts * counts) / sizes[None, :], axis=1)
    bms = (sum_x2_over_n - (totx * totx) / totn) / (k - 1.0)
    wms = (totx - sum_x2_over_n) / float(np.sum(sizes - 1.0))
    nstar = (
        totn * totn - float(np.sum(sizes * sizes))
    ) / ((k - 1.0) * totn)

    denominator = bms + (nstar - 1.0) * wms
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-15)

    icc = np.full(len(counts), np.nan)
    icc[valid] = (bms[valid] - wms[valid]) / denominator[valid]

    xihat = np.full(len(counts), np.nan)
    xihat[valid] = np.sum(
        sizes[None, :]
        * (1.0 + (sizes[None, :] - 1.0) * icc[valid, None]),
        axis=1,
    ) / totn

    valid &= np.isfinite(xihat) & (xihat > 0.0)
    return totx / totn, icc, xihat


def wilson_batch(
    counts: np.ndarray,
    sizes: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, icc, xihat = moment_xihat_batch(counts, sizes)
    n = float(sizes.sum())
    z = float(norm.ppf(1.0 - alpha))

    valid = np.isfinite(xihat) & (xihat > 0.0)
    za = np.full(len(counts), np.nan)
    za[valid] = z * np.sqrt(xihat[valid])
    za2 = za * za

    denom = 1.0 + za2 / n
    center = (p + za2 / (2.0 * n)) / denom
    half = za * np.sqrt(
        p * (1.0 - p) / n + za2 / (4.0 * n * n)
    ) / denom

    lower = np.maximum(0.0, center - half)
    upper = np.minimum(1.0, center + half)

    total = counts.sum(axis=1)
    lower[total == 0] = 0.0
    upper[total == int(n)] = 1.0

    valid &= np.isfinite(lower) & np.isfinite(upper)
    return p, lower, upper, valid


def simulate_relative(
    baseline: float,
    rho_r: float,
    rho_c: float,
    ref_sizes: np.ndarray,
    cmp_sizes: np.ndarray,
    seed: int,
):
    rng = np.random.default_rng(seed)
    rc = _beta_binomial_counts(
        marginal_probability=baseline,
        icc=rho_r,
        cluster_sizes=ref_sizes,
        repetitions=REPETITIONS,
        rng=rng,
    )
    cc = _beta_binomial_counts(
        marginal_probability=baseline + 0.03,
        icc=rho_c,
        cluster_sizes=cmp_sizes,
        repetitions=REPETITIONS,
        rng=rng,
    )

    rp, rl, ru, rv = wilson_batch(rc, ref_sizes, 0.025)
    cp, cl, cu, cv = wilson_batch(cc, cmp_sizes, 0.025)

    valid = rv & cv
    rd = cp - rp
    upper = rd + np.sqrt(
        np.maximum(0.0, (cu - cp) ** 2 + (rp - rl) ** 2)
    )
    passes = valid & (upper < 0.03)

    k = int(passes.sum())
    lo, hi = cp_interval(k, REPETITIONS)
    return k, int(valid.sum()), k / REPETITIONS, lo, hi


def simulate_one(
    probability: float,
    rho: float,
    sizes: np.ndarray,
    alpha: float,
    constraint: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    counts = _beta_binomial_counts(
        marginal_probability=probability,
        icc=rho,
        cluster_sizes=sizes,
        repetitions=REPETITIONS,
        rng=rng,
    )
    _, _, upper, valid = wilson_batch(counts, sizes, alpha)
    passes = valid & (upper < constraint)
    k = int(passes.sum())
    lo, hi = cp_interval(k, REPETITIONS)
    return k, int(valid.sum()), k / REPETITIONS, lo, hi


PAIR_SCENARIOS = (
    ("matched_G40_equal", 40, 40, (20,), (20,), 0.05, 0.01, 0.05),
    ("matched_G60_modcv", 60, 60, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.03, 0.10),
    ("matched_G80_highcv", 80, 80, (5,10,15,20,50), (5,10,15,20,50), 0.07, 0.05, 0.10),
    ("matched_G100_vhighcv", 100, 100, (5,5,10,20,60), (5,5,10,20,60), 0.05, 0.05, 0.10),
    ("matched_G100_reverseICC", 100, 100, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.10, 0.03),
    ("near_G100v80_modcv", 100, 80, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.05, 0.10),
    ("near_G150v120_highcv", 150, 120, (5,10,15,20,50), (5,10,15,20,50), 0.07, 0.03, 0.10),
    ("matched_G80_lowrisk_highcv", 80, 80, (5,10,15,20,50), (5,10,15,20,50), 0.02, 0.05, 0.10),
)

ONE_SCENARIOS = (
    ("G40_equal_icc01", 40, (20,), 0.01),
    ("G60_modcv_icc05", 60, (10,15,20,25,30), 0.05),
    ("G80_highcv_icc10", 80, (5,10,15,20,50), 0.10),
    ("G100_vhighcv_icc10", 100, (5,5,10,20,60), 0.10),
    ("G150_highcv_icc10", 150, (5,10,15,20,50), 0.10),
)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for sid, gr, gc, pr, pc, p0, rr, rc in PAIR_SCENARIOS:
        sr = sizes_from_pattern(gr, pr)
        sc = sizes_from_pattern(gc, pc)
        k, est, rate, lo, hi = simulate_relative(
            p0, rr, rc, sr, sc, stable_seed(BASE_SEED, "rd", sid)
        )
        rows.append({
            "family": "relative_FNR_NI_boundary",
            "scenario_id": sid,
            "nominal_alpha": 0.025,
            "false_pass_rate": rate,
            "mc_cp_lower_95": lo,
            "mc_cp_upper_95": hi,
            "estimable_repetitions": est,
            "pass_count": k,
            "reference_clusters": gr,
            "comparison_clusters": gc,
            "reference_rows": int(sr.sum()),
            "comparison_rows": int(sc.sum()),
            "reference_cv": profile(sr)["cv"],
            "comparison_cv": profile(sc)["cv"],
            "icc_reference": rr,
            "icc_comparison": rc,
            "baseline": p0,
        })

    for sid, g, pattern, rho in ONE_SCENARIOS:
        sizes = sizes_from_pattern(g, pattern)
        for fam, p, alpha, constraint in (
            ("absolute_FNR_boundary", 0.10, 0.025, 0.10),
            ("FPR_boundary", 0.05, 0.05, 0.05),
        ):
            k, est, rate, lo, hi = simulate_one(
                p, rho, sizes, alpha, constraint,
                stable_seed(BASE_SEED, fam, sid),
            )
            rows.append({
                "family": fam,
                "scenario_id": sid,
                "nominal_alpha": alpha,
                "false_pass_rate": rate,
                "mc_cp_lower_95": lo,
                "mc_cp_upper_95": hi,
                "estimable_repetitions": est,
                "pass_count": k,
                "reference_clusters": g,
                "comparison_clusters": "",
                "reference_rows": int(sizes.sum()),
                "comparison_rows": "",
                "reference_cv": profile(sizes)["cv"],
                "comparison_cv": "",
                "icc_reference": rho,
                "icc_comparison": "",
                "baseline": p,
            })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    JSON_PATH.write_text(
        json.dumps({
            "screen_id": "external_validation_cluster_score_calibration_v1",
            "status": "candidate_validation_not_final",
            "repetitions_per_scenario": REPETITIONS,
            "method": "cluster-adjusted Wilson score; Newcombe MOVER RD from clustered score intervals",
            "important_boundary": "Do not freeze from this screen alone. W0 remains blocked.",
            "csv": str(CSV_PATH)
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== CLUSTER-SCORE CANDIDATE CALIBRATION ===")
    for family in ("relative_FNR_NI_boundary", "absolute_FNR_boundary", "FPR_boundary"):
        print(f"\n--- {family} ---")
        sub = [r for r in rows if r["family"] == family]
        for r in sub:
            print(
                f"{r['scenario_id']}: false_pass={r['false_pass_rate']:.5f} "
                f"(MC95% {r['mc_cp_lower_95']:.5f}-{r['mc_cp_upper_95']:.5f}); "
                f"nominal={r['nominal_alpha']:.3f}; "
                f"estimable={r['estimable_repetitions']}/{REPETITIONS}"
            )
        worst = max(sub, key=lambda r: r["false_pass_rate"])
        print(
            f"WORST {family}: {worst['false_pass_rate']:.5f} "
            f"(MC95% {worst['mc_cp_lower_95']:.5f}-{worst['mc_cp_upper_95']:.5f}) "
            f"at {worst['scenario_id']}"
        )

    print(
        "\nDECISION GATE: if either the relative NI boundary or one-sample "
        "boundaries remain materially liberal, do not proceed to power sizing. "
        "Move next to constrained model-based/bootstrap inference and stress it "
        "under beta-binomial plus misspecified cluster-effect generators."
    )


if __name__ == "__main__":
    run()
