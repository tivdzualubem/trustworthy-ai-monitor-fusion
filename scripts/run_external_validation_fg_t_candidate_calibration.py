from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import t

from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    _validate_sizes,
    stable_seed,
)


BASE_SEED = 20260909
REPETITIONS = 20000
FG_BOUND = 0.75
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_fg_t_candidate_calibration_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_fg_t_candidate_calibration_v1.json"


def sizes_from_pattern(g: int, pattern: tuple[int, ...]) -> np.ndarray:
    return np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=np.int64)


def profile(sizes: np.ndarray) -> dict[str, float]:
    arr = sizes.astype(float)
    return {
        "clusters": int(len(arr)),
        "rows": int(arr.sum()),
        "mean": float(arr.mean()),
        "cv": float(arr.std(ddof=0) / arr.mean()),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def cp_interval(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def group_fg_mean_var(
    counts: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = _validate_sizes(np.asarray(sizes)).astype(np.float64)
    n = float(sizes.sum())
    means = counts.sum(axis=1) / n
    residual_sum = counts - means[:, None] * sizes[None, :]

    leverage = sizes / n
    h = (1.0 - np.minimum(FG_BOUND, leverage)) ** (-0.5)
    adjusted = residual_sum * h[None, :]
    variance = np.sum(adjusted * adjusted, axis=1) / (n * n)
    return means, variance


def simulate_relative(
    *,
    baseline: float,
    icc_ref: float,
    icc_cmp: float,
    ref_sizes: np.ndarray,
    cmp_sizes: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    ref_counts = _beta_binomial_counts(
        marginal_probability=baseline,
        icc=icc_ref,
        cluster_sizes=ref_sizes,
        repetitions=REPETITIONS,
        rng=rng,
    )
    cmp_counts = _beta_binomial_counts(
        marginal_probability=baseline + 0.03,
        icc=icc_cmp,
        cluster_sizes=cmp_sizes,
        repetitions=REPETITIONS,
        rng=rng,
    )

    ref_mean, ref_var = group_fg_mean_var(ref_counts, ref_sizes)
    cmp_mean, cmp_var = group_fg_mean_var(cmp_counts, cmp_sizes)
    diff = cmp_mean - ref_mean
    variance = ref_var + cmp_var

    df = len(ref_sizes) + len(cmp_sizes) - 2
    critical = float(t.ppf(0.975, df=df))
    estimable = np.isfinite(variance) & (variance > 0.0)
    upper = np.full(REPETITIONS, np.inf)
    upper[estimable] = diff[estimable] + critical * np.sqrt(variance[estimable])
    passes = estimable & (upper < 0.03)

    k = int(passes.sum())
    lo, hi = cp_interval(k, REPETITIONS)
    return {
        "pass_count": k,
        "estimable": int(estimable.sum()),
        "rate": k / REPETITIONS,
        "mc_lo": lo,
        "mc_hi": hi,
        "df": df,
    }


def simulate_one_sample(
    *,
    probability: float,
    alpha: float,
    constraint: float,
    icc: float,
    sizes: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    counts = _beta_binomial_counts(
        marginal_probability=probability,
        icc=icc,
        cluster_sizes=sizes,
        repetitions=REPETITIONS,
        rng=rng,
    )
    mean, variance = group_fg_mean_var(counts, sizes)
    df = len(sizes) - 1
    critical = float(t.ppf(1.0 - alpha, df=df))
    estimable = np.isfinite(variance) & (variance > 0.0)
    upper = np.full(REPETITIONS, np.inf)
    upper[estimable] = mean[estimable] + critical * np.sqrt(variance[estimable])
    passes = estimable & (upper < constraint)

    k = int(passes.sum())
    lo, hi = cp_interval(k, REPETITIONS)
    return {
        "pass_count": k,
        "estimable": int(estimable.sum()),
        "rate": k / REPETITIONS,
        "mc_lo": lo,
        "mc_hi": hi,
        "df": df,
    }


PAIR_SCENARIOS = (
    ("matched_G40_equal", 40, 40, (20,), (20,), 0.05, 0.01, 0.05),
    ("matched_G60_modcv", 60, 60, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.03, 0.10),
    ("matched_G80_highcv", 80, 80, (5,10,15,20,50), (5,10,15,20,50), 0.07, 0.05, 0.10),
    ("matched_G100_vhighcv", 100, 100, (5,5,10,20,60), (5,5,10,20,60), 0.05, 0.05, 0.10),
    ("matched_G100_reverseICC", 100, 100, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.10, 0.03),
    ("near_G100v80_modcv", 100, 80, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.05, 0.10),
    ("near_G150v120_highcv", 150, 120, (5,10,15,20,50), (5,10,15,20,50), 0.07, 0.03, 0.10),
    ("matched_G80_lowrisk_highcv", 80, 80, (5,10,15,20,50), (5,10,15,20,50), 0.02, 0.05, 0.10)
)

ONE_SAMPLE_SCENARIOS = (
    ("G40_equal_icc01", 40, (20,), 0.01),
    ("G60_modcv_icc05", 60, (10,15,20,25,30), 0.05),
    ("G80_highcv_icc10", 80, (5,10,15,20,50), 0.10),
    ("G100_vhighcv_icc10", 100, (5,5,10,20,60), 0.10),
    ("G150_highcv_icc10", 150, (5,10,15,20,50), 0.10)
)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for sid, gr, gc, pr, pc, baseline, rho_r, rho_c in PAIR_SCENARIOS:
        ref_sizes = sizes_from_pattern(gr, pr)
        cmp_sizes = sizes_from_pattern(gc, pc)
        result = simulate_relative(
            baseline=baseline,
            icc_ref=rho_r,
            icc_cmp=rho_c,
            ref_sizes=ref_sizes,
            cmp_sizes=cmp_sizes,
            seed=stable_seed(BASE_SEED, "relative", sid),
        )
        rp, cp = profile(ref_sizes), profile(cmp_sizes)
        rows.append({
            "family": "relative_FNR_NI_boundary",
            "scenario_id": sid,
            "true_reference_rate": baseline,
            "true_comparison_rate": baseline + 0.03,
            "true_difference": 0.03,
            "nominal_alpha": 0.025,
            "icc_reference": rho_r,
            "icc_comparison": rho_c,
            "reference_clusters": gr,
            "comparison_clusters": gc,
            "reference_rows": rp["rows"],
            "comparison_rows": cp["rows"],
            "reference_cluster_cv": rp["cv"],
            "comparison_cluster_cv": cp["cv"],
            "estimable_repetitions": result["estimable"],
            "pass_count": result["pass_count"],
            "false_pass_rate": result["rate"],
            "mc_cp_lower_95": result["mc_lo"],
            "mc_cp_upper_95": result["mc_hi"],
            "degrees_of_freedom": result["df"],
        })

    for sid, g, pattern, rho in ONE_SAMPLE_SCENARIOS:
        sizes = sizes_from_pattern(g, pattern)
        sp = profile(sizes)

        for family, prob, alpha, constraint in (
            ("absolute_FNR_boundary", 0.10, 0.025, 0.10),
            ("FPR_boundary", 0.05, 0.05, 0.05),
        ):
            result = simulate_one_sample(
                probability=prob,
                alpha=alpha,
                constraint=constraint,
                icc=rho,
                sizes=sizes,
                seed=stable_seed(BASE_SEED, family, sid),
            )
            rows.append({
                "family": family,
                "scenario_id": sid,
                "true_reference_rate": prob,
                "true_comparison_rate": "",
                "true_difference": "",
                "nominal_alpha": alpha,
                "icc_reference": rho,
                "icc_comparison": "",
                "reference_clusters": g,
                "comparison_clusters": "",
                "reference_rows": sp["rows"],
                "comparison_rows": "",
                "reference_cluster_cv": sp["cv"],
                "comparison_cluster_cv": "",
                "estimable_repetitions": result["estimable"],
                "pass_count": result["pass_count"],
                "false_pass_rate": result["rate"],
                "mc_cp_lower_95": result["mc_lo"],
                "mc_cp_upper_95": result["mc_hi"],
                "degrees_of_freedom": result["df"],
            })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    JSON_PATH.write_text(
        json.dumps(
            {
                "screen_id": "external_validation_fg_t_candidate_calibration_v1",
                "status": "candidate_validation_not_final",
                "repetitions_per_scenario": REPETITIONS,
                "method": "working-independence identity-link IEE/GEE + Fay-Graubard sandwich + t(df=clusters-parameters)",
                "bound": FG_BOUND,
                "important_boundary": (
                    "This run validates a candidate procedure only. It does not "
                    "freeze the final FNR/FPR method, quotas, cluster minima/caps, "
                    "candidate caps, terminal-batch rule, or W0."
                ),
                "csv": str(CSV_PATH),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")
    print()
    print("=== FG+t CANDIDATE CALIBRATION ===")

    for family in (
        "relative_FNR_NI_boundary",
        "absolute_FNR_boundary",
        "FPR_boundary",
    ):
        print()
        print(f"--- {family} ---")
        subset = [r for r in rows if r["family"] == family]
        for row in subset:
            cmp_g = row["comparison_clusters"]
            cmp_label = cmp_g if cmp_g != "" else "-"
            print(
                f"{row['scenario_id']}: "
                f"false_pass={float(row['false_pass_rate']):.5f} "
                f"(MC95% {float(row['mc_cp_lower_95']):.5f}-"
                f"{float(row['mc_cp_upper_95']):.5f}); "
                f"nominal={float(row['nominal_alpha']):.3f}; "
                f"G={row['reference_clusters']}/{cmp_label}; "
                f"CV={float(row['reference_cluster_cv']):.2f}"
            )

        worst = max(subset, key=lambda r: float(r["false_pass_rate"]))
        print(
            f"WORST {family}: {float(worst['false_pass_rate']):.5f} "
            f"(MC95% {float(worst['mc_cp_lower_95']):.5f}-"
            f"{float(worst['mc_cp_upper_95']):.5f}) "
            f"at {worst['scenario_id']}"
        )

    print()
    print(
        "DECISION GATE: do not freeze the method from this screen alone. "
        "If calibration is acceptable, proceed to the full professor-requested "
        "type-I/coverage/power design grid. If a boundary is materially "
        "anti-conservative, revise the procedure before power design."
    )


if __name__ == "__main__":
    run()
