from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from monitor_fusion.external_validation.beta_binomial_profile import (
    BetaBinomialProfileNotEstimable,
)
from monitor_fusion.external_validation.cluster_ni_power import stable_seed
from monitor_fusion.external_validation.constrained_null_mc import (
    one_sample_constrained_null_mc,
    two_sample_ni_constrained_null_mc,
)
from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
    draw_logit_normal_counts,
)

BASE_SEED = 20260909
OUTER_REPETITIONS = 10000
INNER_REPETITIONS = 999
MAX_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))

OUT_DIR = Path("results")
PART_DIR = OUT_DIR / "external_validation_final_inference_validation_v1_parts"
CSV_PATH = OUT_DIR / "external_validation_final_inference_validation_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_final_inference_validation_v1.json"


def sizes_from_pattern(g: int, pattern: tuple[int, ...]) -> np.ndarray:
    return np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=int)


def cp_interval(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def two_point_counts(rng, p, rho, sizes):
    variance = rho * p * (1.0 - p)
    high = min(1.0, p + variance / p)
    probability_high = p / high
    probabilities = np.where(
        rng.random(len(sizes)) < probability_high,
        high,
        0.0,
    )
    return rng.binomial(sizes, probabilities).astype(int)


def beta_binomial_counts(rng, p, rho, sizes):
    concentration = (1.0 / rho) - 1.0
    a = p * concentration
    b = (1.0 - p) * concentration
    probabilities = rng.beta(a, b, size=len(sizes))
    return rng.binomial(sizes, probabilities).astype(int)


def draw_counts(rng, p, rho, sizes, generator, calibration=None):
    if generator == "beta_binomial":
        return beta_binomial_counts(rng, p, rho, sizes)
    if generator == "two_point":
        return two_point_counts(rng, p, rho, sizes)
    if generator == "logit_normal":
        return draw_logit_normal_counts(
            rng=rng,
            marginal_probability=p,
            response_scale_icc=rho,
            cluster_sizes=sizes,
            calibration=calibration,
        )
    raise ValueError(generator)


# Hardest exact-matched relative-NI scenarios plus a small-cluster check.
PAIR_TASKS = (
    (
        "R_G80_lowrisk_highcv_mismatch",
        80,
        (5,10,15,20,50),
        (5,5,10,20,60),
        0.02,
        0.05,
        0.10,
    ),
    (
        "R_G100_reverseICC_mismatched_sizes",
        100,
        (10,15,20,25,30),
        (5,10,15,20,50),
        0.05,
        0.10,
        0.03,
    ),
    (
        "R_G40_equal_vs_modcv",
        40,
        (20,),
        (10,15,20,25,30),
        0.05,
        0.01,
        0.10,
    ),
)

# Hardest one-sample boundary case plus the smallest-cluster case.
ONE_TASKS = (
    ("A_G150_vhighcv_icc10", "absolute_FNR_boundary", 150, (5,5,10,20,60), 0.10, 0.10, 0.025),
    ("A_G40_equal_icc01", "absolute_FNR_boundary", 40, (20,), 0.10, 0.01, 0.025),
    ("F_G150_vhighcv_icc10", "FPR_boundary", 150, (5,5,10,20,60), 0.05, 0.10, 0.05),
    ("F_G40_equal_icc01", "FPR_boundary", 40, (20,), 0.05, 0.01, 0.05),
)

GENERATORS = ("beta_binomial", "two_point", "logit_normal")


def run_pair_task(task):
    sid, g, ref_pattern, cmp_pattern, p0, rho_r, rho_c, generator = task
    sr = sizes_from_pattern(g, ref_pattern)
    sc = sizes_from_pattern(g, cmp_pattern)
    if len(sr) != len(sc):
        raise RuntimeError("exact matched-cluster-count invariant violated")

    cal_r = None
    cal_c = None
    if generator == "logit_normal":
        cal_r = calibrate_logit_normal(
            marginal_probability=p0,
            response_scale_icc=rho_r,
        )
        cal_c = calibrate_logit_normal(
            marginal_probability=p0 + 0.03,
            response_scale_icc=rho_c,
        )

    outer_rng = np.random.default_rng(
        stable_seed(BASE_SEED, "final", generator, sid, "outer")
    )
    rejects = 0
    estimable = 0

    for rep in range(OUTER_REPETITIONS):
        kr = draw_counts(
            outer_rng, p0, rho_r, sr, generator, calibration=cal_r
        )
        kc = draw_counts(
            outer_rng, p0 + 0.03, rho_c, sc, generator, calibration=cal_c
        )
        inner_rng = np.random.default_rng(
            stable_seed(BASE_SEED, "final", generator, sid, rep, "inner")
        )
        try:
            out = two_sample_ni_constrained_null_mc(
                reference_events=kr,
                reference_sizes=sr,
                comparison_events=kc,
                comparison_sizes=sc,
                margin=0.03,
                alpha=0.025,
                bootstrap_repetitions=INNER_REPETITIONS,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue

        estimable += 1
        rejects += int(out.reject_noninferiority_null)

    lo, hi = cp_interval(rejects, OUTER_REPETITIONS)
    return {
        "generator": generator,
        "family": "relative_FNR_NI_boundary",
        "scenario_id": sid,
        "nominal_alpha": 0.025,
        "false_pass_rate": rejects / OUTER_REPETITIONS,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
        "estimable_repetitions": estimable,
        "outer_repetitions": OUTER_REPETITIONS,
        "inner_repetitions": INNER_REPETITIONS,
        "reference_clusters": g,
        "comparison_clusters": g,
        "icc_reference": rho_r,
        "icc_comparison": rho_c,
        "baseline_rate": p0,
    }


def run_one_task(task):
    sid, family, g, pattern, p, rho, alpha, generator = task
    sizes = sizes_from_pattern(g, pattern)

    calibration = None
    if generator == "logit_normal":
        calibration = calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )

    outer_rng = np.random.default_rng(
        stable_seed(BASE_SEED, "final", generator, sid, "outer")
    )
    rejects = 0
    estimable = 0

    for rep in range(OUTER_REPETITIONS):
        k = draw_counts(
            outer_rng, p, rho, sizes, generator, calibration=calibration
        )
        inner_rng = np.random.default_rng(
            stable_seed(BASE_SEED, "final", generator, sid, rep, "inner")
        )
        try:
            out = one_sample_constrained_null_mc(
                events=k,
                sizes=sizes,
                boundary=p,
                alpha=alpha,
                bootstrap_repetitions=INNER_REPETITIONS,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue

        estimable += 1
        rejects += int(out.reject_below_boundary)

    lo, hi = cp_interval(rejects, OUTER_REPETITIONS)
    return {
        "generator": generator,
        "family": family,
        "scenario_id": sid,
        "nominal_alpha": alpha,
        "false_pass_rate": rejects / OUTER_REPETITIONS,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
        "estimable_repetitions": estimable,
        "outer_repetitions": OUTER_REPETITIONS,
        "inner_repetitions": INNER_REPETITIONS,
        "reference_clusters": g,
        "comparison_clusters": "",
        "icc_reference": rho,
        "icc_comparison": "",
        "baseline_rate": p,
    }


def task_key(kind, sid, generator):
    return f"{kind}__{sid}__{generator}"


def save_part(key, row):
    PART_DIR.mkdir(parents=True, exist_ok=True)
    (PART_DIR / f"{key}.json").write_text(
        json.dumps(row, indent=2) + "\n",
        encoding="utf-8",
    )


def load_part(key):
    path = PART_DIR / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    completed_rows = []

    for generator in GENERATORS:
        for base in PAIR_TASKS:
            sid = base[0]
            key = task_key("pair", sid, generator)
            cached = load_part(key)
            if cached is not None:
                completed_rows.append(cached)
            else:
                jobs.append(("pair", key, (*base, generator)))

        for base in ONE_TASKS:
            sid = base[0]
            key = task_key("one", sid, generator)
            cached = load_part(key)
            if cached is not None:
                completed_rows.append(cached)
            else:
                jobs.append(("one", key, (*base, generator)))

    print("=== FINAL INFERENCE VALIDATION ===")
    print(
        f"tasks total={len(PAIR_TASKS)*len(GENERATORS)+len(ONE_TASKS)*len(GENERATORS)} "
        f"cached={len(completed_rows)} pending={len(jobs)} "
        f"workers={MAX_WORKERS}"
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {}
            for kind, key, task in jobs:
                fn = run_pair_task if kind == "pair" else run_one_task
                future = executor.submit(fn, task)
                future_map[future] = (kind, key, task[0], task[-1])

            for future in as_completed(future_map):
                kind, key, sid, generator = future_map[future]
                row = future.result()
                save_part(key, row)
                completed_rows.append(row)
                print(
                    f"DONE {generator} {row['family']} {sid}: "
                    f"false_pass={row['false_pass_rate']:.5f} "
                    f"(MC95% {row['mc_cp_lower_95']:.5f}-"
                    f"{row['mc_cp_upper_95']:.5f})"
                )

    rows = sorted(
        completed_rows,
        key=lambda r: (r["generator"], r["family"], r["scenario_id"]),
    )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    JSON_PATH.write_text(
        json.dumps(
            {
                "validation_id": "external_validation_final_inference_validation_v1",
                "status": "final_focused_validation_complete",
                "outer_repetitions_per_task": OUTER_REPETITIONS,
                "inner_null_repetitions": INNER_REPETITIONS,
                "generators": list(GENERATORS),
                "exact_matched_cluster_counts": True,
                "important": (
                    "Passing this focused validation supports freezing the "
                    "inference/design architecture, not the final sample size. "
                    "The next professor-required step is the full cluster "
                    "power/type-I grid."
                ),
                "csv": str(CSV_PATH),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    for generator in GENERATORS:
        print(f"\n### GENERATOR: {generator}")
        for family in (
            "relative_FNR_NI_boundary",
            "absolute_FNR_boundary",
            "FPR_boundary",
        ):
            sub = [
                row
                for row in rows
                if row["generator"] == generator and row["family"] == family
            ]
            worst = max(sub, key=lambda row: float(row["false_pass_rate"]))
            print(
                f"WORST {generator} {family}: "
                f"{float(worst['false_pass_rate']):.5f} "
                f"(MC95% {float(worst['mc_cp_lower_95']):.5f}-"
                f"{float(worst['mc_cp_upper_95']):.5f}) "
                f"at {worst['scenario_id']}"
            )

    print(
        "\nDECISION GATE: if no scenario has a clear anti-conservative signal "
        "(its Monte Carlo uncertainty is compatible with the nominal alpha), "
        "freeze exact matched cluster counts plus the restricted-null Monte "
        "Carlo inference architecture and move directly to the full "
        "professor-requested cluster power/type-I grid. Do not add another "
        "exploratory inference method."
    )


if __name__ == "__main__":
    run()
