from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import expit

from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    stable_seed,
)
from monitor_fusion.external_validation.cluster_wcr_current import (
    fixed_webb_weights,
    wcr_c_one_sample_batch,
    wcr_c_two_sample_batch,
)
from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
)

BASE_SEED = 20260909
OUTER_REPETITIONS = 10000
BOOTSTRAP_REPETITIONS = 999
CHUNK_SIZE = 500
MAX_WORKERS = min(3, max(1, (os.cpu_count() or 2) - 1))

OUT_DIR = Path("results")
PART_DIR = OUT_DIR / "external_validation_wcr_current_design_validation_v1_parts"
CSV_PATH = OUT_DIR / "external_validation_wcr_current_design_validation_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_wcr_current_design_validation_v1.json"


def sizes_from_pattern(g, pattern):
    return np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=np.int64)


def draw_counts(rng, p, rho, sizes, repetitions, generator, calibration=None):
    if generator == "beta_binomial":
        return _beta_binomial_counts(
            marginal_probability=p,
            icc=rho,
            cluster_sizes=sizes,
            repetitions=repetitions,
            rng=rng,
        )

    if generator == "two_point":
        variance = rho * p * (1.0 - p)
        high = min(1.0, p + variance / p)
        probability_high = p / high
        probabilities = np.where(
            rng.random((repetitions, len(sizes))) < probability_high,
            high,
            0.0,
        )
        return rng.binomial(sizes[None, :], probabilities).astype(np.int64)

    if generator == "logit_normal":
        cal = calibration or calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )
        z = rng.normal(size=(repetitions, len(sizes)))
        probabilities = expit(cal.intercept + cal.random_intercept_sd * z)
        return rng.binomial(sizes[None, :], probabilities).astype(np.int64)

    raise ValueError(generator)


PAIR_TASKS = (
    ("R_G40_equal_vs_modcv", 40, (20,), (10,15,20,25,30), 0.05, 0.01, 0.10),
    ("R_G80_lowrisk_highcv_mismatch", 80, (5,10,15,20,50), (5,5,10,20,60), 0.02, 0.05, 0.10),
    ("R_G100_reverseICC_mismatched_sizes", 100, (10,15,20,25,30), (5,10,15,20,50), 0.05, 0.10, 0.03),
)

ONE_TASKS = (
    ("A_G40_equal_icc01", "absolute_FNR_boundary", 40, (20,), 0.10, 0.01, 0.025),
    ("A_G150_vhighcv_icc10", "absolute_FNR_boundary", 150, (5,5,10,20,60), 0.10, 0.10, 0.025),
    ("F_G40_equal_icc01", "FPR_boundary", 40, (20,), 0.05, 0.01, 0.05),
    ("F_G150_vhighcv_icc10", "FPR_boundary", 150, (5,5,10,20,60), 0.05, 0.10, 0.05),
)

GENERATORS = ("beta_binomial", "two_point", "logit_normal")


def aggregate_parts(parts):
    repetitions = sum(p["repetitions"] for p in parts)
    rejects = sum(p["reject_count"] for p in parts)
    estimable = sum(p["estimable_count"] for p in parts)

    from scipy.stats import beta as beta_dist
    if rejects == 0:
        lo = 0.0
    else:
        lo = float(beta_dist.ppf(0.025, rejects, repetitions - rejects + 1))
    if rejects == repetitions:
        hi = 1.0
    else:
        hi = float(beta_dist.ppf(0.975, rejects + 1, repetitions - rejects))

    return {
        "repetitions": repetitions,
        "reject_count": rejects,
        "estimable_count": estimable,
        "false_pass_rate": rejects / repetitions,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
    }


def run_pair_task(task):
    sid, g, ref_pattern, cmp_pattern, p0, rho_r, rho_c, generator = task
    sr = sizes_from_pattern(g, ref_pattern)
    sc = sizes_from_pattern(g, cmp_pattern)
    if len(sr) != len(sc):
        raise RuntimeError("current design requires exact equal cluster counts")

    wr = fixed_webb_weights(
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        cluster_count=g,
        seed=stable_seed(BASE_SEED, "weights", generator, sid, "ref"),
    )
    wc = fixed_webb_weights(
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        cluster_count=g,
        seed=stable_seed(BASE_SEED, "weights", generator, sid, "cmp"),
    )

    cal_r = cal_c = None
    if generator == "logit_normal":
        cal_r = calibrate_logit_normal(
            marginal_probability=p0,
            response_scale_icc=rho_r,
        )
        cal_c = calibrate_logit_normal(
            marginal_probability=p0 + 0.03,
            response_scale_icc=rho_c,
        )

    rng = np.random.default_rng(
        stable_seed(BASE_SEED, "outer", generator, sid)
    )
    parts = []

    for start in range(0, OUTER_REPETITIONS, CHUNK_SIZE):
        reps = min(CHUNK_SIZE, OUTER_REPETITIONS - start)
        ref = draw_counts(rng, p0, rho_r, sr, reps, generator, cal_r)
        cmp_ = draw_counts(rng, p0 + 0.03, rho_c, sc, reps, generator, cal_c)
        result = wcr_c_two_sample_batch(
            reference_counts=ref,
            comparison_counts=cmp_,
            reference_cluster_sizes=sr,
            comparison_cluster_sizes=sc,
            margin=0.03,
            alpha=0.025,
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
            reference_weights=wr,
            comparison_weights=wc,
        )
        parts.append(result.__dict__)

    out = aggregate_parts(parts)
    return {
        "generator": generator,
        "family": "relative_FNR_NI_boundary",
        "scenario_id": sid,
        "nominal_alpha": 0.025,
        "reference_clusters": g,
        "comparison_clusters": g,
        "icc_reference": rho_r,
        "icc_comparison": rho_c,
        "baseline_rate": p0,
        **out,
    }


def run_one_task(task):
    sid, family, g, pattern, p, rho, alpha, generator = task
    sizes = sizes_from_pattern(g, pattern)
    weights = fixed_webb_weights(
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        cluster_count=g,
        seed=stable_seed(BASE_SEED, "weights", generator, sid),
    )

    calibration = None
    if generator == "logit_normal":
        calibration = calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )

    rng = np.random.default_rng(
        stable_seed(BASE_SEED, "outer", generator, sid)
    )
    parts = []

    for start in range(0, OUTER_REPETITIONS, CHUNK_SIZE):
        reps = min(CHUNK_SIZE, OUTER_REPETITIONS - start)
        counts = draw_counts(
            rng, p, rho, sizes, reps, generator, calibration
        )
        result = wcr_c_one_sample_batch(
            counts=counts,
            cluster_sizes=sizes,
            boundary=p,
            alpha=alpha,
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
            weights=weights,
        )
        parts.append(result.__dict__)

    out = aggregate_parts(parts)
    return {
        "generator": generator,
        "family": family,
        "scenario_id": sid,
        "nominal_alpha": alpha,
        "reference_clusters": g,
        "comparison_clusters": "",
        "icc_reference": rho,
        "icc_comparison": "",
        "baseline_rate": p,
        **out,
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

    completed = []
    jobs = []

    for generator in GENERATORS:
        for base in PAIR_TASKS:
            key = task_key("pair", base[0], generator)
            cached = load_part(key)
            if cached is not None:
                completed.append(cached)
            else:
                jobs.append(("pair", key, (*base, generator)))

        for base in ONE_TASKS:
            key = task_key("one", base[0], generator)
            cached = load_part(key)
            if cached is not None:
                completed.append(cached)
            else:
                jobs.append(("one", key, (*base, generator)))

    print("=== CURRENT-DESIGN WCR-C VALIDATION ===")
    print(
        f"tasks total={21} cached={len(completed)} pending={len(jobs)} "
        f"workers={MAX_WORKERS} outer/task={OUTER_REPETITIONS} "
        f"bootstrap={BOOTSTRAP_REPETITIONS}"
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {}
            for kind, key, task in jobs:
                fn = run_pair_task if kind == "pair" else run_one_task
                fut = executor.submit(fn, task)
                future_map[fut] = (key, task[0], task[-1])

            for fut in as_completed(future_map):
                key, sid, generator = future_map[fut]
                row = fut.result()
                save_part(key, row)
                completed.append(row)
                print(
                    f"DONE {generator} {row['family']} {sid}: "
                    f"false_pass={row['false_pass_rate']:.5f} "
                    f"(MC95% {row['mc_cp_lower_95']:.5f}-"
                    f"{row['mc_cp_upper_95']:.5f})"
                )

    rows = sorted(
        completed,
        key=lambda r: (r["generator"], r["family"], r["scenario_id"]),
    )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    clear_failures = [
        r for r in rows
        if float(r["mc_cp_lower_95"]) > float(r["nominal_alpha"])
    ]

    JSON_PATH.write_text(
        json.dumps(
            {
                "validation_id": "external_validation_wcr_current_design_validation_v1",
                "status": (
                    "failed_clear_anti_conservative_signal"
                    if clear_failures
                    else "passed_no_clear_anti_conservative_signal"
                ),
                "method": "WCR-C restricted wild-cluster bootstrap-t with CRV1 and Webb six-point weights",
                "outer_repetitions_per_task": OUTER_REPETITIONS,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "generators": list(GENERATORS),
                "exact_matched_cluster_counts_for_relative_NI": True,
                "clear_failure_rule": "MC95% lower endpoint > nominal one-sided alpha",
                "clear_failure_scenarios": [
                    {
                        "generator": r["generator"],
                        "family": r["family"],
                        "scenario_id": r["scenario_id"],
                        "false_pass_rate": r["false_pass_rate"],
                        "mc_cp_lower_95": r["mc_cp_lower_95"],
                        "nominal_alpha": r["nominal_alpha"],
                    }
                    for r in clear_failures
                ],
                "next_step_if_pass": (
                    "freeze cluster-aware FNR/FPR inference and exact matched "
                    "relative-NI cluster-count requirement, then run the full "
                    "professor-requested cluster type-I/power grid"
                ),
                "next_step_if_fail": (
                    "do not continue estimator hunting; reconsider the "
                    "inferential architecture/design against the professor requirement"
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
                r for r in rows
                if r["generator"] == generator and r["family"] == family
            ]
            worst = max(sub, key=lambda r: float(r["false_pass_rate"]))
            print(
                f"WORST {generator} {family}: "
                f"{float(worst['false_pass_rate']):.5f} "
                f"(MC95% {float(worst['mc_cp_lower_95']):.5f}-"
                f"{float(worst['mc_cp_upper_95']):.5f}) "
                f"nominal={float(worst['nominal_alpha']):.3f} "
                f"at {worst['scenario_id']}"
            )

    print()
    if clear_failures:
        print("DECISION: FAIL — at least one scenario is clearly anti-conservative.")
    else:
        print(
            "DECISION: PASS — no scenario shows a clear anti-conservative "
            "signal. Freeze inference architecture and move directly to the "
            "full professor-requested cluster type-I/power grid."
        )


if __name__ == "__main__":
    run()
