from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist

from monitor_fusion.external_validation.beta_binomial_profile import (
    BetaBinomialProfileNotEstimable,
)
from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    stable_seed,
)
from monitor_fusion.external_validation.constrained_null_mc import (
    one_sample_constrained_null_mc,
    two_sample_ni_constrained_null_mc,
)
from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
)

BASE_SEED = 20260913
OUTER = 500
INNER = 199
MAX_WORKERS = min(3, max(1, (os.cpu_count() or 2) - 1))

PATTERN = (15, 18, 20, 22, 25)
G_GRID = (60, 80, 100, 120, 140, 160, 180, 200)
TYPEI_GENERATORS = ("beta_binomial", "logit_normal")

OUT_DIR = Path("results")
PART_DIR = OUT_DIR / "external_validation_cluster_power_typeI_grid_v2_parts"
CSV_PATH = OUT_DIR / "external_validation_cluster_power_typeI_grid_v2.csv"
JSON_PATH = OUT_DIR / "external_validation_cluster_power_typeI_grid_v2.json"


def sizes(g: int) -> np.ndarray:
    return np.asarray([PATTERN[i % len(PATTERN)] for i in range(g)], dtype=np.int64)


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k+1, n-k))
    return lo, hi


def draw(
    rng: np.random.Generator,
    p: float,
    rho: float,
    s: np.ndarray,
    generator: str,
    calibration=None,
) -> np.ndarray:
    if generator == "beta_binomial":
        return _beta_binomial_counts(
            marginal_probability=p,
            icc=rho,
            cluster_sizes=s,
            repetitions=1,
            rng=rng,
        )[0].astype(np.int64)

    if generator == "logit_normal":
        cal = calibration or calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )
        z = rng.normal(size=len(s))
        probs = expit(cal.intercept + cal.random_intercept_sd * z)
        return rng.binomial(s, probs).astype(np.int64)

    raise ValueError(generator)


def result_row(*, kind, g, scenario, generator, nominal_alpha, success, estimable):
    lo, hi = cp(success, OUTER)
    return {
        "kind": kind,
        "G_per_cell": g,
        "rows_per_cell": int(sizes(g).sum()),
        "scenario": scenario,
        "generator": generator,
        "nominal_alpha_or_power_target": nominal_alpha,
        "success_rate": success / OUTER,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
        "estimable_repetitions": estimable,
        "outer_repetitions": OUTER,
        "inner_repetitions": INNER,
    }


def run_two_sample(task):
    kind, g, scenario, generator, ref_p, cmp_p, rho_r, rho_c = task
    s = sizes(g)

    cal_r = cal_c = None
    if generator == "logit_normal":
        cal_r = calibrate_logit_normal(
            marginal_probability=ref_p,
            response_scale_icc=rho_r,
        )
        cal_c = calibrate_logit_normal(
            marginal_probability=cmp_p,
            response_scale_icc=rho_c,
        )

    rng = np.random.default_rng(
        stable_seed(BASE_SEED, kind, g, scenario, generator, "outer")
    )
    success = 0
    estimable = 0

    for rep in range(OUTER):
        r = draw(rng, ref_p, rho_r, s, generator, cal_r)
        c = draw(rng, cmp_p, rho_c, s, generator, cal_c)
        inner_rng = np.random.default_rng(
            stable_seed(BASE_SEED, kind, g, scenario, generator, rep, "inner")
        )
        try:
            out = two_sample_ni_constrained_null_mc(
                reference_events=r,
                reference_sizes=s,
                comparison_events=c,
                comparison_sizes=s,
                margin=0.03,
                alpha=0.025,
                bootstrap_repetitions=INNER,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue
        estimable += 1
        success += int(out.reject_noninferiority_null)

    return result_row(
        kind=kind,
        g=g,
        scenario=scenario,
        generator=generator,
        nominal_alpha=(0.025 if kind == "typeI" else 0.80),
        success=success,
        estimable=estimable,
    )


def run_one_sample(task):
    kind, g, scenario, generator, true_p, boundary, rho, alpha = task
    s = sizes(g)

    calibration = None
    if generator == "logit_normal":
        calibration = calibrate_logit_normal(
            marginal_probability=true_p,
            response_scale_icc=rho,
        )

    rng = np.random.default_rng(
        stable_seed(BASE_SEED, kind, g, scenario, generator, "outer")
    )
    success = 0
    estimable = 0

    for rep in range(OUTER):
        x = draw(rng, true_p, rho, s, generator, calibration)
        inner_rng = np.random.default_rng(
            stable_seed(BASE_SEED, kind, g, scenario, generator, rep, "inner")
        )
        try:
            out = one_sample_constrained_null_mc(
                events=x,
                sizes=s,
                boundary=boundary,
                alpha=alpha,
                bootstrap_repetitions=INNER,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue
        estimable += 1
        success += int(out.reject_below_boundary)

    return result_row(
        kind=kind,
        g=g,
        scenario=scenario,
        generator=generator,
        nominal_alpha=(alpha if kind == "typeI" else 0.80),
        success=success,
        estimable=estimable,
    )


def key(task):
    return "__".join(str(x).replace(".", "p") for x in task[:4])


def save_part(k, row):
    PART_DIR.mkdir(parents=True, exist_ok=True)
    (PART_DIR / f"{k}.json").write_text(
        json.dumps(row, indent=2) + "\n",
        encoding="utf-8",
    )


def load_part(k):
    p = PART_DIR / f"{k}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_tasks():
    tasks = []

    # Type-I calibration at each candidate G.
    for g in G_GRID:
        for generator in TYPEI_GENERATORS:
            tasks.extend([
                (
                    "two",
                    ("typeI", g, "relative_NI_lowrisk_boundary", generator, 0.02, 0.05, 0.05, 0.10),
                ),
                (
                    "two",
                    ("typeI", g, "relative_NI_typical_boundary", generator, 0.05, 0.08, 0.05, 0.10),
                ),
                (
                    "one",
                    ("typeI", g, "absolute_FNR_boundary", generator, 0.10, 0.10, 0.10, 0.025),
                ),
                (
                    "one",
                    ("typeI", g, "FPR_boundary", generator, 0.05, 0.05, 0.10, 0.05),
                ),
            ])

    # Power under the design model. These preserve the existing project
    # convention of an 80% planning target and report 90% attainment too.
    for g in G_GRID:
        tasks.extend([
            (
                "two",
                ("power", g, "relative_NI_zero_diff_lowrisk", "beta_binomial", 0.02, 0.02, 0.05, 0.10),
            ),
            (
                "two",
                ("power", g, "relative_NI_zero_diff_primary", "beta_binomial", 0.05, 0.05, 0.05, 0.10),
            ),
            (
                "two",
                ("power_sensitivity", g, "relative_NI_zero_diff_near_ceiling", "beta_binomial", 0.08, 0.08, 0.05, 0.10),
            ),
            (
                "one",
                ("power", g, "absolute_FNR_true_0p05", "beta_binomial", 0.05, 0.10, 0.10, 0.025),
            ),
            (
                "one",
                ("power_sensitivity", g, "absolute_FNR_true_0p075", "beta_binomial", 0.075, 0.10, 0.10, 0.025),
            ),
            (
                "one",
                ("power", g, "FPR_true_0p025", "beta_binomial", 0.025, 0.05, 0.10, 0.05),
            ),
        ])

    return tasks


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    jobs = []

    for mode, task in build_tasks():
        k = key(task)
        cached = load_part(k)
        if cached is not None:
            rows.append(cached)
        else:
            jobs.append((mode, k, task))

    print("=== BROAD CLUSTER TYPE-I / POWER GRID ===")
    print(
        f"tasks total={len(rows)+len(jobs)} cached={len(rows)} "
        f"pending={len(jobs)} workers={MAX_WORKERS} "
        f"outer/task={OUTER} inner={INNER}"
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fmap = {}
            for mode, k, task in jobs:
                fn = run_two_sample if mode == "two" else run_one_sample
                fut = ex.submit(fn, task)
                fmap[fut] = (k, task[0], task[1], task[2], task[3])

            for fut in as_completed(fmap):
                k, kind, g, scenario, generator = fmap[fut]
                row = fut.result()
                save_part(k, row)
                rows.append(row)
                print(
                    f"DONE {kind} G={g} {generator} {scenario}: "
                    f"{row['success_rate']:.3f} "
                    f"(MC95% {row['mc_cp_lower_95']:.3f}-"
                    f"{row['mc_cp_upper_95']:.3f})"
                )

    rows = sorted(
        rows,
        key=lambda r: (r["G_per_cell"], r["kind"], r["scenario"], r["generator"]),
    )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for g in G_GRID:
        sub = [r for r in rows if r["G_per_cell"] == g]
        typei = [r for r in sub if r["kind"] == "typeI"]
        primary_power = [
            r for r in sub
            if r["kind"] == "power"
            and r["scenario"] in {
                "relative_NI_zero_diff_lowrisk",
                "relative_NI_zero_diff_primary",
                "absolute_FNR_true_0p05",
                "FPR_true_0p025",
            }
        ]

        clear_typei_failures = [
            r for r in typei
            if float(r["mc_cp_lower_95"])
            > float(r["nominal_alpha_or_power_target"])
        ]
        min_primary_power = min(float(r["success_rate"]) for r in primary_power)
        min_primary_power_lower = min(float(r["mc_cp_lower_95"]) for r in primary_power)

        summary.append({
            "G_per_cell": g,
            "rows_per_cell": int(sizes(g).sum()),
            "typeI_clear_failure_count": len(clear_typei_failures),
            "typeI_screen_admissible": len(clear_typei_failures) == 0,
            "minimum_primary_power_point": min_primary_power,
            "minimum_primary_power_mc95_lower": min_primary_power_lower,
            "meets_80pct_primary_power_point": min_primary_power >= 0.80,
            "meets_90pct_primary_power_point": min_primary_power >= 0.90,
            "broad_screen_candidate_80": (
                len(clear_typei_failures) == 0 and min_primary_power >= 0.80
            ),
        })

    candidate = next(
        (x for x in summary if x["broad_screen_candidate_80"]),
        None,
    )

    JSON_PATH.write_text(
        json.dumps(
            {
                "grid_id": "external_validation_cluster_power_typeI_grid_v2",
                "status": "broad_grid_complete_not_frozen",
                "tight_cluster_size_pattern": list(PATTERN),
                "mean_cluster_size": 20,
                "primary_power_target": 0.80,
                "report_90_percent_sensitivity": True,
                "summary": summary,
                "smallest_broad_screen_candidate_80": candidate,
                "next_step": (
                    "Run targeted high-repetition confirmation at the smallest "
                    "screen candidate and its neighboring G values using "
                    "beta-binomial, two-point, and logit-normal generators. "
                    "Only after that confirmation freeze inference, cluster "
                    "minimum/cap, and Y1/Y0 quotas."
                ),
                "csv": str(CSV_PATH),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n=== GRID SUMMARY ===")
    for x in summary:
        print(
            f"G={x['G_per_cell']} rows/cell={x['rows_per_cell']}: "
            f"typeI={'OK' if x['typeI_screen_admissible'] else 'FAIL'}; "
            f"min-primary-power={x['minimum_primary_power_point']:.3f}; "
            f"80%={'YES' if x['meets_80pct_primary_power_point'] else 'NO'}; "
            f"90%={'YES' if x['meets_90pct_primary_power_point'] else 'NO'}"
        )

    print()
    if candidate is None:
        print(
            "DECISION: no G in this broad grid satisfies both the type-I "
            "screen and all primary 80% planning-power conditions."
        )
    else:
        print(
            "SMALLEST BROAD-SCREEN 80% CANDIDATE: "
            f"G={candidate['G_per_cell']} per cell "
            f"({candidate['rows_per_cell']} eligible class-specific "
            "dependency representatives per cell at the tight pattern)."
        )
        print(
            "This is NOT frozen. Next: high-repetition confirmation at this "
            "G and neighboring G values with all three generators."
        )


if __name__ == "__main__":
    run()
