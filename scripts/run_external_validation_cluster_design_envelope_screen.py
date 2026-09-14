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

BASE_SEED = 20260909
OUTER = 1000
INNER = 199
MAX_WORKERS = min(3, max(1, (os.cpu_count() or 2) - 1))

OUT_DIR = Path("results")
PART_DIR = OUT_DIR / "external_validation_cluster_design_envelope_screen_v1_parts"
CSV_PATH = OUT_DIR / "external_validation_cluster_design_envelope_screen_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_cluster_design_envelope_screen_v1.json"

ENVELOPES = {
    "tight": (15,18,20,22,25),
    "moderate": (10,15,20,25,30),
    "wider": (8,14,20,26,32),
}
G_GRID = (20, 40, 60)
GENERATORS = ("beta_binomial", "logit_normal")


def sizes_from_pattern(g, pattern):
    return np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=np.int64)


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k+1, n-k))
    return lo, hi


def draw(rng, p, rho, sizes, generator, calibration=None):
    if generator == "beta_binomial":
        return _beta_binomial_counts(
            marginal_probability=p,
            icc=rho,
            cluster_sizes=sizes,
            repetitions=1,
            rng=rng,
        )[0]

    if generator == "logit_normal":
        cal = calibration or calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )
        z = rng.normal(size=len(sizes))
        probs = expit(cal.intercept + cal.random_intercept_sd * z)
        return rng.binomial(sizes, probs).astype(np.int64)

    raise ValueError(generator)


def run_relative(task):
    envelope, g, family, p0, rho_r, rho_c, generator = task
    sizes = sizes_from_pattern(g, ENVELOPES[envelope])

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

    outer_rng = np.random.default_rng(
        stable_seed(BASE_SEED, envelope, g, family, generator, "outer")
    )
    reject = 0
    estimable = 0

    for rep in range(OUTER):
        r = draw(outer_rng, p0, rho_r, sizes, generator, cal_r)
        c = draw(outer_rng, p0 + 0.03, rho_c, sizes, generator, cal_c)
        inner_rng = np.random.default_rng(
            stable_seed(BASE_SEED, envelope, g, family, generator, rep, "inner")
        )
        try:
            out = two_sample_ni_constrained_null_mc(
                reference_events=r,
                reference_sizes=sizes,
                comparison_events=c,
                comparison_sizes=sizes,
                margin=0.03,
                alpha=0.025,
                bootstrap_repetitions=INNER,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue
        estimable += 1
        reject += int(out.reject_noninferiority_null)

    lo, hi = cp(reject, OUTER)
    return {
        "envelope": envelope,
        "G_per_cell": g,
        "generator": generator,
        "family": family,
        "nominal_alpha": 0.025,
        "false_pass_rate": reject / OUTER,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
        "estimable_repetitions": estimable,
        "outer_repetitions": OUTER,
        "inner_repetitions": INNER,
        "clear_fail": lo > 0.025,
    }


def run_one(task):
    envelope, g, family, p, rho, alpha, generator = task
    sizes = sizes_from_pattern(g, ENVELOPES[envelope])

    calibration = None
    if generator == "logit_normal":
        calibration = calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )

    outer_rng = np.random.default_rng(
        stable_seed(BASE_SEED, envelope, g, family, generator, "outer")
    )
    reject = 0
    estimable = 0

    for rep in range(OUTER):
        x = draw(outer_rng, p, rho, sizes, generator, calibration)
        inner_rng = np.random.default_rng(
            stable_seed(BASE_SEED, envelope, g, family, generator, rep, "inner")
        )
        try:
            out = one_sample_constrained_null_mc(
                events=x,
                sizes=sizes,
                boundary=p,
                alpha=alpha,
                bootstrap_repetitions=INNER,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue
        estimable += 1
        reject += int(out.reject_below_boundary)

    lo, hi = cp(reject, OUTER)
    return {
        "envelope": envelope,
        "G_per_cell": g,
        "generator": generator,
        "family": family,
        "nominal_alpha": alpha,
        "false_pass_rate": reject / OUTER,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
        "estimable_repetitions": estimable,
        "outer_repetitions": OUTER,
        "inner_repetitions": INNER,
        "clear_fail": lo > alpha,
    }


def key(kind, envelope, g, family, generator):
    return f"{kind}__{envelope}__G{g}__{family}__{generator}"


def save_part(k, row):
    PART_DIR.mkdir(parents=True, exist_ok=True)
    (PART_DIR / f"{k}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def load_part(k):
    p = PART_DIR / f"{k}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    rows = []

    for envelope in ENVELOPES:
        for g in G_GRID:
            for generator in GENERATORS:
                for family, p0, rr, rc in (
                    ("relative_NI_lowrisk", 0.02, 0.05, 0.10),
                    ("relative_NI_typical", 0.05, 0.05, 0.10),
                ):
                    k = key("relative", envelope, g, family, generator)
                    cached = load_part(k)
                    if cached is not None:
                        rows.append(cached)
                    else:
                        jobs.append(("relative", k, (envelope, g, family, p0, rr, rc, generator)))

                for family, p, rho, alpha in (
                    ("absolute_FNR_boundary", 0.10, 0.10, 0.025),
                    ("FPR_boundary", 0.05, 0.10, 0.05),
                ):
                    k = key("one", envelope, g, family, generator)
                    cached = load_part(k)
                    if cached is not None:
                        rows.append(cached)
                    else:
                        jobs.append(("one", k, (envelope, g, family, p, rho, alpha, generator)))

    print("=== CLUSTER DESIGN ENVELOPE SCREEN ===")
    print(
        f"tasks total=72 cached={len(rows)} pending={len(jobs)} "
        f"workers={MAX_WORKERS} outer/task={OUTER} inner={INNER}"
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fmap = {}
            for kind, k, task in jobs:
                fn = run_relative if kind == "relative" else run_one
                fut = ex.submit(fn, task)
                fmap[fut] = (k, task[0], task[1], task[2], task[-1])

            for fut in as_completed(fmap):
                k, envelope, g, family, generator = fmap[fut]
                row = fut.result()
                save_part(k, row)
                rows.append(row)
                print(
                    f"DONE {envelope} G={g} {generator} {family}: "
                    f"{row['false_pass_rate']:.4f} "
                    f"(MC95% {row['mc_cp_lower_95']:.4f}-"
                    f"{row['mc_cp_upper_95']:.4f}) "
                    f"{'FAIL' if row['clear_fail'] else 'screen-ok'}"
                )

    rows = sorted(rows, key=lambda r: (
        r["envelope"], r["G_per_cell"], r["generator"], r["family"]
    ))

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    design_summary = []
    for envelope in ENVELOPES:
        for g in G_GRID:
            sub = [r for r in rows if r["envelope"] == envelope and r["G_per_cell"] == g]
            clear_failures = [r for r in sub if r["clear_fail"]]
            design_summary.append({
                "envelope": envelope,
                "G_per_cell": g,
                "screen_admissible": len(clear_failures) == 0,
                "clear_failure_count": len(clear_failures),
                "worst_alpha_ratio": max(
                    float(r["false_pass_rate"]) / float(r["nominal_alpha"])
                    for r in sub
                ),
            })

    JSON_PATH.write_text(
        json.dumps(
            {
                "screen_id": "external_validation_cluster_design_envelope_screen_v1",
                "status": "screen_complete_not_frozen",
                "outer_repetitions_per_task": OUTER,
                "inner_null_repetitions": INNER,
                "design_summary": design_summary,
                "next_step": (
                    "High-repetition confirmation only for the narrowest feasible "
                    "screen-admissible envelope/G region, adding the two-point "
                    "generator. If none survives, do not continue estimator hunting; "
                    "the confirmatory architecture requires supervisor-level redesign."
                ),
                "csv": str(CSV_PATH),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n=== DESIGN SUMMARY ===")
    for d in design_summary:
        print(
            f"{d['envelope']} G={d['G_per_cell']}: "
            f"{'SCREEN-ADMISSIBLE' if d['screen_admissible'] else 'FAIL'} "
            f"(clear failures={d['clear_failure_count']}, "
            f"worst alpha-ratio={d['worst_alpha_ratio']:.2f})"
        )


if __name__ == "__main__":
    run()
