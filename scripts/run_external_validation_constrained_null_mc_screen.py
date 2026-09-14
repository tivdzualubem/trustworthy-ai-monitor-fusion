from __future__ import annotations

import csv
import json
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

BASE_SEED = 20260909
OUTER_REPETITIONS = 800
INNER_REPETITIONS = 399
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_constrained_null_mc_screen_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_constrained_null_mc_screen_v1.json"


def sizes_from_pattern(g, pattern):
    return np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=int)


def cp_interval(k, n):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k+1, n-k))
    return lo, hi


def draw_counts(rng, p, rho, sizes, generator):
    if generator == "beta_binomial":
        concentration = (1.0 / rho) - 1.0
        a = p * concentration
        b = (1.0 - p) * concentration
        probs = rng.beta(a, b, size=len(sizes))
    elif generator == "two_point":
        variance = rho * p * (1.0 - p)
        high = p + variance / p
        high = min(high, 1.0)
        prob_high = p / high
        probs = np.where(rng.random(len(sizes)) < prob_high, high, 0.0)
    else:
        raise ValueError(generator)

    return rng.binomial(sizes, probs).astype(int)


PAIR_SCENARIOS = (
    ("G40_equal", 40, 40, (20,), (20,), 0.05, 0.01, 0.05),
    ("G60_modcv", 60, 60, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.03, 0.10),
    ("G80_highcv", 80, 80, (5,10,15,20,50), (5,10,15,20,50), 0.07, 0.05, 0.10),
    ("G100_vhighcv", 100, 100, (5,5,10,20,60), (5,5,10,20,60), 0.05, 0.05, 0.10),
    ("G100v80_modcv", 100, 80, (10,15,20,25,30), (10,15,20,25,30), 0.05, 0.05, 0.10),
    ("G80_lowrisk_highcv", 80, 80, (5,10,15,20,50), (5,10,15,20,50), 0.02, 0.05, 0.10),
)

ONE_SCENARIOS = (
    ("G40_equal_icc01", 40, (20,), 0.01),
    ("G60_modcv_icc05", 60, (10,15,20,25,30), 0.05),
    ("G80_highcv_icc10", 80, (5,10,15,20,50), 0.10),
    ("G100_vhighcv_icc10", 100, (5,5,10,20,60), 0.10),
)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for generator in ("beta_binomial", "two_point"):
        for sid, gr, gc, pr, pc, p0, rr, rc in PAIR_SCENARIOS:
            sr = sizes_from_pattern(gr, pr)
            sc = sizes_from_pattern(gc, pc)
            outer_rng = np.random.default_rng(
                stable_seed(BASE_SEED, generator, "outer_rd", sid)
            )
            rejects = 0
            estimable = 0

            for rep in range(OUTER_REPETITIONS):
                kr = draw_counts(outer_rng, p0, rr, sr, generator)
                kc = draw_counts(outer_rng, p0 + 0.03, rc, sc, generator)
                inner_rng = np.random.default_rng(
                    stable_seed(BASE_SEED, generator, "inner_rd", sid, rep)
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

            rate = rejects / OUTER_REPETITIONS
            lo, hi = cp_interval(rejects, OUTER_REPETITIONS)
            rows.append({
                "generator": generator,
                "family": "relative_FNR_NI_boundary",
                "scenario_id": sid,
                "nominal_alpha": 0.025,
                "false_pass_rate": rate,
                "mc_cp_lower_95": lo,
                "mc_cp_upper_95": hi,
                "estimable_repetitions": estimable,
                "outer_repetitions": OUTER_REPETITIONS,
                "inner_repetitions": INNER_REPETITIONS,
            })

        for sid, g, pattern, rho in ONE_SCENARIOS:
            sizes = sizes_from_pattern(g, pattern)

            for fam, p, alpha in (
                ("absolute_FNR_boundary", 0.10, 0.025),
                ("FPR_boundary", 0.05, 0.05),
            ):
                outer_rng = np.random.default_rng(
                    stable_seed(BASE_SEED, generator, fam, sid)
                )
                rejects = 0
                estimable = 0

                for rep in range(OUTER_REPETITIONS):
                    k = draw_counts(outer_rng, p, rho, sizes, generator)
                    inner_rng = np.random.default_rng(
                        stable_seed(BASE_SEED, generator, fam, sid, rep, "inner")
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

                rate = rejects / OUTER_REPETITIONS
                lo, hi = cp_interval(rejects, OUTER_REPETITIONS)
                rows.append({
                    "generator": generator,
                    "family": fam,
                    "scenario_id": sid,
                    "nominal_alpha": alpha,
                    "false_pass_rate": rate,
                    "mc_cp_lower_95": lo,
                    "mc_cp_upper_95": hi,
                    "estimable_repetitions": estimable,
                    "outer_repetitions": OUTER_REPETITIONS,
                    "inner_repetitions": INNER_REPETITIONS,
                })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    JSON_PATH.write_text(
        json.dumps({
            "screen_id": "external_validation_constrained_null_mc_screen_v1",
            "status": "candidate_screen_not_final",
            "outer_repetitions_per_scenario": OUTER_REPETITIONS,
            "inner_null_repetitions": INNER_REPETITIONS,
            "method": (
                "restricted-null beta-binomial nuisance fit + plug-in "
                "parametric Monte Carlo tail test of row-marginal proportion/RD"
            ),
            "generators": ["beta_binomial", "two_point"],
            "important": (
                "This screen does not freeze the method or sample sizes. "
                "W0 remains blocked."
            ),
            "csv": str(CSV_PATH)
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== CONSTRAINED-NULL PARAMETRIC MONTE CARLO SCREEN ===")
    for generator in ("beta_binomial", "two_point"):
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
            print(f"\n--- {family} ---")
            for r in sub:
                print(
                    f"{r['scenario_id']}: false_pass={r['false_pass_rate']:.4f} "
                    f"(MC95% {r['mc_cp_lower_95']:.4f}-{r['mc_cp_upper_95']:.4f}); "
                    f"nominal={r['nominal_alpha']:.3f}; "
                    f"estimable={r['estimable_repetitions']}/{OUTER_REPETITIONS}"
                )

            worst = max(sub, key=lambda r: r["false_pass_rate"])
            print(
                f"WORST {generator} {family}: "
                f"{worst['false_pass_rate']:.4f} "
                f"(MC95% {worst['mc_cp_lower_95']:.4f}-"
                f"{worst['mc_cp_upper_95']:.4f}) "
                f"at {worst['scenario_id']}"
            )

    print(
        "\nDECISION GATE: only if the restricted-null Monte Carlo candidate is "
        "acceptably calibrated under both generators do we invest in the final "
        "20k-repetition validation/power implementation. Otherwise stop this "
        "path and reconsider the inferential architecture."
    )


if __name__ == "__main__":
    run()
