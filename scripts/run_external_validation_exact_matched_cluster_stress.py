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
OUTER_REPETITIONS = 3000
INNER_REPETITIONS = 999
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_exact_matched_cluster_stress_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_exact_matched_cluster_stress_v1.json"


def sizes_from_pattern(g: int, pattern: tuple[int, ...]) -> np.ndarray:
    out = np.asarray([pattern[i % len(pattern)] for i in range(g)], dtype=int)
    if len(out) != g or np.any(out <= 0):
        raise RuntimeError("invalid cluster-size pattern")
    return out


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


def draw_counts(
    rng: np.random.Generator,
    p: float,
    rho: float,
    sizes: np.ndarray,
    generator: str,
) -> np.ndarray:
    if generator == "beta_binomial":
        concentration = (1.0 / rho) - 1.0
        a = p * concentration
        b = (1.0 - p) * concentration
        probs = rng.beta(a, b, size=len(sizes))
    elif generator == "two_point":
        # Two-point random probability with the same E[P]=p and
        # Var(P)=rho*p*(1-p), hence the same Bernoulli ICC.
        variance = rho * p * (1.0 - p)
        high = min(1.0, p + variance / p)
        probability_high = p / high
        probs = np.where(
            rng.random(len(sizes)) < probability_high,
            high,
            0.0,
        )
    else:
        raise ValueError(generator)

    return rng.binomial(sizes, probs).astype(int)


# All pair scenarios have EXACTLY equal numbers of independent provenance
# clusters in reference and comparison cells. Cluster-size distributions are
# deliberately allowed to differ.
PAIR_SCENARIOS = (
    # id, G, ref pattern, cmp pattern, ref FNR, ref ICC, cmp ICC
    ("G40_equal_vs_modcv", 40, (20,), (10,15,20,25,30), 0.05, 0.01, 0.10),
    ("G60_modcv_vs_highcv", 60, (10,15,20,25,30), (5,10,15,20,50), 0.05, 0.03, 0.10),
    ("G80_highcv_vs_modcv", 80, (5,10,15,20,50), (10,15,20,25,30), 0.07, 0.05, 0.10),
    ("G80_lowrisk_highcv_mismatch", 80, (5,10,15,20,50), (5,5,10,20,60), 0.02, 0.05, 0.10),
    ("G100_vhighcv_vs_equal", 100, (5,5,10,20,60), (20,), 0.05, 0.05, 0.10),
    ("G100_reverseICC_mismatched_sizes", 100, (10,15,20,25,30), (5,10,15,20,50), 0.05, 0.10, 0.03),
    ("G120_different_means", 120, (8,12,16,20,24), (10,20,30,40,50), 0.05, 0.03, 0.10),
    ("G150_high_vs_vhigh_lowrisk", 150, (5,10,15,20,50), (5,5,10,20,60), 0.02, 0.05, 0.10),
)

ONE_SAMPLE_SCENARIOS = (
    ("G40_equal_icc01", 40, (20,), 0.01),
    ("G60_modcv_icc05", 60, (10,15,20,25,30), 0.05),
    ("G80_highcv_icc10", 80, (5,10,15,20,50), 0.10),
    ("G100_vhighcv_icc10", 100, (5,5,10,20,60), 0.10),
    ("G120_highcv_icc05", 120, (5,10,15,20,50), 0.05),
    ("G150_vhighcv_icc10", 150, (5,5,10,20,60), 0.10),
)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for generator in ("beta_binomial", "two_point"):
        for sid, g, ref_pattern, cmp_pattern, p0, rho_r, rho_c in PAIR_SCENARIOS:
            sr = sizes_from_pattern(g, ref_pattern)
            sc = sizes_from_pattern(g, cmp_pattern)
            if len(sr) != len(sc):
                raise RuntimeError("exact matched-cluster-count invariant violated")

            outer_rng = np.random.default_rng(
                stable_seed(BASE_SEED, generator, "matched_rd_outer", sid)
            )
            rejects = 0
            estimable = 0

            for rep in range(OUTER_REPETITIONS):
                kr = draw_counts(outer_rng, p0, rho_r, sr, generator)
                kc = draw_counts(outer_rng, p0 + 0.03, rho_c, sc, generator)
                inner_rng = np.random.default_rng(
                    stable_seed(
                        BASE_SEED,
                        generator,
                        "matched_rd_inner",
                        sid,
                        rep,
                    )
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
            rp = profile(sr)
            cp = profile(sc)
            rows.append({
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
                "reference_rows": rp["rows"],
                "comparison_rows": cp["rows"],
                "reference_cv": rp["cv"],
                "comparison_cv": cp["cv"],
                "reference_mean_cluster_size": rp["mean"],
                "comparison_mean_cluster_size": cp["mean"],
                "icc_reference": rho_r,
                "icc_comparison": rho_c,
                "baseline_rate": p0,
            })

        for sid, g, pattern, rho in ONE_SAMPLE_SCENARIOS:
            sizes = sizes_from_pattern(g, pattern)
            sp = profile(sizes)

            for family, p, alpha in (
                ("absolute_FNR_boundary", 0.10, 0.025),
                ("FPR_boundary", 0.05, 0.05),
            ):
                outer_rng = np.random.default_rng(
                    stable_seed(BASE_SEED, generator, family, sid, "outer")
                )
                rejects = 0
                estimable = 0

                for rep in range(OUTER_REPETITIONS):
                    k = draw_counts(outer_rng, p, rho, sizes, generator)
                    inner_rng = np.random.default_rng(
                        stable_seed(
                            BASE_SEED,
                            generator,
                            family,
                            sid,
                            rep,
                            "inner",
                        )
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
                rows.append({
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
                    "reference_rows": sp["rows"],
                    "comparison_rows": "",
                    "reference_cv": sp["cv"],
                    "comparison_cv": "",
                    "reference_mean_cluster_size": sp["mean"],
                    "comparison_mean_cluster_size": "",
                    "icc_reference": rho,
                    "icc_comparison": "",
                    "baseline_rate": p,
                })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    JSON_PATH.write_text(
        json.dumps(
            {
                "screen_id": "external_validation_exact_matched_cluster_stress_v1",
                "status": "design_stress_test_not_final",
                "outer_repetitions_per_scenario": OUTER_REPETITIONS,
                "inner_null_repetitions": INNER_REPETITIONS,
                "exact_cluster_count_match": True,
                "generators": ["beta_binomial", "two_point"],
                "important": (
                    "This is a design/inference stress test. It does not freeze "
                    "the final procedure, cluster minima/caps, sample sizes, or W0."
                ),
                "csv": str(CSV_PATH),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=== EXACT-MATCHED CLUSTER STRESS CALIBRATION ===")
    for generator in ("beta_binomial", "two_point"):
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
            print(f"\n--- {family} ---")
            for row in sub:
                print(
                    f"{row['scenario_id']}: "
                    f"false_pass={float(row['false_pass_rate']):.5f} "
                    f"(MC95% {float(row['mc_cp_lower_95']):.5f}-"
                    f"{float(row['mc_cp_upper_95']):.5f}); "
                    f"nominal={float(row['nominal_alpha']):.3f}; "
                    f"estimable={row['estimable_repetitions']}/{OUTER_REPETITIONS}"
                )

            worst = max(sub, key=lambda row: float(row["false_pass_rate"]))
            print(
                f"WORST {generator} {family}: "
                f"{float(worst['false_pass_rate']):.5f} "
                f"(MC95% {float(worst['mc_cp_lower_95']):.5f}-"
                f"{float(worst['mc_cp_upper_95']):.5f}) "
                f"at {worst['scenario_id']}"
            )

    print(
        "\nDECISION GATE: exact matching can become a candidate design "
        "requirement only if the relative NI boundary no longer shows a clear "
        "anti-conservative signal under the harder unequal-ICC and unequal-size "
        "stress scenarios. A successful run still requires one final "
        "high-repetition validation with an additional misspecification "
        "generator before method/design freeze and power sizing."
    )


if __name__ == "__main__":
    run()
