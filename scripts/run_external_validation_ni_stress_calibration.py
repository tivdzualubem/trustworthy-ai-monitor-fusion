from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from monitor_fusion.external_validation.cluster_ni_calibration import (
    simulate_boundary_calibration,
)
from monitor_fusion.external_validation.cluster_ni_power import stable_seed


BASE_SEED = 20260907
REPETITIONS = 20000
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_ni_stress_calibration_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_ni_stress_calibration_v1.json"


def sizes_from_pattern(cluster_count: int, pattern: tuple[int, ...]) -> list[int]:
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if not pattern or any(value <= 0 for value in pattern):
        raise ValueError("pattern must contain positive sizes")
    return [pattern[i % len(pattern)] for i in range(cluster_count)]


def profile_stats(sizes: list[int]) -> dict[str, float]:
    arr = np.asarray(sizes, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=0))
    return {
        "clusters": int(len(arr)),
        "rows": int(arr.sum()),
        "mean_cluster_size": mean,
        "cluster_size_cv": float(sd / mean),
        "min_cluster_size": int(arr.min()),
        "max_cluster_size": int(arr.max()),
    }


# These scenarios deliberately stress the issues not covered by the first
# equal-size symmetric calibration: unequal cluster sizes, unequal human/model
# cluster counts, unequal ICCs, and low/high event probabilities.
SCENARIOS = (
    {
        "id": "equal_lowrisk_control",
        "baseline": 0.05,
        "icc_ref": 0.01,
        "icc_cmp": 0.01,
        "ref_clusters": 75,
        "cmp_clusters": 75,
        "ref_pattern": (20,),
        "cmp_pattern": (20,),
    },
    {
        "id": "moderate_cv_symmetric",
        "baseline": 0.05,
        "icc_ref": 0.05,
        "icc_cmp": 0.05,
        "ref_clusters": 100,
        "cmp_clusters": 100,
        "ref_pattern": (10, 15, 20, 25, 30),
        "cmp_pattern": (10, 15, 20, 25, 30),
    },
    {
        "id": "high_cv_symmetric",
        "baseline": 0.10,
        "icc_ref": 0.10,
        "icc_cmp": 0.10,
        "ref_clusters": 100,
        "cmp_clusters": 100,
        "ref_pattern": (5, 5, 10, 20, 60),
        "cmp_pattern": (5, 5, 10, 20, 60),
    },
    {
        "id": "asymmetric_100v50_equalrows",
        "baseline": 0.05,
        "icc_ref": 0.05,
        "icc_cmp": 0.10,
        "ref_clusters": 100,
        "cmp_clusters": 50,
        "ref_pattern": (20,),
        "cmp_pattern": (40,),
    },
    {
        "id": "asymmetric_150v75_equalrows",
        "baseline": 0.10,
        "icc_ref": 0.05,
        "icc_cmp": 0.10,
        "ref_clusters": 150,
        "cmp_clusters": 75,
        "ref_pattern": (20,),
        "cmp_pattern": (40,),
    },
    {
        "id": "asymmetric_highcv_150v75",
        "baseline": 0.10,
        "icc_ref": 0.10,
        "icc_cmp": 0.10,
        "ref_clusters": 150,
        "cmp_clusters": 75,
        "ref_pattern": (5, 10, 15, 20, 50),
        "cmp_pattern": (10, 20, 30, 40, 100),
    },
    {
        "id": "asymmetric_200v100_equalrows",
        "baseline": 0.15,
        "icc_ref": 0.05,
        "icc_cmp": 0.10,
        "ref_clusters": 200,
        "cmp_clusters": 100,
        "ref_pattern": (15,),
        "cmp_pattern": (30,),
    },
    {
        "id": "hard_300v100_equalrows",
        "baseline": 0.20,
        "icc_ref": 0.10,
        "icc_cmp": 0.10,
        "ref_clusters": 300,
        "cmp_clusters": 100,
        "ref_pattern": (10,),
        "cmp_pattern": (30,),
    },
)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in SCENARIOS:
        ref_sizes = sizes_from_pattern(
            scenario["ref_clusters"], scenario["ref_pattern"]
        )
        cmp_sizes = sizes_from_pattern(
            scenario["cmp_clusters"], scenario["cmp_pattern"]
        )
        ref_stats = profile_stats(ref_sizes)
        cmp_stats = profile_stats(cmp_sizes)

        for correction in ("KC", "MD"):
            seed = stable_seed(
                BASE_SEED,
                "stress_boundary_calibration",
                scenario["id"],
                correction,
            )
            result = simulate_boundary_calibration(
                reference_fnr=scenario["baseline"],
                icc_reference=scenario["icc_ref"],
                icc_comparison=scenario["icc_cmp"],
                reference_cluster_sizes=ref_sizes,
                comparison_cluster_sizes=cmp_sizes,
                repetitions=REPETITIONS,
                seed=seed,
                correction=correction,
            )

            rows.append(
                {
                    "scenario_id": scenario["id"],
                    "correction": correction,
                    "baseline_fnr": scenario["baseline"],
                    "comparison_fnr": scenario["baseline"] + 0.03,
                    "icc_reference": scenario["icc_ref"],
                    "icc_comparison": scenario["icc_cmp"],
                    "reference_clusters": ref_stats["clusters"],
                    "comparison_clusters": cmp_stats["clusters"],
                    "reference_rows": ref_stats["rows"],
                    "comparison_rows": cmp_stats["rows"],
                    "reference_mean_cluster_size": ref_stats["mean_cluster_size"],
                    "comparison_mean_cluster_size": cmp_stats["mean_cluster_size"],
                    "reference_cluster_size_cv": ref_stats["cluster_size_cv"],
                    "comparison_cluster_size_cv": cmp_stats["cluster_size_cv"],
                    "reference_min_cluster_size": ref_stats["min_cluster_size"],
                    "reference_max_cluster_size": ref_stats["max_cluster_size"],
                    "comparison_min_cluster_size": cmp_stats["min_cluster_size"],
                    "comparison_max_cluster_size": cmp_stats["max_cluster_size"],
                    "repetitions": result.repetitions,
                    "estimable_repetitions": result.estimable_count,
                    "type_I_pass_rate": result.pass_rate,
                    "mc_se": result.mc_se,
                    "mc_cp_lower_95": result.cp_lower_95,
                    "mc_cp_upper_95": result.cp_upper_95,
                }
            )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "screen_id": "external_validation_ni_stress_calibration_v1",
        "status": "precollection_method_stress_test_not_final",
        "primary_margin": 0.03,
        "one_sided_alpha": 0.025,
        "repetitions_per_method_scenario": REPETITIONS,
        "methods": ["KC", "MD"],
        "stress_dimensions": [
            "unequal cluster sizes",
            "high cluster-size CV",
            "unequal reference/comparison cluster counts",
            "unequal ICC across cells",
            "baseline FNR from 0.05 to 0.20",
        ],
        "important_boundary": (
            "This screen does not freeze the final inference method, "
            "absolute FNR ceiling, sample size, author/generator-batch "
            "minima/caps, or W0."
        ),
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")
    print()
    print("=== 20,000-REPETITION STRESS CALIBRATION ===")
    print("Nominal one-sided alpha = 0.025")
    print()

    for scenario in SCENARIOS:
        print(scenario["id"])
        subset = [row for row in rows if row["scenario_id"] == scenario["id"]]
        for row in subset:
            print(
                f"  {row['correction']}: "
                f"typeI={row['type_I_pass_rate']:.5f} "
                f"(MC95% {row['mc_cp_lower_95']:.5f}-"
                f"{row['mc_cp_upper_95']:.5f}); "
                f"Gref/Gcmp={row['reference_clusters']}/"
                f"{row['comparison_clusters']}, "
                f"CVref/CVcmp={row['reference_cluster_size_cv']:.2f}/"
                f"{row['comparison_cluster_size_cv']:.2f}"
            )

    print()
    for correction in ("KC", "MD"):
        subset = [row for row in rows if row["correction"] == correction]
        worst = max(subset, key=lambda row: row["type_I_pass_rate"])
        print(
            f"{correction} MAX TYPE-I: {worst['type_I_pass_rate']:.5f} "
            f"(MC95% {worst['mc_cp_lower_95']:.5f}-"
            f"{worst['mc_cp_upper_95']:.5f}) "
            f"at {worst['scenario_id']}"
        )

    print()
    print(
        "Do not freeze W0 from this screen. If MD remains near nominal while "
        "KC is more anti-conservative, the next step is to freeze the "
        "cluster-inference method and run the final design-power grid."
    )


if __name__ == "__main__":
    run()
