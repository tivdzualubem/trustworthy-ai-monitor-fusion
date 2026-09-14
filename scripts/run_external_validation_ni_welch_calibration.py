from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from monitor_fusion.external_validation.cluster_ni_power import stable_seed
from monitor_fusion.external_validation.cluster_ni_welch import (
    simulate_welch_cluster_boundary,
)


BASE_SEED = 20260907
REPETITIONS = 20000
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_ni_welch_calibration_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_ni_welch_calibration_v1.json"


def sizes_from_pattern(cluster_count: int, pattern: tuple[int, ...]) -> list[int]:
    return [pattern[i % len(pattern)] for i in range(cluster_count)]


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


def profile(sizes):
    arr = np.asarray(sizes, dtype=float)
    return {
        "clusters": len(arr),
        "rows": int(arr.sum()),
        "cv": float(arr.std(ddof=0) / arr.mean()),
    }


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in SCENARIOS:
        ref_sizes = sizes_from_pattern(
            scenario["ref_clusters"], scenario["ref_pattern"]
        )
        cmp_sizes = sizes_from_pattern(
            scenario["cmp_clusters"], scenario["cmp_pattern"]
        )
        ref = profile(ref_sizes)
        cmp_ = profile(cmp_sizes)

        for correction in ("CRV1", "CRV3"):
            seed = stable_seed(
                BASE_SEED,
                "welch_cluster_boundary",
                scenario["id"],
                correction,
            )

            result = simulate_welch_cluster_boundary(
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
                    "reference_clusters": ref["clusters"],
                    "comparison_clusters": cmp_["clusters"],
                    "reference_rows": ref["rows"],
                    "comparison_rows": cmp_["rows"],
                    "reference_cluster_size_cv": ref["cv"],
                    "comparison_cluster_size_cv": cmp_["cv"],
                    "repetitions": result.repetitions,
                    "estimable_repetitions": result.estimable_count,
                    "type_I_reject_count": result.reject_count,
                    "type_I_rate": result.type_I_rate,
                    "mc_se": result.mc_se,
                    "mc_cp_lower_95": result.cp_lower_95,
                    "mc_cp_upper_95": result.cp_upper_95,
                    "df_min": result.df_min,
                    "df_median": result.df_median,
                    "df_max": result.df_max,
                }
            )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "screen_id": "external_validation_ni_welch_calibration_v1",
        "status": "precollection_candidate_method_calibration_not_final",
        "method": (
            "Two-sample Welch-Satterthwaite inference for the difference of "
            "independent row-marginal FNRs, using cell-specific cluster-robust "
            "variance components"
        ),
        "corrections": ["CRV1", "CRV3"],
        "primary_margin": 0.03,
        "one_sided_alpha": 0.025,
        "repetitions_per_scenario": REPETITIONS,
        "important_boundary": (
            "This screen does not freeze the Welch method, sample sizes, "
            "cluster minima/caps, absolute FNR ceiling, or W0."
        ),
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")
    print()
    print("=== CLUSTER-WELCH 20,000-REPETITION TYPE-I CALIBRATION ===")
    print("Nominal one-sided alpha = 0.025")
    print()

    for scenario in SCENARIOS:
        print(scenario["id"])
        subset = [row for row in rows if row["scenario_id"] == scenario["id"]]
        for row in subset:
            print(
                f"  {row['correction']}: "
                f"typeI={row['type_I_rate']:.5f} "
                f"(MC95% {row['mc_cp_lower_95']:.5f}-"
                f"{row['mc_cp_upper_95']:.5f}); "
                f"df median={row['df_median']:.2f}"
            )

    print()
    for correction in ("CRV1", "CRV3"):
        subset = [row for row in rows if row["correction"] == correction]
        worst = max(subset, key=lambda row: row["type_I_rate"])
        print(
            f"WELCH-{correction} MAX TYPE-I: "
            f"{worst['type_I_rate']:.5f} "
            f"(MC95% {worst['mc_cp_lower_95']:.5f}-"
            f"{worst['mc_cp_upper_95']:.5f}) "
            f"at {worst['scenario_id']}"
        )

    print()
    print(
        "Do not freeze W0 from this screen. If a Welch variant fixes the "
        "asymmetric inflation, validate it in a broader design grid before "
        "freezing final inference and power."
    )


if __name__ == "__main__":
    run()
