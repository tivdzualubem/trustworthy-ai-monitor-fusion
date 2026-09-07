from __future__ import annotations

import csv
import json
from pathlib import Path

from monitor_fusion.external_validation.cluster_ni_calibration import (
    simulate_boundary_calibration,
)
from monitor_fusion.external_validation.cluster_ni_power import stable_seed


BASE_SEED = 20260907
REPETITIONS = 20000
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_ni_method_calibration_screen_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_ni_method_calibration_screen_v1.json"

# Target the worst type-I configuration found for each design in the first
# 2,000-repetition screen rather than rerunning the whole design grid.
SCENARIOS = (
    ("G050_M20", 50, 20, 0.05, 0.01),
    ("G075_M20", 75, 20, 0.05, 0.00),
    ("G100_M20", 100, 20, 0.05, 0.00),
    ("G125_M20", 125, 20, 0.05, 0.00),
    ("G150_M20", 150, 20, 0.20, 0.00),
    ("G200_M15", 200, 15, 0.05, 0.05),
    ("G250_M12", 250, 12, 0.05, 0.05),
    ("G300_M10", 300, 10, 0.20, 0.10),
)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for design_id, clusters, rows_per_cluster, baseline, icc in SCENARIOS:
        sizes = [rows_per_cluster] * clusters

        for correction in ("KC", "MD"):
            seed = stable_seed(
                BASE_SEED,
                "targeted_boundary_calibration",
                design_id,
                baseline,
                icc,
                correction,
            )
            result = simulate_boundary_calibration(
                reference_fnr=baseline,
                icc_reference=icc,
                icc_comparison=icc,
                reference_cluster_sizes=sizes,
                comparison_cluster_sizes=sizes,
                repetitions=REPETITIONS,
                seed=seed,
                correction=correction,
            )

            rows.append(
                {
                    "design_id": design_id,
                    "correction": correction,
                    "baseline_fnr": baseline,
                    "comparison_fnr": baseline + 0.03,
                    "icc": icc,
                    "clusters_per_cell": clusters,
                    "rows_per_cluster": rows_per_cluster,
                    "rows_per_cell": clusters * rows_per_cluster,
                    "repetitions": result.repetitions,
                    "estimable_repetitions": result.estimable_count,
                    "pass_count": result.pass_count,
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
        "screen_id": "external_validation_ni_method_calibration_screen_v1",
        "status": "targeted_precollection_method_calibration_not_final",
        "primary_margin": 0.03,
        "one_sided_alpha": 0.025,
        "repetitions_per_method_scenario": REPETITIONS,
        "methods": {
            "KC": "Kauermann-Carroll leverage-adjusted cluster sandwich",
            "MD": "Mancl-DeRouen leverage-adjusted cluster sandwich",
        },
        "degrees_of_freedom": "total provenance clusters across the pair minus 2",
        "scope": (
            "Targets the worst boundary scenarios from the first screen. "
            "This does not freeze the final inference method, sample sizes, "
            "cluster minima/caps, FNR ceiling, or W0."
        ),
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")
    print()
    print("=== TARGETED 20,000-REPETITION TYPE-I CALIBRATION ===")
    print("Nominal one-sided alpha = 0.025")
    print()

    for design_id, *_ in SCENARIOS:
        subset = [row for row in rows if row["design_id"] == design_id]
        print(design_id)
        for row in subset:
            print(
                f"  {row['correction']}: "
                f"typeI={row['type_I_pass_rate']:.5f} "
                f"(MC95% {row['mc_cp_lower_95']:.5f}-"
                f"{row['mc_cp_upper_95']:.5f})"
            )

    print()
    print("DECISION RULE FOR THIS SCREEN:")
    print(
        "Do not freeze a method merely because its point estimate is <=0.025. "
        "Use these targeted results to decide whether a broader calibration "
        "grid / different inference family is required."
    )


if __name__ == "__main__":
    run()
