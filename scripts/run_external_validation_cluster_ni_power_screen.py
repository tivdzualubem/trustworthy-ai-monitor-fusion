from __future__ import annotations

import csv
import json
from pathlib import Path

from monitor_fusion.external_validation.cluster_ni_power import (
    simulate_pair_pass_rate_vectorized,
    stable_seed,
)


BASE_SEED = 20260907
REPETITIONS = 2000

BASELINE_FNRS = (0.05, 0.10, 0.20)
ICCS = (0.00, 0.01, 0.03, 0.05, 0.10)

# Symmetric human-human designs first. These cover the temporal A_val-vs-T
# comparison and the human F-vs-T comparison, which are the natural
# bottleneck before separately checking human-vs-model batch asymmetry.
DESIGNS = (
    ("G050_M20", 50, 20),
    ("G075_M20", 75, 20),
    ("G100_M20", 100, 20),
    ("G125_M20", 125, 20),
    ("G150_M20", 150, 20),
    ("G200_M15", 200, 15),
    ("G250_M12", 250, 12),
    ("G300_M10", 300, 10),
)

OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_cluster_ni_power_screen_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_cluster_ni_power_screen_v1.json"


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for baseline in BASELINE_FNRS:
        for icc in ICCS:
            for design_id, clusters_per_cell, rows_per_cluster in DESIGNS:
                sizes = [rows_per_cluster] * clusters_per_cell

                for scenario, comparison_fnr in (
                    ("preservation_zero_difference", baseline),
                    ("type_I_boundary_plus_0.03", baseline + 0.03),
                ):
                    seed = stable_seed(
                        BASE_SEED,
                        scenario,
                        baseline,
                        icc,
                        design_id,
                    )

                    result = simulate_pair_pass_rate_vectorized(
                        reference_fnr=baseline,
                        comparison_fnr=comparison_fnr,
                        icc_reference=icc,
                        icc_comparison=icc,
                        reference_cluster_sizes=sizes,
                        comparison_cluster_sizes=sizes,
                        repetitions=REPETITIONS,
                        seed=seed,
                    )

                    rows.append(
                        {
                            "scenario": scenario,
                            "baseline_fnr": baseline,
                            "comparison_fnr": comparison_fnr,
                            "true_difference": comparison_fnr - baseline,
                            "icc": icc,
                            "design_id": design_id,
                            "clusters_per_cell": clusters_per_cell,
                            "rows_per_cluster": rows_per_cluster,
                            "rows_per_cell": clusters_per_cell * rows_per_cluster,
                            "repetitions": result.repetitions,
                            "estimable_repetitions": result.estimable_count,
                            "pass_count": result.pass_count,
                            "pass_rate": result.pass_rate_all,
                            "mc_se": result.mc_se_all,
                            "mc_cp_lower_95": result.cp_lower_95,
                            "mc_cp_upper_95": result.cp_upper_95,
                        }
                    )

    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "screen_id": "external_validation_cluster_ni_power_screen_v1",
        "status": "exploratory_design_screen_not_final",
        "analysis_method": "candidate KC/CR2 cluster sandwich + t(df=G_total-2)",
        "primary_margin": 0.03,
        "one_sided_alpha": 0.025,
        "repetitions_per_scenario": REPETITIONS,
        "base_seed": BASE_SEED,
        "baseline_fnrs": list(BASELINE_FNRS),
        "iccs": list(ICCS),
        "designs": [
            {
                "design_id": d,
                "clusters_per_cell": g,
                "rows_per_cluster": m,
                "rows_per_cell": g * m,
            }
            for d, g, m in DESIGNS
        ],
        "scope": (
            "Balanced symmetric provenance-cluster screen only. "
            "No author/generator minima, caps, sample size, absolute FNR ceiling, "
            "or W0 state is frozen by this screen."
        ),
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")
    print()

    # Print compact worst-case summaries by design.
    print("=== WORST-CASE PRESERVATION POWER ACROSS BASELINE FNR / ICC ===")
    power_rows = [
        row for row in rows
        if row["scenario"] == "preservation_zero_difference"
    ]
    for design_id, _, _ in DESIGNS:
        subset = [row for row in power_rows if row["design_id"] == design_id]
        worst = min(subset, key=lambda row: row["pass_rate"])
        print(
            f"{design_id}: worst_power={worst['pass_rate']:.4f} "
            f"(p={worst['baseline_fnr']:.2f}, ICC={worst['icc']:.2f}, "
            f"n/cell={worst['rows_per_cell']})"
        )

    print()
    print("=== WORST TYPE-I PASS RATE AT +0.03 BOUNDARY ===")
    null_rows = [
        row for row in rows
        if row["scenario"] == "type_I_boundary_plus_0.03"
    ]
    for design_id, _, _ in DESIGNS:
        subset = [row for row in null_rows if row["design_id"] == design_id]
        worst = max(subset, key=lambda row: row["pass_rate"])
        print(
            f"{design_id}: max_typeI={worst['pass_rate']:.4f} "
            f"(MC95% {worst['mc_cp_lower_95']:.4f}-"
            f"{worst['mc_cp_upper_95']:.4f}; "
            f"p={worst['baseline_fnr']:.2f}, ICC={worst['icc']:.2f})"
        )

    print()
    print("NOTE: This is a screening grid only. Do not freeze W0 from these results.")


if __name__ == "__main__":
    run()
