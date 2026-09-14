from __future__ import annotations

import csv
import json
from pathlib import Path

from monitor_fusion.external_validation.cluster_ni_cr2_satt import (
    simulate_cr2_satterthwaite_boundary,
)
from monitor_fusion.external_validation.cluster_ni_power import stable_seed


BASE_SEED = 20260907
REPETITIONS = 20000
OUT_DIR = Path("results")
CSV_PATH = OUT_DIR / "external_validation_ni_asymmetry_diagnosis_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_ni_asymmetry_diagnosis_v1.json"

# Factorize the failure observed in the 100-vs-50 stress case.
# Only one structural feature is changed at a time where possible.
SCENARIOS = (
    {
        "id": "A_balanced_control",
        "g_ref": 100, "g_cmp": 100,
        "m_ref": 20, "m_cmp": 20,
        "rho_ref": 0.05, "rho_cmp": 0.05,
    },
    {
        "id": "B_unequal_ICC_only",
        "g_ref": 100, "g_cmp": 100,
        "m_ref": 20, "m_cmp": 20,
        "rho_ref": 0.05, "rho_cmp": 0.10,
    },
    {
        "id": "C_unequal_cluster_count_only",
        "g_ref": 100, "g_cmp": 50,
        "m_ref": 20, "m_cmp": 20,
        "rho_ref": 0.05, "rho_cmp": 0.05,
    },
    {
        "id": "D_unequal_cluster_size_only",
        "g_ref": 100, "g_cmp": 100,
        "m_ref": 20, "m_cmp": 40,
        "rho_ref": 0.05, "rho_cmp": 0.05,
    },
    {
        "id": "E_unequal_G_equal_total_rows_same_ICC",
        "g_ref": 100, "g_cmp": 50,
        "m_ref": 20, "m_cmp": 40,
        "rho_ref": 0.05, "rho_cmp": 0.05,
    },
    {
        "id": "F_equal_G_unequal_m_and_ICC",
        "g_ref": 100, "g_cmp": 100,
        "m_ref": 20, "m_cmp": 40,
        "rho_ref": 0.05, "rho_cmp": 0.10,
    },
    {
        "id": "G_original_100v50_failure",
        "g_ref": 100, "g_cmp": 50,
        "m_ref": 20, "m_cmp": 40,
        "rho_ref": 0.05, "rho_cmp": 0.10,
    },
    {
        "id": "H_balanced_G150_unequal_ICC",
        "g_ref": 150, "g_cmp": 150,
        "m_ref": 20, "m_cmp": 20,
        "rho_ref": 0.05, "rho_cmp": 0.10,
    },
)

BASELINE_FNR = 0.05


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in SCENARIOS:
        ref_sizes = [scenario["m_ref"]] * scenario["g_ref"]
        cmp_sizes = [scenario["m_cmp"]] * scenario["g_cmp"]

        seed = stable_seed(
            BASE_SEED,
            "asymmetry_diagnosis",
            scenario["id"],
        )

        result = simulate_cr2_satterthwaite_boundary(
            reference_fnr=BASELINE_FNR,
            icc_reference=scenario["rho_ref"],
            icc_comparison=scenario["rho_cmp"],
            reference_cluster_sizes=ref_sizes,
            comparison_cluster_sizes=cmp_sizes,
            repetitions=REPETITIONS,
            seed=seed,
        )

        rows.append(
            {
                "scenario_id": scenario["id"],
                "baseline_fnr": BASELINE_FNR,
                "comparison_fnr": BASELINE_FNR + 0.03,
                "reference_clusters": scenario["g_ref"],
                "comparison_clusters": scenario["g_cmp"],
                "reference_rows_per_cluster": scenario["m_ref"],
                "comparison_rows_per_cluster": scenario["m_cmp"],
                "reference_rows": scenario["g_ref"] * scenario["m_ref"],
                "comparison_rows": scenario["g_cmp"] * scenario["m_cmp"],
                "icc_reference": scenario["rho_ref"],
                "icc_comparison": scenario["rho_cmp"],
                "satterthwaite_df": result.degrees_of_freedom,
                "repetitions": result.repetitions,
                "type_I_rate": result.type_I_rate,
                "mc_cp_lower_95": result.cp_lower_95,
                "mc_cp_upper_95": result.cp_upper_95,
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "screen_id": "external_validation_ni_asymmetry_diagnosis_v1",
        "status": "precollection_diagnostic_not_final",
        "baseline_fnr": BASELINE_FNR,
        "boundary_difference": 0.03,
        "nominal_one_sided_alpha": 0.025,
        "repetitions_per_scenario": REPETITIONS,
        "method": "CR2 plus coefficient-specific Satterthwaite",
        "purpose": (
            "Factorize the observed asymmetric type-I inflation into cluster "
            "count imbalance, per-cluster size imbalance, and ICC mismatch."
        ),
        "important_boundary": (
            "This diagnostic does not freeze an inference method, sample size, "
            "cluster minima/caps, absolute FNR ceiling, or W0."
        ),
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {JSON_PATH}")
    print()
    print("=== ASYMMETRY FAILURE DIAGNOSIS ===")
    print("Nominal one-sided alpha = 0.025")
    print()

    for row in rows:
        print(
            f"{row['scenario_id']}: "
            f"typeI={row['type_I_rate']:.5f} "
            f"(MC95% {row['mc_cp_lower_95']:.5f}-"
            f"{row['mc_cp_upper_95']:.5f}); "
            f"G={row['reference_clusters']}/{row['comparison_clusters']}, "
            f"m={row['reference_rows_per_cluster']}/"
            f"{row['comparison_rows_per_cluster']}, "
            f"ICC={row['icc_reference']:.2f}/{row['icc_comparison']:.2f}, "
            f"df={row['satterthwaite_df']:.2f}"
        )

    print()
    print(
        "INTERPRETATION GATE: use this screen to decide whether the study can "
        "avoid the problematic regime by balancing provenance-cluster design, "
        "or whether a different model-based inferential family is required."
    )


if __name__ == "__main__":
    run()
