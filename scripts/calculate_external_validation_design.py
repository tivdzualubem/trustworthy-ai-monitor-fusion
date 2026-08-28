#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logit
from scipy.stats import beta, binom, norm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/safety_monitor_external_validation_preregistration_v1.json"
OUT = ROOT / "reports/safety_monitor_external_validation_preregistration_v1"


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial count")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def kmax_for_upper_bound(n: int, risk_limit: float, alpha: float) -> int:
    valid = [
        k
        for k in range(n + 1)
        if float(binom.cdf(k, n, risk_limit)) <= alpha
    ]
    return max(valid) if valid else -1


def fpr_pass_probability(
    n: int,
    true_fpr: float,
    risk_limit: float,
    alpha: float,
) -> float:
    kmax = kmax_for_upper_bound(n, risk_limit, alpha)
    if kmax < 0:
        return 0.0
    return float(binom.cdf(kmax, n, true_fpr))


def factorial_wald_power(
    p00: float,
    p10: float,
    p01: float,
    p11: float,
    n_per_cell: int,
    term: int,
    alpha: float,
    design_effect: float = 1.0,
) -> float:
    # Effect coding: source, attack in {-0.5, +0.5}; interaction is product.
    X = np.array(
        [
            [1.0, -0.5, -0.5, 0.25],
            [1.0,  0.5, -0.5, -0.25],
            [1.0, -0.5,  0.5, -0.25],
            [1.0,  0.5,  0.5, 0.25],
        ],
        dtype=float,
    )
    p = np.array([p00, p10, p01, p11], dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("all planning probabilities must be strictly between 0 and 1")
    n_eff = float(n_per_cell) / float(design_effect)
    W = np.diag(n_eff * p * (1.0 - p))
    covariance = np.linalg.inv(X.T @ W @ X)
    beta_vec = np.linalg.solve(X, logit(p))
    noncentral_z = abs(float(beta_vec[term])) / math.sqrt(float(covariance[term, term]))
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    return float(
        norm.cdf(noncentral_z - critical)
        + norm.cdf(-noncentral_z - critical)
    )


def main_effect_mde(
    baseline_fnr: float,
    n_per_cell: int,
    alpha: float,
    target_power: float,
    design_effect: float,
) -> float:
    def objective(shifted: float) -> float:
        power = factorial_wald_power(
            baseline_fnr,
            shifted,
            baseline_fnr,
            shifted,
            n_per_cell,
            term=1,
            alpha=alpha,
            design_effect=design_effect,
        )
        return power - target_power

    return float(
        brentq(
            objective,
            baseline_fnr + 1e-8,
            min(0.95, baseline_fnr + 0.60),
        )
    )


def interaction_mde(
    baseline_fnr: float,
    n_per_cell: int,
    alpha: float,
    target_power: float,
    design_effect: float,
) -> float:
    def objective(joint_cell_fnr: float) -> float:
        power = factorial_wald_power(
            baseline_fnr,
            baseline_fnr,
            baseline_fnr,
            joint_cell_fnr,
            n_per_cell,
            term=3,
            alpha=alpha,
            design_effect=design_effect,
        )
        return power - target_power

    return float(
        brentq(
            objective,
            baseline_fnr + 1e-8,
            min(0.95, baseline_fnr + 0.80),
        )
    )


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

    fpr = cfg["fpr_design"]
    n0 = int(fpr["n_Y0_per_validation_cell"])
    risk_limit = float(cfg["outcomes"]["overblocking_outcome"]["individual_cell_operating_constraint"])
    alpha = float(cfg["outcomes"]["overblocking_outcome"]["one_sided_alpha"])
    true_fpr = 0.025

    kmax = kmax_for_upper_bound(n0, risk_limit, alpha)
    upper = cp_upper(kmax, n0, alpha)
    pass_prob = fpr_pass_probability(n0, true_fpr, risk_limit, alpha)

    if kmax != int(fpr["maximum_false_positives_for_individual_5pct_constraint"]):
        raise RuntimeError("FPR acceptance count does not reproduce")
    if abs(upper - float(fpr["cp_upper_at_11"])) > 1e-14:
        raise RuntimeError("FPR CP upper bound does not reproduce")
    if abs(pass_prob - float(fpr["probability_pass_if_true_fpr_0_025"])) > 1e-14:
        raise RuntimeError("FPR pass probability does not reproduce")

    fnr = cfg["fnr_design"]
    n1 = int(fnr["n_Y1_per_validation_cell"])
    planning_alpha = float(fnr["planning_alpha"])
    target_power = float(fnr["planning_power"])
    de_sens = float(fnr["clustering_design_effect_sensitivity"])

    rows = []
    for baseline in [0.05, 0.10, 0.20]:
        for de in [1.0, de_sens]:
            rows.append(
                {
                    "baseline_fnr": baseline,
                    "design_effect": de,
                    "n_positive_per_W1_cell": n1,
                    "planning_alpha_two_sided": planning_alpha,
                    "target_power": target_power,
                    "detectable_main_effect_cell_fnr": main_effect_mde(
                        baseline, n1, planning_alpha, target_power, de
                    ),
                    "detectable_joint_interaction_cell_fnr": interaction_mde(
                        baseline, n1, planning_alpha, target_power, de
                    ),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "design_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "artifact": "safety_monitor_external_validation_preregistration_design",
        "status": "analytic_precollection_design_only",
        "W0_collection_started": False,
        "fresh_monitor_scoring_started": False,
        "FPR": {
            "n_negative_per_validation_cell": n0,
            "risk_limit": risk_limit,
            "one_sided_alpha": alpha,
            "true_fpr_planning_scenario": true_fpr,
            "maximum_false_positives_for_constraint": kmax,
            "cp_upper_at_kmax": upper,
            "pass_probability_at_true_fpr": pass_prob,
            "interpretation": "individual-cell operating-constraint pass probability, not source/attack power",
        },
        "FNR": {
            "n_positive_per_validation_cell": n1,
            "primary_test_count": int(fnr["primary_test_count"]),
            "familywise_alpha": float(fnr["familywise_alpha"]),
            "multiplicity": fnr["multiplicity"],
            "planning_alpha": planning_alpha,
            "target_power": target_power,
            "design_effect_sensitivity": de_sens,
        },
    }
    (OUT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("EXTERNAL_VALIDATION_DESIGN=PASS")
    print(f"fpr_n={n0}")
    print(f"fpr_kmax={kmax}")
    print(f"fpr_cp_upper={upper:.15f}")
    print(f"fpr_pass_probability_at_0.025={pass_prob:.15f}")
    print(f"fnr_n_per_validation_cell={n1}")
    print(f"primary_fnr_tests={int(fnr['primary_test_count'])}")
    print(f"planning_alpha={planning_alpha:.15f}")
    print("W0_collection_started=false")
    print("fresh_monitor_scoring_started=false")


if __name__ == "__main__":
    main()
