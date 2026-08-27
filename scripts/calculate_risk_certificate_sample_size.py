#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/risk_certificate_transport_preregistration_v1.json"
OUT = ROOT / "reports/risk_certificate_transport_preregistration_v1"


def cp_upper(k: int, n: int, alpha: float) -> float:
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial count")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def kmax_for_certificate(n: int, risk_limit: float, alpha: float) -> int:
    # U_CP(k,n) <= risk_limit iff BinomCDF(k; n, risk_limit) <= alpha.
    valid = [
        k
        for k in range(n + 1)
        if float(binom.cdf(k, n, risk_limit)) <= alpha
    ]
    return max(valid) if valid else -1


def certificate_power(
    n: int,
    true_fpr: float,
    risk_limit: float,
    alpha: float,
) -> float:
    kmax = kmax_for_certificate(n, risk_limit, alpha)
    if kmax < 0:
        return 0.0
    return float(binom.cdf(kmax, n, true_fpr))


def minimum_n_for_power(
    true_fpr: float,
    target_power: float,
    risk_limit: float,
    alpha: float,
    max_n: int = 10000,
) -> int:
    for n in range(1, max_n + 1):
        if certificate_power(n, true_fpr, risk_limit, alpha) >= target_power:
            return n
    raise RuntimeError("sample-size search exceeded max_n")


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    design = cfg["sample_size_power"]

    risk_limit = float(design["risk_limit"])
    alpha = float(design["one_sided_alpha"])
    true_fpr = float(design["design_true_fpr"])
    target_power = float(design["target_power"])

    n = minimum_n_for_power(
        true_fpr=true_fpr,
        target_power=target_power,
        risk_limit=risk_limit,
        alpha=alpha,
    )
    kmax = kmax_for_certificate(n, risk_limit, alpha)
    upper = cp_upper(kmax, n, alpha)
    achieved = certificate_power(n, true_fpr, risk_limit, alpha)

    if n != int(design["minimum_independent_negative_units_per_certificate_cell"]):
        raise RuntimeError("protocol sample size does not reproduce")
    if kmax != int(design["maximum_false_positives_at_n_for_certificate"]):
        raise RuntimeError("protocol false-positive acceptance count does not reproduce")
    if abs(upper - float(design["exact_upper_bound_at_max_false_positives"])) > 1e-14:
        raise RuntimeError("protocol exact upper bound does not reproduce")
    if abs(achieved - float(design["achieved_power_at_design_fpr"])) > 1e-14:
        raise RuntimeError("protocol design power does not reproduce")

    rows = []
    for p in [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]:
        rows.append(
            {
                "true_fpr": p,
                "n_negative_independent_units": n,
                "max_false_positives_for_95pct_certificate": kmax,
                "certificate_power": certificate_power(
                    n, p, risk_limit, alpha
                ),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "sample_size_power.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "artifact": "risk_certificate_transport_preregistration_sample_size",
        "status": "analytic_design_only_no_fresh_data",
        "risk_limit": risk_limit,
        "one_sided_alpha": alpha,
        "design_true_fpr": true_fpr,
        "target_power": target_power,
        "minimum_independent_negative_units_per_certificate_cell": n,
        "maximum_false_positives_at_n_for_certificate": kmax,
        "exact_upper_bound_at_max_false_positives": upper,
        "achieved_power_at_design_fpr": achieved,
        "fresh_data_read": False,
        "historical_outcome_data_read": False,
        "monitor_scoring_performed": False,
    }
    (OUT / "sample_size_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("RISK_CERTIFICATE_SAMPLE_SIZE=PASS")
    print(f"primary_n={n}")
    print(f"primary_kmax={kmax}")
    print(f"primary_upper={upper:.15f}")
    print(f"primary_power={achieved:.15f}")
    print("fresh_data_read=false")
    print("historical_outcome_data_read=false")
    print("monitor_scoring_performed=false")


if __name__ == "__main__":
    main()
