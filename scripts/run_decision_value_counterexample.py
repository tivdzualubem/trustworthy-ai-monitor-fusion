#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_counterexample(
    context_probability: float = 0.5,
    informative_accuracy: float = 0.9,
    acquisition_cost: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if abs(context_probability - 0.5) > 1e-12:
        raise ValueError("This matched-rate example uses equal contexts")
    if not 0.5 < informative_accuracy <= 1.0:
        raise ValueError("informative_accuracy must lie in (0.5, 1]")
    if acquisition_cost < 0.0:
        raise ValueError("acquisition_cost must be nonnegative")

    cheap_risk = 0.5
    informative_post_risk = 1.0 - informative_accuracy
    redundant_post_risk = 0.5

    rows = pd.DataFrame(
        [
            {
                "context": "A_informative",
                "cheap_p_y1": 0.5,
                "cheap_uncertainty_bayes_risk": cheap_risk,
                "expensive_monitor_accuracy": informative_accuracy,
                "post_acquisition_bayes_risk": informative_post_risk,
                "decision_value": cheap_risk - informative_post_risk,
                "acquisition_cost": acquisition_cost,
            },
            {
                "context": "B_redundant",
                "cheap_p_y1": 0.5,
                "cheap_uncertainty_bayes_risk": cheap_risk,
                "expensive_monitor_accuracy": 0.5,
                "post_acquisition_bayes_risk": redundant_post_risk,
                "decision_value": cheap_risk - redundant_post_risk,
                "acquisition_cost": acquisition_cost,
            },
        ]
    )
    rows["optimal_acquire"] = (
        rows["decision_value"] > rows["acquisition_cost"]
    )

    oracle_decision_loss = 0.5 * informative_post_risk + 0.5 * cheap_risk
    uncertainty_decision_loss = (
        0.5 * (0.5 * informative_post_risk + 0.5 * cheap_risk)
        + 0.5 * cheap_risk
    )
    acquisition_rate = 0.5
    average_acquisition_cost = acquisition_rate * acquisition_cost

    summary = {
        "matched_acquisition_rate": acquisition_rate,
        "oracle_decision_loss": oracle_decision_loss,
        "uncertainty_decision_loss": uncertainty_decision_loss,
        "decision_loss_gap_uncertainty_minus_oracle": (
            uncertainty_decision_loss - oracle_decision_loss
        ),
        "average_acquisition_cost_both_policies": average_acquisition_cost,
        "oracle_total_risk_with_cost": (
            oracle_decision_loss + average_acquisition_cost
        ),
        "uncertainty_total_risk_with_cost": (
            uncertainty_decision_loss + average_acquisition_cost
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/decision_value_theory"),
    )
    args = parser.parse_args()

    rows, summary = build_counterexample()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "counterexample_results.csv"
    json_path = args.output_dir / "counterexample_summary.json"
    rows.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert rows["cheap_p_y1"].nunique() == 1
    assert rows["cheap_uncertainty_bayes_risk"].nunique() == 1
    assert abs(float(rows.loc[0, "decision_value"]) - 0.4) < 1e-12
    assert float(rows.loc[1, "decision_value"]) == 0.0
    assert bool(rows.loc[0, "optimal_acquire"])
    assert not bool(rows.loc[1, "optimal_acquire"])
    assert abs(summary["oracle_decision_loss"] - 0.3) < 1e-12
    assert abs(summary["uncertainty_decision_loss"] - 0.4) < 1e-12
    assert abs(
        summary["decision_loss_gap_uncertainty_minus_oracle"] - 0.1
    ) < 1e-12

    print("decision-value counterexample verified")
    print(rows.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    print()
    print("csv:", csv_path)
    print("json:", json_path)


if __name__ == "__main__":
    main()
