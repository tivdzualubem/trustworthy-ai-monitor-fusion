from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_decision_value_counterexample.py"

spec = importlib.util.spec_from_file_location(
    "decision_value_counterexample",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {SCRIPT}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

build_counterexample = module.build_counterexample


def test_acquire_if_and_only_if_value_exceeds_cost() -> None:
    rows, _ = build_counterexample(acquisition_cost=0.1)

    informative = rows.loc[
        rows["context"].eq("A_informative")
    ].iloc[0]
    redundant = rows.loc[
        rows["context"].eq("B_redundant")
    ].iloc[0]

    assert informative["decision_value"] == 0.4
    assert bool(informative["optimal_acquire"])

    assert redundant["decision_value"] == 0.0
    assert not bool(redundant["optimal_acquire"])


def test_identical_uncertainty_does_not_imply_identical_value() -> None:
    rows, summary = build_counterexample()

    assert rows["cheap_p_y1"].nunique() == 1
    assert rows["cheap_uncertainty_bayes_risk"].nunique() == 1
    assert rows["decision_value"].nunique() == 2

    assert summary["matched_acquisition_rate"] == 0.5
    assert abs(summary["oracle_decision_loss"] - 0.3) < 1e-12
    assert abs(summary["uncertainty_decision_loss"] - 0.4) < 1e-12
    assert abs(
        summary["decision_loss_gap_uncertainty_minus_oracle"] - 0.1
    ) < 1e-12
