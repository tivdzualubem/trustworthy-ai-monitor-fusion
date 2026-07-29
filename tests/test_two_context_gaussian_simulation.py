from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/run_two_context_gaussian_simulation.py"
)

spec = importlib.util.spec_from_file_location(
    "two_context_gaussian_simulation",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {SCRIPT}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

SimulationConfig = module.SimulationConfig
generate_dataset = module.generate_dataset
run_seed = module.run_seed


def test_context_a_is_informative_and_context_b_is_redundant() -> None:
    config = SimulationConfig(
        train_n=1000,
        test_n=30000,
        seeds=1,
    )
    frame = generate_dataset(
        seed=31,
        n=config.test_n,
        config=config,
    )

    context_a = frame["context_a"].eq(1)
    context_b = ~context_a

    assert (
        frame.loc[context_a, "full_loss"].mean()
        < frame.loc[context_a, "cheap_loss"].mean()
    )
    assert np.array_equal(
        frame.loc[
            context_b,
            "full_prediction_signed",
        ].to_numpy(),
        frame.loc[
            context_b,
            "cheap_prediction_signed",
        ].to_numpy(),
    )
    assert (
        frame.loc[
            context_b,
            "oracle_decision_value",
        ]
        == 0.0
    ).all()


def test_cheap_uncertainty_is_matched_across_contexts() -> None:
    config = SimulationConfig(
        train_n=1000,
        test_n=50000,
        seeds=1,
    )
    frame = generate_dataset(
        seed=47,
        n=config.test_n,
        config=config,
    )

    means = frame.groupby(
        "context",
        observed=True,
    )["cheap_uncertainty"].mean()

    assert abs(
        means["A_informative"]
        - means["B_redundant"]
    ) < 0.01


def test_all_policies_have_identical_acquisition_counts() -> None:
    config = SimulationConfig(
        train_n=3000,
        test_n=6000,
        seeds=1,
        budgets=(0.10, 0.25, 0.50),
    )
    metrics, _ = run_seed(
        seed=5,
        config=config,
    )

    counts = metrics.groupby(
        "requested_budget",
        observed=True,
    )["acquired_n"].nunique()

    assert counts.eq(1).all()


def test_learned_value_beats_uncertainty_at_matched_budget() -> None:
    config = SimulationConfig(
        train_n=6000,
        test_n=12000,
        seeds=1,
        budgets=(0.25, 0.50),
    )
    metrics, _ = run_seed(
        seed=11,
        config=config,
    )

    wide = metrics.pivot(
        index="requested_budget",
        columns="policy",
        values="decision_loss",
    )

    assert (
        wide.loc[0.25, "learned_value"]
        < wide.loc[0.25, "uncertainty"]
    )
    assert (
        wide.loc[0.50, "learned_value"]
        < wide.loc[0.50, "uncertainty"]
    )
