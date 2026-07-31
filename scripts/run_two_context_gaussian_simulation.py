#!/usr/bin/env python3
"""Two-context Gaussian decision-value acquisition experiment.

The cheap signal has the same conditional distribution in both contexts.
The optional monitor is conditionally informative in context A and redundant
in context B. Policies are compared at exactly matched acquisition rates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, spearmanr, t
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class SimulationConfig:
    train_n: int = 6000
    test_n: int = 12000
    seeds: int = 30
    context_a_probability: float = 0.5
    cheap_separation: float = 0.8
    expensive_separation_context_a: float = 2.0
    redundant_noise_sd_context_b: float = 0.25
    acquisition_cost: float = 0.02
    budgets: tuple[float, ...] = (
        0.0,
        0.05,
        0.10,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.75,
        1.00,
    )


def generate_dataset(
    seed: int,
    n: int,
    config: SimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    context_a = (
        rng.random(n) < config.context_a_probability
    )
    y_signed = rng.choice(np.array([-1, 1]), size=n)

    h = (
        config.cheap_separation * y_signed
        + rng.normal(size=n)
    )

    z = np.empty(n, dtype=float)
    z[context_a] = (
        config.expensive_separation_context_a
        * y_signed[context_a]
        + rng.normal(size=int(context_a.sum()))
    )
    z[~context_a] = (
        h[~context_a]
        + config.redundant_noise_sd_context_b
        * rng.normal(size=int((~context_a).sum()))
    )

    cheap_log_odds = 2.0 * config.cheap_separation * h
    cheap_probability = expit(cheap_log_odds)
    cheap_prediction = np.where(
        cheap_log_odds >= 0.0,
        1,
        -1,
    )

    full_log_odds = cheap_log_odds.copy()
    full_log_odds[context_a] += (
        2.0
        * config.expensive_separation_context_a
        * z[context_a]
    )
    full_prediction = np.where(
        full_log_odds >= 0.0,
        1,
        -1,
    )

    # In context B, Z is generated conditional on H and is independent of Y
    # given H. The Bayes-optimal full decision therefore ignores Z.
    full_prediction[~context_a] = cheap_prediction[~context_a]

    cheap_loss = (cheap_prediction != y_signed).astype(float)
    full_loss = (full_prediction != y_signed).astype(float)

    cheap_bayes_risk = np.minimum(
        cheap_probability,
        1.0 - cheap_probability,
    )

    # Analytic ex-ante decision value in context A. Given H=h,
    # Z|Y=1 ~ N(mu_z,1), Z|Y=-1 ~ N(-mu_z,1). The posterior decision
    # boundary is z >= -logit(p)/(2 mu_z).
    mu_z = config.expensive_separation_context_a
    boundary = -cheap_log_odds / (2.0 * mu_z)
    post_acquisition_risk_a = (
        cheap_probability * norm.cdf(boundary - mu_z)
        + (1.0 - cheap_probability)
        * (1.0 - norm.cdf(boundary + mu_z))
    )

    oracle_value = np.where(
        context_a,
        cheap_bayes_risk - post_acquisition_risk_a,
        0.0,
    )

    return pd.DataFrame(
        {
            "context": np.where(
                context_a,
                "A_informative",
                "B_redundant",
            ),
            "context_a": context_a.astype(int),
            "y_signed": y_signed,
            "y": (y_signed == 1).astype(int),
            "h": h,
            "z": z,
            "cheap_probability": cheap_probability,
            "cheap_uncertainty": cheap_bayes_risk,
            "cheap_prediction_signed": cheap_prediction,
            "full_prediction_signed": full_prediction,
            "cheap_loss": cheap_loss,
            "full_loss": full_loss,
            "realized_decision_improvement": (
                cheap_loss - full_loss
            ),
            "oracle_decision_value": oracle_value,
        }
    )


def pre_acquisition_features(frame: pd.DataFrame) -> pd.DataFrame:
    h = frame["h"].to_numpy(dtype=float)
    context_a = frame["context_a"].to_numpy(dtype=float)
    probability = frame[
        "cheap_probability"
    ].to_numpy(dtype=float)
    uncertainty = frame[
        "cheap_uncertainty"
    ].to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "h": h,
            "abs_h": np.abs(h),
            "h_squared": h * h,
            "context_a": context_a,
            "cheap_probability": probability,
            "cheap_uncertainty": uncertainty,
            "context_x_h": context_a * h,
            "context_x_abs_h": context_a * np.abs(h),
            "context_x_uncertainty": (
                context_a * uncertainty
            ),
        }
    )


def fit_value_estimator(
    train: pd.DataFrame,
    seed: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=30,
        l2_regularization=0.01,
        random_state=seed,
    )
    model.fit(
        pre_acquisition_features(train),
        train[
            "realized_decision_improvement"
        ].to_numpy(dtype=float),
    )
    return model


def select_top_k(
    scores: np.ndarray,
    k: int,
    tie_breaker: np.ndarray,
) -> np.ndarray:
    n = len(scores)
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0,{n}], got {k}")

    selected = np.zeros(n, dtype=bool)
    if k == 0:
        return selected
    if k == n:
        selected[:] = True
        return selected

    order = np.lexsort(
        (
            tie_breaker,
            np.asarray(scores, dtype=float),
        )
    )
    selected[order[-k:]] = True
    return selected


def evaluate_selection(
    test: pd.DataFrame,
    selected: np.ndarray,
    policy: str,
    requested_budget: float,
    acquisition_cost: float,
) -> dict[str, float | int | str]:
    y = test["y"].to_numpy(dtype=int)
    cheap_prediction = (
        test["cheap_prediction_signed"].to_numpy()
        == 1
    ).astype(int)
    full_prediction = (
        test["full_prediction_signed"].to_numpy()
        == 1
    ).astype(int)

    prediction = np.where(
        selected,
        full_prediction,
        cheap_prediction,
    )
    loss = (prediction != y).astype(float)

    negative = y == 0
    positive = y == 1

    false_positive = (
        (prediction == 1) & negative
    ).sum()
    true_positive = (
        (prediction == 1) & positive
    ).sum()

    actual_rate = float(selected.mean())
    context_a = (
        test["context_a"].to_numpy(dtype=int) == 1
    )
    context_b = ~context_a

    cheap_mean_loss = float(
        test["cheap_loss"].mean()
    )
    decision_loss = float(loss.mean())

    return {
        "policy": policy,
        "requested_budget": requested_budget,
        "acquired_n": int(selected.sum()),
        "acquisition_rate": actual_rate,
        "decision_loss": decision_loss,
        "accuracy": 1.0 - decision_loss,
        "total_risk_with_cost": (
            decision_loss
            + acquisition_cost * actual_rate
        ),
        "fpr": float(false_positive / negative.sum()),
        "recall": float(true_positive / positive.sum()),
        "fnr": float(
            1.0 - true_positive / positive.sum()
        ),
        "decision_loss_reduction_vs_never": (
            cheap_mean_loss - decision_loss
        ),
        "context_a_acquisition_rate": float(
            selected[context_a].mean()
        ),
        "context_b_acquisition_rate": float(
            selected[context_b].mean()
        ),
        "oracle_value_captured_per_example": float(
            (
                selected
                * test[
                    "oracle_decision_value"
                ].to_numpy(dtype=float)
            ).mean()
        ),
    }


def run_seed(
    seed: int,
    config: SimulationConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = generate_dataset(
        seed=2 * seed,
        n=config.train_n,
        config=config,
    )
    test = generate_dataset(
        seed=2 * seed + 1,
        n=config.test_n,
        config=config,
    )

    model = fit_value_estimator(train, seed=seed)
    learned_value = model.predict(
        pre_acquisition_features(test)
    )

    oracle_value = test[
        "oracle_decision_value"
    ].to_numpy(dtype=float)
    uncertainty = test[
        "cheap_uncertainty"
    ].to_numpy(dtype=float)

    rng = np.random.default_rng(100_000 + seed)
    tie_breaker = rng.random(config.test_n)
    random_score = rng.random(config.test_n)

    policy_scores = {
        "random": random_score,
        "uncertainty": uncertainty,
        "oracle_decision_value": oracle_value,
        "learned_value": learned_value,
    }

    metric_rows: list[
        dict[str, float | int | str]
    ] = []

    for budget in config.budgets:
        acquired_n = int(
            round(budget * config.test_n)
        )

        for policy, scores in policy_scores.items():
            selected = select_top_k(
                scores=scores,
                k=acquired_n,
                tie_breaker=tie_breaker,
            )
            row = evaluate_selection(
                test=test,
                selected=selected,
                policy=policy,
                requested_budget=budget,
                acquisition_cost=config.acquisition_cost,
            )
            row["seed"] = seed
            metric_rows.append(row)

    # Value-estimation diagnostics are secondary. Matched-budget policy
    # outcomes remain the primary evaluation.
    pearson = float(
        np.corrcoef(
            learned_value,
            oracle_value,
        )[0, 1]
    )
    spearman = float(
        spearmanr(
            learned_value,
            oracle_value,
        ).statistic
    )
    mse = float(
        np.mean(
            (learned_value - oracle_value) ** 2
        )
    )

    context_summary = (
        test.groupby(
            "context",
            observed=True,
        )[
            [
                "cheap_loss",
                "full_loss",
                "cheap_uncertainty",
                "oracle_decision_value",
            ]
        ]
        .mean()
    )

    diagnostics = {
        "seed": seed,
        "value_mse": mse,
        "value_pearson": pearson,
        "value_spearman": spearman,
        "context_a_cheap_loss": float(
            context_summary.loc[
                "A_informative",
                "cheap_loss",
            ]
        ),
        "context_a_full_loss": float(
            context_summary.loc[
                "A_informative",
                "full_loss",
            ]
        ),
        "context_b_cheap_loss": float(
            context_summary.loc[
                "B_redundant",
                "cheap_loss",
            ]
        ),
        "context_b_full_loss": float(
            context_summary.loc[
                "B_redundant",
                "full_loss",
            ]
        ),
        "context_a_mean_uncertainty": float(
            context_summary.loc[
                "A_informative",
                "cheap_uncertainty",
            ]
        ),
        "context_b_mean_uncertainty": float(
            context_summary.loc[
                "B_redundant",
                "cheap_uncertainty",
            ]
        ),
        "context_a_mean_oracle_value": float(
            context_summary.loc[
                "A_informative",
                "oracle_decision_value",
            ]
        ),
        "context_b_mean_oracle_value": float(
            context_summary.loc[
                "B_redundant",
                "oracle_decision_value",
            ]
        ),
    }

    return pd.DataFrame(metric_rows), diagnostics


def summarize_with_ci(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    if group_columns:
        grouped = frame.groupby(
            group_columns,
            sort=True,
            observed=True,
        )
    else:
        grouped = [((), frame)]

    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {
            column: value
            for column, value in zip(
                group_columns,
                group_values,
            )
        }
        row["seed_n"] = int(group["seed"].nunique())

        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            if len(values) > 1:
                standard_error = float(
                    np.std(values, ddof=1)
                    / math.sqrt(len(values))
                )
                critical = float(
                    t.ppf(
                        0.975,
                        df=len(values) - 1,
                    )
                )
                lower = mean - critical * standard_error
                upper = mean + critical * standard_error
            else:
                standard_error = 0.0
                lower = mean
                upper = mean

            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = standard_error
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper

        rows.append(row)

    return pd.DataFrame(rows)


def paired_policy_differences(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    wide = metrics.pivot(
        index=["seed", "requested_budget"],
        columns="policy",
        values="decision_loss",
    ).reset_index()

    rows: list[dict[str, float | int | str]] = []
    for comparator in [
        "random",
        "oracle_decision_value",
        "learned_value",
    ]:
        work = wide[
            ["seed", "requested_budget"]
        ].copy()
        work["comparator"] = comparator
        work["loss_reduction_vs_uncertainty"] = (
            wide["uncertainty"]
            - wide[comparator]
        )
        rows.extend(work.to_dict("records"))

    paired = pd.DataFrame(rows)
    return summarize_with_ci(
        paired,
        group_columns=[
            "requested_budget",
            "comparator",
        ],
        metric_columns=[
            "loss_reduction_vs_uncertainty",
        ],
    )


def make_figures(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.2))
    for policy, group in summary.groupby(
        "policy",
        sort=False,
    ):
        group = group.sort_values(
            "requested_budget"
        )
        plt.plot(
            group["requested_budget"],
            group["decision_loss_mean"],
            marker="o",
            label=policy.replace("_", " "),
        )
        plt.fill_between(
            group["requested_budget"],
            group["decision_loss_ci95_lower"],
            group["decision_loss_ci95_upper"],
            alpha=0.15,
        )
    plt.xlabel("Acquisition rate")
    plt.ylabel("Downstream decision loss")
    plt.title(
        "Two-context Gaussian simulation: matched-budget decision loss"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "matched_budget_decision_loss.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for comparator, group in paired.groupby(
        "comparator",
        sort=False,
    ):
        group = group.sort_values(
            "requested_budget"
        )
        plt.plot(
            group["requested_budget"],
            group[
                "loss_reduction_vs_uncertainty_mean"
            ],
            marker="o",
            label=comparator.replace("_", " "),
        )
        plt.fill_between(
            group["requested_budget"],
            group[
                "loss_reduction_vs_uncertainty_ci95_lower"
            ],
            group[
                "loss_reduction_vs_uncertainty_ci95_upper"
            ],
            alpha=0.15,
        )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Acquisition rate")
    plt.ylabel(
        "Decision-loss reduction relative to uncertainty"
    )
    plt.title(
        "Matched-budget advantage over uncertainty routing"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir
        / "matched_budget_advantage_over_uncertainty.png",
        dpi=180,
    )
    plt.close()

    budget_half = summary[
        np.isclose(
            summary["requested_budget"],
            0.50,
        )
    ].copy()
    positions = np.arange(len(budget_half))
    width = 0.36

    plt.figure(figsize=(8.5, 5.2))
    plt.bar(
        positions - width / 2,
        budget_half[
            "context_a_acquisition_rate_mean"
        ],
        width,
        label="context A informative",
    )
    plt.bar(
        positions + width / 2,
        budget_half[
            "context_b_acquisition_rate_mean"
        ],
        width,
        label="context B redundant",
    )
    plt.xticks(
        positions,
        [
            policy.replace("_", " ")
            for policy in budget_half["policy"]
        ],
        rotation=20,
        ha="right",
    )
    plt.ylabel("Acquisition rate within context")
    plt.title(
        "Context allocation at 50% total acquisition"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "context_allocation_at_50pct.png",
        dpi=180,
    )
    plt.close()


def write_summary(
    config: SimulationConfig,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    def result_row(
        frame: pd.DataFrame,
        budget: float,
        policy: str,
    ) -> pd.Series:
        return frame[
            np.isclose(
                frame["requested_budget"],
                budget,
            )
            & frame["policy"].eq(policy)
        ].iloc[0]

    learned_25 = result_row(
        summary,
        0.25,
        "learned_value",
    )
    uncertainty_25 = result_row(
        summary,
        0.25,
        "uncertainty",
    )
    oracle_25 = result_row(
        summary,
        0.25,
        "oracle_decision_value",
    )

    learned_50 = result_row(
        summary,
        0.50,
        "learned_value",
    )
    uncertainty_50 = result_row(
        summary,
        0.50,
        "uncertainty",
    )
    oracle_50 = result_row(
        summary,
        0.50,
        "oracle_decision_value",
    )

    learned_advantage = paired[
        paired["comparator"].eq("learned_value")
        & np.isclose(
            paired["requested_budget"],
            0.25,
        )
    ].iloc[0]

    diagnostic_means = diagnostics.mean(
        numeric_only=True
    )

    summary_text = f"""# Two-context Gaussian decision-value simulation

## Scope

This is a controlled toy experiment. It does not establish a new routing
method and does not use the real-data test partition.

## Data-generating process

- Equal class prior.
- Context A probability: {config.context_a_probability:.2f}.
- Cheap signal: H given Y is Gaussian with class means plus or minus
  {config.cheap_separation:.2f} and unit variance.
- The cheap-signal distribution is identical across contexts.
- In context A, the optional monitor is conditionally independent of H given
  Y and has Gaussian class separation
  {config.expensive_separation_context_a:.2f}.
- In context B, the optional monitor is H plus independent Gaussian noise and
  is conditionally redundant given H.
- Train examples per seed: {config.train_n}.
- Test examples per seed: {config.test_n}.
- Independent seeds: {config.seeds}.

## Policies

All policies acquire exactly the same number of optional-monitor outputs at
each requested acquisition rate.

1. Random routing.
2. Cheap-model uncertainty routing.
3. Oracle decision-value routing using the analytic conditional Bayes-risk
   reduction.
4. A learned value estimator trained on realized decision-loss reduction
   using only pre-acquisition H and the observed context.

## Primary matched-budget results

At 25% acquisition:

- uncertainty decision loss:
  {uncertainty_25["decision_loss_mean"]:.6f};
- learned-value decision loss:
  {learned_25["decision_loss_mean"]:.6f};
- oracle decision-value loss:
  {oracle_25["decision_loss_mean"]:.6f};
- learned reduction relative to uncertainty:
  {learned_advantage["loss_reduction_vs_uncertainty_mean"]:.6f};
- 95% paired interval:
  [{learned_advantage["loss_reduction_vs_uncertainty_ci95_lower"]:.6f},
   {learned_advantage["loss_reduction_vs_uncertainty_ci95_upper"]:.6f}].

At 50% acquisition:

- uncertainty decision loss:
  {uncertainty_50["decision_loss_mean"]:.6f};
- learned-value decision loss:
  {learned_50["decision_loss_mean"]:.6f};
- oracle decision-value loss:
  {oracle_50["decision_loss_mean"]:.6f}.

## Diagnostic checks

Across seeds:

- mean context-A cheap decision loss:
  {diagnostic_means["context_a_cheap_loss"]:.6f};
- mean context-A full-information loss:
  {diagnostic_means["context_a_full_loss"]:.6f};
- mean context-B cheap decision loss:
  {diagnostic_means["context_b_cheap_loss"]:.6f};
- mean context-B full-information loss:
  {diagnostic_means["context_b_full_loss"]:.6f};
- mean context-A cheap uncertainty:
  {diagnostic_means["context_a_mean_uncertainty"]:.6f};
- mean context-B cheap uncertainty:
  {diagnostic_means["context_b_mean_uncertainty"]:.6f};
- learned-value Spearman correlation with analytic value:
  {diagnostic_means["value_spearman"]:.6f}.

The cheap uncertainty distribution is matched across contexts, but the
optional monitor has positive decision value only in context A. The learned
value estimator uses the context to allocate acquisition to informative
examples and improves decision loss relative to uncertainty routing at the
same acquisition rate.

## Interpretation

The experiment demonstrates the mechanism requested by the professor:
ordinary uncertainty can be identical across examples while optional-monitor
quality differs. A value estimator can exploit legitimate pre-acquisition
context to improve matched-budget routing.

This is only a toy validation. The next step is the development-only,
cross-fitted real-data decision-value diagnostic.
"""

    (output_dir / "summary.md").write_text(
        summary_text,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/decision_value_gaussian_simulation"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--train-n",
        type=int,
        default=6000,
    )
    parser.add_argument(
        "--test-n",
        type=int,
        default=12000,
    )
    args = parser.parse_args()

    config = SimulationConfig(
        train_n=args.train_n,
        test_n=args.test_n,
        seeds=args.seeds,
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, float]] = []

    for seed in range(config.seeds):
        metrics, diagnostics = run_seed(
            seed=seed,
            config=config,
        )
        metric_frames.append(metrics)
        diagnostic_rows.append(diagnostics)
        print(
            f"completed seed {seed + 1}/{config.seeds}",
            flush=True,
        )

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )
    diagnostics = pd.DataFrame(diagnostic_rows)

    # Exact matched-budget validation.
    budget_counts = (
        metrics.groupby(
            ["seed", "requested_budget"],
            observed=True,
        )["acquired_n"]
        .nunique()
    )
    if not budget_counts.eq(1).all():
        raise RuntimeError(
            "Policies did not use identical acquisition counts"
        )

    summary = summarize_with_ci(
        metrics,
        group_columns=[
            "requested_budget",
            "policy",
        ],
        metric_columns=[
            "acquisition_rate",
            "decision_loss",
            "accuracy",
            "total_risk_with_cost",
            "fpr",
            "recall",
            "fnr",
            "decision_loss_reduction_vs_never",
            "context_a_acquisition_rate",
            "context_b_acquisition_rate",
            "oracle_value_captured_per_example",
        ],
    )
    paired = paired_policy_differences(metrics)
    diagnostic_summary = summarize_with_ci(
        diagnostics,
        group_columns=[],
        metric_columns=[
            column
            for column in diagnostics.columns
            if column != "seed"
        ],
    )

    metrics.to_csv(
        args.output_dir / "seed_metrics.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir
        / "matched_budget_summary.csv",
        index=False,
    )
    paired.to_csv(
        args.output_dir
        / "paired_policy_differences.csv",
        index=False,
    )
    diagnostics.to_csv(
        args.output_dir
        / "value_prediction_diagnostics_by_seed.csv",
        index=False,
    )
    diagnostic_summary.to_csv(
        args.output_dir
        / "value_prediction_diagnostics_summary.csv",
        index=False,
    )

    make_figures(
        summary=summary,
        paired=paired,
        output_dir=args.output_dir,
    )
    write_summary(
        config=config,
        summary=summary,
        paired=paired,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
    )

    learned_25 = paired[
        paired["comparator"].eq("learned_value")
        & np.isclose(
            paired["requested_budget"],
            0.25,
        )
    ].iloc[0]
    learned_50 = paired[
        paired["comparator"].eq("learned_value")
        & np.isclose(
            paired["requested_budget"],
            0.50,
        )
    ].iloc[0]

    if (
        learned_25[
            "loss_reduction_vs_uncertainty_ci95_lower"
        ]
        <= 0.0
    ):
        raise RuntimeError(
            "Learned value did not beat uncertainty at 25% "
            "with a positive paired 95% interval"
        )
    if (
        learned_50[
            "loss_reduction_vs_uncertainty_ci95_lower"
        ]
        <= 0.0
    ):
        raise RuntimeError(
            "Learned value did not beat uncertainty at 50% "
            "with a positive paired 95% interval"
        )

    diagnostic_means = diagnostics.mean(
        numeric_only=True
    )
    if not (
        diagnostic_means["context_a_full_loss"]
        < diagnostic_means["context_a_cheap_loss"]
    ):
        raise RuntimeError(
            "The optional monitor was not informative in context A"
        )
    if not math.isclose(
        diagnostic_means["context_b_full_loss"],
        diagnostic_means["context_b_cheap_loss"],
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "The optional monitor was not redundant in context B"
        )
    if (
        abs(
            diagnostic_means[
                "context_a_mean_uncertainty"
            ]
            - diagnostic_means[
                "context_b_mean_uncertainty"
            ]
        )
        > 0.01
    ):
        raise RuntimeError(
            "Cheap uncertainty was not matched across contexts"
        )

    manifest = {
        "artifact": (
            "two_context_gaussian_decision_value_simulation"
        ),
        "status": "toy_experiment_completed",
        "scope": (
            "controlled simulation only; no new routing-method claim"
        ),
        "config": {
            **asdict(config),
            "budgets": list(config.budgets),
        },
        "policies": [
            "random",
            "uncertainty",
            "oracle_decision_value",
            "learned_value",
        ],
        "matched_acquisition_counts_verified": True,
        "primary_evaluation": (
            "downstream decision loss at identical acquisition rates"
        ),
        "secondary_diagnostics": (
            "value prediction correlations and context allocation"
        ),
        "key_checks": {
            "context_a_monitor_informative": True,
            "context_b_monitor_redundant": True,
            "cheap_uncertainty_matched_across_contexts": True,
            "learned_beats_uncertainty_at_25pct": True,
            "learned_beats_uncertainty_at_50pct": True,
        },
        "outputs": [
            "seed_metrics.csv",
            "matched_budget_summary.csv",
            "paired_policy_differences.csv",
            "value_prediction_diagnostics_by_seed.csv",
            "value_prediction_diagnostics_summary.csv",
            "matched_budget_decision_loss.png",
            "matched_budget_advantage_over_uncertainty.png",
            "context_allocation_at_50pct.png",
            "summary.md",
        ],
    }
    (
        args.output_dir / "simulation_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== PRIMARY 25% BUDGET COMPARISON ===")
    print(
        summary[
            np.isclose(
                summary["requested_budget"],
                0.25,
            )
        ][
            [
                "policy",
                "acquisition_rate_mean",
                "decision_loss_mean",
                "decision_loss_ci95_lower",
                "decision_loss_ci95_upper",
                "context_a_acquisition_rate_mean",
                "context_b_acquisition_rate_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "=== PAIRED ADVANTAGE OVER UNCERTAINTY "
        "AT 25% ==="
    )
    print(
        paired[
            np.isclose(
                paired["requested_budget"],
                0.25,
            )
        ].to_string(index=False)
    )

    print()
    print(
        "=== CONTEXT AND VALUE-ESTIMATOR DIAGNOSTICS ==="
    )
    print(
        diagnostic_summary.to_string(index=False)
    )

    print()
    print(
        "simulation completed:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
