#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CACHE_PATH = Path("data/processed/monitor_score_cache.parquet")
OUT_DIR = Path("reports/policy_selection")

TRAIN_SPLIT = "policy_train"
SELECTION_SPLIT = "policy_selection"
TARGET_FPR = 0.05

MONITORS = {
    "rule_filter_v1": {
        "score": "rule_score",
        "latency": "rule_latency_ms",
    },
    "koala_text_moderation": {
        "score": "compact_unsafe_score",
        "latency": "compact_latency_ms",
    },
    "qwen3guard_gen_4b": {
        "score": "judge_unsafe_score",
        "latency": "judge_latency_ms",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def threshold_grid(scores: pd.Series) -> list[float]:
    values = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    qs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0]
    grid = sorted(set(float(np.quantile(values, q)) for q in qs))
    grid.append(float(np.nextafter(values.max(), float("inf"))))
    return sorted(set(grid))


def threshold_for_max_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    candidates = threshold_grid(pd.Series(score))
    best_t = float(np.nextafter(np.nanmax(score), float("inf")))
    best_recall = -1.0
    best_fpr = 1.0

    for t in candidates:
        pred = score >= t
        pos = y == 1
        neg = y == 0
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & neg))
        fn = int(np.sum((~pred) & pos))
        tn = int(np.sum((~pred) & neg))
        fpr = fp / (fp + tn) if (fp + tn) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan

        if fpr <= max_fpr + 1e-12:
            if recall > best_recall or (recall == best_recall and fpr < best_fpr):
                best_t = float(t)
                best_recall = float(recall)
                best_fpr = float(fpr)

    return best_t


def metrics_for_policy(
    *,
    family: str,
    policy_id: str,
    pred: np.ndarray,
    cost_ms: np.ndarray,
    selection: pd.DataFrame,
    details: dict,
    rule_median_latency_ms: float,
) -> dict:
    y = selection["y"].to_numpy(dtype=int)
    pred = np.asarray(pred, dtype=bool)
    cost_ms = np.asarray(cost_ms, dtype=float)

    pos = y == 1
    neg = y == 0

    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    fn = int(np.sum((~pred) & pos))

    recall = tp / (tp + fn) if (tp + fn) else math.nan
    fpr = fp / (fp + tn) if (fp + tn) else math.nan
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    intercept_rate = float(np.mean(pred))

    avg_cost = float(np.mean(cost_ms))
    median_cost = float(np.median(cost_ms))
    p95_cost = float(np.quantile(cost_ms, 0.95))

    return {
        "family": family,
        "policy_id": policy_id,
        "selection_split": SELECTION_SPLIT,
        "n": int(len(selection)),
        "positives": int(np.sum(pos)),
        "negatives": int(np.sum(neg)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "intercept_rate": intercept_rate,
        "avg_cost_ms": avg_cost,
        "median_cost_ms": median_cost,
        "p95_cost_ms": p95_cost,
        "normalized_avg_cost_vs_rule": avg_cost / rule_median_latency_ms,
        "normalized_median_cost_vs_rule": median_cost / rule_median_latency_ms,
        "policy_details_json": json.dumps(details, sort_keys=True),
    }


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    values = df[["recall", "fpr", "avg_cost_ms"]].to_numpy(dtype=float)

    for i, row in df.iterrows():
        recall, fpr, cost = values[df.index.get_loc(i)]
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            other_recall = other["recall"]
            other_fpr = other["fpr"]
            other_cost = other["avg_cost_ms"]

            at_least_as_good = (
                other_recall >= recall - 1e-12
                and other_fpr <= fpr + 1e-12
                and other_cost <= cost + 1e-12
            )
            strictly_better = (
                other_recall > recall + 1e-12
                or other_fpr < fpr - 1e-12
                or other_cost < cost - 1e-12
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(i)

    return df.loc[rows].sort_values(["fpr", "avg_cost_ms", "recall"], ascending=[True, True, False])


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df[columns].copy()

    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            if math.isinf(x):
                return "inf"
            return f"{x:.4g}"
        return str(x)

    rows = [[fmt(v) for v in row] for row in table.to_numpy()]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]

    def line(vals):
        return "| " + " | ".join(str(vals[i]).ljust(widths[i]) for i in range(len(vals))) + " |"

    out = [line(columns), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CACHE_PATH)
    required = {"example_id", "split", "y"}
    for spec in MONITORS.values():
        required.add(spec["score"])
        required.add(spec["latency"])

    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    train = df[df["split"] == TRAIN_SPLIT].copy()
    selection = df[df["split"] == SELECTION_SPLIT].copy()

    if train.empty or selection.empty:
        raise SystemExit("Missing policy_train or policy_selection rows.")

    y_train = train["y"].to_numpy(dtype=int)
    rule_median_latency_ms = float(train["rule_latency_ms"].median())

    candidates: list[dict] = []

    # 1. Cheapest monitor only: rule filter.
    for t in threshold_grid(train["rule_score"]):
        pred = selection["rule_score"].to_numpy(dtype=float) >= t
        cost = selection["rule_latency_ms"].to_numpy(dtype=float)
        candidates.append(
            metrics_for_policy(
                family="cheapest_monitor_only",
                policy_id=f"rule_only_t_{t:.8g}",
                pred=pred,
                cost_ms=cost,
                selection=selection,
                details={"monitor": "rule_filter_v1", "threshold": t},
                rule_median_latency_ms=rule_median_latency_ms,
            )
        )

    # 2. Strongest single monitor: Qwen3Guard judge.
    for t in threshold_grid(train["judge_unsafe_score"]):
        pred = selection["judge_unsafe_score"].to_numpy(dtype=float) >= t
        cost = selection["judge_latency_ms"].to_numpy(dtype=float)
        candidates.append(
            metrics_for_policy(
                family="strongest_single_monitor",
                policy_id=f"judge_only_t_{t:.8g}",
                pred=pred,
                cost_ms=cost,
                selection=selection,
                details={"monitor": "qwen3guard_gen_4b", "threshold": t},
                rule_median_latency_ms=rule_median_latency_ms,
            )
        )

    # 3. All monitors always on: max score threshold over all three scores.
    train_all_max = train[["rule_score", "compact_unsafe_score", "judge_unsafe_score"]].max(axis=1)
    selection_all_max = selection[["rule_score", "compact_unsafe_score", "judge_unsafe_score"]].max(axis=1)
    all_cost = (
        selection["rule_latency_ms"].to_numpy(dtype=float)
        + selection["compact_latency_ms"].to_numpy(dtype=float)
        + selection["judge_latency_ms"].to_numpy(dtype=float)
    )
    for t in threshold_grid(train_all_max):
        pred = selection_all_max.to_numpy(dtype=float) >= t
        candidates.append(
            metrics_for_policy(
                family="all_monitors_always_on",
                policy_id=f"all_max_t_{t:.8g}",
                pred=pred,
                cost_ms=all_cost,
                selection=selection,
                details={
                    "monitors": ["rule_filter_v1", "koala_text_moderation", "qwen3guard_gen_4b"],
                    "aggregation": "max_score",
                    "threshold": t,
                },
                rule_median_latency_ms=rule_median_latency_ms,
            )
        )

    # 4. Fixed cascade: rule gate, then judge only for rule-flagged examples.
    rule_gate = threshold_for_max_fpr(
        y_train,
        train["rule_score"].to_numpy(dtype=float),
        max_fpr=0.10,
    )
    judge_threshold = threshold_for_max_fpr(
        y_train,
        train["judge_unsafe_score"].to_numpy(dtype=float),
        max_fpr=0.05,
    )
    route_to_judge = selection["rule_score"].to_numpy(dtype=float) >= rule_gate
    pred = route_to_judge & (selection["judge_unsafe_score"].to_numpy(dtype=float) >= judge_threshold)
    cost = selection["rule_latency_ms"].to_numpy(dtype=float) + route_to_judge.astype(float) * selection["judge_latency_ms"].to_numpy(dtype=float)
    candidates.append(
        metrics_for_policy(
            family="fixed_cascade",
            policy_id="fixed_rule_gate_010_to_judge_005",
            pred=pred,
            cost_ms=cost,
            selection=selection,
            details={
                "route": "run rule; if rule_score >= train_10pct_fpr_threshold, run judge; intercept if judge_score >= train_5pct_fpr_threshold",
                "rule_gate_threshold": rule_gate,
                "judge_threshold": judge_threshold,
            },
            rule_median_latency_ms=rule_median_latency_ms,
        )
    )

    # 5. Cost-tuned cascade: rule low/high, compact low/high, judge threshold.
    rule_low_grid = sorted(set(float(np.quantile(train["rule_score"], q)) for q in [0.00, 0.05, 0.10, 0.25, 0.50, 0.75]))
    compact_low_grid = sorted(set(float(np.quantile(train["compact_unsafe_score"], q)) for q in [0.00, 0.10, 0.25, 0.50, 0.75]))
    rule_high_grid = [
        threshold_for_max_fpr(y_train, train["rule_score"].to_numpy(dtype=float), f)
        for f in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]
    compact_high_grid = [
        threshold_for_max_fpr(y_train, train["compact_unsafe_score"].to_numpy(dtype=float), f)
        for f in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]
    judge_grid = [
        threshold_for_max_fpr(y_train, train["judge_unsafe_score"].to_numpy(dtype=float), f)
        for f in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]

    s_rule = selection["rule_score"].to_numpy(dtype=float)
    s_compact = selection["compact_unsafe_score"].to_numpy(dtype=float)
    s_judge = selection["judge_unsafe_score"].to_numpy(dtype=float)
    c_rule = selection["rule_latency_ms"].to_numpy(dtype=float)
    c_compact = selection["compact_latency_ms"].to_numpy(dtype=float)
    c_judge = selection["judge_latency_ms"].to_numpy(dtype=float)

    for r_low in rule_low_grid:
        for r_high in rule_high_grid:
            if r_low >= r_high:
                continue
            for c_low in compact_low_grid:
                for c_high in compact_high_grid:
                    if c_low >= c_high:
                        continue
                    for j_thr in judge_grid:
                        pred = np.zeros(len(selection), dtype=bool)
                        cost = c_rule.copy()

                        rule_intercept = s_rule >= r_high
                        rule_allow = s_rule < r_low
                        go_compact = ~(rule_intercept | rule_allow)

                        pred[rule_intercept] = True
                        cost[go_compact] += c_compact[go_compact]

                        compact_intercept = go_compact & (s_compact >= c_high)
                        compact_allow = go_compact & (s_compact < c_low)
                        go_judge = go_compact & ~(compact_intercept | compact_allow)

                        pred[compact_intercept] = True
                        cost[go_judge] += c_judge[go_judge]
                        pred[go_judge] = s_judge[go_judge] >= j_thr

                        candidates.append(
                            metrics_for_policy(
                                family="cost_tuned_cascade",
                                policy_id=f"cascade_rl{r_low:.4g}_rh{r_high:.4g}_cl{c_low:.4g}_ch{c_high:.4g}_j{j_thr:.4g}",
                                pred=pred,
                                cost_ms=cost,
                                selection=selection,
                                details={
                                    "rule_low_allow_threshold": r_low,
                                    "rule_high_intercept_threshold": r_high,
                                    "compact_low_allow_threshold": c_low,
                                    "compact_high_intercept_threshold": c_high,
                                    "judge_threshold": j_thr,
                                    "route": "rule allow/intercept bands; compact allow/intercept bands; judge only for unresolved examples",
                                },
                                rule_median_latency_ms=rule_median_latency_ms,
                            )
                        )

    # 6. Learned stacker/router: train on policy_train, select thresholds on policy_selection.
    cheap_features = ["rule_score", "compact_unsafe_score"]
    full_features = ["rule_score", "compact_unsafe_score", "judge_unsafe_score"]

    router = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )

    router.fit(train[cheap_features], y_train)
    stacker.fit(train[full_features], y_train)

    train_router_prob = router.predict_proba(train[cheap_features])[:, 1]
    train_stacker_prob = stacker.predict_proba(train[full_features])[:, 1]
    selection_router_prob = router.predict_proba(selection[cheap_features])[:, 1]
    selection_stacker_prob = stacker.predict_proba(selection[full_features])[:, 1]

    router_low_grid = sorted(set(float(np.quantile(train_router_prob, q)) for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]))
    router_high_grid = [
        threshold_for_max_fpr(y_train, train_router_prob, f)
        for f in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]
    stacker_grid = [
        threshold_for_max_fpr(y_train, train_stacker_prob, f)
        for f in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]

    base_cheap_cost = selection["rule_latency_ms"].to_numpy(dtype=float) + selection["compact_latency_ms"].to_numpy(dtype=float)

    for low in router_low_grid:
        for high in router_high_grid:
            if low >= high:
                continue
            for final_t in stacker_grid:
                cheap_intercept = selection_router_prob >= high
                cheap_allow = selection_router_prob <= low
                call_judge = ~(cheap_intercept | cheap_allow)

                pred = cheap_intercept.copy()
                pred[call_judge] = selection_stacker_prob[call_judge] >= final_t

                cost = base_cheap_cost.copy()
                cost[call_judge] += selection["judge_latency_ms"].to_numpy(dtype=float)[call_judge]

                candidates.append(
                    metrics_for_policy(
                        family="learned_stacker_router",
                        policy_id=f"learned_router_low{low:.4g}_high{high:.4g}_final{final_t:.4g}",
                        pred=pred,
                        cost_ms=cost,
                        selection=selection,
                        details={
                            "cheap_features": cheap_features,
                            "full_features": full_features,
                            "router_low_allow_threshold": low,
                            "router_high_intercept_threshold": high,
                            "stacker_final_threshold": final_t,
                            "route": "run rule and compact; allow/intercept if cheap router confident; otherwise run judge and full stacker",
                        },
                        rule_median_latency_ms=rule_median_latency_ms,
                    )
                )

    coeff_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_split": TRAIN_SPLIT,
        "selection_split": SELECTION_SPLIT,
        "router_features": cheap_features,
        "stacker_features": full_features,
        "router_pipeline": str(router),
        "stacker_pipeline": str(stacker),
        "router_logistic_coefficients_after_scaling": router.named_steps["logisticregression"].coef_.tolist(),
        "router_logistic_intercept": router.named_steps["logisticregression"].intercept_.tolist(),
        "stacker_logistic_coefficients_after_scaling": stacker.named_steps["logisticregression"].coef_.tolist(),
        "stacker_logistic_intercept": stacker.named_steps["logisticregression"].intercept_.tolist(),
    }
    (OUT_DIR / "learned_router_coefficients.json").write_text(
        json.dumps(coeff_payload, indent=2),
        encoding="utf-8",
    )

    all_candidates = pd.DataFrame(candidates)
    all_candidates = (
        all_candidates.sort_values(
            ["family", "policy_id", "recall", "fpr", "avg_cost_ms"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(subset=["family", "policy_id"], keep="first")
        .sort_values(
            ["family", "fpr", "avg_cost_ms", "recall"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )

    all_candidates.to_csv(OUT_DIR / "all_candidate_policies.csv", index=False)

    selected_rows = []
    for family, group in all_candidates.groupby("family", sort=True):
        feasible = group[group["fpr"] <= TARGET_FPR + 1e-12].copy()
        if feasible.empty:
            chosen = group.sort_values(["fpr", "avg_cost_ms", "recall"], ascending=[True, True, False]).iloc[0]
        else:
            chosen = feasible.sort_values(
                ["recall", "avg_cost_ms", "fpr"],
                ascending=[False, True, True],
            ).iloc[0]
        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows).sort_values(
        ["recall", "avg_cost_ms"],
        ascending=[False, True],
    )
    selected.to_csv(OUT_DIR / "selected_baselines_fpr05.csv", index=False)

    frontier = pareto_frontier(all_candidates)
    frontier.to_csv(OUT_DIR / "selection_pareto_frontier.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_cache": str(CACHE_PATH),
        "input_cache_sha256": sha256_file(CACHE_PATH),
        "training_split": TRAIN_SPLIT,
        "selection_split": SELECTION_SPLIT,
        "target_fpr_for_selected_baselines": TARGET_FPR,
        "calibration_final_and_shift_splits_used": False,
        "families": sorted(all_candidates["family"].unique().tolist()),
        "num_candidate_policies": int(len(all_candidates)),
        "num_pareto_frontier_policies": int(len(frontier)),
        "rule_median_latency_ms_from_train": rule_median_latency_ms,
    }
    (OUT_DIR / "policy_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    summary = f"""# Baseline policy selection

Generated at `{manifest["created_at"]}`.

This stage trains or configures baseline policies using `{TRAIN_SPLIT}` and compares candidate operating points on `{SELECTION_SPLIT}`. It does **not** use `calibration`, `final_test`, or `held_out_shift`.

The selected table below chooses the highest-recall candidate within each required baseline family subject to selection FPR <= `{TARGET_FPR}`; ties prefer lower average cost.

## Selected required baselines at FPR <= {TARGET_FPR}

{markdown_table(selected, ["family", "policy_id", "recall", "fpr", "precision", "avg_cost_ms", "normalized_avg_cost_vs_rule"])}

## Pareto frontier preview

{markdown_table(frontier.head(30), ["family", "policy_id", "recall", "fpr", "precision", "avg_cost_ms", "normalized_avg_cost_vs_rule"])}

## Required baseline coverage

- `cheapest_monitor_only`: rule filter only.
- `strongest_single_monitor`: Qwen3Guard judge only.
- `all_monitors_always_on`: rule + compact + judge always run, max-score decision.
- `fixed_cascade`: fixed rule gate into judge.
- `cost_tuned_cascade`: grid-searched rule/compact/judge cascade.
- `learned_stacker_router`: logistic cheap router plus full logistic stacker when routed to judge.

## Files

- `all_candidate_policies.csv`
- `selected_baselines_fpr05.csv`
- `selection_pareto_frontier.csv`
- `learned_router_coefficients.json`
- `policy_selection_manifest.json`
"""
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    print("=== SELECTED BASELINES @ FPR <= 0.05 ===")
    print(selected[["family", "policy_id", "recall", "fpr", "precision", "avg_cost_ms", "normalized_avg_cost_vs_rule"]].to_string(index=False))

    print("\n=== PARETO FRONTIER HEAD ===")
    print(frontier[["family", "policy_id", "recall", "fpr", "precision", "avg_cost_ms", "normalized_avg_cost_vs_rule"]].head(30).to_string(index=False))

    print("\n=== COUNTS ===")
    print(json.dumps(manifest, indent=2))

    print("\n=== WROTE ===")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
