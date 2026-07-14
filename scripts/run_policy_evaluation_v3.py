#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta

CACHE = Path("data/processed/monitor_score_cache_v3.parquet")
BUNDLE = Path("artifacts/fusion_models_v3/fusion_bundle.joblib")
TIMING_DIR = Path(
    "data/interim/controlled_timing_benchmark_v2/timing_benchmark_v2"
)

SELECTION_DIR = Path("reports/policy_selection_v3")
FROZEN_DIR = Path("artifacts/frozen_policies_v3")
EVALUATION_DIR = Path("reports/policy_evaluation_v3")
FUSION_DIR = Path("reports/fusion_comparison_v3")
OPERATING_POINTS = (
    Path("artifacts/fusion_models_v3/frozen_operating_points_v3.json")
)
MANIFEST = Path("data/metadata/policy_evaluation_v3_manifest.json")

TRAIN_SPLIT = "policy_train"
SELECTION_SPLIT = "policy_selection"
EVAL_SPLITS = ["calibration", "final_test", "held_out_shift"]

BASELINE_TARGET_FPR = 0.05
COMMON_TARGET_FPRS = [0.01, 0.025, 0.05, 0.10]
BOOTSTRAP_REPS = 2000
RNG_SEED = 20260714
ALPHA = 0.05

CHEAP_FEATURES = ["rule_score", "compact_unsafe_score"]
FULL_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]
REQUIRED_FAMILIES = {
    "cheapest_monitor_only",
    "strongest_single_monitor",
    "all_monitors_always_on",
    "fixed_cascade",
    "cost_tuned_cascade",
    "learned_stacker_router",
}

# Verified controlled T4 medians. These are used only if the corresponding
# controlled-timing parquet cannot be parsed automatically.
TIMING_FALLBACK_MS = {
    "rule": 1.035,
    "compact": 38.087,
    "qwen_prompt_response": 1368.64,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def threshold_grid(values: pd.Series) -> list[float]:
    scores = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    scores = scores[np.isfinite(scores)]
    quantiles = [
        0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0,
    ]
    grid = {float(np.quantile(scores, q)) for q in quantiles}
    grid.add(float(np.nextafter(scores.max(), np.inf)))
    return sorted(grid)


def threshold_for_max_fpr(
    y: np.ndarray,
    scores: np.ndarray,
    max_fpr: float,
) -> float:
    best_threshold = float(np.nextafter(np.nanmax(scores), np.inf))
    best_key: tuple[float, float] | None = None
    for threshold in threshold_grid(pd.Series(scores)):
        result = binary_metrics(y, scores >= threshold)
        if result["fpr"] <= max_fpr + 1e-12:
            key = (float(result["recall"]), -float(result["fpr"]))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
    return best_threshold


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    pos = y == 1
    neg = y == 0

    tp = int(np.sum(pred & pos))
    fn = int(np.sum((~pred) & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    positive_n = int(pos.sum())
    negative_n = int(neg.sum())

    return {
        "n": int(len(y)),
        "positive_n": positive_n,
        "negative_n": negative_n,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / positive_n if positive_n else math.nan,
        "fpr": fp / negative_n if negative_n else math.nan,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "accuracy": (tp + tn) / len(y) if len(y) else math.nan,
        "intercept_rate": float(np.mean(pred)) if len(pred) else math.nan,
    }


def exact_ci(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(
        beta.ppf(1 - alpha / 2, k + 1, n - k)
    )
    return lower, upper


def exact_upper(k: int, n: int, alpha: float = ALPHA) -> float:
    if n == 0:
        return math.nan
    if k == n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def finite_sample_metrics(
    y: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float | int]:
    result = binary_metrics(y, pred)
    recall_low, recall_high = exact_ci(
        int(result["tp"]), int(result["positive_n"])
    )
    fpr_low, fpr_high = exact_ci(
        int(result["fp"]), int(result["negative_n"])
    )
    result.update(
        {
            "recall_ci95_low": recall_low,
            "recall_ci95_high": recall_high,
            "fpr_ci95_low": fpr_low,
            "fpr_ci95_high": fpr_high,
            "fpr_one_sided95_upper": exact_upper(
                int(result["fp"]), int(result["negative_n"])
            ),
        }
    )
    return result


def find_latency_column(
    frame: pd.DataFrame,
    preferred: list[str],
) -> str | None:
    for column in preferred:
        if column in frame.columns:
            return column
    candidates = [
        column
        for column in frame.columns
        if "latency" in column.lower() and column.lower().endswith("_ms")
    ]
    total = [c for c in candidates if "total" in c.lower()]
    return total[0] if total else (candidates[0] if candidates else None)


def controlled_timing_medians() -> tuple[dict[str, float], dict]:
    values = dict(TIMING_FALLBACK_MS)
    provenance: dict[str, object] = {
        "timing_directory": str(TIMING_DIR),
        "fallback_values_ms": TIMING_FALLBACK_MS,
        "parsed_files": {},
        "fallback_used_for": [],
    }

    if TIMING_DIR.exists():
        for path in sorted(TIMING_DIR.rglob("*.parquet")):
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue

            name = path.name.lower()

            if "rule_latency" in name:
                column = find_latency_column(
                    frame,
                    ["external_total_latency_ms", "rule_total_latency_ms"],
                )
                if column:
                    values["rule"] = float(
                        pd.to_numeric(frame[column], errors="coerce").median()
                    )
                    provenance["parsed_files"]["rule"] = {
                        "path": str(path),
                        "column": column,
                        "sha256": sha256(path),
                    }

            elif "compact_latency" in name:
                column = find_latency_column(
                    frame,
                    ["compact_total_latency_ms", "external_total_latency_ms"],
                )
                if column:
                    values["compact"] = float(
                        pd.to_numeric(frame[column], errors="coerce").median()
                    )
                    provenance["parsed_files"]["compact"] = {
                        "path": str(path),
                        "column": column,
                        "sha256": sha256(path),
                    }

            elif "qwen" in name and "latency" in name:
                mode_column = next(
                    (
                        c
                        for c in ["mode", "qwen_mode", "input_mode"]
                        if c in frame.columns
                    ),
                    None,
                )
                column = find_latency_column(
                    frame,
                    [
                        "qwen_total_latency_ms",
                        "external_total_latency_ms",
                        "total_latency_ms",
                    ],
                )
                if column:
                    subset = frame
                    if mode_column:
                        modes = (
                            frame[mode_column]
                            .astype(str)
                            .str.replace("-", "_", regex=False)
                        )
                        subset = frame[modes.eq("prompt_response")]
                    if not subset.empty:
                        values["qwen_prompt_response"] = float(
                            pd.to_numeric(
                                subset[column], errors="coerce"
                            ).median()
                        )
                        provenance["parsed_files"]["qwen_prompt_response"] = {
                            "path": str(path),
                            "column": column,
                            "mode_column": mode_column,
                            "sha256": sha256(path),
                        }

    for key in values:
        if key not in provenance["parsed_files"]:
            provenance["fallback_used_for"].append(key)

    if any(not np.isfinite(v) or v <= 0 for v in values.values()):
        raise SystemExit(f"Invalid controlled timing medians: {values}")

    provenance["selected_p50_ms"] = values
    return values, provenance


def policy_costs(
    n: int,
    timing: dict[str, float],
    *,
    rule: bool = False,
    compact: bool = False,
    qwen_mask: np.ndarray | None = None,
    qwen_always: bool = False,
) -> np.ndarray:
    cost = np.zeros(n, dtype=float)
    if rule:
        cost += timing["rule"]
    if compact:
        cost += timing["compact"]
    if qwen_always:
        cost += timing["qwen_prompt_response"]
    elif qwen_mask is not None:
        cost += (
            np.asarray(qwen_mask, dtype=float)
            * timing["qwen_prompt_response"]
        )
    return cost


def selection_record(
    *,
    family: str,
    policy_id: str,
    y: np.ndarray,
    pred: np.ndarray,
    cost: np.ndarray,
    details: dict,
    runtime: dict[str, float],
) -> dict:
    result = binary_metrics(y, pred)
    return {
        "family": family,
        "policy_id": policy_id,
        **result,
        "avg_cost_ms": float(np.mean(cost)),
        "median_cost_ms": float(np.median(cost)),
        "p95_cost_ms": float(np.quantile(cost, 0.95)),
        **runtime,
        "policy_details_json": json.dumps(details, sort_keys=True),
    }


def choose_family_candidate(group: pd.DataFrame) -> pd.Series:
    feasible = group[group["fpr"] <= BASELINE_TARGET_FPR + 1e-12]
    if feasible.empty:
        return group.sort_values(
            ["fpr", "avg_cost_ms", "recall"],
            ascending=[True, True, False],
        ).iloc[0]
    return feasible.sort_values(
        ["recall", "avg_cost_ms", "fpr", "precision"],
        ascending=[False, True, True, False],
    ).iloc[0]


def build_baseline_candidates(
    train: pd.DataFrame,
    selection: pd.DataFrame,
    bundle: dict,
    timing: dict[str, float],
) -> pd.DataFrame:
    y_train = train["y"].to_numpy(int)
    y_selection = selection["y"].to_numpy(int)

    train_rule = train["rule_score"].to_numpy(float)
    train_compact = train["compact_unsafe_score"].to_numpy(float)
    train_qwen = train["qwen_prompt_response_score"].to_numpy(float)

    sel_rule = selection["rule_score"].to_numpy(float)
    sel_compact = selection["compact_unsafe_score"].to_numpy(float)
    sel_qwen = selection["qwen_prompt_response_score"].to_numpy(float)

    cheap_model = bundle["cheap_router"]
    full_model = bundle["full_information_fusion"]

    train_cheap = cheap_model.predict_proba(
        train[bundle["cheap_features"]]
    )[:, 1]
    train_full = full_model.predict_proba(
        train[bundle["full_features"]]
    )[:, 1]
    sel_cheap = cheap_model.predict_proba(
        selection[bundle["cheap_features"]]
    )[:, 1]
    sel_full = full_model.predict_proba(
        selection[bundle["full_features"]]
    )[:, 1]

    candidates: list[dict] = []

    # 1. Cheapest monitor only.
    for threshold in threshold_grid(train["rule_score"]):
        pred = sel_rule >= threshold
        cost = policy_costs(len(selection), timing, rule=True)
        candidates.append(
            selection_record(
                family="cheapest_monitor_only",
                policy_id=f"rule_only_t_{threshold:.8g}",
                y=y_selection,
                pred=pred,
                cost=cost,
                details={
                    "monitor": "rule_filter_v1",
                    "threshold": float(threshold),
                },
                runtime={
                    "rule_call_rate": 1.0,
                    "compact_call_rate": 0.0,
                    "qwen_call_rate": 0.0,
                },
            )
        )

    # 2. Strongest single monitor.
    for threshold in threshold_grid(train["qwen_prompt_response_score"]):
        pred = sel_qwen >= threshold
        cost = policy_costs(
            len(selection), timing, qwen_always=True
        )
        candidates.append(
            selection_record(
                family="strongest_single_monitor",
                policy_id=f"qwen_only_t_{threshold:.8g}",
                y=y_selection,
                pred=pred,
                cost=cost,
                details={
                    "monitor": "qwen3guard_gen_4b_prompt_response",
                    "threshold": float(threshold),
                },
                runtime={
                    "rule_call_rate": 0.0,
                    "compact_call_rate": 0.0,
                    "qwen_call_rate": 1.0,
                },
            )
        )

    # 3. All monitors always on, prespecified max-score aggregation.
    train_all_max = np.maximum.reduce(
        [train_rule, train_compact, train_qwen]
    )
    sel_all_max = np.maximum.reduce([sel_rule, sel_compact, sel_qwen])
    for threshold in threshold_grid(pd.Series(train_all_max)):
        pred = sel_all_max >= threshold
        cost = policy_costs(
            len(selection),
            timing,
            rule=True,
            compact=True,
            qwen_always=True,
        )
        candidates.append(
            selection_record(
                family="all_monitors_always_on",
                policy_id=f"all_max_t_{threshold:.8g}",
                y=y_selection,
                pred=pred,
                cost=cost,
                details={
                    "aggregation": "max_score",
                    "monitors": [
                        "rule_filter_v1",
                        "koala_text_moderation",
                        "qwen3guard_gen_4b_prompt_response",
                    ],
                    "threshold": float(threshold),
                },
                runtime={
                    "rule_call_rate": 1.0,
                    "compact_call_rate": 1.0,
                    "qwen_call_rate": 1.0,
                },
            )
        )

    # 4. Fixed rule gate into Qwen.
    rule_gate = threshold_for_max_fpr(
        y_train, train_rule, 0.10
    )
    qwen_threshold = threshold_for_max_fpr(
        y_train, train_qwen, 0.05
    )
    call_qwen = sel_rule >= rule_gate
    pred = call_qwen & (sel_qwen >= qwen_threshold)
    cost = policy_costs(
        len(selection), timing, rule=True, qwen_mask=call_qwen
    )
    candidates.append(
        selection_record(
            family="fixed_cascade",
            policy_id="fixed_rule_gate_010_to_qwen_005",
            y=y_selection,
            pred=pred,
            cost=cost,
            details={
                "rule_gate_threshold": float(rule_gate),
                "qwen_threshold": float(qwen_threshold),
                "route": (
                    "run rule; if rule score reaches the training 10% FPR "
                    "gate, run Qwen; intercept at the training 5% FPR "
                    "Qwen threshold"
                ),
            },
            runtime={
                "rule_call_rate": 1.0,
                "compact_call_rate": 0.0,
                "qwen_call_rate": float(np.mean(call_qwen)),
            },
        )
    )

    # 5. Cost-tuned rule/compact/Qwen cascade.
    rule_low_grid = sorted(
        {
            float(np.quantile(train_rule, q))
            for q in [0.00, 0.05, 0.10, 0.25, 0.50, 0.75]
        }
    )
    compact_low_grid = sorted(
        {
            float(np.quantile(train_compact, q))
            for q in [0.00, 0.10, 0.25, 0.50, 0.75]
        }
    )
    rule_high_grid = [
        threshold_for_max_fpr(y_train, train_rule, fpr)
        for fpr in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]
    compact_high_grid = [
        threshold_for_max_fpr(y_train, train_compact, fpr)
        for fpr in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]
    qwen_grid = [
        threshold_for_max_fpr(y_train, train_qwen, fpr)
        for fpr in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]

    for rule_low in rule_low_grid:
        for rule_high in rule_high_grid:
            if rule_low >= rule_high:
                continue
            for compact_low in compact_low_grid:
                for compact_high in compact_high_grid:
                    if compact_low >= compact_high:
                        continue
                    for qwen_final in qwen_grid:
                        pred = np.zeros(len(selection), dtype=bool)

                        rule_intercept = sel_rule >= rule_high
                        rule_allow = sel_rule < rule_low
                        go_compact = ~(rule_intercept | rule_allow)

                        pred[rule_intercept] = True

                        compact_intercept = (
                            go_compact & (sel_compact >= compact_high)
                        )
                        compact_allow = (
                            go_compact & (sel_compact < compact_low)
                        )
                        go_qwen = (
                            go_compact
                            & ~(compact_intercept | compact_allow)
                        )

                        pred[compact_intercept] = True
                        pred[go_qwen] = sel_qwen[go_qwen] >= qwen_final

                        cost = policy_costs(
                            len(selection),
                            timing,
                            rule=True,
                            compact=False,
                        )
                        cost += (
                            go_compact.astype(float) * timing["compact"]
                        )
                        cost += (
                            go_qwen.astype(float)
                            * timing["qwen_prompt_response"]
                        )

                        candidates.append(
                            selection_record(
                                family="cost_tuned_cascade",
                                policy_id=(
                                    f"cascade_rl{rule_low:.4g}"
                                    f"_rh{rule_high:.4g}"
                                    f"_cl{compact_low:.4g}"
                                    f"_ch{compact_high:.4g}"
                                    f"_q{qwen_final:.4g}"
                                ),
                                y=y_selection,
                                pred=pred,
                                cost=cost,
                                details={
                                    "rule_low_allow_threshold": float(rule_low),
                                    "rule_high_intercept_threshold": float(
                                        rule_high
                                    ),
                                    "compact_low_allow_threshold": float(
                                        compact_low
                                    ),
                                    "compact_high_intercept_threshold": float(
                                        compact_high
                                    ),
                                    "qwen_threshold": float(qwen_final),
                                    "route": (
                                        "rule allow/intercept bands; compact "
                                        "allow/intercept bands; Qwen only for "
                                        "unresolved examples"
                                    ),
                                },
                                runtime={
                                    "rule_call_rate": 1.0,
                                    "compact_call_rate": float(
                                        np.mean(go_compact)
                                    ),
                                    "qwen_call_rate": float(np.mean(go_qwen)),
                                },
                            )
                        )

    # 6. Serialized learned router/full-fusion policy. No refitting.
    router_low_grid = sorted(
        {
            float(np.quantile(train_cheap, q))
            for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        }
    )
    router_high_grid = [
        threshold_for_max_fpr(y_train, train_cheap, fpr)
        for fpr in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]
    full_grid = [
        threshold_for_max_fpr(y_train, train_full, fpr)
        for fpr in [0.01, 0.03, 0.05, 0.10, 0.20]
    ]

    for low in router_low_grid:
        for high in router_high_grid:
            if low >= high:
                continue
            for full_threshold in full_grid:
                cheap_intercept = sel_cheap >= high
                cheap_allow = sel_cheap <= low
                call_qwen = ~(cheap_intercept | cheap_allow)

                pred = cheap_intercept.copy()
                pred[call_qwen] = (
                    sel_full[call_qwen] >= full_threshold
                )

                cost = policy_costs(
                    len(selection),
                    timing,
                    rule=True,
                    compact=True,
                    qwen_mask=call_qwen,
                )

                candidates.append(
                    selection_record(
                        family="learned_stacker_router",
                        policy_id=(
                            f"serialized_router_low{low:.4g}"
                            f"_high{high:.4g}"
                            f"_full{full_threshold:.4g}"
                        ),
                        y=y_selection,
                        pred=pred,
                        cost=cost,
                        details={
                            "cheap_features": bundle["cheap_features"],
                            "full_features": bundle["full_features"],
                            "router_low_allow_threshold": float(low),
                            "router_high_intercept_threshold": float(high),
                            "full_fusion_threshold": float(full_threshold),
                            "serialized_bundle": str(BUNDLE),
                            "route": (
                                "run rule and compact; allow/intercept when "
                                "the serialized cheap router is confident; "
                                "otherwise run Qwen and use the serialized "
                                "full fusion model"
                            ),
                        },
                        runtime={
                            "rule_call_rate": 1.0,
                            "compact_call_rate": 1.0,
                            "qwen_call_rate": float(np.mean(call_qwen)),
                        },
                    )
                )

    result = pd.DataFrame(candidates)
    result = (
        result.sort_values(
            ["family", "policy_id", "recall", "fpr", "avg_cost_ms"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(["family", "policy_id"], keep="first")
        .reset_index(drop=True)
    )
    if set(result["family"]) != REQUIRED_FAMILIES:
        raise SystemExit(
            f"Baseline family mismatch: {set(result['family'])}"
        )
    return result


def freeze_baselines(
    candidates: pd.DataFrame,
    timing: dict[str, float],
    timing_provenance: dict,
) -> dict:
    selected_rows = [
        choose_family_candidate(group)
        for _, group in candidates.groupby("family", sort=True)
    ]
    selected = pd.DataFrame(selected_rows).sort_values(
        ["recall", "avg_cost_ms", "fpr"],
        ascending=[False, True, True],
    )
    selected.to_csv(
        SELECTION_DIR / "selected_baselines_fpr05.csv",
        index=False,
    )

    primary = selected.iloc[0]
    policies = []
    for row in selected.to_dict(orient="records"):
        policies.append(
            {
                "family": row["family"],
                "policy_id": row["policy_id"],
                "frozen_role": (
                    "primary_budget_aware_policy"
                    if row["policy_id"] == primary["policy_id"]
                    else "required_baseline"
                ),
                "selection_metrics": {
                    key: (
                        None
                        if not np.isfinite(float(row[key]))
                        else float(row[key])
                    )
                    for key in [
                        "recall",
                        "fpr",
                        "precision",
                        "avg_cost_ms",
                        "median_cost_ms",
                        "p95_cost_ms",
                        "rule_call_rate",
                        "compact_call_rate",
                        "qwen_call_rate",
                    ]
                },
                "policy_details": json.loads(
                    row["policy_details_json"]
                ),
            }
        )

    frozen = {
        "artifact": "frozen_prespecified_policies_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "author_reviewed_labels_locked_not_risk_certified",
        "freeze_rule": (
            "Within each prespecified family, select the highest-recall "
            "candidate on policy_selection subject to FPR <= 0.05; ties "
            "prefer lower controlled estimated cost, then lower FPR. "
            "The overall primary is chosen by the same ordering."
        ),
        "primary_policy_family": str(primary["family"]),
        "primary_policy_id": str(primary["policy_id"]),
        "required_families": sorted(REQUIRED_FAMILIES),
        "allowed_selection_data": [TRAIN_SPLIT, SELECTION_SPLIT],
        "forbidden_for_policy_selection": EVAL_SPLITS,
        "cache_path": str(CACHE),
        "cache_sha256": sha256(CACHE),
        "bundle_path": str(BUNDLE),
        "bundle_sha256": sha256(BUNDLE),
        "controlled_timing_p50_ms": timing,
        "controlled_timing_provenance": timing_provenance,
        "policies": policies,
    }
    (FROZEN_DIR / "frozen_policies.json").write_text(
        json.dumps(frozen, indent=2),
        encoding="utf-8",
    )
    return frozen


def apply_frozen_policy(
    policy: dict,
    frame: pd.DataFrame,
    bundle: dict,
    timing: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    family = policy["family"]
    details = policy["policy_details"]
    n = len(frame)

    rule = frame["rule_score"].to_numpy(float)
    compact = frame["compact_unsafe_score"].to_numpy(float)
    qwen = frame["qwen_prompt_response_score"].to_numpy(float)

    runtime = {
        "rule_call_rate": 0.0,
        "compact_call_rate": 0.0,
        "qwen_call_rate": 0.0,
    }

    if family == "cheapest_monitor_only":
        pred = rule >= float(details["threshold"])
        cost = policy_costs(n, timing, rule=True)
        runtime["rule_call_rate"] = 1.0

    elif family == "strongest_single_monitor":
        pred = qwen >= float(details["threshold"])
        cost = policy_costs(n, timing, qwen_always=True)
        runtime["qwen_call_rate"] = 1.0

    elif family == "all_monitors_always_on":
        score = np.maximum.reduce([rule, compact, qwen])
        pred = score >= float(details["threshold"])
        cost = policy_costs(
            n, timing, rule=True, compact=True, qwen_always=True
        )
        runtime = {
            "rule_call_rate": 1.0,
            "compact_call_rate": 1.0,
            "qwen_call_rate": 1.0,
        }

    elif family == "fixed_cascade":
        call_qwen = rule >= float(details["rule_gate_threshold"])
        pred = call_qwen & (
            qwen >= float(details["qwen_threshold"])
        )
        cost = policy_costs(
            n, timing, rule=True, qwen_mask=call_qwen
        )
        runtime["rule_call_rate"] = 1.0
        runtime["qwen_call_rate"] = float(np.mean(call_qwen))

    elif family == "cost_tuned_cascade":
        rule_intercept = (
            rule >= float(details["rule_high_intercept_threshold"])
        )
        rule_allow = (
            rule < float(details["rule_low_allow_threshold"])
        )
        go_compact = ~(rule_intercept | rule_allow)

        compact_intercept = go_compact & (
            compact >= float(
                details["compact_high_intercept_threshold"]
            )
        )
        compact_allow = go_compact & (
            compact < float(details["compact_low_allow_threshold"])
        )
        go_qwen = go_compact & ~(
            compact_intercept | compact_allow
        )

        pred = np.zeros(n, dtype=bool)
        pred[rule_intercept] = True
        pred[compact_intercept] = True
        pred[go_qwen] = (
            qwen[go_qwen] >= float(details["qwen_threshold"])
        )

        cost = policy_costs(n, timing, rule=True)
        cost += go_compact.astype(float) * timing["compact"]
        cost += (
            go_qwen.astype(float)
            * timing["qwen_prompt_response"]
        )
        runtime = {
            "rule_call_rate": 1.0,
            "compact_call_rate": float(np.mean(go_compact)),
            "qwen_call_rate": float(np.mean(go_qwen)),
        }

    elif family == "learned_stacker_router":
        cheap_probability = bundle["cheap_router"].predict_proba(
            frame[bundle["cheap_features"]]
        )[:, 1]
        full_probability = bundle[
            "full_information_fusion"
        ].predict_proba(
            frame[bundle["full_features"]]
        )[:, 1]

        low = float(details["router_low_allow_threshold"])
        high = float(details["router_high_intercept_threshold"])
        final_threshold = float(details["full_fusion_threshold"])

        cheap_allow = cheap_probability <= low
        cheap_intercept = cheap_probability >= high
        call_qwen = ~(cheap_allow | cheap_intercept)

        pred = cheap_intercept.copy()
        pred[call_qwen] = (
            full_probability[call_qwen] >= final_threshold
        )
        cost = policy_costs(
            n,
            timing,
            rule=True,
            compact=True,
            qwen_mask=call_qwen,
        )
        runtime = {
            "rule_call_rate": 1.0,
            "compact_call_rate": 1.0,
            "qwen_call_rate": float(np.mean(call_qwen)),
        }

    else:
        raise ValueError(f"Unknown policy family: {family}")

    return pred.astype(bool), cost, runtime


def evaluate_frozen_baselines(
    cache: pd.DataFrame,
    bundle: dict,
    frozen: dict,
    timing: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []

    for split in EVAL_SPLITS:
        frame = cache[cache["split"].eq(split)].copy()
        y = frame["y"].to_numpy(int)

        for policy in frozen["policies"]:
            pred, cost, runtime = apply_frozen_policy(
                policy, frame, bundle, timing
            )
            result = finite_sample_metrics(y, pred)
            metric_rows.append(
                {
                    "split": split,
                    "family": policy["family"],
                    "policy_id": policy["policy_id"],
                    "frozen_role": policy["frozen_role"],
                    **result,
                    "avg_estimated_cost_ms": float(np.mean(cost)),
                    "median_estimated_cost_ms": float(np.median(cost)),
                    "p95_estimated_cost_ms": float(
                        np.quantile(cost, 0.95)
                    ),
                    **runtime,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "example_id": frame["example_id"].astype(str),
                        "split": split,
                        "y": y,
                        "family": policy["family"],
                        "policy_id": policy["policy_id"],
                        "intercept_pred": pred.astype(int),
                        "estimated_cost_ms": cost,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions


def select_full_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    candidates = np.unique(
        np.concatenate(
            [
                probability,
                [0.0, 1.0, np.nextafter(probability.max(), np.inf)],
            ]
        )
    )
    feasible = []
    for threshold in candidates:
        result = binary_metrics(y, probability >= threshold)
        if result["fpr"] <= target_fpr + 1e-12:
            precision = (
                float(result["precision"])
                if np.isfinite(result["precision"])
                else -1.0
            )
            feasible.append(
                (
                    float(result["recall"]),
                    precision,
                    -float(result["fpr"]),
                    float(threshold),
                )
            )
    if not feasible:
        raise RuntimeError(
            f"No full-information threshold for target {target_fpr}"
        )
    best = max(feasible)
    return {
        "threshold": best[3],
        "selection_recall": best[0],
        "selection_fpr": -best[2],
        "selection_precision": best[1],
    }


def select_selective_policy(
    y: np.ndarray,
    cheap_probability: np.ndarray,
    full_probability: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    low_grid = np.unique(
        np.quantile(cheap_probability, np.linspace(0.02, 0.75, 18))
    )
    high_grid = np.unique(
        np.quantile(cheap_probability, np.linspace(0.25, 0.98, 18))
    )
    final_grid = np.unique(
        np.concatenate(
            [
                np.quantile(
                    full_probability, np.linspace(0.0, 1.0, 121)
                ),
                [
                    0.0,
                    1.0,
                    np.nextafter(full_probability.max(), np.inf),
                ],
            ]
        )
    )

    best_key = None
    best_payload = None

    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue
            cheap_allow = cheap_probability <= low
            cheap_intercept = cheap_probability >= high
            call_qwen = ~(cheap_allow | cheap_intercept)

            for final_threshold in final_grid:
                pred = cheap_intercept.copy()
                pred[call_qwen] = (
                    full_probability[call_qwen] >= final_threshold
                )
                result = binary_metrics(y, pred)
                if result["fpr"] > target_fpr + 1e-12:
                    continue

                qwen_call_rate = float(np.mean(call_qwen))
                precision = (
                    float(result["precision"])
                    if np.isfinite(result["precision"])
                    else -1.0
                )
                key = (
                    float(result["recall"]),
                    -qwen_call_rate,
                    precision,
                    -float(result["fpr"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        "router_low_allow_threshold": float(low),
                        "router_high_intercept_threshold": float(high),
                        "full_fusion_threshold": float(final_threshold),
                        "selection_recall": float(result["recall"]),
                        "selection_fpr": float(result["fpr"]),
                        "selection_precision": float(
                            result["precision"]
                        ),
                        "selection_qwen_call_rate": qwen_call_rate,
                    }

    if best_payload is None:
        raise RuntimeError(
            f"No selective policy for target {target_fpr}"
        )
    return best_payload


def stratified_paired_bootstrap(
    *,
    y: np.ndarray,
    full_pred: np.ndarray,
    selective_pred: np.ndarray,
    selective_call: np.ndarray,
    full_cost: np.ndarray,
    selective_cost: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    recall_diff = np.empty(BOOTSTRAP_REPS)
    fpr_diff = np.empty(BOOTSTRAP_REPS)
    accuracy_diff = np.empty(BOOTSTRAP_REPS)
    call_reduction = np.empty(BOOTSTRAP_REPS)
    cost_reduction = np.empty(BOOTSTRAP_REPS)

    for i in range(BOOTSTRAP_REPS):
        sampled_pos = rng.choice(pos, size=len(pos), replace=True)
        sampled_neg = rng.choice(neg, size=len(neg), replace=True)
        index = np.concatenate([sampled_pos, sampled_neg])

        y_i = y[index]
        full_i = full_pred[index]
        selective_i = selective_pred[index]

        recall_diff[i] = (
            np.mean(selective_i[y_i == 1])
            - np.mean(full_i[y_i == 1])
        )
        fpr_diff[i] = (
            np.mean(selective_i[y_i == 0])
            - np.mean(full_i[y_i == 0])
        )
        accuracy_diff[i] = (
            np.mean(selective_i == y_i)
            - np.mean(full_i == y_i)
        )
        call_reduction[i] = 1.0 - np.mean(selective_call[index])
        cost_reduction[i] = 1.0 - (
            np.mean(selective_cost[index])
            / np.mean(full_cost[index])
        )

    def interval(values: np.ndarray) -> tuple[float, float]:
        return (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )

    recall_low, recall_high = interval(recall_diff)
    fpr_low, fpr_high = interval(fpr_diff)
    accuracy_low, accuracy_high = interval(accuracy_diff)
    call_low, call_high = interval(call_reduction)
    cost_low, cost_high = interval(cost_reduction)

    return {
        "recall_diff_ci95_low": recall_low,
        "recall_diff_ci95_high": recall_high,
        "fpr_diff_ci95_low": fpr_low,
        "fpr_diff_ci95_high": fpr_high,
        "accuracy_diff_ci95_low": accuracy_low,
        "accuracy_diff_ci95_high": accuracy_high,
        "qwen_call_reduction_ci95_low": call_low,
        "qwen_call_reduction_ci95_high": call_high,
        "estimated_cost_reduction_ci95_low": cost_low,
        "estimated_cost_reduction_ci95_high": cost_high,
    }


def fusion_comparison(
    cache: pd.DataFrame,
    bundle: dict,
    timing: dict[str, float],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = cache[
        cache["split"].eq(SELECTION_SPLIT)
    ].copy()
    y_selection = selection["y"].to_numpy(int)

    cheap_selection = bundle["cheap_router"].predict_proba(
        selection[bundle["cheap_features"]]
    )[:, 1]
    full_selection = bundle[
        "full_information_fusion"
    ].predict_proba(
        selection[bundle["full_features"]]
    )[:, 1]

    frozen = {
        "artifact": "fusion_operating_points_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "author_reviewed_labels_locked_not_risk_certified",
        "selection_split": SELECTION_SPLIT,
        "target_fprs": COMMON_TARGET_FPRS,
        "bundle_path": str(BUNDLE),
        "bundle_sha256": sha256(BUNDLE),
        "operating_points": {},
    }

    selection_rows = []
    for target in COMMON_TARGET_FPRS:
        full_op = select_full_threshold(
            y_selection, full_selection, target
        )
        selective_op = select_selective_policy(
            y_selection,
            cheap_selection,
            full_selection,
            target,
        )
        key = f"{target:.3f}"
        frozen["operating_points"][key] = {
            "target_fpr": target,
            "full_information": full_op,
            "selective_acquisition": selective_op,
        }
        selection_rows.extend(
            [
                {
                    "target_fpr": target,
                    "policy": "full_information_always_on",
                    **full_op,
                    "selection_qwen_call_rate": 1.0,
                },
                {
                    "target_fpr": target,
                    "policy": "selective_acquisition",
                    **selective_op,
                },
            ]
        )

    metric_rows = []
    bootstrap_rows = []
    prediction_frames = []

    full_constant_cost = (
        timing["rule"]
        + timing["compact"]
        + timing["qwen_prompt_response"]
    )

    for split in EVAL_SPLITS:
        frame = cache[cache["split"].eq(split)].copy()
        y = frame["y"].to_numpy(int)

        cheap_probability = bundle["cheap_router"].predict_proba(
            frame[bundle["cheap_features"]]
        )[:, 1]
        full_probability = bundle[
            "full_information_fusion"
        ].predict_proba(
            frame[bundle["full_features"]]
        )[:, 1]

        for target in COMMON_TARGET_FPRS:
            op = frozen["operating_points"][f"{target:.3f}"]
            full_threshold = float(
                op["full_information"]["threshold"]
            )
            selective = op["selective_acquisition"]

            full_pred = full_probability >= full_threshold

            low = float(
                selective["router_low_allow_threshold"]
            )
            high = float(
                selective["router_high_intercept_threshold"]
            )
            final_threshold = float(
                selective["full_fusion_threshold"]
            )

            cheap_allow = cheap_probability <= low
            cheap_intercept = cheap_probability >= high
            call_qwen = ~(cheap_allow | cheap_intercept)

            selective_pred = cheap_intercept.copy()
            selective_pred[call_qwen] = (
                full_probability[call_qwen] >= final_threshold
            )

            full_cost = np.full(len(frame), full_constant_cost)
            selective_cost = policy_costs(
                len(frame),
                timing,
                rule=True,
                compact=True,
                qwen_mask=call_qwen,
            )

            full_metrics = finite_sample_metrics(y, full_pred)
            selective_metrics = finite_sample_metrics(
                y, selective_pred
            )

            metric_rows.extend(
                [
                    {
                        "split": split,
                        "target_fpr": target,
                        "policy": "full_information_always_on",
                        **full_metrics,
                        "qwen_call_rate": 1.0,
                        "avg_estimated_cost_ms": float(
                            np.mean(full_cost)
                        ),
                        "estimated_cost_reduction": 0.0,
                    },
                    {
                        "split": split,
                        "target_fpr": target,
                        "policy": "selective_acquisition",
                        **selective_metrics,
                        "qwen_call_rate": float(
                            np.mean(call_qwen)
                        ),
                        "avg_estimated_cost_ms": float(
                            np.mean(selective_cost)
                        ),
                        "estimated_cost_reduction": float(
                            1.0
                            - np.mean(selective_cost)
                            / np.mean(full_cost)
                        ),
                    },
                ]
            )

            paired = stratified_paired_bootstrap(
                y=y,
                full_pred=full_pred,
                selective_pred=selective_pred,
                selective_call=call_qwen,
                full_cost=full_cost,
                selective_cost=selective_cost,
                seed=(
                    RNG_SEED
                    + int(target * 1000)
                    + sum(ord(char) for char in split)
                ),
            )
            bootstrap_rows.append(
                {
                    "split": split,
                    "target_fpr": target,
                    "recall_diff_selective_minus_full": (
                        selective_metrics["recall"]
                        - full_metrics["recall"]
                    ),
                    "fpr_diff_selective_minus_full": (
                        selective_metrics["fpr"]
                        - full_metrics["fpr"]
                    ),
                    "accuracy_diff_selective_minus_full": (
                        selective_metrics["accuracy"]
                        - full_metrics["accuracy"]
                    ),
                    "qwen_call_reduction": float(
                        1.0 - np.mean(call_qwen)
                    ),
                    "estimated_cost_reduction": float(
                        1.0
                        - np.mean(selective_cost)
                        / np.mean(full_cost)
                    ),
                    **paired,
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "example_id": frame["example_id"].astype(str),
                        "split": split,
                        "y": y,
                        "target_fpr": target,
                        "full_probability": full_probability,
                        "cheap_probability": cheap_probability,
                        "full_intercept_pred": full_pred.astype(int),
                        "selective_intercept_pred": (
                            selective_pred.astype(int)
                        ),
                        "selective_qwen_called": call_qwen.astype(int),
                        "full_estimated_cost_ms": full_cost,
                        "selective_estimated_cost_ms": selective_cost,
                    }
                )
            )

    return (
        frozen,
        pd.DataFrame(selection_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(bootstrap_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def main() -> None:
    for path in [CACHE, BUNDLE]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    cache = pd.read_parquet(CACHE).copy()
    cache["example_id"] = cache["example_id"].astype(str)
    bundle = joblib.load(BUNDLE)

    if len(cache) != 2159 or cache["example_id"].nunique() != 2159:
        raise SystemExit("Cache v3 row/ID validation failed")
    if bundle["provenance"]["cache_sha256"] != sha256(CACHE):
        raise SystemExit("Bundle was not trained from current cache v3")
    if (
        bundle["provenance"]["status"]
        != "author_reviewed_labels_locked"
    ):
        raise SystemExit("Bundle status is not final author-reviewed")
    if bundle["cheap_features"] != CHEAP_FEATURES:
        raise SystemExit("Unexpected cheap feature order")
    if bundle["full_features"] != FULL_FEATURES:
        raise SystemExit("Unexpected full feature order")

    expected_counts = {
        "policy_train": 844,
        "policy_selection": 421,
        "calibration": 422,
        "final_test": 422,
        "held_out_shift": 50,
    }
    actual_counts = cache["split"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise SystemExit(
            f"Unexpected split counts: {actual_counts}"
        )

    timing, timing_provenance = controlled_timing_medians()
    print("controlled timing p50 ms:", timing)
    print(
        "timing fallback used for:",
        timing_provenance["fallback_used_for"],
    )

    train = cache[cache["split"].eq(TRAIN_SPLIT)].copy()
    selection = cache[
        cache["split"].eq(SELECTION_SPLIT)
    ].copy()

    for directory in [
        SELECTION_DIR,
        FROZEN_DIR,
        EVALUATION_DIR,
        FUSION_DIR,
        OPERATING_POINTS.parent,
        MANIFEST.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    candidates = build_baseline_candidates(
        train, selection, bundle, timing
    )
    candidates.to_csv(
        SELECTION_DIR / "all_candidate_policies.csv",
        index=False,
    )
    frozen = freeze_baselines(
        candidates, timing, timing_provenance
    )

    selected = pd.read_csv(
        SELECTION_DIR / "selected_baselines_fpr05.csv"
    )

    selection_manifest = {
        "artifact": "policy_selection_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "author_reviewed_labels_locked_not_risk_certified",
        "input_cache": str(CACHE),
        "input_cache_sha256": sha256(CACHE),
        "serialized_bundle": str(BUNDLE),
        "serialized_bundle_sha256": sha256(BUNDLE),
        "training_split": TRAIN_SPLIT,
        "selection_split": SELECTION_SPLIT,
        "calibration_final_and_shift_splits_used": False,
        "required_families": sorted(REQUIRED_FAMILIES),
        "candidate_count": int(len(candidates)),
        "selected_policy_count": int(len(selected)),
        "target_fpr": BASELINE_TARGET_FPR,
        "learned_models_refit_during_selection": False,
        "controlled_timing_p50_ms": timing,
        "controlled_timing_provenance": timing_provenance,
    }
    (SELECTION_DIR / "policy_selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2),
        encoding="utf-8",
    )

    baseline_metrics, baseline_predictions = (
        evaluate_frozen_baselines(
            cache, bundle, frozen, timing
        )
    )
    baseline_metrics.to_csv(
        EVALUATION_DIR / "metrics_by_split.csv",
        index=False,
    )
    baseline_predictions.to_csv(
        EVALUATION_DIR / "predictions.csv",
        index=False,
    )

    calibration_bounds = baseline_metrics[
        baseline_metrics["split"].eq("calibration")
    ].copy()
    calibration_bounds.to_csv(
        EVALUATION_DIR / "calibration_diagnostic_bounds.csv",
        index=False,
    )

    fusion_result = fusion_comparison(cache, bundle, timing)
    (
        frozen_operating_points,
        selected_operating_points,
        fusion_metrics,
        fusion_bootstrap,
        fusion_predictions,
    ) = fusion_result

    OPERATING_POINTS.write_text(
        json.dumps(frozen_operating_points, indent=2),
        encoding="utf-8",
    )
    selected_operating_points.to_csv(
        FUSION_DIR / "selected_operating_points.csv",
        index=False,
    )
    fusion_metrics.to_csv(
        FUSION_DIR / "metrics_by_split.csv",
        index=False,
    )
    fusion_bootstrap.to_csv(
        FUSION_DIR / "paired_bootstrap.csv",
        index=False,
    )
    fusion_predictions.to_csv(
        FUSION_DIR / "predictions.csv",
        index=False,
    )

    methodology = """# Policy evaluation v3 methodology

This evaluation uses the final author-reviewed labels and the complete
serialized v3 fusion pipelines.

## Data-use boundaries

- `policy_train`: model fitting and train-derived candidate grids.
- `policy_selection`: threshold and policy selection.
- `calibration`, `final_test`, `held_out_shift`: evaluation only.
- No model is refit during policy selection or evaluation.

## Prespecified policy families

1. Cheapest monitor only.
2. Strongest single monitor.
3. All monitors always on with max-score aggregation.
4. Fixed rule-to-Qwen cascade.
5. Cost-tuned rule/compact/Qwen cascade.
6. Serialized learned cheap-router/full-fusion policy.

## Full-information comparison

Full-information always-on fusion and selective expensive-monitor acquisition
are compared at common selection target FPRs of 1%, 2.5%, 5%, and 10%.
Paired stratified bootstrap confidence intervals use 2,000 replicates.

## Cost

Estimated policy cost uses component p50 values from the controlled,
CUDA-synchronized T4 benchmark. These are not replacements for the separate
measured end-to-end policy timing benchmark.

## Statistical scope

The exact binomial intervals and paired bootstrap intervals are descriptive
evaluation intervals. They are not a formal Neyman-Pearson or
Learn-then-Test risk certificate. Formal risk control is a later,
separate professor requirement.
"""
    (EVALUATION_DIR / "methodology.md").write_text(
        methodology,
        encoding="utf-8",
    )

    manifest = {
        "artifact": "policy_evaluation_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_not_risk_certified",
        "cache": {
            "path": str(CACHE),
            "sha256": sha256(CACHE),
            "rows": int(len(cache)),
            "unique_example_id": int(
                cache["example_id"].nunique()
            ),
            "split_counts": {
                key: int(value)
                for key, value in actual_counts.items()
            },
        },
        "bundle": {
            "path": str(BUNDLE),
            "sha256": sha256(BUNDLE),
            "status": bundle["provenance"]["status"],
        },
        "selection": {
            "training_split": TRAIN_SPLIT,
            "selection_split": SELECTION_SPLIT,
            "evaluation_splits_used": False,
            "learned_models_refit": False,
        },
        "prespecified_families": sorted(REQUIRED_FAMILIES),
        "baseline_target_fpr": BASELINE_TARGET_FPR,
        "common_fusion_target_fprs": COMMON_TARGET_FPRS,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": RNG_SEED,
        "controlled_timing": timing_provenance,
        "risk_control_status": (
            "not certified; formal risk control is a later stage"
        ),
        "outputs": {},
    }

    output_paths = [
        SELECTION_DIR / "all_candidate_policies.csv",
        SELECTION_DIR / "selected_baselines_fpr05.csv",
        SELECTION_DIR / "policy_selection_manifest.json",
        FROZEN_DIR / "frozen_policies.json",
        EVALUATION_DIR / "metrics_by_split.csv",
        EVALUATION_DIR / "predictions.csv",
        EVALUATION_DIR / "calibration_diagnostic_bounds.csv",
        EVALUATION_DIR / "methodology.md",
        OPERATING_POINTS,
        FUSION_DIR / "selected_operating_points.csv",
        FUSION_DIR / "metrics_by_split.csv",
        FUSION_DIR / "paired_bootstrap.csv",
        FUSION_DIR / "predictions.csv",
    ]
    manifest["outputs"] = {
        str(path): sha256(path) for path in output_paths
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== SELECTED PRESPECIFIED POLICIES ===")
    print(
        selected[
            [
                "family",
                "policy_id",
                "recall",
                "fpr",
                "avg_cost_ms",
                "qwen_call_rate",
            ]
        ].to_string(index=False)
    )

    print("\n=== BASELINE METRICS ===")
    print(
        baseline_metrics[
            [
                "split",
                "family",
                "recall",
                "fpr",
                "fpr_one_sided95_upper",
                "avg_estimated_cost_ms",
                "qwen_call_rate",
            ]
        ].to_string(index=False)
    )

    print("\n=== FULL VS SELECTIVE AT TARGET FPR 0.05 ===")
    print(
        fusion_metrics[
            fusion_metrics["target_fpr"].eq(0.05)
        ][
            [
                "split",
                "policy",
                "recall",
                "fpr",
                "qwen_call_rate",
                "avg_estimated_cost_ms",
                "estimated_cost_reduction",
            ]
        ].to_string(index=False)
    )

    print("\n=== PAIRED BOOTSTRAP AT TARGET FPR 0.05 ===")
    print(
        fusion_bootstrap[
            fusion_bootstrap["target_fpr"].eq(0.05)
        ].to_string(index=False)
    )

    print("\nmanifest:", MANIFEST)


if __name__ == "__main__":
    main()
