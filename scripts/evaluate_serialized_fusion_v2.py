#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CACHE = Path("data/processed/monitor_score_cache_v2.parquet")
BUNDLE = Path("artifacts/fusion_models_v2/fusion_bundle.joblib")
OUT_DIR = Path("reports/fusion_comparison_v2")
OPERATING_POINTS = Path("artifacts/fusion_models_v2/frozen_operating_points.json")
MANIFEST = Path("data/metadata/fusion_comparison_v2_manifest.json")

SELECTION_SPLIT = "policy_selection"
EVAL_SPLITS = ["calibration", "final_test", "held_out_shift"]
TARGET_FPRS = [0.01, 0.025, 0.05, 0.10]
BOOTSTRAP_REPS = 2000
RNG_SEED = 20260711


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = y.astype(int)
    pred = pred.astype(bool)

    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))

    return {
        "n": int(len(y)),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
    }


def select_full_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    candidates = np.unique(
        np.concatenate(
            [
                probability,
                [0.0, 1.0, np.nextafter(float(np.max(probability)), np.inf)],
            ]
        )
    )

    feasible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        m = metrics(y, probability >= threshold)
        if m["fpr"] <= target_fpr + 1e-12:
            feasible.append(
                (
                    float(m["recall"]),
                    float(m["precision"])
                    if np.isfinite(m["precision"])
                    else -1.0,
                    -float(m["fpr"]),
                    float(threshold),
                )
            )

    if not feasible:
        raise RuntimeError(f"No full-information threshold for FPR {target_fpr}")

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
                np.quantile(full_probability, np.linspace(0.0, 1.0, 121)),
                [
                    0.0,
                    1.0,
                    np.nextafter(float(np.max(full_probability)), np.inf),
                ],
            ]
        )
    )

    best_key = None
    best_payload = None

    for low in low_grid:
        for high in high_grid:
            if not low < high:
                continue

            cheap_allow = cheap_probability <= low
            cheap_intercept = cheap_probability >= high
            call_expensive = ~(cheap_allow | cheap_intercept)

            for final_threshold in final_grid:
                pred = cheap_intercept.copy()
                pred[call_expensive] = (
                    full_probability[call_expensive] >= final_threshold
                )

                m = metrics(y, pred)
                if m["fpr"] > target_fpr + 1e-12:
                    continue

                judge_call_rate = float(np.mean(call_expensive))
                precision = (
                    float(m["precision"])
                    if np.isfinite(m["precision"])
                    else -1.0
                )

                # Maximize recall, then minimize expensive-monitor use,
                # then maximize precision, then minimize FPR.
                key = (
                    float(m["recall"]),
                    -judge_call_rate,
                    precision,
                    -float(m["fpr"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        "router_low_allow_threshold": float(low),
                        "router_high_intercept_threshold": float(high),
                        "full_fusion_threshold": float(final_threshold),
                        "selection_recall": float(m["recall"]),
                        "selection_fpr": float(m["fpr"]),
                        "selection_precision": float(m["precision"]),
                        "selection_expensive_call_rate": judge_call_rate,
                    }

    if best_payload is None:
        raise RuntimeError(f"No selective policy for FPR {target_fpr}")

    return best_payload


def stratified_paired_bootstrap(
    y: np.ndarray,
    full_pred: np.ndarray,
    selective_pred: np.ndarray,
    selective_call: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    if len(pos) == 0 or len(neg) == 0:
        return {
            "recall_diff_ci_low": float("nan"),
            "recall_diff_ci_high": float("nan"),
            "fpr_diff_ci_low": float("nan"),
            "fpr_diff_ci_high": float("nan"),
            "accuracy_diff_ci_low": float("nan"),
            "accuracy_diff_ci_high": float("nan"),
            "expensive_call_reduction_ci_low": float("nan"),
            "expensive_call_reduction_ci_high": float("nan"),
        }

    recall_diff = np.empty(reps)
    fpr_diff = np.empty(reps)
    accuracy_diff = np.empty(reps)
    call_reduction = np.empty(reps)

    for i in range(reps):
        pos_sample = rng.choice(pos, size=len(pos), replace=True)
        neg_sample = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])

        recall_diff[i] = (
            np.mean(selective_pred[pos_sample])
            - np.mean(full_pred[pos_sample])
        )
        fpr_diff[i] = (
            np.mean(selective_pred[neg_sample])
            - np.mean(full_pred[neg_sample])
        )
        accuracy_diff[i] = (
            np.mean(selective_pred[idx] == y[idx])
            - np.mean(full_pred[idx] == y[idx])
        )
        call_reduction[i] = 1.0 - np.mean(selective_call[idx])

    def q(values: np.ndarray, p: float) -> float:
        return float(np.quantile(values, p))

    return {
        "recall_diff_ci_low": q(recall_diff, 0.025),
        "recall_diff_ci_high": q(recall_diff, 0.975),
        "fpr_diff_ci_low": q(fpr_diff, 0.025),
        "fpr_diff_ci_high": q(fpr_diff, 0.975),
        "accuracy_diff_ci_low": q(accuracy_diff, 0.025),
        "accuracy_diff_ci_high": q(accuracy_diff, 0.975),
        "expensive_call_reduction_ci_low": q(call_reduction, 0.025),
        "expensive_call_reduction_ci_high": q(call_reduction, 0.975),
    }


for path in [CACHE, BUNDLE]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")

cache = pd.read_parquet(CACHE).copy()
bundle = joblib.load(BUNDLE)

expected_cache_hash = bundle["provenance"]["cache_sha256"]
actual_cache_hash = sha256(CACHE)
if expected_cache_hash != actual_cache_hash:
    raise SystemExit(
        "Serialized bundle was not trained from the current repaired cache"
    )

cheap_features = bundle["cheap_features"]
full_features = bundle["full_features"]
cheap_model = bundle["cheap_router"]
full_model = bundle["full_information_fusion"]

selection = cache[cache["split"].eq(SELECTION_SPLIT)].copy()
if selection.empty:
    raise SystemExit("Selection split is empty")

y_selection = selection["y"].to_numpy(dtype=int)
cheap_selection = cheap_model.predict_proba(
    selection[cheap_features]
)[:, 1]
full_selection = full_model.predict_proba(
    selection[full_features]
)[:, 1]

frozen = {
    "artifact": "fusion_operating_points_v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "provisional_labels_pending_author_review",
    "selection_split": SELECTION_SPLIT,
    "target_fprs": TARGET_FPRS,
    "bundle_path": str(BUNDLE),
    "bundle_sha256": sha256(BUNDLE),
    "operating_points": {},
}

selection_rows = []
for target in TARGET_FPRS:
    full_op = select_full_threshold(
        y_selection,
        full_selection,
        target,
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

    selection_rows.append(
        {
            "target_fpr": target,
            "policy": "full_information_always_on",
            **full_op,
            "selection_expensive_call_rate": 1.0,
        }
    )
    selection_rows.append(
        {
            "target_fpr": target,
            "policy": "selective_acquisition",
            **selective_op,
        }
    )

OPERATING_POINTS.write_text(
    json.dumps(frozen, indent=2),
    encoding="utf-8",
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
selection_df = pd.DataFrame(selection_rows)
selection_df.to_csv(
    OUT_DIR / "selected_operating_points.csv",
    index=False,
)

metric_rows: list[dict[str, object]] = []
bootstrap_rows: list[dict[str, object]] = []
prediction_frames: list[pd.DataFrame] = []

for split_name in EVAL_SPLITS:
    frame = cache[cache["split"].eq(split_name)].copy()
    if frame.empty:
        raise SystemExit(f"Evaluation split is empty: {split_name}")

    y = frame["y"].to_numpy(dtype=int)
    cheap_prob = cheap_model.predict_proba(frame[cheap_features])[:, 1]
    full_prob = full_model.predict_proba(frame[full_features])[:, 1]

    for target in TARGET_FPRS:
        op = frozen["operating_points"][f"{target:.3f}"]
        full_threshold = float(
            op["full_information"]["threshold"]
        )
        selective = op["selective_acquisition"]

        full_pred = full_prob >= full_threshold

        low = float(selective["router_low_allow_threshold"])
        high = float(selective["router_high_intercept_threshold"])
        final_threshold = float(selective["full_fusion_threshold"])

        cheap_allow = cheap_prob <= low
        cheap_intercept = cheap_prob >= high
        call_expensive = ~(cheap_allow | cheap_intercept)

        selective_pred = cheap_intercept.copy()
        selective_pred[call_expensive] = (
            full_prob[call_expensive] >= final_threshold
        )

        full_m = metrics(y, full_pred)
        selective_m = metrics(y, selective_pred)

        metric_rows.append(
            {
                "split": split_name,
                "target_fpr": target,
                "policy": "full_information_always_on",
                **full_m,
                "expensive_call_rate": 1.0,
                "expensive_call_reduction": 0.0,
            }
        )
        metric_rows.append(
            {
                "split": split_name,
                "target_fpr": target,
                "policy": "selective_acquisition",
                **selective_m,
                "expensive_call_rate": float(np.mean(call_expensive)),
                "expensive_call_reduction": float(
                    1.0 - np.mean(call_expensive)
                ),
            }
        )

        paired = stratified_paired_bootstrap(
            y=y,
            full_pred=full_pred,
            selective_pred=selective_pred,
            selective_call=call_expensive,
            reps=BOOTSTRAP_REPS,
            seed=RNG_SEED
            + int(target * 1000)
            + sum(ord(c) for c in split_name),
        )

        bootstrap_rows.append(
            {
                "split": split_name,
                "target_fpr": target,
                "recall_diff_selective_minus_full": (
                    selective_m["recall"] - full_m["recall"]
                ),
                "fpr_diff_selective_minus_full": (
                    selective_m["fpr"] - full_m["fpr"]
                ),
                "accuracy_diff_selective_minus_full": (
                    selective_m["accuracy"] - full_m["accuracy"]
                ),
                "expensive_call_reduction": float(
                    1.0 - np.mean(call_expensive)
                ),
                **paired,
            }
        )

        pred_frame = frame[
            [
                "example_id",
                "split",
                "y",
                "source_dataset",
                "attack_family",
            ]
        ].copy()
        pred_frame["target_fpr"] = target
        pred_frame["cheap_router_probability"] = cheap_prob
        pred_frame["full_information_probability"] = full_prob
        pred_frame["full_information_pred"] = full_pred.astype(int)
        pred_frame["selective_pred"] = selective_pred.astype(int)
        pred_frame["selective_called_expensive"] = (
            call_expensive.astype(int)
        )
        prediction_frames.append(pred_frame)

metrics_df = pd.DataFrame(metric_rows)
bootstrap_df = pd.DataFrame(bootstrap_rows)
predictions_df = pd.concat(prediction_frames, ignore_index=True)

metrics_path = OUT_DIR / "evaluation_metrics.csv"
bootstrap_path = OUT_DIR / "paired_bootstrap_differences.csv"
predictions_path = OUT_DIR / "locked_predictions.parquet"

metrics_df.to_csv(metrics_path, index=False)
bootstrap_df.to_csv(bootstrap_path, index=False)
predictions_df.to_parquet(predictions_path, index=False)

summary_lines = [
    "# Serialized full fusion versus selective acquisition",
    "",
    "Both policies load the exact serialized fitted bundle. No model is",
    "reconstructed or refit during evaluation.",
    "",
    "Operating points were selected only on `policy_selection` and then locked",
    "before evaluation on calibration, final-test, and held-out-shift splits.",
    "",
    "The expensive-monitor comparison is reported as call rate and call reduction.",
    "Latency claims are intentionally deferred until a controlled synchronized",
    "timing benchmark is completed.",
    "",
    "These are provisional development results because the assistant-assisted",
    "label audit still requires author review. They are not a formal 5% FPR",
    "risk certificate.",
    "",
    "## Evaluation",
    "",
]

for row in metrics_df.itertuples(index=False):
    summary_lines.append(
        f"- {row.split}, target={row.target_fpr:.3f}, {row.policy}: "
        f"recall={row.recall:.4f}, FPR={row.fpr:.4f}, "
        f"precision={row.precision:.4f}, "
        f"expensive-call-rate={row.expensive_call_rate:.4f}"
    )

summary_lines.extend(
    [
        "",
        "## Paired comparison",
        "",
        "Differences are selective minus full-information. Confidence intervals",
        f"use {BOOTSTRAP_REPS} stratified paired bootstrap replicates.",
        "",
    ]
)

for row in bootstrap_df.itertuples(index=False):
    summary_lines.append(
        f"- {row.split}, target={row.target_fpr:.3f}: "
        f"Δrecall={row.recall_diff_selective_minus_full:.4f} "
        f"[{row.recall_diff_ci_low:.4f}, {row.recall_diff_ci_high:.4f}], "
        f"ΔFPR={row.fpr_diff_selective_minus_full:.4f} "
        f"[{row.fpr_diff_ci_low:.4f}, {row.fpr_diff_ci_high:.4f}], "
        f"expensive-call reduction={row.expensive_call_reduction:.4f} "
        f"[{row.expensive_call_reduction_ci_low:.4f}, "
        f"{row.expensive_call_reduction_ci_high:.4f}]"
    )

(OUT_DIR / "summary.md").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

manifest = {
    "artifact": "fusion_comparison_v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "provisional_labels_pending_author_review",
    "model_loading": "serialized_bundle_only_no_refit",
    "bundle": {
        "path": str(BUNDLE),
        "sha256": sha256(BUNDLE),
    },
    "cache": {
        "path": str(CACHE),
        "sha256": sha256(CACHE),
    },
    "selection_split": SELECTION_SPLIT,
    "evaluation_splits": EVAL_SPLITS,
    "target_fprs": TARGET_FPRS,
    "bootstrap_replicates": BOOTSTRAP_REPS,
    "outputs": {
        str(OPERATING_POINTS): sha256(OPERATING_POINTS),
        str(metrics_path): sha256(metrics_path),
        str(bootstrap_path): sha256(bootstrap_path),
        str(predictions_path): sha256(predictions_path),
        str(OUT_DIR / "summary.md"): sha256(OUT_DIR / "summary.md"),
    },
    "limitations": [
        "Assistant-assisted label audit requires author review.",
        "No formal Neyman-Pearson or Learn-then-Test risk certification.",
        "No controlled latency comparison in this artifact.",
        "Existing final-test split previously influenced development and is not a newly locked test.",
    ],
}
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("serialized bundle loaded:", BUNDLE)
print("selection rows:", len(selection))
print("\nSelected operating points:")
print(selection_df.to_string(index=False))
print("\nEvaluation metrics:")
print(
    metrics_df[
        [
            "split",
            "target_fpr",
            "policy",
            "recall",
            "fpr",
            "precision",
            "expensive_call_rate",
        ]
    ].to_string(index=False)
)
print("\nPaired bootstrap differences:")
print(bootstrap_df.to_string(index=False))
