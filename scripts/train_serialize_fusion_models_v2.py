#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("data/processed/monitor_score_cache_v2.parquet")
ARTIFACT_DIR = Path("artifacts/fusion_models_v2")
REPORT_DIR = Path("reports/fusion_models_v2")

TRAIN_SPLIT = "policy_train"
SELECTION_SPLIT = "policy_selection"

CHEAP_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
]
FULL_FEATURES = [
    "rule_score",
    "compact_unsafe_score",
    "qwen_prompt_response_score",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def id_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_calibrated_logistic() -> CalibratedClassifierCV:
    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=5,
        ensemble=True,
    )


if not CACHE.exists():
    raise SystemExit(f"Missing repaired cache: {CACHE}")

df = pd.read_parquet(CACHE).copy()

required = {
    "example_id",
    "split",
    "y",
    *CHEAP_FEATURES,
    *FULL_FEATURES,
}
missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

if len(df) != 2159 or df["example_id"].nunique() != 2159:
    raise SystemExit("Expected 2159 unique examples in repaired cache")

for col in sorted(set(CHEAP_FEATURES + FULL_FEATURES)):
    if df[col].isna().any():
        raise SystemExit(f"Feature contains missing values: {col}")
    if not np.isfinite(df[col].astype(float)).all():
        raise SystemExit(f"Feature contains non-finite values: {col}")

train = df[df["split"].eq(TRAIN_SPLIT)].copy()
selection = df[df["split"].eq(SELECTION_SPLIT)].copy()

if train.empty or selection.empty:
    raise SystemExit("Training or selection split is empty")
if train["y"].nunique() != 2:
    raise SystemExit("policy_train must contain both classes")

cheap_router = build_calibrated_logistic()
full_fusion = build_calibrated_logistic()

cheap_router.fit(train[CHEAP_FEATURES], train["y"].astype(int))
full_fusion.fit(train[FULL_FEATURES], train["y"].astype(int))

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

cheap_path = ARTIFACT_DIR / "cheap_router.joblib"
full_path = ARTIFACT_DIR / "full_information_fusion.joblib"
bundle_path = ARTIFACT_DIR / "fusion_bundle.joblib"
manifest_path = ARTIFACT_DIR / "manifest.json"

joblib.dump(cheap_router, cheap_path, compress=3)
joblib.dump(full_fusion, full_path, compress=3)

provenance = {
    "artifact_version": "fusion_models_v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "provisional_labels_pending_author_review",
    "training_split": TRAIN_SPLIT,
    "selection_split_reserved_for_threshold_selection": SELECTION_SPLIT,
    "cache_path": str(CACHE),
    "cache_sha256": sha256(CACHE),
    "train_rows": int(len(train)),
    "train_positive_n": int(train["y"].sum()),
    "train_negative_n": int((train["y"] == 0).sum()),
    "train_example_id_sha256": id_hash(train["example_id"]),
    "cheap_features": CHEAP_FEATURES,
    "full_features": FULL_FEATURES,
    "model_family": "5-fold sigmoid-calibrated class-balanced logistic regression",
    "random_state": 0,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "joblib_version": joblib.__version__,
}

bundle = {
    "cheap_router": cheap_router,
    "full_information_fusion": full_fusion,
    "cheap_features": CHEAP_FEATURES,
    "full_features": FULL_FEATURES,
    "provenance": provenance,
}
joblib.dump(bundle, bundle_path, compress=3)

# Reload from disk and verify exact score reproduction.
loaded = joblib.load(bundle_path)

for split_name, frame in [
    (TRAIN_SPLIT, train),
    (SELECTION_SPLIT, selection),
]:
    cheap_mem = cheap_router.predict_proba(frame[CHEAP_FEATURES])[:, 1]
    cheap_disk = loaded["cheap_router"].predict_proba(
        frame[loaded["cheap_features"]]
    )[:, 1]
    full_mem = full_fusion.predict_proba(frame[FULL_FEATURES])[:, 1]
    full_disk = loaded["full_information_fusion"].predict_proba(
        frame[loaded["full_features"]]
    )[:, 1]

    if not np.allclose(cheap_mem, cheap_disk, atol=0.0, rtol=0.0):
        raise SystemExit(f"Serialized cheap router mismatch on {split_name}")
    if not np.allclose(full_mem, full_disk, atol=0.0, rtol=0.0):
        raise SystemExit(f"Serialized full fusion mismatch on {split_name}")

    out = frame[
        ["example_id", "split", "y", "source_dataset", "attack_family"]
    ].copy()
    out["cheap_router_probability"] = cheap_disk
    out["full_information_probability"] = full_disk
    out.to_csv(
        REPORT_DIR / f"{split_name}_serialized_scores.csv",
        index=False,
    )

manifest = {
    **provenance,
    "artifacts": {
        "cheap_router": {
            "path": str(cheap_path),
            "sha256": sha256(cheap_path),
        },
        "full_information_fusion": {
            "path": str(full_path),
            "sha256": sha256(full_path),
        },
        "fusion_bundle": {
            "path": str(bundle_path),
            "sha256": sha256(bundle_path),
        },
    },
    "verification": {
        "reloaded_from_disk": True,
        "exact_probability_match": True,
        "evaluation_reconstruction_required": False,
    },
    "notes": [
        "The bundle contains fitted scalers, logistic estimators, and calibration folds.",
        "Evaluation must load this bundle rather than refit or reconstruct models.",
        "No thresholds were selected in this step.",
        "The artifact must be refit after author-approved label corrections are locked.",
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

summary = f"""# Serialized fusion models v2

Two complete fitted pipelines were trained on `{TRAIN_SPLIT}` using the repaired
official-Qwen score cache.

## Artifacts

- `{cheap_path}`: calibrated cheap router using {CHEAP_FEATURES}
- `{full_path}`: calibrated full-information fusion model using {FULL_FEATURES}
- `{bundle_path}`: complete serialized bundle used by downstream evaluation

The bundle includes fitted standardization, logistic estimators, calibration
objects, feature order, and provenance. Reloaded probabilities exactly match
the in-memory fitted models on both `{TRAIN_SPLIT}` and `{SELECTION_SPLIT}`.

No operating threshold was selected in this step. The current artifact is
marked provisional because the assistant-assisted label audit still requires
author review before corrected labels are locked.
"""
(REPORT_DIR / "summary.md").write_text(summary, encoding="utf-8")

print("train rows:", len(train))
print("selection rows:", len(selection))
print("cheap artifact:", cheap_path)
print("full artifact:", full_path)
print("bundle:", bundle_path)
print("manifest:", manifest_path)
print("serialization verification: exact match")
