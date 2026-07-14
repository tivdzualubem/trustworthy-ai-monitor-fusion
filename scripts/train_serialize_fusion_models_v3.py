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

CACHE = Path("data/processed/monitor_score_cache_v3.parquet")
ARTIFACT_DIR = Path("artifacts/fusion_models_v3")
REPORT_DIR = Path("reports/fusion_models_v3")

RULE_MANIFEST = Path("data/metadata/rule_scores_v2_manifest.json")
COMPACT_MANIFEST = Path("data/metadata/compact_scores_v2_manifest.json")
QWEN_MANIFEST = Path("data/metadata/monitor_score_cache_v2_manifest.json")
LABEL_MANIFEST = Path("data/metadata/label_audited_dataset_v1_manifest.json")

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

RANDOM_STATE = 0
CALIBRATION_FOLDS = 5


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
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=CALIBRATION_FOLDS,
        ensemble=True,
    )


for required_path in [
    CACHE,
    RULE_MANIFEST,
    COMPACT_MANIFEST,
    QWEN_MANIFEST,
    LABEL_MANIFEST,
]:
    if not required_path.exists():
        raise SystemExit(f"Missing required input: {required_path}")

df = pd.read_parquet(CACHE).copy()
df["example_id"] = df["example_id"].astype(str)

required_columns = {
    "example_id",
    "split",
    "y",
    *CHEAP_FEATURES,
    *FULL_FEATURES,
}
missing = sorted(required_columns - set(df.columns))
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

if len(df) != 2159 or df["example_id"].nunique() != 2159:
    raise SystemExit("Expected 2159 unique examples in cache v3")

for column in sorted(set(CHEAP_FEATURES + FULL_FEATURES)):
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise SystemExit(f"Feature contains missing values: {column}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise SystemExit(f"Feature contains non-finite values: {column}")

if not set(df["y"].astype(int).unique()).issubset({0, 1}):
    raise SystemExit("Target labels must be binary")

train = df[df["split"].eq(TRAIN_SPLIT)].copy()
selection = df[df["split"].eq(SELECTION_SPLIT)].copy()

if len(train) != 844:
    raise SystemExit(f"Expected 844 policy_train rows, found {len(train)}")
if len(selection) != 421:
    raise SystemExit(
        f"Expected 421 policy_selection rows, found {len(selection)}"
    )
if train["y"].nunique() != 2:
    raise SystemExit("policy_train must contain both classes")

cheap_router = build_calibrated_logistic()
full_fusion = build_calibrated_logistic()

cheap_router.fit(
    train[CHEAP_FEATURES],
    train["y"].astype(int),
)
full_fusion.fit(
    train[FULL_FEATURES],
    train["y"].astype(int),
)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

cheap_path = ARTIFACT_DIR / "cheap_router.joblib"
full_path = ARTIFACT_DIR / "full_information_fusion.joblib"
bundle_path = ARTIFACT_DIR / "fusion_bundle.joblib"
manifest_path = ARTIFACT_DIR / "manifest.json"

joblib.dump(cheap_router, cheap_path, compress=3)
joblib.dump(full_fusion, full_path, compress=3)

created_at = datetime.now(timezone.utc).isoformat()

provenance = {
    "artifact_version": "fusion_models_v3",
    "created_at": created_at,
    "status": "author_reviewed_labels_locked",
    "supersedes": "artifacts/fusion_models_v2",
    "training_split": TRAIN_SPLIT,
    "selection_split_reserved_for_threshold_selection": SELECTION_SPLIT,
    "cache_path": str(CACHE),
    "cache_sha256": sha256(CACHE),
    "train_rows": int(len(train)),
    "train_positive_n": int(train["y"].astype(int).sum()),
    "train_negative_n": int((train["y"].astype(int) == 0).sum()),
    "train_example_id_sha256": id_hash(train["example_id"]),
    "selection_rows": int(len(selection)),
    "selection_example_id_sha256": id_hash(selection["example_id"]),
    "cheap_features": CHEAP_FEATURES,
    "full_features": FULL_FEATURES,
    "model_family": (
        "5-fold sigmoid-calibrated class-balanced logistic regression"
    ),
    "calibration_method": "sigmoid",
    "calibration_folds": CALIBRATION_FOLDS,
    "calibration_ensemble": True,
    "random_state": RANDOM_STATE,
    "thresholds_selected": False,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "joblib_version": joblib.__version__,
    "input_manifests": {
        "rule_scores": {
            "path": str(RULE_MANIFEST),
            "sha256": sha256(RULE_MANIFEST),
        },
        "compact_scores": {
            "path": str(COMPACT_MANIFEST),
            "sha256": sha256(COMPACT_MANIFEST),
        },
        "qwen_and_cache_v2": {
            "path": str(QWEN_MANIFEST),
            "sha256": sha256(QWEN_MANIFEST),
        },
        "final_label_audit": {
            "path": str(LABEL_MANIFEST),
            "sha256": sha256(LABEL_MANIFEST),
        },
    },
}

bundle = {
    "cheap_router": cheap_router,
    "full_information_fusion": full_fusion,
    "cheap_features": CHEAP_FEATURES,
    "full_features": FULL_FEATURES,
    "provenance": provenance,
}
joblib.dump(bundle, bundle_path, compress=3)

# Reload all three persisted artifacts and require exact probability reproduction.
loaded_cheap = joblib.load(cheap_path)
loaded_full = joblib.load(full_path)
loaded_bundle = joblib.load(bundle_path)

verification_rows = []
for split_name, frame in df.groupby("split", sort=True):
    cheap_memory = cheap_router.predict_proba(
        frame[CHEAP_FEATURES]
    )[:, 1]
    cheap_disk = loaded_cheap.predict_proba(
        frame[CHEAP_FEATURES]
    )[:, 1]
    cheap_bundle = loaded_bundle["cheap_router"].predict_proba(
        frame[loaded_bundle["cheap_features"]]
    )[:, 1]

    full_memory = full_fusion.predict_proba(
        frame[FULL_FEATURES]
    )[:, 1]
    full_disk = loaded_full.predict_proba(
        frame[FULL_FEATURES]
    )[:, 1]
    full_bundle = loaded_bundle[
        "full_information_fusion"
    ].predict_proba(
        frame[loaded_bundle["full_features"]]
    )[:, 1]

    checks = {
        "cheap_standalone_exact": bool(
            np.array_equal(cheap_memory, cheap_disk)
        ),
        "cheap_bundle_exact": bool(
            np.array_equal(cheap_memory, cheap_bundle)
        ),
        "full_standalone_exact": bool(
            np.array_equal(full_memory, full_disk)
        ),
        "full_bundle_exact": bool(
            np.array_equal(full_memory, full_bundle)
        ),
    }

    if not all(checks.values()):
        raise SystemExit(
            f"Serialized probability mismatch on split {split_name}: {checks}"
        )

    verification_rows.append(
        {
            "split": str(split_name),
            "rows": int(len(frame)),
            **checks,
        }
    )

verification = pd.DataFrame(verification_rows)
verification_path = REPORT_DIR / "serialization_verification.csv"
verification.to_csv(verification_path, index=False)

# Save scores only for the two development splits used by this stage.
for split_name, frame in [
    (TRAIN_SPLIT, train),
    (SELECTION_SPLIT, selection),
]:
    out = frame[
        [
            "example_id",
            "split",
            "y",
            "source_dataset",
            "attack_family",
        ]
    ].copy()
    out["cheap_router_probability"] = loaded_bundle[
        "cheap_router"
    ].predict_proba(
        frame[loaded_bundle["cheap_features"]]
    )[:, 1]
    out["full_information_probability"] = loaded_bundle[
        "full_information_fusion"
    ].predict_proba(
        frame[loaded_bundle["full_features"]]
    )[:, 1]
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
        "reloaded_standalone_artifacts": True,
        "reloaded_bundle": True,
        "exact_probability_match_all_splits": True,
        "verification_table": {
            "path": str(verification_path),
            "sha256": sha256(verification_path),
        },
        "evaluation_reconstruction_required": False,
    },
    "notes": [
        (
            "The bundle contains fitted standardizers, logistic estimators, "
            "five calibration folds, feature order, and provenance."
        ),
        (
            "Downstream evaluation must load this bundle and must not refit "
            "or reconstruct either pipeline."
        ),
        "No operating threshold was selected during this step.",
        (
            "Final-test and held-out labels were not used for fitting or "
            "threshold selection."
        ),
    ],
}
manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

summary = f"""# Serialized fusion models v3

These models were trained on `{TRAIN_SPLIT}` from the final
author-reviewed cache `{CACHE}`.

## Complete serialized artifacts

- `{cheap_path}` uses {CHEAP_FEATURES}
- `{full_path}` uses {FULL_FEATURES}
- `{bundle_path}` contains both fitted pipelines, feature order, and provenance

Each pipeline contains fitted standardization, class-balanced logistic
regression, and five-fold sigmoid calibration. Reloaded probabilities matched
the in-memory fitted models exactly on every split.

No operating threshold was selected in this stage. The
`{SELECTION_SPLIT}` split remains reserved for the next professor-required
policy-selection step.
"""
(REPORT_DIR / "summary.md").write_text(
    summary,
    encoding="utf-8",
)

print("cache:", CACHE)
print("train rows:", len(train))
print(
    "train labels:",
    train["y"].astype(int).value_counts().sort_index().to_dict(),
)
print("selection rows:", len(selection))
print("cheap artifact:", cheap_path)
print("full artifact:", full_path)
print("bundle:", bundle_path)
print("manifest:", manifest_path)
print("serialization verification: exact match on every split")
print("\nverification table:")
print(verification.to_string(index=False))
