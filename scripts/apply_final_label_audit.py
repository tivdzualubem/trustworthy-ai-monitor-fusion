#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ORIGINAL_DATASET = Path("data/processed/unified_dataset.parquet")
CACHE_V2 = Path("data/processed/monitor_score_cache_v2.parquet")
FINAL_DECISIONS = Path("reports/label_audit/audit_decisions_final.csv")

AUDITED_DATASET = Path(
    "data/processed/unified_dataset_label_audited_v1.parquet"
)
CACHE_V3 = Path("data/processed/monitor_score_cache_v3.parquet")

REPORT_DIR = Path("reports/label_audit")
PROMPT_REPORT_DIR = Path("reports/prompt_contamination_v3")
MANIFEST = Path(
    "data/metadata/label_audited_dataset_v1_manifest.json"
)

QWEN_MODES = ["prompt_only", "response_only", "prompt_response"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_auc(y: pd.Series, score: pd.Series) -> tuple[float, float]:
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y.astype(int), score.astype(float))),
        float(average_precision_score(y.astype(int), score.astype(float))),
    )


def binary_metrics(y: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    y = y.astype(int)
    pred = pred.astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    return {
        "n": int(len(y)),
        "positive_n": int((y == 1).sum()),
        "negative_n": int((y == 0).sum()),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }


for path in [ORIGINAL_DATASET, CACHE_V2, FINAL_DECISIONS]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")

dataset = pd.read_parquet(ORIGINAL_DATASET).copy()
cache = pd.read_parquet(CACHE_V2).copy()
decisions = pd.read_csv(FINAL_DECISIONS).copy()

for frame in [dataset, cache, decisions]:
    frame["example_id"] = frame["example_id"].astype(str)

if len(dataset) != 2159 or dataset["example_id"].nunique() != 2159:
    raise SystemExit("Original dataset must contain 2159 unique examples")

if len(cache) != 2159 or cache["example_id"].nunique() != 2159:
    raise SystemExit("Cache v2 must contain 2159 unique examples")

if len(decisions) != 200 or decisions["example_id"].nunique() != 200:
    raise SystemExit("Final decisions must contain 200 unique examples")

dataset_ids = set(dataset["example_id"])
cache_ids = set(cache["example_id"])
decision_ids = set(decisions["example_id"])

if dataset_ids != cache_ids:
    raise SystemExit("Dataset and cache v2 IDs do not match")
if not decision_ids.issubset(dataset_ids):
    raise SystemExit("Audit decisions contain unknown example IDs")

check = dataset[["example_id", "y"]].merge(
    decisions[["example_id", "old_y"]],
    on="example_id",
    how="inner",
    validate="one_to_one",
)

if not (check["y"].astype(int) == check["old_y"].astype(int)).all():
    raise SystemExit("Audit old_y values do not match the original dataset")

audit_public = decisions[
    [
        "example_id",
        "audit_group",
        "old_y",
        "audited_y",
        "changed",
        "auditor_label_valid",
        "auditor_proxy_label_issue",
        "auditor_prompt_contamination",
        "auditor_confidence",
    ]
].copy()

audit_public = audit_public.rename(
    columns={
        "audit_group": "label_audit_group",
        "changed": "label_audit_changed",
        "auditor_label_valid": "label_audit_label_valid",
        "auditor_proxy_label_issue": "label_audit_proxy_issue",
        "auditor_prompt_contamination": "label_audit_prompt_contamination",
        "auditor_confidence": "label_audit_confidence",
    }
)

audited = dataset.copy()
audited["y_original"] = audited["y"].astype(int)

audited = audited.merge(
    audit_public,
    on="example_id",
    how="left",
    validate="one_to_one",
)

reviewed = audited["audited_y"].notna()
audited.loc[reviewed, "y"] = audited.loc[reviewed, "audited_y"].astype(int)

audited["y"] = audited["y"].astype(int)
audited["label_audit_reviewed"] = reviewed
audited["label_audit_version"] = ""
audited.loc[reviewed, "label_audit_version"] = "author_reviewed_v1"

audited["label_audit_changed"] = (
    audited["label_audit_changed"]
    .fillna(False)
    .astype(bool)
)

if int(audited["label_audit_reviewed"].sum()) != 200:
    raise SystemExit("Expected exactly 200 reviewed rows")

if int(audited["label_audit_changed"].sum()) != 19:
    raise SystemExit("Expected exactly 19 corrected labels")

AUDITED_DATASET.parent.mkdir(parents=True, exist_ok=True)
audited.to_parquet(AUDITED_DATASET, index=False)

cache_v3 = cache.drop(
    columns=[
        c
        for c in [
            "y",
            "y_original",
            "label_audit_reviewed",
            "label_audit_version",
            "label_audit_group",
            "label_audit_changed",
            "label_audit_label_valid",
            "label_audit_proxy_issue",
            "label_audit_prompt_contamination",
            "label_audit_confidence",
            "old_y",
            "audited_y",
        ]
        if c in cache.columns
    ]
).copy()

cache_v3 = cache_v3.merge(
    audited[
        [
            "example_id",
            "y",
            "y_original",
            "label_audit_reviewed",
            "label_audit_version",
            "label_audit_group",
            "label_audit_changed",
            "label_audit_label_valid",
            "label_audit_proxy_issue",
            "label_audit_prompt_contamination",
            "label_audit_confidence",
        ]
    ],
    on="example_id",
    how="inner",
    validate="one_to_one",
)

if len(cache_v3) != 2159:
    raise SystemExit("Cache v3 row count mismatch")

CACHE_V3.parent.mkdir(parents=True, exist_ok=True)
cache_v3.to_parquet(CACHE_V3, index=False)

change_summary = (
    audited[audited["label_audit_changed"]]
    .groupby(
        [
            "label_audit_group",
            "y_original",
            "y",
            "split",
        ],
        dropna=False,
    )
    .size()
    .rename("n")
    .reset_index()
    .sort_values(
        [
            "label_audit_group",
            "y_original",
            "y",
            "split",
        ]
    )
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
change_summary.to_csv(
    REPORT_DIR / "final_label_change_summary.csv",
    index=False,
)

split_counts = (
    audited.groupby(["split", "y"], dropna=False)
    .size()
    .rename("n")
    .reset_index()
)
split_counts.to_csv(
    REPORT_DIR / "audited_split_label_counts.csv",
    index=False,
)

metric_rows = []
groups = [("overall", "all", cache_v3)]
groups.extend(
    ("split", str(split), frame)
    for split, frame in cache_v3.groupby("split", dropna=False)
)

for scope_type, scope_value, frame in groups:
    y = frame["y"].astype(int)

    for mode in QWEN_MODES:
        score_col = f"qwen_{mode}_score"
        label_col = f"qwen_{mode}_label"
        roc_auc, ap = safe_auc(y, frame[score_col])

        decision_rules = {
            "unsafe_only": frame[label_col].eq("Unsafe"),
            "unsafe_or_controversial": frame[label_col].isin(
                ["Unsafe", "Controversial"]
            ),
        }

        for decision_rule, pred in decision_rules.items():
            row = {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "mode": mode,
                "decision_rule": decision_rule,
                "roc_auc": roc_auc,
                "average_precision": ap,
            }
            row.update(binary_metrics(y, pred))
            metric_rows.append(row)

metrics = pd.DataFrame(metric_rows)
PROMPT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
metrics.to_csv(
    PROMPT_REPORT_DIR / "qwen_mode_metrics.csv",
    index=False,
)

overall = metrics[
    (metrics["scope_type"] == "overall")
    & (metrics["decision_rule"] == "unsafe_only")
].copy()

old_counts = dataset["y"].value_counts().sort_index().to_dict()
new_counts = audited["y"].value_counts().sort_index().to_dict()

summary_lines = [
    "# Final audited-label dataset",
    "",
    "The final author-reviewed audit decisions were applied to the unified dataset.",
    "",
    "## Counts",
    "",
    f"- Total examples: {len(audited)}",
    f"- Author-reviewed examples: {int(audited['label_audit_reviewed'].sum())}",
    f"- Corrected labels: {int(audited['label_audit_changed'].sum())}",
    f"- Original class counts: {old_counts}",
    f"- Audited class counts: {new_counts}",
    "",
    "## Corrected-label Qwen metrics",
    "",
]

for row in overall.itertuples(index=False):
    summary_lines.append(
        f"- {row.mode}: ROC-AUC={row.roc_auc:.4f}, "
        f"AP={row.average_precision:.4f}, "
        f"recall={row.recall:.4f}, "
        f"FPR={row.fpr:.4f}, "
        f"precision={row.precision:.4f}"
    )

summary_lines.extend(
    [
        "",
        "The monitor scores were not regenerated because label correction does not",
        "change monitor inference outputs. The score cache was rebuilt with the",
        "final audited target labels and audit provenance.",
        "",
    ]
)

(REPORT_DIR / "applied_final_label_audit.md").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

manifest = {
    "artifact": "label_audited_dataset_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "author_reviewed_labels_applied",
    "rows": int(len(audited)),
    "unique_example_id": int(audited["example_id"].nunique()),
    "reviewed_rows": int(audited["label_audit_reviewed"].sum()),
    "corrected_rows": int(audited["label_audit_changed"].sum()),
    "original_class_counts": {
        str(k): int(v) for k, v in old_counts.items()
    },
    "audited_class_counts": {
        str(k): int(v) for k, v in new_counts.items()
    },
    "sources": {
        str(ORIGINAL_DATASET): sha256(ORIGINAL_DATASET),
        str(CACHE_V2): sha256(CACHE_V2),
        str(FINAL_DECISIONS): sha256(FINAL_DECISIONS),
    },
    "outputs": {
        str(AUDITED_DATASET): sha256(AUDITED_DATASET),
        str(CACHE_V3): sha256(CACHE_V3),
        str(REPORT_DIR / "final_label_change_summary.csv"): sha256(
            REPORT_DIR / "final_label_change_summary.csv"
        ),
        str(REPORT_DIR / "audited_split_label_counts.csv"): sha256(
            REPORT_DIR / "audited_split_label_counts.csv"
        ),
        str(PROMPT_REPORT_DIR / "qwen_mode_metrics.csv"): sha256(
            PROMPT_REPORT_DIR / "qwen_mode_metrics.csv"
        ),
    },
    "note": (
        "Existing monitor inference outputs are preserved; only target labels "
        "and audit provenance were updated."
    ),
}

MANIFEST.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("original class counts:", old_counts)
print("audited class counts:", new_counts)
print("reviewed rows:", int(audited["label_audit_reviewed"].sum()))
print("corrected rows:", int(audited["label_audit_changed"].sum()))
print("\nchange summary:")
print(change_summary.to_string(index=False))
print("\ncorrected-label Qwen metrics:")
print(
    overall[
        [
            "mode",
            "roc_auc",
            "average_precision",
            "recall",
            "fpr",
            "precision",
        ]
    ].to_string(index=False)
)
print("\naudited dataset:", AUDITED_DATASET)
print("cache v3:", CACHE_V3)
print("manifest:", MANIFEST)
