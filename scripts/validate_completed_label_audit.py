#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PRIVATE_SHEET = Path("data/audit_private/label_validity_audit_sheet.csv")
PRIVATE_REVIEW = Path("data/audit_private/label_validity_review_required.csv")
PUBLIC_DECISIONS = Path("reports/label_audit/audit_decisions_public.csv")
PUBLIC_CORRECTIONS = Path("reports/label_audit/label_corrections_v1.csv")
SUMMARY = Path("reports/label_audit/completed_summary.md")
MANIFEST = Path("data/metadata/label_audit_completed_manifest.json")

REQUIRED_COLUMNS = [
    "example_id",
    "audit_group",
    "source_dataset",
    "attack_family",
    "split",
    "y",
    "auditor_response_harmful",
    "auditor_should_intercept",
    "auditor_label_valid",
    "auditor_proxy_label_issue",
    "auditor_prompt_contamination",
    "auditor_confidence",
    "auditor_notes",
]

VALID_VALUES = {
    "auditor_response_harmful": {"0", "1"},
    "auditor_should_intercept": {"0", "1"},
    "auditor_label_valid": {"yes", "no"},
    "auditor_proxy_label_issue": {"yes", "no"},
    "auditor_prompt_contamination": {"yes", "no", "unclear"},
    "auditor_confidence": {"high", "medium", "low"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if not PRIVATE_SHEET.exists():
    raise SystemExit(f"Missing private audit sheet: {PRIVATE_SHEET}")

df = pd.read_csv(PRIVATE_SHEET, dtype=str, keep_default_na=False)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

if len(df) != 200:
    raise SystemExit(f"Expected 200 audit rows, found {len(df)}")

if df["example_id"].nunique() != 200:
    raise SystemExit("example_id must be unique across all 200 rows")

for col, allowed in VALID_VALUES.items():
    normalized = df[col].astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise SystemExit(f"{col} has invalid values: {invalid}")
    df[col] = normalized

if (df["auditor_notes"].astype(str).str.strip() == "").any():
    raise SystemExit("Every row must contain an auditor note")

df["old_y"] = df["y"].astype(int)
df["audited_y"] = df["auditor_should_intercept"].astype(int)
df["changed"] = df["old_y"] != df["audited_y"]

expected_valid = (~df["changed"]).map({True: "yes", False: "no"})
if not (df["auditor_label_valid"] == expected_valid).all():
    bad = df.loc[
        df["auditor_label_valid"] != expected_valid,
        ["example_id", "old_y", "audited_y", "auditor_label_valid"],
    ]
    raise SystemExit(
        "auditor_label_valid is inconsistent with old_y versus audited_y:\n"
        + bad.to_string(index=False)
    )

review_mask = (
    df["changed"]
    | df["auditor_confidence"].isin(["medium", "low"])
    | df["audit_group"].eq("GCG_negative")
)
review = df.loc[review_mask].copy()
review.to_csv(PRIVATE_REVIEW, index=False)

public_cols = [
    "example_id",
    "audit_group",
    "source_dataset",
    "attack_family",
    "split",
    "old_y",
    "audited_y",
    "changed",
    "auditor_label_valid",
    "auditor_proxy_label_issue",
    "auditor_prompt_contamination",
    "auditor_confidence",
]
public = df[public_cols].copy()
PUBLIC_DECISIONS.parent.mkdir(parents=True, exist_ok=True)
public.to_csv(PUBLIC_DECISIONS, index=False)

corrections = public[public["changed"]].copy()
corrections.to_csv(PUBLIC_CORRECTIONS, index=False)

group_summary = (
    df.groupby("audit_group", dropna=False)
    .agg(
        n=("example_id", "size"),
        changed_n=("changed", "sum"),
        audited_positive_n=("audited_y", "sum"),
    )
    .reset_index()
)
group_summary["changed_rate"] = group_summary["changed_n"] / group_summary["n"]

confidence_summary = (
    df["auditor_confidence"]
    .value_counts(dropna=False)
    .rename_axis("confidence")
    .reset_index(name="n")
)

lines = [
    "# Completed label-audit summary",
    "",
    "The current worksheet contains an assistant-assisted first-pass adjudication.",
    "It is not treated as final human ground truth until the review-required subset",
    "has been checked by the project author.",
    "",
    "## Counts",
    "",
    f"- Total audited rows: {len(df)}",
    f"- Proposed label corrections: {int(df['changed'].sum())}",
    f"- Review-required rows: {len(review)}",
    f"- GCG negative rows: {int(df['audit_group'].eq('GCG_negative').sum())}",
    "",
    "## By audit group",
    "",
]
for row in group_summary.itertuples(index=False):
    lines.append(
        f"- {row.audit_group}: n={row.n}, corrections={row.changed_n}, "
        f"correction_rate={row.changed_rate:.4f}, "
        f"audited_positive={row.audited_positive_n}"
    )

lines.extend(["", "## Confidence", ""])
for row in confidence_summary.itertuples(index=False):
    lines.append(f"- {row.confidence}: {row.n}")

lines.extend(
    [
        "",
        "## Required author review",
        "",
        f"Review the local file `{PRIVATE_REVIEW}` before applying the proposed",
        "corrections to the canonical dataset. This file is private and ignored by Git.",
        "",
        "The tracked public artifacts contain IDs and decisions only; raw prompt and",
        "response text remains private.",
        "",
    ]
)
SUMMARY.write_text("\n".join(lines), encoding="utf-8")

manifest = {
    "artifact": "completed_label_audit_first_pass",
    "status": "assistant_assisted_first_pass_human_review_pending",
    "rows": int(len(df)),
    "unique_example_id": int(df["example_id"].nunique()),
    "proposed_corrections": int(df["changed"].sum()),
    "review_required_rows": int(len(review)),
    "audit_group_counts": {
        str(k): int(v)
        for k, v in df["audit_group"].value_counts(dropna=False).to_dict().items()
    },
    "confidence_counts": {
        str(k): int(v)
        for k, v in df["auditor_confidence"].value_counts(dropna=False).to_dict().items()
    },
    "private_files": {
        str(PRIVATE_SHEET): {
            "sha256": sha256(PRIVATE_SHEET),
            "contains_raw_text": True,
            "tracked": False,
        },
        str(PRIVATE_REVIEW): {
            "sha256": sha256(PRIVATE_REVIEW),
            "contains_raw_text": True,
            "tracked": False,
        },
    },
    "public_files": {
        str(PUBLIC_DECISIONS): sha256(PUBLIC_DECISIONS),
        str(PUBLIC_CORRECTIONS): sha256(PUBLIC_CORRECTIONS),
        str(SUMMARY): sha256(SUMMARY),
    },
}
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("rows:", len(df))
print("proposed corrections:", int(df["changed"].sum()))
print("review required:", len(review))
print("\nBy group:")
print(group_summary.to_string(index=False))
print("\nConfidence:")
print(confidence_summary.to_string(index=False))
print("\nPrivate review file:", PRIVATE_REVIEW)
print("Public corrections:", PUBLIC_CORRECTIONS)
