#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(".")
DATASET = ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
OLD_SCORES = ROOT / "data/interim/rule_scores.parquet"
OUT_DIR = ROOT / "data/interim/rule_scores_v2"
OUT_PARQUET = OUT_DIR / "rule_scores_v2.parquet"
OUT_CSV = OUT_DIR / "rule_scores_v2.csv"
MANIFEST = ROOT / "data/metadata/rule_scores_v2_manifest.json"

RULE_SOURCE = ROOT / "src/monitor_fusion/monitors/rule_filter.py"
SCRIPT_SOURCE = ROOT / "scripts/regenerate_rule_scores_v2.py"

sys.path.insert(0, str(ROOT / "src"))
from monitor_fusion.monitors.rule_filter import RULES, score_prompt_response  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


for path in [DATASET, RULE_SOURCE]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")

df = pd.read_parquet(DATASET).copy()
required = {"example_id", "prompt", "response"}
missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"Dataset missing columns: {missing}")

df["example_id"] = df["example_id"].astype(str)

if len(df) != 2159 or df["example_id"].nunique() != 2159:
    raise SystemExit("Expected 2159 unique dataset examples")

records = []
for i, row in enumerate(df.itertuples(index=False), start=1):
    result = score_prompt_response(
        "" if pd.isna(row.prompt) else str(row.prompt),
        "" if pd.isna(row.response) else str(row.response),
    )
    records.append(
        {
            "example_id": str(row.example_id),
            "rule_score": float(result["rule_score"]),
            "rule_weighted_sum": float(result["rule_weighted_sum"]),
            "rule_match_count": int(result["rule_match_count"]),
            "rule_matches": json.dumps(
                result["rule_matches"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "rule_latency_ms_uncontrolled": float(result["rule_latency_ms"]),
        }
    )
    if i % 250 == 0 or i == len(df):
        print(f"scored {i}/{len(df)}")

scores = pd.DataFrame(records)

if len(scores) != 2159 or scores["example_id"].nunique() != 2159:
    raise SystemExit("Regenerated rule score output failed row/ID validation")

if scores[
    [
        "rule_score",
        "rule_weighted_sum",
        "rule_match_count",
        "rule_latency_ms_uncontrolled",
    ]
].isna().any().any():
    raise SystemExit("Regenerated rule scores contain missing values")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

scores.to_parquet(OUT_PARQUET, index=False)
scores.to_csv(OUT_CSV, index=False)

old_comparison = {
    "available": OLD_SCORES.exists(),
    "rows_compared": 0,
    "exact_score_match": None,
    "max_abs_rule_score_difference": None,
    "exact_weighted_sum_match": None,
    "exact_match_count_match": None,
}

if OLD_SCORES.exists():
    old = pd.read_parquet(OLD_SCORES).copy()
    old["example_id"] = old["example_id"].astype(str)
    compare = scores.merge(
        old[
            [
                "example_id",
                "rule_score",
                "rule_weighted_sum",
                "rule_match_count",
            ]
        ],
        on="example_id",
        how="inner",
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )

    if len(compare) != 2159:
        raise SystemExit("Could not compare all regenerated scores with old cache")

    score_diff = (
        compare["rule_score_new"] - compare["rule_score_old"]
    ).abs()

    old_comparison = {
        "available": True,
        "rows_compared": int(len(compare)),
        "exact_score_match": bool(
            (compare["rule_score_new"] == compare["rule_score_old"]).all()
        ),
        "max_abs_rule_score_difference": float(score_diff.max()),
        "exact_weighted_sum_match": bool(
            (
                compare["rule_weighted_sum_new"]
                == compare["rule_weighted_sum_old"]
            ).all()
        ),
        "exact_match_count_match": bool(
            (
                compare["rule_match_count_new"]
                == compare["rule_match_count_old"]
            ).all()
        ),
    }

    if not all(
        [
            old_comparison["exact_score_match"],
            old_comparison["exact_weighted_sum_match"],
            old_comparison["exact_match_count_match"],
        ]
    ):
        raise SystemExit(
            "Regenerated deterministic rule scores do not exactly match old cache"
        )

manifest = {
    "artifact": "rule_scores_v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "monitor_id": "rule_filter_v1",
    "monitor_type": "deterministic_weighted_lexical_filter",
    "prediction_unit": "prompt_response_pair",
    "rows": int(len(scores)),
    "unique_example_id": int(scores["example_id"].nunique()),
    "input_fields": ["example_id", "prompt", "response"],
    "forbidden_fields_used_during_inference": [],
    "threshold_selection_during_scoring": False,
    "rule_count": int(len(RULES)),
    "implementation": {
        "source_path": str(RULE_SOURCE),
        "source_sha256": sha256(RULE_SOURCE),
        "regeneration_script_path": str(SCRIPT_SOURCE),
        "regeneration_script_sha256": sha256(SCRIPT_SOURCE),
        "python_version": platform.python_version(),
    },
    "input": {
        "path": str(DATASET),
        "sha256": sha256(DATASET),
    },
    "outputs": {
        "parquet": {
            "path": str(OUT_PARQUET),
            "sha256": sha256(OUT_PARQUET),
        },
        "csv": {
            "path": str(OUT_CSV),
            "sha256": sha256(OUT_CSV),
        },
    },
    "score_validation": {
        "min": float(scores["rule_score"].min()),
        "max": float(scores["rule_score"].max()),
        "nonmissing": int(scores["rule_score"].notna().sum()),
    },
    "old_cache_exact_reproduction": old_comparison,
    "timing_note": (
        "rule_latency_ms_uncontrolled is score-generation provenance only. "
        "Use the controlled timing benchmark for latency claims."
    ),
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("\nrows:", len(scores))
print("unique IDs:", scores["example_id"].nunique())
print("score range:", scores["rule_score"].min(), scores["rule_score"].max())
print("exact old-cache reproduction:", old_comparison)
print("output:", OUT_PARQUET)
print("manifest:", MANIFEST)
