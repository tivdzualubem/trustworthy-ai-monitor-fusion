#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PINNED_REVISION = "838ade0edb66dcffc5532d08ff6ed5c899abfb5c"
ROOT = Path(".")
DEST = ROOT / "data/interim/compact_monitor_regeneration_v2"
FINAL_MANIFEST = ROOT / "data/metadata/compact_scores_v2_manifest.json"
REPORT = ROOT / "reports/monitor_reproducibility/compact_monitor_v2_validation.md"
REPRO_SCRIPT = ROOT / "scripts/score_compact_monitor_colab_v2.py"

OLD_CACHE = ROOT / "data/interim/compact_scores.parquet"
CACHE_V3 = ROOT / "data/processed/monitor_score_cache_v3.parquet"
AUDITED_DATASET = (
    ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_members(archive: zipfile.ZipFile) -> None:
    for member in archive.infolist():
        target = (ROOT / member.filename).resolve()
        if not str(target).startswith(str(ROOT.resolve())):
            raise SystemExit(f"Unsafe ZIP path: {member.filename}")


def compare_scores(
    regenerated: pd.DataFrame,
    reference_path: Path,
    reference_score_column: str = "compact_unsafe_score",
) -> dict:
    if not reference_path.exists():
        return {
            "path": str(reference_path),
            "available": False,
        }

    reference = pd.read_parquet(reference_path).copy()
    reference["example_id"] = reference["example_id"].astype(str)

    merged = regenerated[
        ["example_id", "compact_unsafe_score", "compact_label"]
    ].merge(
        reference[
            ["example_id", reference_score_column, "compact_label"]
        ],
        on="example_id",
        how="inner",
        suffixes=("_regenerated", "_reference"),
        validate="one_to_one",
    )

    if len(merged) != 2159:
        raise SystemExit(
            f"Only {len(merged)} IDs matched {reference_path}"
        )

    differences = (
        merged["compact_unsafe_score_regenerated"]
        - merged[f"{reference_score_column}_reference"]
    ).abs()

    label_match = (
        merged["compact_label_regenerated"]
        == merged["compact_label_reference"]
    )

    result = {
        "path": str(reference_path),
        "available": True,
        "rows_compared": int(len(merged)),
        "exact_score_match": bool((differences == 0).all()),
        "max_abs_score_difference": float(differences.max()),
        "mean_abs_score_difference": float(differences.mean()),
        "score_match_atol_1e_6": bool(
            np.allclose(
                merged["compact_unsafe_score_regenerated"],
                merged[f"{reference_score_column}_reference"],
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "exact_label_match": bool(label_match.all()),
        "label_match_rate": float(label_match.mean()),
    }

    if not result["score_match_atol_1e_6"]:
        raise SystemExit(
            f"Regenerated scores differ materially from {reference_path}: "
            f"max difference={result['max_abs_score_difference']}"
        )
    if not result["exact_label_match"]:
        raise SystemExit(
            f"Regenerated labels differ from {reference_path}"
        )

    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: import_compact_monitor_v2.py <zip-path>"
        )

    zip_path = Path(sys.argv[1]).resolve()
    if not zip_path.exists():
        raise SystemExit(f"ZIP not found: {zip_path}")

    temp = ROOT / ".tmp_compact_monitor_regeneration_v2"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as archive:
        validate_zip_members(archive)
        archive.extractall(temp)

    candidates = list(
        temp.rglob("compact_scores_v2.parquet")
    )
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one compact_scores_v2.parquet, found {len(candidates)}"
        )

    source_dir = candidates[0].parent
    required_names = {
        "compact_scores_v2.parquet",
        "compact_scores_v2.csv",
        "compact_raw_outputs.jsonl",
        "run_manifest.json",
        "README.md",
    }
    missing = sorted(
        name for name in required_names
        if not (source_dir / name).exists()
    )
    if missing:
        raise SystemExit(f"Imported bundle missing: {missing}")

    run_manifest_path = source_dir / "run_manifest.json"
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )

    output_key_to_name = {
        "scores_parquet": "compact_scores_v2.parquet",
        "scores_csv": "compact_scores_v2.csv",
        "raw_outputs_jsonl": "compact_raw_outputs.jsonl",
    }
    for key, name in output_key_to_name.items():
        expected = run_manifest["outputs"][key]["sha256"]
        actual = sha256(source_dir / name)
        if expected != actual:
            raise SystemExit(
                f"Hash mismatch for {name}: {actual} != {expected}"
            )

    if sha256(AUDITED_DATASET) != run_manifest["dataset"]["sha256"]:
        raise SystemExit(
            "Imported run used a different audited dataset hash"
        )

    scores = pd.read_parquet(
        source_dir / "compact_scores_v2.parquet"
    ).copy()
    scores["example_id"] = scores["example_id"].astype(str)

    if len(scores) != 2159 or scores["example_id"].nunique() != 2159:
        raise SystemExit("Score row/ID validation failed")
    if scores["compact_unsafe_score"].isna().any():
        raise SystemExit("Missing compact scores")
    if not scores["compact_unsafe_score"].between(0, 1).all():
        raise SystemExit("Compact score outside [0, 1]")
    if not (
        scores["compact_model_revision"].astype(str)
        == PINNED_REVISION
    ).all():
        raise SystemExit("Unexpected compact model revision")

    raw_ids = []
    with (
        source_dir / "compact_raw_outputs.jsonl"
    ).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            raw_ids.append(str(record["example_id"]))
            if record["model_revision"] != PINNED_REVISION:
                raise SystemExit(
                    f"Wrong raw-output revision on line {line_number}"
                )

    if len(raw_ids) != 2159 or len(set(raw_ids)) != 2159:
        raise SystemExit("Raw-output row/ID validation failed")
    if set(raw_ids) != set(scores["example_id"]):
        raise SystemExit("Raw-output and score IDs differ")

    comparisons = {
        "old_interim_cache": compare_scores(
            scores,
            OLD_CACHE,
        ),
        "audited_cache_v3": compare_scores(
            scores,
            CACHE_V3,
        ),
    }

    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(source_dir, DEST)

    final_manifest = {
        "artifact": "compact_scores_v2_validated_import",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "regenerated_and_validated",
        "monitor_id": "koala_text_moderation",
        "model_id": "KoalaAI/Text-Moderation",
        "model_revision": PINNED_REVISION,
        "rows": 2159,
        "unique_example_id": 2159,
        "source_zip": {
            "original_path": str(zip_path),
            "sha256": sha256(zip_path),
        },
        "audited_dataset": {
            "path": str(AUDITED_DATASET),
            "sha256": sha256(AUDITED_DATASET),
        },
        "imported_run_manifest": run_manifest,
        "reproduction_script": {
            "path": str(REPRO_SCRIPT),
            "sha256": sha256(REPRO_SCRIPT),
            "note": (
                "Canonical script documenting the successful Colab "
                "standard-HTTP scoring procedure."
            ),
        },
        "imported_outputs": {
            name: {
                "path": str(DEST / name),
                "sha256": sha256(DEST / name),
            }
            for name in sorted(required_names)
        },
        "comparisons": comparisons,
        "latency_note": (
            "Scoring-run latency is provenance only. Controlled synchronized "
            "batch-size-1 measurements remain the source for latency claims."
        ),
    }

    FINAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MANIFEST.write_text(
        json.dumps(final_manifest, indent=2),
        encoding="utf-8",
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    old = comparisons["old_interim_cache"]
    cache_v3 = comparisons["audited_cache_v3"]

    REPORT.write_text(
        "\n".join(
            [
                "# Compact monitor v2 validation",
                "",
                "- Status: regenerated and validated",
                "- Rows: 2,159",
                "- Unique example IDs: 2,159",
                f"- Model revision: `{PINNED_REVISION}`",
                "- Raw logits and label probabilities preserved: yes",
                "- Inference fields: example_id, prompt, response",
                "- Labels/splits used during inference: no",
                "",
                "## Comparison with previous compact outputs",
                "",
                (
                    "- Old interim cache exact score match: "
                    f"{old.get('exact_score_match')}"
                ),
                (
                    "- Old interim cache maximum absolute difference: "
                    f"{old.get('max_abs_score_difference')}"
                ),
                (
                    "- Old interim cache exact label match: "
                    f"{old.get('exact_label_match')}"
                ),
                (
                    "- Audited cache v3 exact score match: "
                    f"{cache_v3.get('exact_score_match')}"
                ),
                (
                    "- Audited cache v3 maximum absolute difference: "
                    f"{cache_v3.get('max_abs_score_difference')}"
                ),
                (
                    "- Audited cache v3 exact label match: "
                    f"{cache_v3.get('exact_label_match')}"
                ),
                "",
                "The controlled timing benchmark, rather than this batch-16",
                "scoring run, remains the basis for compact-monitor latency.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    shutil.rmtree(temp)

    print("rows:", len(scores))
    print("unique IDs:", scores["example_id"].nunique())
    print("raw outputs:", len(raw_ids))
    print("model revision:", PINNED_REVISION)
    print("comparison with old interim cache:")
    print(json.dumps(comparisons["old_interim_cache"], indent=2))
    print("comparison with audited cache v3:")
    print(json.dumps(comparisons["audited_cache_v3"], indent=2))
    print("imported bundle:", DEST)
    print("final manifest:", FINAL_MANIFEST)
    print("report:", REPORT)


if __name__ == "__main__":
    main()
