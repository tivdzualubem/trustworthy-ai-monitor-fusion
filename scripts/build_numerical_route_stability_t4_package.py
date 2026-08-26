#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/mnt/c/Users/NOAH/Downloads")
OUTZIP = DOWNLOADS / "numerical_route_stability_t4_package.zip"
STAGE = ROOT / ".tmp_numerical_route_stability_t4_package"

PROTOCOL = ROOT / "configs/numerical_route_stability_v1.json"
BENCHMARK = ROOT / "scripts/benchmark_numerical_route_stability_t4.py"
DEFS = ROOT / "reports/evaluation_measurement_pilot_v1/cpu/primary_policy_definitions.json"
PRED = ROOT / "reports/evaluation_measurement_pilot_v1/t4/prediction_mismatches.csv"
DEV = ROOT / "data/processed/v2_development_view/unified_dataset_label_audited_v1.development.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "code").mkdir(parents=True)
    (STAGE / "data").mkdir(parents=True)

    pred = pd.read_csv(PRED)
    if len(pred) != 5:
        raise RuntimeError("Expected five prediction mismatch rows.")
    ids = sorted(pred["example_id"].astype(str).unique())
    if len(ids) != 5:
        raise RuntimeError("Expected five unique mismatch examples.")

    # Package only operational fields required for stability testing; no labels.
    dev = pd.read_parquet(DEV)
    required = ["example_id", "source_dataset", "prompt", "response"]
    examples = dev.loc[
        dev["example_id"].astype(str).isin(ids), required
    ].copy()
    if len(examples) != 5:
        raise RuntimeError("Could not recover all five development mismatch examples.")
    forbidden = {"y", "y_original", "label"}
    if forbidden.intersection(examples.columns):
        raise RuntimeError("Labels must not enter the hardware-stability package.")

    mismatch_columns = [
        "policy_id",
        "stack",
        "policy_kind",
        "target_rate",
        "acquisition_threshold",
        "decision_threshold",
        "example_id",
        "reference_rule_score",
        "reference_compact_unsafe_score",
        "reference_qwen_prompt_response_score",
    ]
    mismatch = pred[mismatch_columns].copy()

    shutil.copy2(BENCHMARK, STAGE / "code" / BENCHMARK.name)
    shutil.copy2(PROTOCOL, STAGE / "data" / PROTOCOL.name)
    shutil.copy2(DEFS, STAGE / "data" / DEFS.name)
    examples.to_parquet(STAGE / "data" / "near_threshold_examples.parquet", index=False)
    mismatch.to_csv(STAGE / "data" / "mismatch_rows.csv", index=False)

    files = {}
    for path in sorted(STAGE.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(STAGE))] = sha256(path)

    manifest = {
        "artifact": "numerical_route_stability_v1_t4_package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "controlled_precision_runtime_hardware_stability_on_five_mismatch_examples",
        "labels_in_package": False,
        "mismatch_examples": 5,
        "files": files,
    }
    (STAGE / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    if OUTZIP.exists():
        OUTZIP.unlink()
    with zipfile.ZipFile(OUTZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE))

    print("NUMERICAL_ROUTE_STABILITY_T4_PACKAGE=PASS")
    print(f"package={OUTZIP}")
    print(f"sha256={sha256(OUTZIP)}")


if __name__ == "__main__":
    main()
