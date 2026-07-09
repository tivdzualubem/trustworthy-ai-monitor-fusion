#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SELECTION_DIR = Path("reports/policy_selection")
OUT_DIR = Path("reports/frozen_policy")
SELECTED_PATH = SELECTION_DIR / "selected_baselines_fpr05.csv"
SELECTION_MANIFEST_PATH = SELECTION_DIR / "policy_selection_manifest.json"
CACHE_PATH = Path("data/processed/monitor_score_cache.parquet")

REQUIRED_FAMILIES = {
    "cheapest_monitor_only",
    "strongest_single_monitor",
    "all_monitors_always_on",
    "fixed_cascade",
    "cost_tuned_cascade",
    "learned_stacker_router",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def as_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(SELECTED_PATH)
    selection_manifest = json.loads(SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))

    families = set(selected["family"])
    missing = sorted(REQUIRED_FAMILIES - families)
    extra = sorted(families - REQUIRED_FAMILIES)
    if missing:
        raise SystemExit(f"Missing required frozen policy families: {missing}")
    if extra:
        raise SystemExit(f"Unexpected policy families: {extra}")

    if selection_manifest.get("calibration_final_and_shift_splits_used") is not False:
        raise SystemExit("Policy selection manifest indicates forbidden splits were used.")

    selected_sorted = selected.sort_values(
        ["recall", "avg_cost_ms", "fpr"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    primary = selected_sorted.iloc[0].to_dict()

    policies = []
    for row in selected.to_dict(orient="records"):
        details_raw = row.get("policy_details_json", "{}")
        try:
            details = json.loads(details_raw)
        except Exception:
            details = {"raw_policy_details_json": details_raw}

        policies.append(
            {
                "family": row["family"],
                "policy_id": row["policy_id"],
                "frozen_role": "primary_budget_aware_policy"
                if row["policy_id"] == primary["policy_id"]
                else "required_baseline",
                "selection_metrics": {
                    "recall": as_float_or_none(row.get("recall")),
                    "fpr": as_float_or_none(row.get("fpr")),
                    "precision": as_float_or_none(row.get("precision")),
                    "avg_cost_ms": as_float_or_none(row.get("avg_cost_ms")),
                    "median_cost_ms": as_float_or_none(row.get("median_cost_ms")),
                    "p95_cost_ms": as_float_or_none(row.get("p95_cost_ms")),
                    "normalized_avg_cost_vs_rule": as_float_or_none(
                        row.get("normalized_avg_cost_vs_rule")
                    ),
                },
                "policy_details": details,
            }
        )

    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_freeze": git_commit(),
        "freeze_rule": "Select the highest-recall candidate on policy_selection among required baseline families subject to policy_selection FPR <= 0.05; ties prefer lower average cost, then lower FPR.",
        "primary_policy_family": primary["family"],
        "primary_policy_id": primary["policy_id"],
        "allowed_selection_data": ["policy_train", "policy_selection"],
        "forbidden_for_policy_selection": ["calibration", "final_test", "held_out_shift"],
        "finite_sample_bounds_not_yet_computed": True,
        "bounds_stage_note": "Calibration/final/shift finite-sample bounds must be computed only after this frozen policy artifact exists.",
        "policy_selection_manifest": str(SELECTION_MANIFEST_PATH),
        "policy_selection_manifest_sha256": sha256_file(SELECTION_MANIFEST_PATH),
        "selected_baselines_path": str(SELECTED_PATH),
        "selected_baselines_sha256": sha256_file(SELECTED_PATH),
        "monitor_score_cache_path": str(CACHE_PATH),
        "monitor_score_cache_sha256": sha256_file(CACHE_PATH),
        "policies": policies,
    }

    frozen_path = OUT_DIR / "frozen_policies.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    summary = f"""# Frozen policies

Frozen at `{frozen["frozen_at"]}` from commit `{frozen["git_commit_at_freeze"]}`.

Policy selection used only `policy_train` and `policy_selection`. The `calibration`, `final_test`, and `held_out_shift` splits remain unused for policy selection.

## Primary frozen budget-aware policy

- Family: `{frozen["primary_policy_family"]}`
- Policy ID: `{frozen["primary_policy_id"]}`
- Freeze rule: {frozen["freeze_rule"]}

## Frozen selected policies

| family | policy_id | role | selection_recall | selection_fpr | selection_avg_cost_ms |
| --- | --- | --- | ---: | ---: | ---: |
"""

    for policy in policies:
        metrics = policy["selection_metrics"]
        summary += (
            f"| {policy['family']} | {policy['policy_id']} | {policy['frozen_role']} | "
            f"{metrics['recall']:.6g} | {metrics['fpr']:.6g} | {metrics['avg_cost_ms']:.6g} |\n"
        )

    summary += """
## Next stage

Use this frozen artifact for calibration, final-test, and held-out-shift evaluation. Any finite-sample bounds are valid only after this freeze step and only under the stated in-distribution exchangeability assumptions.
"""

    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    print("=== FROZEN PRIMARY POLICY ===")
    print(json.dumps(
        {
            "primary_policy_family": frozen["primary_policy_family"],
            "primary_policy_id": frozen["primary_policy_id"],
            "git_commit_at_freeze": frozen["git_commit_at_freeze"],
        },
        indent=2,
    ))

    print("\n=== FROZEN POLICY TABLE ===")
    print(selected[["family", "policy_id", "recall", "fpr", "avg_cost_ms", "normalized_avg_cost_vs_rule"]].to_string(index=False))

    print("\n=== WROTE ===")
    print(frozen_path)
    print(OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
