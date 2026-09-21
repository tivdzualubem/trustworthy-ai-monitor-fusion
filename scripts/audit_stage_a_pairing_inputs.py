from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
SOURCE_AUDIT = ROOT / "data/metadata/source_audit.json"

OUT_JSON = ROOT / "results/stage_a_pairing_input_audit_v1.json"
OUT_CSV = ROOT / "results/stage_a_jbb_base_case_inventory_v1.csv"

JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "behaviors"
JBB_SPLIT = "harmful"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip().lower()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def main() -> None:
    if not UNIFIED.exists():
        raise FileNotFoundError(f"Missing development dataset: {UNIFIED}")
    if not SOURCE_AUDIT.exists():
        raise FileNotFoundError(f"Missing source audit: {SOURCE_AUDIT}")

    frame = pd.read_parquet(UNIFIED)

    required_stage_a_fields = {
        "base_intent_id",
        "formulation_source",
        "representation",
        "severity",
    }
    observed_fields = set(map(str, frame.columns))
    missing_stage_a_fields = sorted(required_stage_a_fields - observed_fields)

    existing_summary: dict[str, object] = {
        "rows": int(len(frame)),
        "columns": sorted(map(str, frame.columns)),
        "missing_required_stage_a_pairing_fields": missing_stage_a_fields,
        "source_counts": (
            frame["source_dataset"].astype(str).value_counts().sort_index().to_dict()
            if "source_dataset" in frame.columns else {}
        ),
        "nonnull_goal_rows": (
            int(frame["goal"].notna().sum()) if "goal" in frame.columns else 0
        ),
        "nonnull_harm_category_rows": (
            int(frame["harm_category"].notna().sum())
            if "harm_category" in frame.columns else 0
        ),
        "nonnull_attack_family_rows": (
            int(frame["attack_family"].notna().sum())
            if "attack_family" in frame.columns else 0
        ),
    }

    if "goal" in frame.columns:
        goal_frame = frame.loc[frame["goal"].notna(), ["goal"]].copy()
        goal_frame["goal_norm"] = goal_frame["goal"].map(normalize_text)
        counts = goal_frame["goal_norm"].value_counts()
        existing_summary["unique_nonempty_goals"] = int(
            (goal_frame["goal_norm"] != "").sum()
            if goal_frame.empty else goal_frame.loc[
                goal_frame["goal_norm"] != "", "goal_norm"
            ].nunique()
        )
        existing_summary["goals_with_multiple_rows"] = int((counts > 1).sum())
        existing_summary["max_rows_for_same_goal"] = (
            int(counts.max()) if not counts.empty else 0
        )

    natural_pairing_supported = not missing_stage_a_fields
    existing_summary["required_2x2_pairing_supported_as_stored"] = bool(
        natural_pairing_supported
    )

    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    revision = audit["sources"]["jailbreakbench"]["revision"]

    public = load_dataset(
        JBB_REPO,
        JBB_CONFIG,
        split=JBB_SPLIT,
        revision=revision,
    ).to_pandas()

    public.columns = [str(c).strip().lower() for c in public.columns]
    required = {"index", "goal", "target", "behavior", "category", "source"}
    missing = required.difference(public.columns)
    if missing:
        raise ValueError(f"JBB harmful split missing columns: {sorted(missing)}")

    if len(public) != 100:
        raise ValueError(f"Expected 100 JBB harmful behaviors; found {len(public)}")

    if public["index"].nunique() != len(public):
        raise ValueError("JBB harmful behavior index is not unique.")

    if public["goal"].map(normalize_text).nunique() != len(public):
        raise ValueError("JBB harmful goals are not unique after normalization.")

    inventory = pd.DataFrame(
        {
            "base_intent_id": [
                f"stagea-jbb-{int(i):03d}" for i in public["index"].astype(int)
            ],
            "source_dataset": JBB_REPO,
            "source_revision": revision,
            "source_index": public["index"].astype(int),
            "public_direct_text": public["goal"].astype(str),
            "behavior": public["behavior"].astype(str),
            "public_category": public["category"].astype(str),
            "public_source": public["source"].astype(str),
            "variant_hd_required": True,
            "variant_md_required": True,
            "variant_ho_required": True,
            "variant_mo_required": True,
            "semantic_equivalence_status": "pending",
            "severity_status": "pending",
            "guard_scoring_status": "not_started",
        }
    ).sort_values("source_index")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(OUT_CSV, index=False)

    payload = {
        "audit_id": "stage_a_pairing_input_audit_v1",
        "created_at": utc_now(),
        "scope": "development/public data only",
        "w0_authorized": False,
        "fresh_confirmatory_scoring_authorized": False,
        "existing_development_dataset": existing_summary,
        "conclusion": {
            "existing_unified_dataset_directly_satisfies_required_2x2_pairing": bool(
                natural_pairing_supported
            ),
            "reason": (
                "The existing dataset was built as prompt-response evaluation rows and "
                "does not store the full base-intent x formulation-source x representation "
                "structure required for genuine Stage-A pairing."
            ),
            "public_base_case_inventory_ready": True,
            "public_base_case_count": int(len(inventory)),
            "base_case_source": (
                "JailbreakBench/JBB-Behaviors behaviors/harmful, pinned to the "
                "existing project source revision"
            ),
            "next_step": (
                "Construct model-direct formulations of the same base intents, apply "
                "direct/obfuscated variants, then check semantic equivalence and severity "
                "as separate fields before guard-panel scoring."
            ),
        },
        "jbb_inventory": {
            "rows": int(len(inventory)),
            "category_counts": inventory["public_category"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "source_counts": inventory["public_source"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "output_csv": str(OUT_CSV.relative_to(ROOT)),
        },
    }

    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== STAGE-A PAIRING INPUT AUDIT ===")
    print(f"development rows: {len(frame)}")
    print("missing Stage-A pairing fields:", missing_stage_a_fields)
    print(
        "existing dataset directly supports required 2x2 pairing:",
        natural_pairing_supported,
    )
    print()
    print(f"public JBB harmful base cases: {len(inventory)}")
    print("category counts:")
    print(inventory["public_category"].value_counts().sort_index().to_string())
    print()
    print(f"WROTE {OUT_JSON}")
    print(f"WROTE {OUT_CSV}")
    print()
    print("BOUNDARY: no W0, no fresh confirmatory scoring, no prospective freeze.")


if __name__ == "__main__":
    main()
