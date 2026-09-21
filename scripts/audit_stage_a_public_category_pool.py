from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
SOURCE_AUDIT = ROOT / "data/metadata/source_audit.json"
OUT_JSON = ROOT / "results/stage_a_public_category_pool_audit_v1.json"
OUT_TXT = ROOT / "results/stage_a_public_category_pool_audit_v1.txt"

JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "behaviors"
JBB_SPLIT = "harmful"


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\\s+", " ", str(value)).strip()


def counts(series: pd.Series) -> dict[str, int]:
    clean = series.map(norm)
    clean = clean.loc[clean != ""]
    return {str(k): int(v) for k, v in clean.value_counts().sort_index().items()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not SOURCE_AUDIT.exists():
        raise FileNotFoundError(SOURCE_AUDIT)

    df = pd.read_parquet(DATA)

    required = {
        "example_id",
        "prompt",
        "source_dataset",
        "source_record_id",
        "prompt_harmful",
        "harm_category",
        "y",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    harmful_prompt = df.loc[
        pd.to_numeric(df["prompt_harmful"], errors="coerce").eq(1)
    ].copy()

    harmful_prompt["prompt_norm"] = harmful_prompt["prompt"].map(
        lambda x: re.sub(r"\\s+", " ", str(x)).strip().lower()
    )

    source_summary = {}
    for source, g in harmful_prompt.groupby("source_dataset", dropna=False):
        source_summary[str(source)] = {
            "rows": int(len(g)),
            "unique_prompts": int(g["prompt_norm"].nunique()),
            "response_Y1_rows": int(pd.to_numeric(g["y"], errors="coerce").eq(1).sum()),
            "harm_category_counts": counts(g["harm_category"]),
        }

    wild = harmful_prompt.loc[
        harmful_prompt["source_dataset"].astype(str).eq("wildguardtest")
    ].copy()

    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    jbb_revision = audit["sources"]["jailbreakbench"]["revision"]
    jbb = load_dataset(
        JBB_REPO,
        JBB_CONFIG,
        split=JBB_SPLIT,
        revision=jbb_revision,
    ).to_pandas()
    jbb.columns = [str(c).strip().lower() for c in jbb.columns]

    jbb_required = {"index", "goal", "category", "source", "behavior"}
    jbb_missing = sorted(jbb_required - set(jbb.columns))
    if jbb_missing:
        raise ValueError(f"JBB harmful split missing columns: {jbb_missing}")

    payload = {
        "audit_id": "stage_a_public_category_pool_audit_v1",
        "created_at": utc_now(),
        "scope": "development/public data only; no monitor scoring",
        "w0_authorized": False,
        "fresh_confirmatory_scoring_authorized": False,
        "existing_harmful_prompt_pool": {
            "rows": int(len(harmful_prompt)),
            "unique_prompts": int(harmful_prompt["prompt_norm"].nunique()),
            "source_summary": source_summary,
        },
        "wildguard_harmful_prompt_pool": {
            "rows": int(len(wild)),
            "unique_prompts": int(wild["prompt_norm"].nunique()),
            "harm_category_counts": counts(wild["harm_category"]),
        },
        "jailbreakbench_harmful_behavior_pool": {
            "revision": jbb_revision,
            "rows": int(len(jbb)),
            "unique_goals": int(jbb["goal"].map(norm).nunique()),
            "category_counts": counts(jbb["category"]),
            "source_counts": counts(jbb["source"]),
        },
        "important_boundary": {
            "automatic_mapping_to_C1_C8_performed": False,
            "reason": (
                "The public source taxonomies are not identical to the frozen study ontology. "
                "Stage A must use an explicit auditable crosswalk or row-level eligibility rule "
                "before selecting the 100-200 base cases."
            ),
            "next_step": (
                "Define the Stage-A public-source-to-C1-C8 eligibility/crosswalk, then select "
                "a balanced set of genuinely independent base intents before generating paired "
                "human/model and direct/obfuscated variants."
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")

    lines = [
        "=== STAGE-A PUBLIC CATEGORY POOL AUDIT ===",
        f"Existing harmful-prompt rows: {len(harmful_prompt)}",
        f"Existing unique harmful prompts: {harmful_prompt['prompt_norm'].nunique()}",
        "",
        "=== BY SOURCE ===",
    ]
    for source, info in source_summary.items():
        lines.append(
            f"{source}: rows={info['rows']} unique_prompts={info['unique_prompts']} "
            f"response_Y1_rows={info['response_Y1_rows']}"
        )

    lines += [
        "",
        "=== WILDGUARD HARMFUL-PROMPT CATEGORY COUNTS ===",
    ]
    wc = payload["wildguard_harmful_prompt_pool"]["harm_category_counts"]
    lines.extend([f"{k}: {v}" for k, v in wc.items()])

    lines += [
        "",
        "=== JAILBREAKBENCH HARMFUL-BEHAVIOR CATEGORY COUNTS ===",
    ]
    jc = payload["jailbreakbench_harmful_behavior_pool"]["category_counts"]
    lines.extend([f"{k}: {v}" for k, v in jc.items()])

    lines += [
        "",
        "BOUNDARY:",
        "- No automatic C1-C8 mapping was performed.",
        "- No Stage-A pair generation has started.",
        "- No monitor scoring has started.",
        "- W0 remains blocked.",
    ]
    OUT_TXT.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    print("\\n".join(lines))
    print()
    print(f"WROTE {OUT_JSON}")
    print(f"WROTE {OUT_TXT}")


if __name__ == "__main__":
    main()
