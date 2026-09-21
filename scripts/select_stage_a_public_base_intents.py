"""Audit public harmful-intent coverage and select only manually reviewed cases.

This script performs no pair generation, monitor scoring, W0, or fresh confirmation.
Run once to create a review template, fill review_decision/category/rationale, then
run again with --decisions PATH. Selection is fail-closed when coverage is short.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_a_public_crosswalk_v1.json"
ONTOLOGY = ROOT / "configs/external_validation_common_ontology_v1.json"
SOURCE_AUDIT = ROOT / "data/metadata/source_audit.json"
DATA = ROOT / "data/processed/unified_dataset_label_audited_v1.parquet"
OUT = ROOT / "results/stage_a_public_selection_v1"


def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def key(text: str) -> str:
    return hashlib.sha256(norm(text).casefold().encode("utf-8")).hexdigest()


def beavertails_candidates(source: dict) -> pd.DataFrame:
    public = load_dataset(
        source["repo_id"], split=source["split"], revision=source["revision"],
        data_files={source["split"]: source["data_file"]},
    ).to_pandas()
    required = {source["text_field"], "category", "category_id"}
    if required - set(public):
        raise ValueError(f"BeaverTails columns missing: {sorted(required - set(public))}")
    if len(public) != source["expected_split_rows"]:
        raise ValueError("Unexpected BeaverTails split size")
    # Preserve the zero-based row in the pinned source file before filtering.
    public["source_row_index"] = range(len(public))
    public = public.loc[public.category.eq(source["source_category"])].copy()
    if len(public) != source["expected_category_rows"] or not public.category_id.eq(
        source["source_category_id"]
    ).all():
        raise ValueError("Unexpected BeaverTails screening category count or ID")
    if not public[source["text_field"]].map(lambda s: isinstance(s, str) and bool(norm(s))).all():
        raise ValueError("BeaverTails screening prompts must be nonempty strings")
    return pd.DataFrame({
        "candidate_id": public.source_row_index.map(lambda i: f"beavertails_evaluation:{source['split']}:{i}"),
        "source": "beavertails_evaluation",
        "revision": source["revision"],
        "source_label": public.category,
        "text": public[source["text_field"]],
        "source_repo_id": source["repo_id"],
        "source_split": source["split"],
        "source_data_file": source["data_file"],
        "source_row_index": public.source_row_index,
        "source_category_id": public.category_id,
        "source_license": source["license"],
        "source_license_url": source["license_url"],
        "source_license_evidence_url": source["license_evidence_url"],
        "candidate_role": source["candidate_role"],
    })


def candidates() -> pd.DataFrame:
    provenance = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    rev = provenance["sources"]["jailbreakbench"]["revision"]
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful", revision=rev).to_pandas()
    jbb.columns = [str(c).strip().lower() for c in jbb.columns]
    needed = {"index", "goal", "category"}
    if needed - set(jbb):
        raise ValueError(f"JBB columns missing: {sorted(needed - set(jbb))}")
    rows = []
    for _, row in jbb.iterrows():
        rows.append((f"jbb:{row['index']}", "jailbreakbench", rev, norm(row["category"]), norm(row["goal"])))

    wild = pd.read_parquet(DATA)
    needed = {"source_dataset", "source_record_id", "prompt_harmful", "prompt", "harm_category"}
    if needed - set(wild):
        raise ValueError(f"Development columns missing: {sorted(needed - set(wild))}")
    wild = wild.loc[wild.source_dataset.astype(str).eq("wildguardtest") &
                    pd.to_numeric(wild.prompt_harmful, errors="coerce").eq(1)]
    for _, row in wild.iterrows():
        rows.append((f"wildguard:{row['source_record_id']}", "wildguardtest",
                     provenance["sources"]["wildguardmix"]["revision"],
                     norm(row["harm_category"]), norm(row["prompt"])))
    frame = pd.DataFrame(rows, columns=["candidate_id", "source", "revision", "source_label", "text"])
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.concat([
        frame, beavertails_candidates(cfg["public_screening_sources"]["beavertails_evaluation"])
    ], ignore_index=True)
    for column in ["source_row_index", "source_category_id"]:
        frame[column] = frame[column].astype("Int64")
    frame = frame.loc[frame.text.ne("")].copy()
    frame["text_hash"] = frame.text.map(key)
    frame = frame.sort_values(["source", "candidate_id"]).drop_duplicates("text_hash")
    if frame.candidate_id.duplicated().any():
        raise ValueError("Candidate IDs are not unique")
    return frame.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, help="Completed row-level review CSV")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    categories = {c["id"] for c in json.loads(ONTOLOGY.read_text(encoding="utf-8"))["categories"]}
    frame = candidates()
    hints = {k.casefold(): v for k, v in cfg["source_label_hints"].items()}
    frame["category_hint"] = frame.source_label.map(lambda s: hints.get(s.casefold(), ""))
    source = cfg["public_screening_sources"]["beavertails_evaluation"]
    # This source-specific hint never fills reviewed_category or review_decision.
    frame.loc[
        frame.source.eq("beavertails_evaluation") & frame.source_label.eq(source["source_category"]),
        "category_hint",
    ] = source["category_hint"]
    OUT.mkdir(parents=True, exist_ok=True)
    selected_path = OUT / "selected_base_intents.csv"
    if selected_path.exists():
        selected_path.unlink()
    template = frame.assign(review_decision="", reviewed_category="", review_rationale="",
                            reviewer="", independent_intent_group="")
    template.to_csv(OUT / "review_template.csv", index=False)

    report = {
        "crosswalk_id": cfg["crosswalk_id"],
        "candidate_count": len(frame),
        "source_counts": frame.source.value_counts().sort_index().to_dict(),
        "public_screening_sources": cfg["public_screening_sources"],
        "source_label_counts": frame.groupby(["source", "source_label"]).size().to_dict(),
        "hint_counts": frame.category_hint.value_counts().to_dict(),
        "w0_authorized": False,
        "fresh_confirmatory_scoring_authorized": False,
        "pair_generation_authorized": False,
        "guard_scoring_authorized": False,
        "semantic_equivalence_review_status": "not_started",
        "severity_review_status": "not_started",
        "selection_status": "awaiting_row_level_review",
    }
    if args.decisions:
        review = pd.read_csv(args.decisions, dtype=str).fillna("")
        required = {"candidate_id", "text_hash", "review_decision", "reviewed_category",
                    "review_rationale", "reviewer", "independent_intent_group"}
        if required - set(review):
            raise ValueError(f"Decision columns missing: {sorted(required - set(review))}")
        if review.candidate_id.duplicated().any() or set(review.candidate_id) != set(frame.candidate_id):
            raise ValueError("Decision IDs must match candidate IDs exactly once")
        reviewed = frame.merge(review[list(required)], on="candidate_id", validate="one_to_one",
                               suffixes=("", "_review"))
        if (reviewed.text_hash != reviewed.text_hash_review).any():
            raise ValueError("Candidate text changed since review; regenerate decisions")
        if not reviewed.review_decision.isin(["include", "exclude"]).all():
            raise ValueError("Every candidate needs include or exclude")
        included = reviewed.loc[reviewed.review_decision.eq("include")].copy()
        if not included.reviewed_category.isin(categories).all():
            raise ValueError("Included cases require a valid C1-C8 category")
        if (included.source.eq("beavertails_evaluation") &
                included.reviewed_category.ne(source["category_hint"])).any():
            raise ValueError("BeaverTails C5 screening cases with another primary category must be excluded")
        for col in ["review_rationale", "reviewer", "independent_intent_group"]:
            if included[col].str.strip().eq("").any():
                raise ValueError(f"Included cases require {col}")
        if included.independent_intent_group.duplicated().any():
            raise ValueError("Multiple included rows share an underlying intent group")
        counts = included.reviewed_category.value_counts().reindex(sorted(categories), fill_value=0)
        quota = int(cfg["target_per_category"])
        report["eligible_by_category"] = counts.to_dict()
        report["shortfall_by_category"] = {c: max(0, quota - int(n)) for c, n in counts.items()}
        if (counts >= quota).all():
            selected = included.assign(order=included.independent_intent_group.map(
                lambda s: key(f"{cfg['selection_seed']}:{s}")))
            selected = selected.sort_values(["reviewed_category", "order"]).groupby(
                "reviewed_category", group_keys=False).head(quota)
            selected = selected.drop(columns=["order", "text_hash_review"])
            selected.to_csv(selected_path, index=False)
            report["selection_status"] = "selected_pending_semantic_and_severity_review"
            report["selected_count"] = len(selected)
        else:
            report["selection_status"] = "blocked_insufficient_category_coverage"
    report["source_label_counts"] = {f"{a} | {b}": int(n) for (a, b), n in report["source_label_counts"].items()}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
