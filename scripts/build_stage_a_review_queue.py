"""Nominate a small, unreviewed public/development Stage-A queue, offline.

Uses the existing no-decisions candidate CSV. Does not label, select base intents,
generate pairs, review equivalence/severity, score guards, or authorize W0.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_a_review_screening_v1.json"
CROSSWALK = ROOT / "configs/stage_a_public_crosswalk_v1.json"
ONTOLOGY = ROOT / "configs/external_validation_common_ontology_v1.json"
SOURCE_AUDIT = ROOT / "data/metadata/source_audit.json"
INPUT = ROOT / "results/stage_a_public_selection_v1/review_template.csv"
OUT = ROOT / "results/stage_a_targeted_review_v1"
REVIEW_FIELDS = ["review_decision", "reviewed_category", "review_rationale",
                 "reviewer", "independent_intent_group"]
REQUIRED = ["candidate_id", "source", "revision", "source_label", "text", "text_hash"] + REVIEW_FIELDS
SCREEN_FIELDS = ["proposed_screen_category", "screening_reason", "screen_categories",
                 "ambiguity_flag", "screening_priority", "screening_order_key"]
BOUNDARIES = {
    "scope": "development/public candidate nomination only",
    "row_level_review_performed": False,
    "final_category_assignment_performed": False,
    "base_intent_selection_authorized": False,
    "pair_generation_authorized": False,
    "semantic_equivalence_review_performed": False,
    "severity_review_performed": False,
    "guard_scoring_authorized": False,
    "w0_authorized": False,
    "fresh_confirmatory_scoring_authorized": False,
    "prospective_primary_design_frozen": False,
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_key(text: str) -> str:
    # Identical to the existing candidate builder; do not alter exported text.
    return digest(re.sub(r"\s+", " ", text).strip().casefold())


def validate_config(cfg: dict, ontology: dict) -> None:
    categories = {c["id"] for c in ontology["categories"]}
    if set(cfg["rules"]) != categories or categories != {f"C{i}" for i in range(1, 9)}:
        raise ValueError("Screening rules must cover the frozen C1-C8 ontology exactly")
    if not 20 <= cfg["target_queue_per_category"] <= 25:
        raise ValueError("Queue target must be 20-25 per category")
    if cfg["shortage_threshold"] != 20 or cfg["accepted_independent_intents_needed"] != 13:
        raise ValueError("Unexpected Stage-A review targets")
    for rule in cfg["rules"].values():
        for pattern in rule["terms"].values():
            re.compile(pattern, re.I)
    for pattern in cfg["c3_previous_probe"]["patterns"]:
        re.compile(pattern, re.I)


def validate_candidates(rows: list[dict], crosswalk: dict, sources: dict) -> tuple[list[dict], list[dict]]:
    bt = crosswalk["public_screening_sources"]["beavertails_evaluation"]
    allowed = {
        "jailbreakbench": sources["sources"]["jailbreakbench"]["revision"],
        "wildguardtest": sources["sources"]["wildguardmix"]["revision"],
        "beavertails_evaluation": bt["revision"],
    }
    seen_ids, seen_hashes, unique, duplicates = set(), {}, [], []
    for row in sorted(rows, key=lambda r: r.get("candidate_id", "")):
        if set(REQUIRED) - row.keys():
            raise ValueError("Candidate CSV lacks required review/provenance columns")
        if not row["candidate_id"] or row["candidate_id"] in seen_ids:
            raise ValueError("Candidate IDs must be nonempty and unique")
        seen_ids.add(row["candidate_id"])
        if any(row[field].strip() for field in REVIEW_FIELDS):
            raise ValueError("Input already contains scientific review; use an unreviewed candidate snapshot")
        if row["source"] not in allowed or row["revision"] != allowed[row["source"]]:
            raise ValueError("Unknown source or unpinned source revision")
        if not row["text"].strip() or row["text_hash"] != text_key(row["text"]):
            raise ValueError("Empty text or text-hash mismatch")
        if row["source"] == "beavertails_evaluation":
            if row["source_label"] != bt["source_category"]:
                raise ValueError("Only registered BeaverTails self_harm candidates may be screened")
            for column, field in [("source_repo_id", "repo_id"), ("source_split", "split"),
                                  ("source_data_file", "data_file"), ("source_license", "license")]:
                if row.get(column) != bt[field]:
                    raise ValueError(f"BeaverTails provenance mismatch: {column}")
        if row["text_hash"] in seen_hashes:
            kept = seen_hashes[row["text_hash"]]
            duplicates.append({"kept_candidate_id": kept, "removed_candidate": row})
        else:
            seen_hashes[row["text_hash"]] = row["candidate_id"]
            unique.append(row)
    return unique, duplicates


def screen(row: dict, cfg: dict) -> dict[str, dict]:
    if row["source"] == "beavertails_evaluation":
        # It may never nominate another study category, regardless of its text.
        return {"C5": {"priority": 1, "reason": "pinned BeaverTails self_harm source hint only"}}
    matches = {}
    for category, rule in sorted(cfg["rules"].items()):
        labels = rule["labels"].get(row["source"], [])
        label_match = row["source_label"] in labels
        terms = []
        for name, pattern in rule["terms"].items():
            found = re.search(pattern, row["text"], re.I)
            if found:
                terms.append(f"text:{name}={found.group(0)!r}")
        if terms or (label_match and rule["label_only_allowed"]):
            evidence = ([f"source_label:{row['source']}:{row['source_label']}"] if label_match else []) + terms
            matches[category] = {
                "priority": 0 if label_match and terms else (1 if label_match else 2),
                "reason": "; ".join(evidence),
            }
    return matches


def build_queue(rows: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    pools = {c: [] for c in cfg["rules"]}
    nominations = {c: [] for c in cfg["rules"]}
    overlaps, evidence_by_id = [], {}
    for row in rows:
        evidence = screen(row, cfg)
        evidence_by_id[row["candidate_id"]] = evidence
        for category in evidence:
            nominations[category].append(row)
        if len(evidence) > 1:
            overlaps.append({"candidate_id": row["candidate_id"], "text_hash": row["text_hash"],
                             "ambiguity_flag": "multiple_category_signals", "evidence": evidence})
        elif evidence:
            category, match = next(iter(evidence.items()))
            pools[category].append(dict(row, proposed_screen_category=category,
                screening_reason=match["reason"], screen_categories=category, ambiguity_flag="",
                screening_priority=match["priority"],
                screening_order_key=digest(f"{cfg['seed']}|{category}|{row['text_hash']}")))
    queue, summary = [], {}
    for category, pool in sorted(pools.items()):
        pool.sort(key=lambda r: (r["screening_priority"], r["screening_order_key"], r["candidate_id"]))
        chosen = pool[:cfg["target_queue_per_category"]]
        queue.extend(chosen)
        summary[category] = {
            "screened_matches_before_overlap": len(nominations[category]),
            "screened_source_composition": dict(sorted(Counter(r["source"] for r in nominations[category]).items())),
            "unambiguous_screening_pool": len(pool),
            "overlap_nominations_withheld": len(nominations[category]) - len(pool),
            "queued_candidates": len(chosen),
            "queue_source_composition": dict(sorted(Counter(r["source"] for r in chosen).items())),
            "shortfall_to_20": max(0, cfg["shortage_threshold"] - len(chosen)),
            "reserve_candidate_ids": [r["candidate_id"] for r in pool[len(chosen):]],
        }
    probe_patterns = cfg["c3_previous_probe"]["patterns"]
    probe = [r for r in rows if r["source"] != "beavertails_evaluation" and
             any(re.search(p, r["text"], re.I) for p in probe_patterns)]
    queued_ids = {r["candidate_id"] for r in queue}
    expected = set(cfg["c3_previous_probe"]["expected_candidate_ids"])
    observed = {r["candidate_id"] for r in probe}
    probe_audit = {
        "reported_count": cfg["c3_previous_probe"]["reported_candidate_count"],
        "reproduced_count": len(probe), "exact_ids_reproduced": observed == expected,
        "missing_reported_ids": sorted(expected - observed), "additional_ids": sorted(observed - expected),
        "queued_count": sum(r["candidate_id"] in queued_ids for r in probe),
        "overlap_withheld_count": sum(len(evidence_by_id[r["candidate_id"]]) > 1 for r in probe),
        "additional_C3_nominations_beyond_probe": sorted(
            r["candidate_id"] for r in nominations["C3"] if r["candidate_id"] not in observed),
        "source_label_counts": dict(sorted(Counter(f"{r['source']} | {r['source_label']}" for r in probe).items())),
        "candidates": [{"candidate_id": r["candidate_id"], "text_hash": r["text_hash"],
                        "evidence": evidence_by_id[r["candidate_id"]],
                        "in_main_queue": r["candidate_id"] in queued_ids,
                        "ambiguity_flag": "multiple_category_signals" if len(evidence_by_id[r["candidate_id"]]) > 1 else ""}
                       for r in sorted(probe, key=lambda r: r["candidate_id"])],
    }
    return queue, {
        "by_category": summary,
        "categories_below_20": [c for c, s in summary.items() if s["queued_candidates"] < 20],
        "overlap_candidates_withheld": len(overlaps),
        "overlap_nominations_withheld": sum(len(r["evidence"]) for r in overlaps),
        "overlap_details": sorted(overlaps, key=lambda r: r["candidate_id"]),
        "no_screening_signal_count": sum(not e for e in evidence_by_id.values()),
        "c3_previous_probe": probe_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    crosswalk = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    sources = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    validate_config(cfg, json.loads(ONTOLOGY.read_text(encoding="utf-8")))
    with args.input.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError("Candidate CSV must not be empty")
    if set(columns) & set(SCREEN_FIELDS):
        raise ValueError("Input must be the candidate snapshot, not an existing screening queue")
    unique, duplicates = validate_candidates(rows, crosswalk, sources)
    queue, audit = build_queue(unique, cfg)
    queue_path = args.out_dir / "review_queue.csv"
    if queue_path.exists():
        with queue_path.open(encoding="utf-8", newline="") as f:
            if any(any(r.get(field, "").strip() for field in REVIEW_FIELDS) for r in csv.DictReader(f)):
                raise ValueError("Refusing to overwrite a queue containing scientific review")
    audit.update({
        "screening_id": cfg["screening_id"], "seed": cfg["seed"],
        "input_candidate_count": len(rows), "unique_candidate_count": len(unique),
        "input_source_counts": dict(sorted(Counter(r["source"] for r in rows).items())),
        "input_source_label_counts": dict(sorted(Counter(f"{r['source']} | {r['source_label']}" for r in rows).items())),
        "queue_count": len(queue), "exact_text_duplicates_removed": len(duplicates),
        "duplicate_details": duplicates,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [CONFIG, CROSSWALK, ONTOLOGY, SOURCE_AUDIT, Path(__file__)]},
        "public_screening_sources": crosswalk["public_screening_sources"],
        "source_provenance": sources["sources"],
        "execution_boundaries": BOUNDARIES,
        "status": "awaiting_explicit_row_level_review; no accepted intents",
        "limitations": [
            "Screening signals may be incidental, benign, defensive, mixed, or in the wrong primary category; review remains mandatory.",
            "No text-level screen establishes distinct underlying intents. independent_intent_group remains blank for reviewer assignment.",
            "The queue is a purposive screen, not a representative sample or a prospective allocation freeze.",
            "The existing selection --decisions interface still requires the full candidate snapshot. Do not fabricate exclusions for unreviewed rows or pass this subset directly to that interface.",
        ],
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns) + SCREEN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(queue)
    audit["queue_sha256"] = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["Stage-A targeted review queue (screening only)",
             f"Input: {len(rows)}; unique: {len(unique)}; queued: {len(queue)}",
             f"Exact text duplicates removed: {len(duplicates)}; overlapping candidates withheld: {audit['overlap_candidates_withheld']}"]
    for category, info in audit["by_category"].items():
        lines.append(f"{category}: screened={info['screened_matches_before_overlap']} singleton={info['unambiguous_screening_pool']} queued={info['queued_candidates']} sources={info['queue_source_composition']}")
    lines += [f"Categories below 20: {audit['categories_below_20']}",
              f"Original C3 probe: {audit['c3_previous_probe']['reproduced_count']}; exact IDs reproduced={audit['c3_previous_probe']['exact_ids_reproduced']}",
              "Review decisions, categories, rationales, reviewers and independence groups remain blank.",
              "All generation/scoring/W0/confirmation gates remain blocked; no semantic or severity review."]
    lines += [f"{k}: {v}" for k, v in BOUNDARIES.items()] + audit["limitations"]
    (args.out_dir / "audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
