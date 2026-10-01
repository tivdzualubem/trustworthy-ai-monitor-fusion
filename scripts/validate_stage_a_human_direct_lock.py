"""Read-only validation of the Stage-A human/direct lock; opens no downstream gate."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results/stage_a_human_direct_authoring_v1")
CONTRACT = Path("configs/stage_a_human_direct_authoring_contract_v1.json")
SOURCE = Path("results/stage_a_pair_construction_v1/base_intent_spec_worksheet_author_approved.csv")
SOURCE_SHA = "3f962a4d4fac01029046286e978ad2852e80647abc27f3114c8ba6a8700b1e59"
MAP_SHA = "97d4b5cee6fd6eb4aa305d839078c45b2d48189b05e322fb2236203f968acc94"
PREVIOUS_SHA = "8b32f54917e1230a68c80cbdeb687f8cf5805283161decc7c01365ba9a41abf0"
FIELDS = ["authoring_item_id", "base_intent_spec", "human_formulation_author",
          "human_direct_text", "human_direct_review_decision", "human_direct_review_rationale",
          "human_direct_lock_status", "human_direct_text_hash"]
CLOSED_GATES = ["model_direct_generation_authorized", "obfuscation_generation_authorized",
                "semantic_equivalence_review_authorized", "severity_review_authorized",
                "guard_scoring_authorized", "q_estimation_authorized", "w0_authorized",
                "fresh_confirmatory_scoring_authorized", "prospective_design_frozen"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    require(bool(fields) and all(None not in r and None not in r.values() for r in rows),
            "Malformed CSV")
    return fields, rows


def validate(root: Path = ROOT) -> dict:
    contract = json.loads((root / CONTRACT).read_text(encoding="utf-8"))
    lock = contract["human_direct_lock"]
    path = OUT / "human_direct_authoring_worksheet.csv"
    require(lock["path"] == path.as_posix(), "Unexpected lock path")
    require(lock["condition"] == "human/direct", "Wrong experimental condition")
    require(lock["author"] == lock["reviewer"] == "project_author", "Wrong author/reviewer")
    require(lock["rows"] == lock["locked_rows"] == 104, "Wrong lock counts")
    require(lock["previous_worksheet_sha256"] == PREVIOUS_SHA, "Wrong handoff fingerprint")
    require(lock["blinded_map_sha256"] == MAP_SHA, "Wrong map fingerprint")
    require(sha(root / path) == lock["sha256"], "Locked worksheet fingerprint mismatch")
    require(sha(root / SOURCE) == SOURCE_SHA == contract["source_specifications"]["sha256"],
            "Frozen specification fingerprint mismatch")
    require(contract["source_specifications"]["path"] == SOURCE.as_posix(), "Wrong source path")
    require(sha(root / OUT / "human_direct_authoring_map.csv") == MAP_SHA,
            "Blinded map fingerprint mismatch")
    require(contract["human_direct_authoring_rules"]["author"] == "project_author" and
            contract["review_and_lock"]["reviewer"] == "project_author", "Provenance changed")
    boundary = contract["scientific_boundary"]
    require(boundary["human_direct_authoring_authorized"] is True and
            boundary["human_direct_text_currently_generated"] is True, "Human lock state incomplete")
    require(all(boundary[g] is False for g in CLOSED_GATES), "Downstream gate opened")
    require(all(boundary[g] == "pending" for g in
                ["semantic_equivalence_review_status", "severity_review_status"]),
            "Downstream review status changed")
    fields, rows = read_csv(root / path)
    require(fields == FIELDS, "Unexpected author-facing columns")
    expected = [f"H{i:03d}" for i in range(1, 105)]
    require([r["authoring_item_id"] for r in rows] == expected, "Missing, duplicate or reordered IDs")
    _, mapping = read_csv(root / OUT / "human_direct_authoring_map.csv")
    _, source = read_csv(root / SOURCE)
    require([r["authoring_item_id"] for r in mapping] == expected, "Invalid mapping IDs")
    require(len(source) == len({r["base_intent_id"] for r in source}) == 104, "Invalid source IDs")
    require(len({r["base_intent_id"] for r in mapping}) == 104, "Duplicate mapped base intent")
    counts = Counter(r["reviewed_category"] for r in mapping)
    require(counts == Counter({f"C{i}": 13 for i in range(1, 9)}), "Invalid category balance")
    by_id = {r["base_intent_id"]: r for r in source}
    for row, mapped in zip(rows, mapping):
        spec = by_id[mapped["base_intent_id"]]
        require(row["base_intent_spec"] == spec["base_intent_spec"], "Frozen intent alignment changed")
        require(hashlib.sha256(row["base_intent_spec"].encode()).hexdigest() ==
                mapped["base_intent_spec_sha256"], "Mapped specification hash mismatch")
        require(all(mapped[k] == spec[k] for k in
                    ["selected_candidate_id", "reviewed_category", "selection_order_key"]), "Mapping changed")
        require(spec["specification_approved"] == "True" and spec["spec_review_decision"] == "include",
                "Unapproved frozen specification")
        require(row["human_formulation_author"] == "project_author", "Wrong formulation author")
        require(row["human_direct_review_decision"] == "include" and
                row["human_direct_lock_status"] == "locked", "Incomplete review/lock")
        require(bool(row["human_direct_text"].strip()) and
                bool(row["human_direct_review_rationale"].strip()), "Missing text or review rationale")
        require(hashlib.sha256(row["human_direct_text"].encode("utf-8")).hexdigest() ==
                row["human_direct_text_hash"], "Formulation text hash mismatch")
        require(all(not spec[k] for k in ["human_direct_text", "model_direct_text",
                                         "human_obfuscated_text", "model_obfuscated_text"]),
                "Frozen specification artifact contains formulation text")
    pair = json.loads((root / "configs/stage_a_pair_construction_contract_v1.json").read_text())
    require(pair["model_direct"]["generation_authorized"] is False and
            pair["obfuscation"]["generation_authorized"] is False, "Pair generation opened")
    require(pair["semantic_equivalence"]["status"] == pair["severity"]["status"] == "pending" and
            pair["severity"]["severity_review_authorized"] is False, "Pair review gate opened")
    require(all(v is False for k, v in pair["analysis_boundary"].items() if k != "stage_a_exploratory"),
            "Pair analysis gate opened")
    _, construction = read_csv(root / "results/stage_a_pair_construction_v1/base_intent_spec_worksheet.csv")
    require(len(construction) == 104, "Construction row count changed")
    for row in construction:
        require(all(not row[k] for k in ["model_direct_text", "human_obfuscated_text", "model_obfuscated_text"]),
                "Downstream formulation created")
        require(all(row[g] == "False" for g in CLOSED_GATES), "Construction gate opened")
        require(row["semantic_equivalence_review_status"] == row["severity_review_status"] == "pending",
                "Construction review status changed")
    audit = json.loads((root / OUT / "audit.json").read_text())
    require(audit["author_facing_worksheet_sha256"] == lock["sha256"] and
            audit["human_direct_lock"] == lock, "Lock audit mismatch")
    require(audit["scientific_boundary"] == boundary, "Audit gate mismatch")
    require(audit["human_direct_text_nonblank_rows"] == audit["human_direct_locked_rows"] == 104,
            "Audit row counts incomplete")
    require(all(audit[g] is False for g in ["model_direct_text_generated", "obfuscation_performed",
                                          "guard_scoring_performed", "q_estimation_performed"]),
            "Audit downstream activity recorded")
    return {"validation": "PASS", "condition": "human/direct", "author": "project_author",
            "rows": 104, "unique_ids": 104, "locked_rows": 104, "category_counts": dict(counts),
            "artifact": path.as_posix(), "sha256": lock["sha256"], "downstream_gates_opened": False}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
