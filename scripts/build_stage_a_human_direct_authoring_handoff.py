"""Build a blinded Stage-A human/direct authoring handoff.

This script does not generate human formulations, model formulations, obfuscations,
monitor outputs, or q estimates.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/stage_a_pair_construction_v1/base_intent_spec_worksheet_author_approved.csv"
CONTRACT = ROOT / "configs/stage_a_human_direct_authoring_contract_v1.json"
OUT = ROOT / "results/stage_a_human_direct_authoring_v1"

EXPECTED_SOURCE_SHA = "3f962a4d4fac01029046286e978ad2852e80647abc27f3114c8ba6a8700b1e59"
CATEGORIES = [f"C{i}" for i in range(1, 9)]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def spec_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def authoring_order_key(base_intent_id: str) -> str:
    return hashlib.sha256(
        f"stage_a_human_direct_authoring_v1|{base_intent_id}".encode("utf-8")
    ).hexdigest()


def main() -> None:
    if sha256_file(SOURCE) != EXPECTED_SOURCE_SHA:
        raise ValueError("Approved specification artifact fingerprint mismatch")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["source_specifications"]["sha256"] != EXPECTED_SOURCE_SHA:
        raise ValueError("Contract/source fingerprint mismatch")

    with SOURCE.open(encoding="utf-8", newline="") as f:
        src = list(csv.DictReader(f))

    if len(src) != 104:
        raise ValueError(f"Expected 104 approved specifications, found {len(src)}")

    counts = Counter(r["reviewed_category"] for r in src)
    if counts != Counter({c: 13 for c in CATEGORIES}):
        raise ValueError(f"Expected 13/category, found {dict(counts)}")

    if len({r["base_intent_id"] for r in src}) != 104:
        raise ValueError("base_intent_id must be unique")

    for r in src:
        if r["specification_approved"] != "True":
            raise ValueError(f"Unapproved specification: {r['base_intent_id']}")
        if r["spec_review_decision"] != "include":
            raise ValueError(f"Non-included specification: {r['base_intent_id']}")
        if not r["base_intent_spec"].strip():
            raise ValueError(f"Blank approved specification: {r['base_intent_id']}")
        if r["human_direct_text"] or r["model_direct_text"]:
            raise ValueError("Unexpected formulation text in approved specification artifact")

    ordered = sorted(src, key=lambda r: authoring_order_key(r["base_intent_id"]))

    OUT.mkdir(parents=True, exist_ok=True)

    authoring_rows = []
    map_rows = []
    for i, r in enumerate(ordered, start=1):
        item_id = f"H{i:03d}"
        key = authoring_order_key(r["base_intent_id"])
        sh = spec_hash(r["base_intent_spec"])

        authoring_rows.append(
            {
                "authoring_item_id": item_id,
                "base_intent_spec": r["base_intent_spec"],
                "human_formulation_author": "project_author",
                "human_direct_text": "",
                "human_direct_review_decision": "",
                "human_direct_review_rationale": "",
                "human_direct_lock_status": "pending",
                "human_direct_text_hash": "",
            }
        )
        map_rows.append(
            {
                "authoring_item_id": item_id,
                "base_intent_id": r["base_intent_id"],
                "selected_candidate_id": r["selected_candidate_id"],
                "reviewed_category": r["reviewed_category"],
                "base_intent_spec_sha256": sh,
                "authoring_order_key": key,
                "selection_order_key": r["selection_order_key"],
            }
        )

    authoring_path = OUT / "human_direct_authoring_worksheet.csv"
    with authoring_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(authoring_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(authoring_rows)

    map_path = OUT / "human_direct_authoring_map.csv"
    with map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(map_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(map_rows)

    # The author-facing worksheet deliberately excludes provenance, category,
    # reference prompt, model generator/seed, and transform assignment.
    forbidden_authoring_columns = {
        "selected_candidate_id",
        "reviewed_category",
        "reference_source",
        "reference_revision",
        "reference_source_label",
        "reference_text",
        "reference_text_hash",
        "model_generator_id",
        "model_generation_seed",
        "model_direct_text",
        "obfuscation_transform_id",
    }
    if forbidden_authoring_columns.intersection(authoring_rows[0]):
        raise ValueError("Author-facing worksheet leaks protected context")

    audit = {
        "artifact_id": "stage_a_human_direct_authoring_handoff_v1",
        "status": "human_authoring_ready_model_generation_not_authorized",
        "source_specification_path": str(SOURCE.relative_to(ROOT)),
        "source_specification_sha256": sha256_file(SOURCE),
        "rows": 104,
        "category_counts_in_blinded_map": dict(sorted(counts.items())),
        "authoring_order_rule": "ascending SHA256('stage_a_human_direct_authoring_v1|' + base_intent_id)",
        "author_facing_columns": list(authoring_rows[0]),
        "author_facing_worksheet_sha256": sha256_file(authoring_path),
        "blinded_map_sha256": sha256_file(map_path),
        "human_direct_text_nonblank_rows": 0,
        "model_direct_text_generated": False,
        "obfuscation_performed": False,
        "semantic_equivalence_review_status": "pending",
        "severity_review_status": "pending",
        "guard_scoring_performed": False,
        "q_estimation_performed": False,
        "authoring_rules": contract["human_direct_authoring_rules"],
        "scientific_boundary": contract["scientific_boundary"],
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (OUT / "audit.txt").write_text(
        "\n".join(
            [
                "Stage-A human/direct authoring handoff",
                "Rows: 104",
                "Category counts in blinded map: 13 each for C1-C8",
                "Author-facing worksheet contains approved specification only; source/reference/model/transform context is excluded.",
                "Human/direct text rows completed: 0",
                "Human/direct lock status: pending for all rows",
                "Model/direct generation: not authorized",
                "Obfuscation generation: not authorized",
                "Semantic-equivalence review: pending",
                "Severity review: pending",
                "Guard scoring: not authorized",
                "q estimation: not authorized",
                "W0: not authorized",
                "Fresh confirmatory scoring: not authorized",
                "Prospective design frozen: false",
                f"Authoring worksheet SHA256: {sha256_file(authoring_path)}",
                f"Blinded map SHA256: {sha256_file(map_path)}",
                "No formulation text was generated by this builder.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print((OUT / "audit.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
