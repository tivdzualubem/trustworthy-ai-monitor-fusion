"""Build the Stage-A paired-formulation handoff. Performs no prompt generation or scoring."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "results/stage_a_targeted_review_handoff_v1/c8_extension_validation/selected_base_intents.csv"
CONTRACT = ROOT / "configs/stage_a_pair_construction_contract_v1.json"
OUT = ROOT / "results/stage_a_pair_construction_v1"

EXPECTED_SELECTED_SHA = "ad2662f50a65d239a8fabd9cfebfad373264e49fdecfdb533364ca0b9b3c7f15"
CATEGORIES = [f"C{i}" for i in range(1, 9)]
TRANSFORMS = ["O1", "O2", "O3"]
EMPTY_FIELDS = [
    "base_intent_spec",
    "spec_review_decision",
    "spec_review_rationale",
    "spec_reviewer",
    "human_direct_text",
    "human_direct_lock_status",
    "model_direct_text",
    "model_generation_status",
    "human_obfuscated_text",
    "model_obfuscated_text",
]
PENDING_FIELDS = {
    "semantic_equivalence_review_status": "pending",
    "severity_review_status": "pending",
}
GATES = {
    "specification_approved": False,
    "human_direct_generation_authorized": False,
    "model_direct_generation_authorized": False,
    "obfuscation_generation_authorized": False,
    "semantic_equivalence_review_authorized": False,
    "severity_review_authorized": False,
    "guard_scoring_authorized": False,
    "q_estimation_authorized": False,
    "w0_authorized": False,
    "fresh_confirmatory_scoring_authorized": False,
    "prospective_design_frozen": False,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_for(base_intent_id: str) -> str:
    value = int(hashlib.sha256(base_intent_id.encode("utf-8")).hexdigest(), 16)
    return TRANSFORMS[value % len(TRANSFORMS)]


def generation_seed(base_intent_id: str) -> int:
    key = f"stage_a_pair_construction_v1|model_direct|{base_intent_id}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % 2147483647


def main() -> None:
    if sha256_file(SELECTED) != EXPECTED_SELECTED_SHA:
        raise ValueError("Selected base-intent input fingerprint mismatch")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    with SELECTED.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 104:
        raise ValueError(f"Expected 104 selected base intents, found {len(rows)}")
    counts = Counter(r["reviewed_category"] for r in rows)
    if counts != Counter({c: 13 for c in CATEGORIES}):
        raise ValueError(f"Expected exactly 13/category, observed {dict(counts)}")
    if any(r["selection_status"] != "selected_pending_downstream_reviews" for r in rows):
        raise ValueError("Selection status changed")
    if len({r["independent_intent_group"] for r in rows}) != 104:
        raise ValueError("Selected base-intent IDs must be unique")

    output_rows = []
    for r in rows:
        base_id = r["independent_intent_group"]
        row = {
            "base_intent_id": base_id,
            "selected_candidate_id": r["candidate_id"],
            "reviewed_category": r["reviewed_category"],
            "reference_source": r["source"],
            "reference_revision": r["revision"],
            "reference_source_label": r["source_label"],
            "reference_text": r["text"],
            "reference_text_hash": r["text_hash"],
            "selection_order_key": r["selection_order_key"],
            "base_intent_spec": "",
            "spec_review_decision": "",
            "spec_review_rationale": "",
            "spec_reviewer": "",
            "human_formulation_author": "project_author",
            "human_direct_text": "",
            "human_direct_lock_status": "",
            "model_generator_id": contract["model_direct"]["proposed_generator_id"],
            "model_generation_seed": str(generation_seed(base_id)),
            "model_direct_text": "",
            "model_generation_status": "",
            "obfuscation_transform_id": transform_for(base_id),
            "human_obfuscated_text": "",
            "model_obfuscated_text": "",
            **PENDING_FIELDS,
            **GATES,
        }
        output_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    worksheet = OUT / "base_intent_spec_worksheet.csv"
    with worksheet.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    transform_counts = Counter(r["obfuscation_transform_id"] for r in output_rows)
    audit = {
        "artifact_id": "stage_a_pair_construction_v1",
        "status": "awaiting_base_intent_specification_review",
        "selected_input_path": str(SELECTED.relative_to(ROOT)),
        "selected_input_sha256": sha256_file(SELECTED),
        "selected_base_intents": len(output_rows),
        "accepted_per_category": dict(sorted(counts.items())),
        "planned_variants_per_base_intent": 4,
        "planned_prompt_variants_total": len(output_rows) * 4,
        "obfuscation_transform_counts": dict(sorted(transform_counts.items())),
        "worksheet_sha256": sha256_file(worksheet),
        "scientific_review_performed_by_builder": False,
        "prompt_generation_performed": False,
        "obfuscation_performed": False,
        "monitor_scoring_performed": False,
        "q_estimation_performed": False,
        "downstream_gates": GATES,
        "pending_reviews": PENDING_FIELDS,
        "limitations": [
            "The public/development reference prompt is retained only as provenance for the selected base intent; it is not automatically treated as the human-authored formulation.",
            "Base-intent specifications remain blank until explicit review.",
            "Human/direct text must be authored and locked before model/direct output is revealed to the author.",
            "The proposed model generator is not authorized by this artifact.",
            "Semantic equivalence and severity remain separate downstream reviews.",
            "No guard scoring or q estimation is authorized.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "audit.txt").write_text(
        "\n".join([
            "Stage-A pair-construction handoff",
            f"Selected base intents: {len(output_rows)}",
            f"Category counts: {dict(sorted(counts.items()))}",
            f"Planned variants: {len(output_rows) * 4} (human/model x direct/obfuscated)",
            f"Transform assignments: {dict(sorted(transform_counts.items()))}",
            "Base-intent specifications: blank",
            "Human/direct formulations: blank",
            "Model/direct formulations: blank",
            "Obfuscated formulations: blank",
            "Semantic-equivalence review: pending",
            "Severity review: pending",
            "All generation/scoring/q/W0/prospective gates: false",
            "No prompt generation or monitor scoring performed.",
        ]) + "\n",
        encoding="utf-8",
    )

    print((OUT / "audit.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
