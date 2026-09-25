import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/stage_a_pair_construction_v1/base_intent_spec_worksheet_author_approved.csv"
CONTRACT = ROOT / "configs/stage_a_human_direct_authoring_contract_v1.json"
OUT = ROOT / "results/stage_a_human_direct_authoring_v1"
WORKSHEET = OUT / "human_direct_authoring_worksheet.csv"
MAP = OUT / "human_direct_authoring_map.csv"

EXPECTED_SOURCE_SHA = "3f962a4d4fac01029046286e978ad2852e80647abc27f3114c8ba6a8700b1e59"

FORBIDDEN_AUTHOR_COLUMNS = {
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


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_approved_and_pinned():
    assert sha(SOURCE) == EXPECTED_SOURCE_SHA
    rows = read_csv(SOURCE)
    assert len(rows) == 104
    assert Counter(r["reviewed_category"] for r in rows) == Counter(
        {f"C{i}": 13 for i in range(1, 9)}
    )
    assert all(r["specification_approved"] == "True" for r in rows)
    assert all(r["spec_review_decision"] == "include" for r in rows)


def test_contract_authorizes_only_human_authoring():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    b = c["scientific_boundary"]
    assert b["human_direct_authoring_authorized"] is True
    assert b["human_direct_text_currently_generated"] is False
    assert b["model_direct_generation_authorized"] is False
    assert b["obfuscation_generation_authorized"] is False
    assert b["semantic_equivalence_review_authorized"] is False
    assert b["severity_review_authorized"] is False
    assert b["guard_scoring_authorized"] is False
    assert b["q_estimation_authorized"] is False
    assert b["w0_authorized"] is False
    assert b["fresh_confirmatory_scoring_authorized"] is False
    assert b["prospective_design_frozen"] is False


def test_builder_creates_blinded_blank_authoring_handoff():
    source_before = SOURCE.read_bytes()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_stage_a_human_direct_authoring_handoff.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert SOURCE.read_bytes() == source_before

    work = read_csv(WORKSHEET)
    mapping = read_csv(MAP)

    assert len(work) == len(mapping) == 104
    assert len({r["authoring_item_id"] for r in work}) == 104
    assert len({r["authoring_item_id"] for r in mapping}) == 104
    assert len({r["base_intent_id"] for r in mapping}) == 104
    assert Counter(r["reviewed_category"] for r in mapping) == Counter(
        {f"C{i}": 13 for i in range(1, 9)}
    )

    assert not (FORBIDDEN_AUTHOR_COLUMNS & set(work[0]))
    assert all(r["base_intent_spec"].strip() for r in work)
    assert all(r["human_formulation_author"] == "project_author" for r in work)
    assert all(r["human_direct_text"] == "" for r in work)
    assert all(r["human_direct_review_decision"] == "" for r in work)
    assert all(r["human_direct_review_rationale"] == "" for r in work)
    assert all(r["human_direct_lock_status"] == "pending" for r in work)
    assert all(r["human_direct_text_hash"] == "" for r in work)

    first_work = WORKSHEET.read_bytes()
    first_map = MAP.read_bytes()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_stage_a_human_direct_authoring_handoff.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert WORKSHEET.read_bytes() == first_work
    assert MAP.read_bytes() == first_map
