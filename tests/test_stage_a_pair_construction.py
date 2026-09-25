import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "results/stage_a_targeted_review_handoff_v1/c8_extension_validation/selected_base_intents.csv"
CONTRACT = ROOT / "configs/stage_a_pair_construction_contract_v1.json"
OUT = ROOT / "results/stage_a_pair_construction_v1"
WORKSHEET = OUT / "base_intent_spec_worksheet.csv"
EXPECTED_SELECTED_SHA = "ad2662f50a65d239a8fabd9cfebfad373264e49fdecfdb533364ca0b9b3c7f15"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_selected_input_fingerprint_and_shape():
    assert hashlib.sha256(SELECTED.read_bytes()).hexdigest() == EXPECTED_SELECTED_SHA
    rows = read_csv(SELECTED)
    assert len(rows) == 104
    assert Counter(r["reviewed_category"] for r in rows) == Counter({f"C{i}": 13 for i in range(1, 9)})
    assert len({r["independent_intent_group"] for r in rows}) == 104


def test_contract_keeps_generation_and_scoring_closed():
    c = json.loads(CONTRACT.read_text())
    assert c["model_direct"]["generation_authorized"] is False
    assert c["obfuscation"]["generation_authorized"] is False
    assert c["severity"]["severity_review_authorized"] is False
    assert c["analysis_boundary"]["guard_scoring_authorized"] is False
    assert c["analysis_boundary"]["q_estimation_authorized"] is False
    assert c["analysis_boundary"]["w0_authorized"] is False
    assert c["analysis_boundary"]["fresh_confirmatory_scoring_authorized"] is False
    assert c["analysis_boundary"]["prospective_design_frozen"] is False


def test_builder_outputs_blank_handoff_and_deterministic_assignments():
    before = SELECTED.read_bytes()
    subprocess.run([sys.executable, str(ROOT / "scripts/build_stage_a_pair_construction.py")],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    assert SELECTED.read_bytes() == before
    rows = read_csv(WORKSHEET)
    assert len(rows) == 104
    assert Counter(r["reviewed_category"] for r in rows) == Counter({f"C{i}": 13 for i in range(1, 9)})
    assert len({r["base_intent_id"] for r in rows}) == 104
    assert set(r["obfuscation_transform_id"] for r in rows) <= {"O1", "O2", "O3"}

    blank_fields = [
        "base_intent_spec", "spec_review_decision", "spec_review_rationale", "spec_reviewer",
        "human_direct_text", "human_direct_lock_status", "model_direct_text",
        "model_generation_status", "human_obfuscated_text", "model_obfuscated_text",
    ]
    assert all(r[f] == "" for r in rows for f in blank_fields)
    assert all(r["semantic_equivalence_review_status"] == "pending" for r in rows)
    assert all(r["severity_review_status"] == "pending" for r in rows)
    assert all(r["guard_scoring_authorized"] == "False" for r in rows)
    assert all(r["q_estimation_authorized"] == "False" for r in rows)

    first = {r["base_intent_id"]: (r["obfuscation_transform_id"], r["model_generation_seed"]) for r in rows}
    subprocess.run([sys.executable, str(ROOT / "scripts/build_stage_a_pair_construction.py")],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    rows2 = read_csv(WORKSHEET)
    second = {r["base_intent_id"]: (r["obfuscation_transform_id"], r["model_generation_seed"]) for r in rows2}
    assert first == second
