import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("human_lock", ROOT / "scripts/validate_stage_a_human_direct_lock.py")
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)
EXPECTED_SHA = "9d21312869d2b7d5192af7e22356efc522d8b005abe96cdbc004a95c51859d04"
FILES = [lock.CONTRACT, lock.SOURCE, lock.OUT / "human_direct_authoring_worksheet.csv",
         lock.OUT / "human_direct_authoring_map.csv", lock.OUT / "audit.json",
         Path("configs/stage_a_pair_construction_contract_v1.json"),
         Path("results/stage_a_pair_construction_v1/base_intent_spec_worksheet.csv")]


@pytest.fixture
def snapshot(tmp_path):
    for rel in FILES:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    return tmp_path


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_committed_lock_is_complete_pinned_and_read_only():
    before = {p: (ROOT / p).read_bytes() for p in FILES}
    result = lock.validate()
    assert result["rows"] == result["unique_ids"] == result["locked_rows"] == 104
    assert result["sha256"] == EXPECTED_SHA
    assert result["condition"] == "human/direct" and result["author"] == "project_author"
    assert result["category_counts"] == {f"C{i}": 13 for i in range(1, 9)}
    assert result["downstream_gates_opened"] is False
    assert {p: (ROOT / p).read_bytes() for p in FILES} == before


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "spec", "empty",
                                     "author", "review", "rationale", "pending", "hash", "extra_column"])
def test_invalid_rows_cannot_validate_even_with_refreshed_artifact_hash(snapshot, mutation):
    path = snapshot / lock.OUT / "human_direct_authoring_worksheet.csv"
    fields, rows = lock.read_csv(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["authoring_item_id"] = rows[0]["authoring_item_id"]
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "extra_column":
        fields.append("model_direct_text")
        for row in rows:
            row["model_direct_text"] = "unexpected"
    else:
        field, value = {"spec": ("base_intent_spec", "changed"),
                        "empty": ("human_direct_text", ""),
                        "author": ("human_formulation_author", "other"),
                        "review": ("human_direct_review_decision", "needs_revision"),
                        "rationale": ("human_direct_review_rationale", ""),
                        "pending": ("human_direct_lock_status", "pending"),
                        "hash": ("human_direct_text_hash", "0" * 64)}[mutation]
        rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    contract_path = snapshot / lock.CONTRACT
    c = json.loads(contract_path.read_text())
    c["human_direct_lock"]["sha256"] = lock.sha(path)
    write_json(contract_path, c)
    with pytest.raises(ValueError):
        lock.validate(snapshot)


@pytest.mark.parametrize("relative", [lock.SOURCE, lock.OUT / "human_direct_authoring_map.csv",
                                     lock.OUT / "human_direct_authoring_worksheet.csv"])
def test_fingerprints_fail_closed(snapshot, relative):
    path = snapshot / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        lock.validate(snapshot)


@pytest.mark.parametrize("gate", lock.CLOSED_GATES)
def test_human_lock_cannot_open_downstream_gates(snapshot, gate):
    path = snapshot / lock.CONTRACT
    c = json.loads(path.read_text())
    c["scientific_boundary"][gate] = True
    write_json(path, c)
    with pytest.raises(ValueError, match="Downstream gate opened"):
        lock.validate(snapshot)


def test_wrong_condition_rejected(snapshot):
    path = snapshot / lock.CONTRACT
    c = json.loads(path.read_text())
    c["human_direct_lock"]["condition"] = "model/direct"
    write_json(path, c)
    with pytest.raises(ValueError, match="condition"):
        lock.validate(snapshot)


def test_downstream_formulation_rejected(snapshot):
    path = snapshot / "results/stage_a_pair_construction_v1/base_intent_spec_worksheet.csv"
    fields, rows = lock.read_csv(path)
    rows[0]["model_direct_text"] = "unexpected"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Downstream formulation created"):
        lock.validate(snapshot)


def test_pair_generation_gate_rejected(snapshot):
    path = snapshot / "configs/stage_a_pair_construction_contract_v1.json"
    c = json.loads(path.read_text())
    c["model_direct"]["generation_authorized"] = True
    write_json(path, c)
    with pytest.raises(ValueError, match="Pair generation opened"):
        lock.validate(snapshot)


def test_builder_cannot_erase_the_canonical_lock():
    directory = ROOT / lock.OUT
    before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/build_stage_a_human_direct_authoring_handoff.py")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to overwrite locked" in result.stderr
    assert {p: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before


def test_builder_cannot_erase_partially_authored_rows(tmp_path):
    builder = ROOT / "scripts/build_stage_a_human_direct_authoring_handoff.py"
    command = [sys.executable, str(builder), "--output-dir", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True)
    path = tmp_path / "human_direct_authoring_worksheet.csv"
    text = path.read_text().replace("project_author,,,,pending,", "project_author,draft,,,pending,", 1)
    path.write_text(text)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to overwrite nonblank" in result.stderr
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before
