import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

from monitor_fusion.external_validation.attack_transforms import select_transform

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/stage_a_obfuscation_generation_contract_v1.json"
PAIR_CONTRACT = ROOT / "configs/stage_a_pair_construction_contract_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_stage_a_obfuscation_contract_is_frozen_and_downstream_closed():
    c = json.loads(CONTRACT.read_text())

    assert c["status"] == "frozen_pre_generation"
    assert c["transform_registry"]["transform_ids"] == ["O1", "O2", "O3"]

    assert c["assignment"]["same_transform_for_human_and_model"] is True
    assert c["assignment"]["adaptive_selection_allowed"] is False
    assert c["assignment"]["monitor_output_based_selection_allowed"] is False
    assert c["assignment"]["target_response_based_selection_allowed"] is False

    assert c["execution"]["manual_post_transform_editing_allowed"] is False
    assert c["execution"]["human_and_model_direct_text_must_remain_unchanged"] is True

    assert all(v is False for v in c["downstream_boundary"].values())


def test_all_frozen_source_hashes_match():
    c = json.loads(CONTRACT.read_text())

    for spec in c["source_inputs"].values():
        path = ROOT / spec["path"]
        assert path.exists()
        assert sha(path) == spec["sha256"]

    registry = ROOT / c["transform_registry"]["path"]
    assert sha(registry) == c["transform_registry"]["sha256"]


def test_preassigned_transform_assignments_are_unchanged():
    c = json.loads(CONTRACT.read_text())
    p = c["source_inputs"]["approved_pair_worksheet"]
    rows = read_csv(ROOT / p["path"])

    assert len(rows) == 104
    assert len({r["base_intent_id"] for r in rows}) == 104

    counts = Counter(r["obfuscation_transform_id"] for r in rows)
    assert counts == Counter({"O1": 38, "O2": 28, "O3": 38})

    for r in rows:
        assert r["obfuscation_transform_id"] == select_transform(
            r["base_intent_id"]
        )


def test_pair_contract_authorizes_only_obfuscation_execution():
    c = json.loads(PAIR_CONTRACT.read_text())

    o = c["obfuscation"]

    assert o["generation_authorized"] is True
    assert o["status"] == "frozen_generation_authorized"
    assert (
        o["generation_contract"]
        == "configs/stage_a_obfuscation_generation_contract_v1.json"
    )

    assert c["semantic_equivalence"]["status"] == "pending"
    assert c["severity"]["status"] == "pending"

    assert c["analysis_boundary"]["guard_scoring_authorized"] is False
    assert c["analysis_boundary"]["q_estimation_authorized"] is False
    assert c["analysis_boundary"]["w0_authorized"] is False
    assert c["analysis_boundary"]["fresh_confirmatory_scoring_authorized"] is False
