from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_stage_a_readout_transfer_kill_inputs.py"
CONTRACT = ROOT / "configs" / "stage_a_readout_transfer_kill_contract_v1.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("builder", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _synthetic_source() -> pd.DataFrame:
    rows = []
    conditions = [
        "direct_decoded",
        "O1",
        "O2",
        "O3_raw",
        "O3_raw_plus_decoded",
    ]
    for i in range(38):
        pair_id = f"FU{i+1:02d}"
        for case_type in ("harmful", "benign"):
            direct = f"sample {pair_id} {case_type}"
            for condition in conditions:
                text = (
                    direct
                    if condition == "direct_decoded"
                    else f"{condition}:{direct}"
                )
                rows.append({
                    "followup_example_id":
                        f"{pair_id}::{case_type}::{condition}",
                    "pair_id": pair_id,
                    "case_type": case_type,
                    "source_anchor": "x",
                    "base_intent_id": f"{pair_id}-{case_type}",
                    "matched_harmful_base_intent_id": pair_id,
                    "reviewed_category":
                        "C1" if case_type == "harmful" else "BENIGN",
                    "condition": condition,
                    "transform_id": condition,
                    "prompt_text": text,
                    "direct_char_length": len(direct),
                    "matched_harmful_direct_char_length": len(direct),
                    "abs_direct_char_length_difference": 0,
                })
    return pd.DataFrame(rows)


def test_contract_is_frozen_and_o2_primary():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["status"] == "frozen_pre_scoring_design"
    assert c["primary_o2_experiment"]["representations"] == ["direct", "O2"]
    assert c["primary_o2_experiment"]["expected_prompt_rows"] == 152
    assert c["o3_ablation"]["expected_prompt_rows"] == 304
    assert c["boundaries"]["frozen_model_weights"] is True
    assert (
        c["boundaries"]["no_generated_semantic_pairs_for_probe_training"]
        is True
    )


def test_builder_preserves_pairs_and_fold_boundaries():
    b = _load_builder()
    df = _synthetic_source()
    b.require_source(df)
    folds = b.assign_folds(sorted(df["pair_id"].unique()), 5)
    probe = b.build_probe_input(df, folds)
    o3 = b.build_o3_ablation(df, folds)

    assert len(probe) == 152
    assert probe["semantic_case_id"].nunique() == 76
    assert len(o3) == 304

    for _, g in probe.groupby("semantic_case_id"):
        assert set(g["representation"]) == {"direct", "O2"}
        assert g["fold_id"].nunique() == 1

    for _, g in probe.groupby("pair_id"):
        assert g["fold_id"].nunique() == 1
        assert set(g["case_type"]) == {"harmful", "benign"}

    expected_o3 = {
        "direct_plain",
        "base64_payload_only",
        "plain_wrapper",
        "encoded_wrapper",
    }
    for _, g in o3.groupby("semantic_case_id"):
        assert set(g["condition"]) == expected_o3
        assert g["fold_id"].nunique() == 1
