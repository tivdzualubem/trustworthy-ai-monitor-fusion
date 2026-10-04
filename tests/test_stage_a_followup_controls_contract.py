import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/stage_a_followup_controls_contract_v1.json"
INPUT = ROOT / "results/stage_a_followup_controls_v1/followup_prompts.csv"
RUNNER = ROOT / "scripts/run_stage_a_followup_controls_kaggle.py"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data
    ).hexdigest()


def cfg():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_followup_contract_is_frozen_and_nonconfirmatory():
    c = cfg()
    assert c["status"] == "authorized_pre_scoring_followup_control_experiment"
    assert c["scope"] == "development_followup_only"
    assert c["boundaries"]["W0_used"] is False
    assert c["boundaries"]["fresh_confirmatory_scoring"] is False
    assert c["boundaries"]["post_result_case_selection"] is False
    assert c["boundaries"]["threshold_retuning"] is False


def test_followup_input_blob_and_shape():
    c = cfg()
    assert git_blob_sha(INPUT) == c["source"]["input_blob_sha"]

    df = pd.read_csv(INPUT, keep_default_na=False)
    assert len(df) == 380
    assert df["followup_example_id"].nunique() == 380
    assert df["pair_id"].nunique() == 38
    assert df["case_type"].value_counts().to_dict() == {
        "harmful": 190,
        "benign": 190,
    }
    assert df["condition"].value_counts().to_dict() == {
        "direct_decoded": 76,
        "O1": 76,
        "O2": 76,
        "O3_raw": 76,
        "O3_raw_plus_decoded": 76,
    }


def test_each_pair_has_benign_and_harmful_all_conditions():
    df = pd.read_csv(INPUT, keep_default_na=False)
    expected = {
        "direct_decoded",
        "O1",
        "O2",
        "O3_raw",
        "O3_raw_plus_decoded",
    }

    for _, g in df.groupby("pair_id"):
        assert set(g["case_type"]) == {"harmful", "benign"}
        for case_type in ("harmful", "benign"):
            x = g[g["case_type"].eq(case_type)]
            assert len(x) == 5
            assert set(x["condition"]) == expected


def test_same_harmful_intents_receive_o1_o2_o3():
    df = pd.read_csv(INPUT, keep_default_na=False)
    harmful = df[df["case_type"].eq("harmful")]
    assert harmful["base_intent_id"].nunique() == 38

    for _, g in harmful.groupby("base_intent_id"):
        assert set(g["condition"]) == {
            "direct_decoded",
            "O1",
            "O2",
            "O3_raw",
            "O3_raw_plus_decoded",
        }


def test_runner_reuses_frozen_guard_runtime_and_does_not_open_w0():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'import_module("run_stage_a_guard_panel_kaggle")' in text
    assert "base.verify_runtime(runtime_lock)" in text
    assert "base.build_llama_raw(c)" in text
    assert "base.llama_raw_generate(generator, c, prompt)" in text
    assert "guardian_config={\"criteria_id\": \"harm\"}" in text
    assert '"threshold_retuning_performed": False' in text
    assert '"W0_used": False' in text
    assert '"fresh_confirmatory_scoring_performed": False' in text
