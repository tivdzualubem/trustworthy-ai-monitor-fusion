import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/stage_a_guard_panel_prompt_contract_v1.json"
RUNNER = ROOT / "scripts/run_stage_a_guard_panel_kaggle.py"


def test_llama_stage_a_uses_frozen_meta_raw_checkpoint():
    cfg = json.loads(CONTRACT.read_text(encoding="utf-8"))
    c = cfg["monitors"]["llama_guard_3_1b"]

    assert c["provenance"] == "official_meta_direct_checkpoint"
    assert c["runtime"] == (
        "official Meta llama-models raw-checkpoint implementation"
    )
    assert c["llama_models_version"] == "0.3.0"
    assert c["runtime_dtype"] == "bfloat16"
    assert c["quantized"] is False
    assert c["verdict_token_ids"] == {
        "safe": 19193,
        "unsafe": 39257,
    }

    assert c["checkpoint_integrity"]["sha256"]["consolidated.00.pth"] == (
        "10a6c07a5556355be37791a225cdf63352c37de1733910a2ab4b3e70077d8fd3"
    )

    assert "revision" not in c
    assert c["pre_scoring_amendment"]["study_examples_scored_before_amendment"] == 0
    assert c["pre_scoring_amendment"]["q_estimated_before_amendment"] is False
    assert c["pre_scoring_amendment"]["W0_opened_before_amendment"] is False


def test_llama_amendment_does_not_open_stage_a_gates():
    cfg = json.loads(CONTRACT.read_text(encoding="utf-8"))
    b = cfg["boundaries"]

    assert b["stage_a_synthetic_preflight_authorized"] is True
    assert b["stage_a_guard_scoring_authorized"] is False
    assert b["stage_a_q_estimation_authorized"] is False
    assert b["W0_authorized"] is False


def test_runner_uses_raw_llama_not_blocked_hf_adapter():
    text = RUNNER.read_text(encoding="utf-8")

    assert "def build_llama_raw" in text
    assert "Llama3.build(" in text
    assert "def llama_raw_generate" in text

    assert "acf7aafa60f0410f8f42b1fa35e077d705892029" not in text
