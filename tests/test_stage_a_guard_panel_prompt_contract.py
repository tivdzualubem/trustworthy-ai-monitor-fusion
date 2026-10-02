import csv
import json
from collections import Counter
from pathlib import Path

from monitor_fusion.external_validation.stage_a_guard_panel_prompt import (
    clopper_pearson_95,
    parse_granite_guardian_prompt_output,
    parse_llama_guard_prompt_output,
    parse_qwen3guard_prompt_output,
)

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"configs/stage_a_guard_panel_prompt_contract_v1.json"
INPUT=ROOT/"results/stage_a_guard_panel_input_v1/stage_a_guard_panel_prompts.csv"


def cfg(): return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_four_prompt_guards_and_all_scoring_gates_closed():
    c=cfg(); assert c["status"]=="frozen_pre_preflight"
    assert set(c["monitors"])=={"llama_guard_3_1b","shieldgemma_2b","granite_guardian_3_3_8b","qwen3guard_gen_4b"}
    b=c["boundaries"]; assert b["stage_a_synthetic_preflight_authorized"] is True
    for k in ["stage_a_guard_scoring_authorized","stage_a_q_estimation_authorized","common_mode_analysis_authorized","matched_effect_analysis_authorized","W0_authorized","fresh_confirmatory_scoring_authorized","prospective_claim_freeze_authorized"]: assert b[k] is False
    assert b["threshold_retuning_from_stage_a_allowed"] is False


def test_exact_frozen_pair_denominators():
    p=cfg()["analysis_plan_frozen_before_scoring"]["paired_comparisons"]
    assert p["human_representation"]["expected_n"]==104
    assert p["model_representation"]["expected_n"]==55
    assert p["source_direct_strict"]["expected_n"]==52
    assert p["source_obfuscated_strict"]["expected_n"]==52
    assert p["source_direct_semantic_sensitivity"]["expected_n"]==55
    assert p["source_obfuscated_semantic_sensitivity"]["expected_n"]==55


def test_416_input_rows_and_variants():
    with INPUT.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    assert len(rows)==416; assert len({r["stage_a_example_id"] for r in rows})==416; assert len({r["base_intent_id"] for r in rows})==104
    assert Counter(r["variant"] for r in rows)==Counter({"human_direct":104,"model_direct":104,"human_obfuscated":104,"model_obfuscated":104})


def test_parsers_fail_closed_and_accept_expected_shapes():
    assert parse_llama_guard_prompt_output("unsafe\nS2")["decision"]==1
    assert parse_llama_guard_prompt_output("safe")["decision"]==0
    assert not parse_llama_guard_prompt_output("maybe unsafe")["parse_ok"]
    assert parse_granite_guardian_prompt_output("<score> yes </score>")["decision"]==1
    assert parse_granite_guardian_prompt_output("x <score>no</score>")["decision"]==0
    assert not parse_granite_guardian_prompt_output("yes")["parse_ok"]
    q=parse_qwen3guard_prompt_output("Safety: Controversial\nCategories: Jailbreak")
    assert q["decision"]==1 and q["label"]=="Controversial" and "Jailbreak" in q["categories"]
    assert not parse_qwen3guard_prompt_output("Categories: Violent")["parse_ok"]


def test_q_interval_is_exact_binomial_and_bounded():
    lo,hi=clopper_pearson_95(4,100); assert 0<=lo<0.04<hi<=1
    assert clopper_pearson_95(0,52)[0]==0.0
    assert clopper_pearson_95(52,52)[1]==1.0
