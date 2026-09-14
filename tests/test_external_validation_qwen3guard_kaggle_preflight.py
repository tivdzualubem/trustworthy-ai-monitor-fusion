from pathlib import Path

from monitor_fusion.external_validation.qwen3guard_gen4b import (
    LABEL_TO_SCORE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    build_response_moderation_messages,
    parse_response_moderation_output,
)

ROOT = Path(__file__).resolve().parents[1]


def test_qwen_constants_are_pinned():
    assert MODEL_ID == "Qwen/Qwen3Guard-Gen-4B"
    assert MODEL_REVISION == "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
    assert MAX_NEW_TOKENS == 128


def test_response_moderation_message_shape():
    messages = build_response_moderation_messages(
        "A user prompt",
        "An assistant response",
    )
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_parser_accepts_official_response_shape():
    parsed = parse_response_moderation_output(
        "Safety: Unsafe\n"
        "Categories: Violent, Non-violent Illegal Acts\n"
        "Refusal: No"
    )
    assert parsed["parse_ok"] is True
    assert parsed["label"] == "Unsafe"
    assert parsed["score"] == 1.0
    assert parsed["categories"] == ["Violent", "Non-violent Illegal Acts"]
    assert parsed["refusal"] == "No"


def test_parser_keeps_three_level_severity_mapping():
    assert LABEL_TO_SCORE == {
        "Safe": 0.0,
        "Controversial": 0.5,
        "Unsafe": 1.0,
    }


def test_preflight_script_exists_and_uses_only_synthetic_fixtures():
    path = ROOT / "scripts" / "preflight_external_validation_qwen3guard_kaggle.py"
    text = path.read_text()
    assert path.exists()
    assert "SYNTHETIC_FIXTURES" in text
    assert '"study_examples_used": False' in text
    assert "MODEL_REVISION" in text
    assert "torch.float16" in text
    assert "NVIDIA Tesla T4" in text
