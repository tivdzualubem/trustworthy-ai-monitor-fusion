from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/validate_full_text_embedding_protocol.py"
)

spec = importlib.util.spec_from_file_location(
    "validate_full_text_embedding_protocol",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {SCRIPT}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_complete_text_embedding_protocol_is_frozen() -> None:
    result = module.validate()

    assert result["status"] == "PASS"
    assert result["full_text_coverage_required"] is True
    assert result["truncation_allowed"] is False
    assert result["minimum_required_token_coverage"] == 1.0
    assert result["chunk_overlap_tokens"] == 0
    assert result["amendment_id"] == (
        "full_text_embedding_coverage_v1"
    )
