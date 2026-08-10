from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "scripts"
    / "validate_exact_cost_risk_cascade_protocol_v2_amendment.py"
)

spec = importlib.util.spec_from_file_location(
    "validate_v2_protocol_amendment",
    SCRIPT,
)

assert spec is not None
assert spec.loader is not None

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_amended_protocol_contract() -> None:
    result = module.validate()

    assert result["status"] == "PASS"
    assert result["version"] == "2.1.0"
    assert (
        result["amendment_id"]
        == "complete_model_selection_contracts_v2_1"
    )
