from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_decision_value_protocol.py"

spec = importlib.util.spec_from_file_location(
    "validate_decision_value_protocol",
    SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_protocol_matches_committed_development_data() -> None:
    result = module.validate()
    assert result["status"] == "PASS"
    assert result["development_rows"] == 1687
    assert result["negative_n"] == 1396
    assert result["positive_n"] == 291
    assert result["excluded_rows"] == 472
    assert set(result["optional_setups"]) == {
        "compact_after_rule",
        "qwen_after_rule_compact",
    }


def test_excluded_splits_are_not_development_splits() -> None:
    protocol = module.load_protocol()
    development = set(protocol["scope"]["development_splits"])
    excluded = set(protocol["scope"]["excluded_splits"])
    assert development == {
        "policy_train",
        "policy_selection",
        "calibration",
    }
    assert excluded == {"final_test", "held_out_shift"}
    assert development.isdisjoint(excluded)


def test_optional_outputs_are_not_router_inputs() -> None:
    protocol = module.load_protocol()
    predictors = protocol["predictor_families"]

    compact_pre = (
        predictors["cheap_features"]["compact_after_rule"]
        + predictors["runtime_metadata"]["compact_after_rule"]
    )
    qwen_pre = (
        predictors["cheap_features"]["qwen_after_rule_compact"]
        + predictors["runtime_metadata"]["qwen_after_rule_compact"]
    )

    assert not any(name.startswith("compact_") for name in compact_pre)
    assert not any(name.startswith("qwen_") for name in qwen_pre)
