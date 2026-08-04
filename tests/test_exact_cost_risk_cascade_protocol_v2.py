from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_exact_cost_risk_cascade_protocol_v2.py"

spec = importlib.util.spec_from_file_location("validate_v2_protocol", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_frozen_protocol_is_complete_and_internally_consistent() -> None:
    result = module.validate_protocol(module.load_protocol())
    assert result["status"] == "PASS"
    assert result["fold_seed_count"] == 5
    assert result["signed_value_model_family_count"] == 3
    assert result["report_files_changed"] is False
    assert result["project_data_opened"] is False


def test_all_professor_required_baselines_are_frozen() -> None:
    protocol = module.load_protocol()
    baseline_ids = {
        item["policy_id"]
        for item in protocol["policies"]["required_baselines"]
    }
    assert baseline_ids == {
        "threshold_distance",
        "current_error_prediction",
        "random_acquisition",
        "direct_fusion",
    }


def test_joint_risk_requires_FPR_and_exact_cost() -> None:
    protocol = module.load_protocol()
    risks = {
        item["risk"]
        for item in protocol["joint_risk_control"]["constraints"]
    }
    assert risks == {
        "false_positive_rate",
        "mean_total_end_to_end_cost_ms",
    }
    assert protocol["exact_cost_comparison"][
        "recall_comparison_allowed_only_when_cost_equivalence_passes"
    ] is True
