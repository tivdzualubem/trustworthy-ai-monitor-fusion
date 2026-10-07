from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "scripts" / "analyze_stage_a_readout_transfer_kill.py"


def _load_analysis():
    spec = importlib.util.spec_from_file_location("kill_analysis", ANALYSIS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _synthetic_prompts_and_features(tmp_path):
    rows = []
    ids = []
    hidden = []
    native_score = []
    native_decision = []

    # Five folds. Direct encodes the label only in dimension 0; O2 encodes
    # exactly the same label only in dimension 1. A source probe cannot
    # transfer, while a target probe can recover. Native scores are constant,
    # so a threshold change cannot explain the recovery.
    for fold in range(5):
        for j in range(4):
            pair_id = f"F{fold}_{j}"
            for case_type, label in (("benign", 0), ("harmful", 1)):
                semantic = f"{pair_id}::{case_type}"
                sign = -3.0 if label == 0 else 3.0

                for representation in ("direct", "O2"):
                    ex = f"{semantic}::{representation}"
                    rows.append({
                        "experiment_example_id": ex,
                        "pair_id": pair_id,
                        "semantic_case_id": semantic,
                        "case_type": case_type,
                        "label_harmful": label,
                        "reviewed_category":
                            "BENIGN" if label == 0 else "C1",
                        "representation": representation,
                        "fold_id": fold,
                    })
                    ids.append(ex)
                    if representation == "direct":
                        hidden.append([sign, 0.0])
                    else:
                        hidden.append([0.0, sign])
                    native_score.append(0.5)
                    native_decision.append(0)

    prompts = pd.DataFrame(rows)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        experiment_example_id=np.asarray(ids, dtype="U"),
        native_score=np.asarray(native_score, dtype=float),
        native_decision=np.asarray(native_decision, dtype=np.int8),
        hidden=np.asarray(hidden, dtype=np.float32),
    )
    return prompts, feature_path


def test_readout_transfer_screening_logic(tmp_path):
    a = _load_analysis()
    prompts, feature_path = _synthetic_prompts_and_features(tmp_path)

    predictions, metrics, thresholds, flags = a.analyze_guard(
        "synthetic_guard",
        prompts,
        feature_path,
    )

    assert len(thresholds) == 5
    assert set(predictions["fold_id"]) == {0, 1, 2, 3, 4}

    assert flags["source_accessible"] is True
    assert flags["transfer_drop"] is True
    assert flags["target_recovery"] is True
    assert flags["native_threshold_fix"] is False
    assert flags["readout_transfer_signal"] is True

    direct = metrics[
        (metrics["method"] == "source_probe")
        & (metrics["representation"] == "direct")
    ].iloc[0]
    source_o2 = metrics[
        (metrics["method"] == "source_probe")
        & (metrics["representation"] == "O2")
    ].iloc[0]
    target_o2 = metrics[
        (metrics["method"] == "target_probe")
        & (metrics["representation"] == "O2")
    ].iloc[0]

    assert direct["balanced_accuracy"] == 1.0
    assert source_o2["balanced_accuracy"] == 0.5
    assert target_o2["balanced_accuracy"] == 1.0


def test_threshold_selection_uses_only_supplied_training_values():
    a = _load_analysis()
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = a.choose_threshold(y, score)

    assert 0.2 < threshold < 0.8
