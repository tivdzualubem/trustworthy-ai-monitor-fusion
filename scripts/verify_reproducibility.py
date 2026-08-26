#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/metadata/reproducibility_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise AssertionError(f"Missing required path: {path.relative_to(ROOT)}")
    return path


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "pass"}
    )


def verify(strict_hashes: bool) -> None:
    manifest = json.loads(require(MANIFEST).read_text(encoding="utf-8"))

    legacy_status = json.loads(
        require(
            ROOT / "data/metadata/confirmatory_split_provenance.json"
        ).read_text(encoding="utf-8")
    )
    for split_name, expected_rows in {
        "final_test": 422,
        "held_out_shift": 50,
    }.items():
        record = legacy_status["splits"][split_name]
        assert record["historically_evaluated"] is True
        assert record["included_in_label_audit"] is True
        assert record["audited_rows"] == expected_rows
        assert record["used_in_development_only_pilot"] is False
        assert record["fresh_confirmatory_eligible"] is False

    audited_counts = pd.read_csv(
        require(ROOT / "reports/label_audit/audited_split_label_counts.csv")
    )
    totals = audited_counts.groupby("split")["n"].sum().astype(int).to_dict()
    assert totals["final_test"] == 422
    assert totals["held_out_shift"] == 50

    historical_metrics = pd.read_csv(
        require(ROOT / "results/tables/final_prespecified_policy_metrics_v3.csv")
    )
    historical_splits = set(historical_metrics["split"].astype(str))
    assert "final_test" in historical_splits
    assert "held_out_shift" in historical_splits

    if strict_hashes:
        for relative, expected in manifest["files"].items():
            path = require(ROOT / relative)
            actual = sha256(path)
            if actual != expected:
                raise AssertionError(
                    f"Hash mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )

    dataset = pd.read_parquet(
        require(
            ROOT
            / "data/processed/unified_dataset_label_audited_v1.parquet"
        )
    )
    cache = pd.read_parquet(
        require(ROOT / "data/processed/monitor_score_cache_v3.parquet")
    )

    assert len(dataset) == 2159
    assert len(cache) == 2159
    assert cache["y"].astype(int).value_counts().to_dict() == {
        0: 1756,
        1: 403,
    }

    bundle = joblib.load(
        require(ROOT / "artifacts/fusion_models_v3/fusion_bundle.joblib")
    )
    assert bundle is not None

    selected = pd.read_csv(
        require(ROOT / "results/tables/final_selected_policies_v3.csv")
    )
    assert len(selected) == 6

    policy_metrics = pd.read_csv(
        require(
            ROOT
            / "results/tables/final_prespecified_policy_metrics_v3.csv"
        )
    )
    assert policy_metrics["family"].nunique() == 6

    certificates = pd.read_csv(
        require(ROOT / "results/tables/final_ltt_certificates_v1.csv")
    )
    assert len(certificates) == 12
    if "certificate_pass" in certificates:
        assert as_bool(certificates["certificate_pass"]).all()

    qwen_modes = pd.read_csv(
        require(ROOT / "results/tables/final_qwen_mode_metrics_v3.csv")
    )
    assert set(qwen_modes["mode"]) == {
        "prompt_only",
        "response_only",
        "prompt_response",
    }

    timing = json.loads(
        require(
            ROOT / "reports/final_v3_policy_timing/run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert timing["benchmark_rows"] == 128
    assert "T4" in timing["gpu"]["name"]
    assert timing["qwen_parse_rate"] == 1.0
    assert (
        timing["score_validation"]["qwen_mismatch_unique_examples"]
        == 2
    )
    assert (
        timing["policy_comparison"]["mean_reduction_rate"]
        > 0
    )
    assert (
        timing["policy_comparison"]["selective_p95_ms"]
        >= timing["policy_comparison"]["full_p95_ms"]
    )

    decision = require(
        ROOT / "reports/stop_go_v3/decision.md"
    ).read_text(encoding="utf-8").lower()
    assert "no-go" in decision or "no go" in decision
    assert "risk" in decision

    require(
        ROOT
        / "paper/Exact_Cost_Development_Screening_and_Risk_Constrained_Evaluation.pdf"
    )
    require(
        ROOT
        / "paper/Risk_Controlled_Decision_Value_Acquisition_Report.tex"
    )
    require(ROOT / "paper/references.bib")
    require(ROOT / "paper/build_report.sh")
    require(
        ROOT
        / "paper/figures/value_predictability_intervals.pdf"
    )
    require(
        ROOT
        / "paper/figures/common_risk_recall_cost_frontier.pdf"
    )
    require(ROOT / "artifacts/final_v3_policy_timing_results.zip")
    require(ROOT / "exports/final_v3_policy_timing_package.zip")

    print("reproducibility verification passed")
    print(f"strict hashes: {strict_hashes}")
    print(f"dataset rows: {len(dataset)}")
    print("labels: 1756 negative, 403 positive")
    print(f"prespecified policy families: {policy_metrics['family'].nunique()}")
    print(f"LTT certificates: {len(certificates)}")
    print(
        "timing mean reduction rate:",
        timing["policy_comparison"]["mean_reduction_rate"],
    )
    print(
        "Qwen mismatch unique examples:",
        timing["score_validation"]["qwen_mismatch_unique_examples"],
    )
    print("current paper direction: evaluation/measurement of runtime safety-monitor cascades")
    print("historical decision-value milestone: no-go")
    print("current evaluation/measurement pilot: completed (development only)")
    print("router superiority claim: none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-hashes",
        action="store_true",
        help="Verify the committed artifact hashes in addition to structure.",
    )
    args = parser.parse_args()
    verify(strict_hashes=args.strict_hashes)


if __name__ == "__main__":
    main()
