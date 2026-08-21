#!/usr/bin/env python3
"""Fail-closed provenance audit for the historical v2 evidence snapshot.

The Aug-17 scientific review identified that the published v2 CSV evidence
did not have one committed generator/reproduction entry point. This checker
does not pretend that missing raw timing observations can be reconstructed.

It establishes a canonical, machine-readable boundary between:

1. committed upstream artifacts from which later analysis can be rebuilt; and
2. historical published tables whose complete generating provenance was not
   preserved.

The historical v2 tables remain immutable scientific evidence. Corrected
measurement/evaluation work must be written to a new artifact namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports/exact_cost_risk_cascade_v2"

UPSTREAM_DIR = (
    ROOT
    / "data/processed/v2_repeated_grouped_candidate_bundle"
)

PROVENANCE_PATH = (
    REPORT_DIR
    / "historical_v2_provenance_status.json"
)

HISTORICAL_TABLES = (
    "across_seed_summary.csv",
    "cost_equivalence_pass_counts.csv",
    "endpoint_metrics.csv",
    "latency_and_tail_summary.csv",
    "outer_fold_threshold_calibration_and_evaluation.csv",
    "per_seed_metrics.csv",
    "primary_pairwise_development_comparisons.csv",
    "recall_stability_pass_counts.csv",
    "signed_value_family_selection.csv",
)

UPSTREAM_ARTIFACTS = (
    "candidate_bundle.parquet",
    "outer_fold_model_selection.parquet",
    "family_evidence.json",
    "run_manifest.json",
)

REQUIRED_UNPRESERVED_TIMING_COLUMNS = (
    "router_features_ms",
    "signed_ridge_router_inference_ms",
    "signed_hgbr_router_inference_ms",
    "signed_rfr_router_inference_ms",
    "current_error_lr_ms",
    "current_error_hgbc_ms",
    "current_error_rfc_ms",
    "base_policy_inference_ms",
    "augmented_policy_inference_ms",
    "direct_fusion_lr_ms",
    "direct_fusion_hgbc_ms",
    "direct_fusion_rfc_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(
            f"Required file is missing: "
            f"{path.relative_to(ROOT)}"
        )

    return path


def build_status() -> dict[str, object]:
    historical = {}

    for name in HISTORICAL_TABLES:
        path = require_file(REPORT_DIR / name)
        frame = pd.read_csv(path)

        historical[name] = {
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    upstream = {}

    for name in UPSTREAM_ARTIFACTS:
        path = require_file(UPSTREAM_DIR / name)

        upstream[name] = {
            "sha256": sha256_file(path),
        }

    bundle_path = (
        UPSTREAM_DIR
        / "candidate_bundle.parquet"
    )

    bundle = pd.read_parquet(bundle_path)

    missing_timing = [
        column
        for column in REQUIRED_UNPRESERVED_TIMING_COLUMNS
        if column not in bundle.columns
    ]

    if missing_timing != list(
        REQUIRED_UNPRESERVED_TIMING_COLUMNS
    ):
        raise RuntimeError(
            "Historical timing-provenance gap changed; "
            "review before updating this contract"
        )

    run_manifest = json.loads(
        (
            UPSTREAM_DIR
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )

    if (
        run_manifest.get(
            "exact_cost_policy_selection_performed"
        )
        is not False
    ):
        raise RuntimeError(
            "Historical upstream manifest no longer "
            "matches reviewed v2 state"
        )

    return {
        "artifact":
            "historical_v2_evidence_provenance_status",
        "review_context":
            "professor_aug_17_2026_reproducibility_repair",
        "historical_v2_status":
            "published_snapshot_not_fully_regenerable_from_committed_raw_artifacts",
        "historical_tables_are_modified":
            False,
        "router_rescue_attempted":
            False,
        "protected_legacy_data_opened":
            False,
        "fresh_data_used":
            False,
        "upstream_candidate_bundle": {
            "rows": int(len(bundle)),
            "columns": int(len(bundle.columns)),
            "artifacts": upstream,
        },
        "historical_published_tables":
            historical,
        "known_provenance_gap": {
            "raw_downstream_timing_not_preserved":
                True,
            "missing_candidate_bundle_timing_columns":
                missing_timing,
            "historical_latency_summary_exact_regeneration":
                False,
            "reason": (
                "The committed upstream candidate bundle "
                "contains policy predictions and model-selection "
                "evidence but does not preserve the raw downstream "
                "CPU/router timing observations used by the "
                "published component-latency summaries."
            ),
        },
        "scientific_boundary": {
            "historical_component_cost_results_are_direct_e2e":
                False,
            "historical_component_cost_results_are_final_exact_cost":
                False,
            "historical_grouping_protects_near_duplicate_dependence":
                False,
            "historical_35s_bound_is_proven_runtime_enforced":
                False,
            "historical_1pct_equivalence_margin_externally_justified":
                False,
            "fresh_calibration_available":
                False,
            "joint_fpr_cost_certificate_available":
                False,
            "fresh_source_time_confirmation_available":
                False,
            "multi_rater_confirmation_available":
                False,
        },
        "repair_rule": (
            "Do not overwrite or retrospectively regenerate the "
            "historical v2 snapshot with newly measured values. "
            "Corrected E2E, grouping, risk-control, label, and "
            "comparison experiments must use a new evaluation-"
            "measurement artifact namespace and a committed "
            "generator from the start."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write the canonical provenance-status JSON. "
            "Without this flag the checker validates and prints."
        ),
    )

    args = parser.parse_args()

    status = build_status()

    if args.write:
        PROVENANCE_PATH.write_text(
            json.dumps(
                status,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "wrote="
            f"{PROVENANCE_PATH.relative_to(ROOT)}"
        )

    print(
        "historical_v2_status="
        f"{status['historical_v2_status']}"
    )

    print(
        "missing_raw_timing_columns="
        f"{len(status['known_provenance_gap']['missing_candidate_bundle_timing_columns'])}"
    )

    print(
        "historical_component_cost_is_direct_e2e=False"
    )

    print(
        "V2_EVIDENCE_PROVENANCE_CHECK=PASS"
    )


if __name__ == "__main__":
    main()
