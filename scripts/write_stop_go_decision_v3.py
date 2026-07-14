#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METRICS = Path("reports/fusion_comparison_v3/metrics_by_split.csv")
BOOTSTRAP = Path("reports/fusion_comparison_v3/paired_bootstrap.csv")
OPERATING_POINTS = Path(
    "artifacts/fusion_models_v3/frozen_operating_points_v3.json"
)
POLICY_MANIFEST = Path(
    "data/metadata/policy_evaluation_v3_manifest.json"
)

OUT_DIR = Path("reports/stop_go_v3")
TABLE = Path("results/tables/stop_go_target_fpr005_v3.csv")
MANIFEST = Path("data/metadata/stop_go_v3_manifest.json")
CANONICAL_REPORT = Path("reports/stop_go_decision.md")
VERSIONED_REPORT = OUT_DIR / "decision.md"

TARGET_FPR = 0.05
EXPECTED_SPLITS = ["calibration", "final_test", "held_out_shift"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for path in [
    METRICS,
    BOOTSTRAP,
    OPERATING_POINTS,
    POLICY_MANIFEST,
]:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")

metrics = pd.read_csv(METRICS)
bootstrap = pd.read_csv(BOOTSTRAP)

target_metrics = metrics[
    metrics["target_fpr"].round(6).eq(TARGET_FPR)
].copy()
target_bootstrap = bootstrap[
    bootstrap["target_fpr"].round(6).eq(TARGET_FPR)
].copy()

if set(target_metrics["split"]) != set(EXPECTED_SPLITS):
    raise SystemExit("Unexpected metric splits at target FPR 0.05")
if set(target_metrics["policy"]) != {
    "full_information_always_on",
    "selective_acquisition",
}:
    raise SystemExit("Unexpected policies at target FPR 0.05")
if set(target_bootstrap["split"]) != set(EXPECTED_SPLITS):
    raise SystemExit("Unexpected bootstrap splits at target FPR 0.05")

rows = []
for split in EXPECTED_SPLITS:
    full = target_metrics[
        target_metrics["split"].eq(split)
        & target_metrics["policy"].eq(
            "full_information_always_on"
        )
    ].iloc[0]

    selective = target_metrics[
        target_metrics["split"].eq(split)
        & target_metrics["policy"].eq(
            "selective_acquisition"
        )
    ].iloc[0]

    paired = target_bootstrap[
        target_bootstrap["split"].eq(split)
    ].iloc[0]

    rows.append(
        {
            "split": split,
            "target_fpr": TARGET_FPR,
            "full_recall": float(full["recall"]),
            "full_fpr": float(full["fpr"]),
            "full_fpr_one_sided95_upper": float(
                full["fpr_one_sided95_upper"]
            ),
            "full_qwen_call_rate": float(
                full["qwen_call_rate"]
            ),
            "full_avg_estimated_cost_ms": float(
                full["avg_estimated_cost_ms"]
            ),
            "selective_recall": float(selective["recall"]),
            "selective_fpr": float(selective["fpr"]),
            "selective_fpr_one_sided95_upper": float(
                selective["fpr_one_sided95_upper"]
            ),
            "selective_qwen_call_rate": float(
                selective["qwen_call_rate"]
            ),
            "selective_avg_estimated_cost_ms": float(
                selective["avg_estimated_cost_ms"]
            ),
            "selective_estimated_cost_reduction": float(
                selective["estimated_cost_reduction"]
            ),
            "recall_diff_selective_minus_full": float(
                paired["recall_diff_selective_minus_full"]
            ),
            "recall_diff_ci95_low": float(
                paired["recall_diff_ci95_low"]
            ),
            "recall_diff_ci95_high": float(
                paired["recall_diff_ci95_high"]
            ),
            "fpr_diff_selective_minus_full": float(
                paired["fpr_diff_selective_minus_full"]
            ),
            "fpr_diff_ci95_low": float(
                paired["fpr_diff_ci95_low"]
            ),
            "fpr_diff_ci95_high": float(
                paired["fpr_diff_ci95_high"]
            ),
            "estimated_cost_reduction_ci95_low": float(
                paired["estimated_cost_reduction_ci95_low"]
            ),
            "estimated_cost_reduction_ci95_high": float(
                paired["estimated_cost_reduction_ci95_high"]
            ),
        }
    )

decision_table = pd.DataFrame(rows)

# Stop/go rule:
# GO requires a positive cost advantage whose paired 95% lower bound is
# above zero, while the selective policy satisfies the nominal risk limit
# in every evaluation split. Because formal risk certification is a later
# requirement, observed FPR and the one-sided 95% upper bound are both
# checked here as a conservative descriptive gate.
decision_table["cost_advantage_observed"] = (
    decision_table["selective_estimated_cost_reduction"] > 0
)
decision_table["cost_advantage_ci_positive"] = (
    decision_table["estimated_cost_reduction_ci95_low"] > 0
)
decision_table["selective_observed_fpr_within_target"] = (
    decision_table["selective_fpr"] <= TARGET_FPR
)
decision_table["selective_upper_bound_within_target"] = (
    decision_table["selective_fpr_one_sided95_upper"]
    <= TARGET_FPR
)
decision_table["risk_gate_pass"] = (
    decision_table["selective_observed_fpr_within_target"]
    & decision_table["selective_upper_bound_within_target"]
)
decision_table["split_stop_go_pass"] = (
    decision_table["cost_advantage_ci_positive"]
    & decision_table["risk_gate_pass"]
)

cost_advantage_pass = bool(
    decision_table["cost_advantage_ci_positive"].all()
)
risk_gate_pass = bool(decision_table["risk_gate_pass"].all())
overall_go = cost_advantage_pass and risk_gate_pass

if overall_go:
    decision = "GO FOR A ROUTING-PERFORMANCE PAPER"
    pivot = False
else:
    decision = (
        "NO-GO FOR A ROUTING-PERFORMANCE PAPER UNDER THE "
        "CURRENT AUDITED-LABEL EVIDENCE"
    )
    pivot = True

TABLE.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

decision_table.to_csv(TABLE, index=False)

calibration = decision_table[
    decision_table["split"].eq("calibration")
].iloc[0]
final_test = decision_table[
    decision_table["split"].eq("final_test")
].iloc[0]
shift = decision_table[
    decision_table["split"].eq("held_out_shift")
].iloc[0]

report = f"""# Stop/go decision v3

## Decision: {decision}

The selective-acquisition policy has a measurable cost advantage, but it does
not satisfy the required risk gate across the evaluation splits. Therefore,
the project should continue as a **measurement-validity paper**, not as a
routing-performance paper.

This decision uses the final author-reviewed labels, the regenerated monitor
caches, the complete serialized v3 fusion pipelines, the controlled T4 timing
measurements, and the corrected v3 policy comparison.

## Prespecified decision rule

At the common selection target FPR of `{TARGET_FPR:.2f}`, a routing-performance
GO requires both:

1. a positive paired 95% lower confidence bound for estimated cost reduction;
2. selective-policy FPR at or below `{TARGET_FPR:.2f}`, with its one-sided 95%
   upper bound also at or below `{TARGET_FPR:.2f}`, in every evaluated split.

This stop/go gate is deliberately conservative and is not a substitute for
the later formal Neyman-Pearson or Learn-then-Test certificate.

## Cost result

The cost requirement passes descriptively:

- Calibration: estimated reduction
  `{calibration['selective_estimated_cost_reduction']:.4f}`,
  paired 95% CI
  `[{calibration['estimated_cost_reduction_ci95_low']:.4f},
  {calibration['estimated_cost_reduction_ci95_high']:.4f}]`.
- Final test: estimated reduction
  `{final_test['selective_estimated_cost_reduction']:.4f}`,
  paired 95% CI
  `[{final_test['estimated_cost_reduction_ci95_low']:.4f},
  {final_test['estimated_cost_reduction_ci95_high']:.4f}]`.
- Held-out shift: estimated reduction
  `{shift['selective_estimated_cost_reduction']:.4f}`,
  paired 95% CI
  `[{shift['estimated_cost_reduction_ci95_low']:.4f},
  {shift['estimated_cost_reduction_ci95_high']:.4f}]`.

## Risk result

The risk requirement fails:

- Calibration selective FPR:
  `{calibration['selective_fpr']:.4f}`;
  one-sided 95% upper bound:
  `{calibration['selective_fpr_one_sided95_upper']:.4f}`.
- Final-test selective FPR:
  `{final_test['selective_fpr']:.4f}`;
  one-sided 95% upper bound:
  `{final_test['selective_fpr_one_sided95_upper']:.4f}`.
- Held-out-shift selective FPR:
  `{shift['selective_fpr']:.4f}`;
  one-sided 95% upper bound:
  `{shift['selective_fpr_one_sided95_upper']:.4f}`.

The final-test upper bound exceeds 5%, and the held-out-shift observed FPR is
far above 5%. Selective acquisition also has lower point recall and higher
point FPR than full-information fusion on all three evaluated splits.

## Measurement-validity paper direction

The evidence supports a paper centered on:

1. invalid first-token versus official generated Qwen3Guard classification;
2. prompt contamination across prompt-only, response-only, and
   prompt-response inputs;
3. label-proxy errors and the effect of author-reviewed corrections;
4. complete model serialization and reproducible score-cache provenance;
5. controlled timing validity, including synchronization, warm-up, generation,
   tail latency, and end-to-end policy timing;
6. failure of nominal FPR control under attack-family shift.

## Remaining professor-required stages

This decision does not complete the later requirements for:

- a new locked test or nested leave-source/leave-family-out evaluation;
- formal risk control on an untouched risk-control split.

Those stages may refine the strength of the final claims, but the current v3
evidence does not justify a routing-performance claim.
"""

VERSIONED_REPORT.write_text(report, encoding="utf-8")
CANONICAL_REPORT.write_text(report, encoding="utf-8")

manifest = {
    "artifact": "stop_go_decision_v3",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "decision": decision,
    "routing_performance_go": overall_go,
    "measurement_validity_pivot": pivot,
    "target_fpr": TARGET_FPR,
    "decision_rule": {
        "cost_advantage_required": True,
        "paired_cost_reduction_ci95_lower_must_exceed_zero": True,
        "selective_observed_fpr_must_not_exceed_target": True,
        "selective_one_sided95_fpr_upper_must_not_exceed_target": True,
        "required_in_every_evaluation_split": True,
    },
    "gate_results": {
        "cost_advantage_pass": cost_advantage_pass,
        "risk_gate_pass": risk_gate_pass,
        "overall_go": overall_go,
    },
    "risk_control_status": (
        "not formally certified; formal risk control remains a later "
        "professor-required stage"
    ),
    "inputs": {
        str(METRICS): sha256(METRICS),
        str(BOOTSTRAP): sha256(BOOTSTRAP),
        str(OPERATING_POINTS): sha256(OPERATING_POINTS),
        str(POLICY_MANIFEST): sha256(POLICY_MANIFEST),
    },
    "outputs": {
        str(TABLE): sha256(TABLE),
        str(VERSIONED_REPORT): sha256(VERSIONED_REPORT),
        str(CANONICAL_REPORT): sha256(CANONICAL_REPORT),
    },
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("decision:", decision)
print("cost advantage pass:", cost_advantage_pass)
print("risk gate pass:", risk_gate_pass)
print("overall go:", overall_go)
print("\nstop/go table:")
print(
    decision_table[
        [
            "split",
            "selective_estimated_cost_reduction",
            "estimated_cost_reduction_ci95_low",
            "selective_recall",
            "selective_fpr",
            "selective_fpr_one_sided95_upper",
            "recall_diff_selective_minus_full",
            "fpr_diff_selective_minus_full",
            "split_stop_go_pass",
        ]
    ].to_string(index=False)
)
print("\nreport:", VERSIONED_REPORT)
print("canonical report:", CANONICAL_REPORT)
print("manifest:", MANIFEST)
