#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

METRICS = Path("reports/fusion_comparison_v2/evaluation_metrics.csv")
BOOT = Path("reports/fusion_comparison_v2/paired_bootstrap_differences.csv")
PROMPT = Path("reports/prompt_contamination/qwen_mode_metrics.csv")
AUDIT = Path("data/metadata/label_audit_completed_manifest.json")

OUT = Path("reports/stop_go_decision.md")
TABLE = Path("results/tables/stop_go_target_fpr005.csv")

for path in [METRICS, BOOT, PROMPT, AUDIT]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")

metrics = pd.read_csv(METRICS)
boot = pd.read_csv(BOOT)
audit = json.loads(AUDIT.read_text(encoding="utf-8"))

target = 0.05
m = metrics[metrics["target_fpr"].round(6).eq(target)].copy()
b = boot[boot["target_fpr"].round(6).eq(target)].copy()

expected_splits = {"calibration", "final_test", "held_out_shift"}
if set(m["split"]) != expected_splits:
    raise SystemExit("Unexpected evaluation splits at target FPR 0.05")

wide = m.pivot(
    index="split",
    columns="policy",
    values=["recall", "fpr", "precision", "expensive_call_rate"],
)

rows = []
for split in ["calibration", "final_test", "held_out_shift"]:
    full = m[
        (m["split"] == split)
        & (m["policy"] == "full_information_always_on")
    ].iloc[0]
    selective = m[
        (m["split"] == split)
        & (m["policy"] == "selective_acquisition")
    ].iloc[0]
    paired = b[b["split"] == split].iloc[0]

    rows.append(
        {
            "split": split,
            "target_fpr": target,
            "full_recall": full["recall"],
            "selective_recall": selective["recall"],
            "recall_difference_selective_minus_full": paired[
                "recall_diff_selective_minus_full"
            ],
            "recall_difference_ci_low": paired["recall_diff_ci_low"],
            "recall_difference_ci_high": paired["recall_diff_ci_high"],
            "full_fpr": full["fpr"],
            "selective_fpr": selective["fpr"],
            "fpr_difference_selective_minus_full": paired[
                "fpr_diff_selective_minus_full"
            ],
            "fpr_difference_ci_low": paired["fpr_diff_ci_low"],
            "fpr_difference_ci_high": paired["fpr_diff_ci_high"],
            "selective_expensive_call_rate": selective[
                "expensive_call_rate"
            ],
            "selective_expensive_call_reduction": selective[
                "expensive_call_reduction"
            ],
        }
    )

decision_table = pd.DataFrame(rows)
TABLE.parent.mkdir(parents=True, exist_ok=True)
decision_table.to_csv(TABLE, index=False)

shift = decision_table[
    decision_table["split"] == "held_out_shift"
].iloc[0]
final = decision_table[
    decision_table["split"] == "final_test"
].iloc[0]
cal = decision_table[
    decision_table["split"] == "calibration"
].iloc[0]

decision = "PROVISIONAL NO-GO FOR A ROUTING-PERFORMANCE PAPER"

lines = [
    "# Stop/go decision",
    "",
    f"## Decision: {decision}",
    "",
    "The selective-acquisition policy reduces expensive-monitor calls, but the",
    "current evidence does not show that it preserves the intended risk constraint.",
    "Therefore, the project should pivot toward a measurement-validity paper rather",
    "than make a routing-performance claim.",
    "",
    "This decision is provisional because the label audit is assistant-assisted and",
    "still requires author review, no formal Neyman–Pearson or Learn-then-Test",
    "certificate has been produced, and the existing final-test split previously",
    "influenced development.",
    "",
    "## Target FPR 0.05 comparison",
    "",
    f"- Calibration: full recall={cal.full_recall:.4f}, FPR={cal.full_fpr:.4f}; "
    f"selective recall={cal.selective_recall:.4f}, FPR={cal.selective_fpr:.4f}, "
    f"expensive-call reduction={cal.selective_expensive_call_reduction:.4f}.",
    f"- Final test: full recall={final.full_recall:.4f}, FPR={final.full_fpr:.4f}; "
    f"selective recall={final.selective_recall:.4f}, FPR={final.selective_fpr:.4f}, "
    f"expensive-call reduction={final.selective_expensive_call_reduction:.4f}.",
    f"- Held-out shift: full recall={shift.full_recall:.4f}, FPR={shift.full_fpr:.4f}; "
    f"selective recall={shift.selective_recall:.4f}, FPR={shift.selective_fpr:.4f}, "
    f"expensive-call reduction={shift.selective_expensive_call_reduction:.4f}.",
    "",
    "On held-out shift, the selective policy has lower recall and higher FPR than",
    "the full-information predictor. Its FPR is also far above the nominal 5%",
    "target. This fails the stated stop/go requirement that routing retain a cost",
    "advantage while satisfying the risk constraint.",
    "",
    "## Measurement-validity pivot",
    "",
    "The revised paper should center on:",
    "",
    "1. Invalid first-token Qwen scoring versus official generated classification.",
    "2. Prompt contamination across prompt-only, response-only, and prompt-response inputs.",
    "3. Label-proxy errors in JailbreakBench and GCG evaluation.",
    "4. Complete model serialization and reproducible score-cache provenance.",
    "5. Timing-validity requirements: synchronization, warm-up, batch comparability,",
    "   generation latency, tail latency, and end-to-end policy timing.",
    "6. Failure of nominal FPR control under source/family shift.",
    "",
    "## Remaining work before final claims",
    "",
    f"- Author-review the {audit['review_required_rows']} flagged audit rows.",
    "- Lock corrected labels and regenerate the cache, serialized models, and comparisons.",
    "- Run a controlled timing benchmark on fixed hardware and batch size.",
    "- Use a new locked test or nested leave-source/leave-family-out evaluation.",
    "- Apply formal risk control only on an untouched risk-control split.",
    "",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")

print("decision:", decision)
print("report:", OUT)
print("table:", TABLE)
print("\nTarget FPR 0.05:")
print(decision_table.to_string(index=False))
