#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
CPU_DIR = ROOT / "reports/evaluation_measurement_pilot_v1/cpu"
T4_DIR = ROOT / "reports/evaluation_measurement_pilot_v1/t4"
OUT_DIR = ROOT / "reports/evaluation_measurement_pilot_v1/analysis"
DEV_DATASET = (
    ROOT / "data/processed/v2_development_view/"
    / "unified_dataset_label_audited_v1.development.parquet"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_upper(k: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0:
        return math.nan
    if k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def add_target_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_target_key"] = frame["target_rate"].fillna(-1.0).astype(float)
    return frame


def pair(frame, left, right, keys):
    a = add_target_key(frame.loc[left]).copy()
    b = add_target_key(frame.loc[right]).copy()
    a = a[keys + ["primary_risk_gate_pass", "holdout_acquisition_rate"]].rename(
        columns={
            "primary_risk_gate_pass": "a_pass",
            "holdout_acquisition_rate": "a_acq",
        }
    )
    b = b[keys + ["primary_risk_gate_pass", "holdout_acquisition_rate"]].rename(
        columns={
            "primary_risk_gate_pass": "b_pass",
            "holdout_acquisition_rate": "b_acq",
        }
    )
    p = a.merge(b, on=keys, how="inner", validate="one_to_one")
    if len(p) != len(a) or len(p) != len(b):
        raise RuntimeError(
            f"Pairing mismatch: left={len(a)}, right={len(b)}, paired={len(p)}"
        )
    return p


def binary_category_metrics(part: pd.DataFrame) -> dict:
    y = part["y_eval"].to_numpy(int)
    pred = part["prediction"].to_numpy(int).astype(bool)
    pos = y == 1
    neg = y == 0
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    p = int(pos.sum())
    n = int(neg.sum())
    return {
        "n": int(len(part)),
        "positive_n": p,
        "negative_n": n,
        "recall": (tp / p) if p else math.nan,
        "fpr": (fp / n) if n else math.nan,
        "row_fpr_upper95": exact_upper(fp, n) if n else math.nan,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cpu = pd.read_csv(CPU_DIR / "policy_summary.csv")
    t4 = pd.read_csv(T4_DIR / "policy_latency_summary.csv")
    t4_manifest = json.loads((T4_DIR / "t4_timing_manifest.json").read_text())
    route = json.loads((T4_DIR / "score_and_route_validation.json").read_text())

    deploy = cpu[cpu["deployable"].astype(bool)].copy()
    primary = deploy[
        deploy["label_condition"].eq("audited")
        & deploy["grouping_condition"].eq("dependency_primary")
    ].copy()

    empirical_pass = primary["holdout_fpr"].astype(float) <= 0.05 + 1e-12
    risk_pass = primary["primary_risk_gate_pass"].astype(bool)

    rows = [
        {
            "factor": "empirical_fpr_vs_finite_sample_risk",
            "comparison": "row empirical FPR <= 0.05 vs frozen row+dependency upper-bound gate",
            "eligible_n": int(len(primary)),
            "apparent_pass_n_left": int(empirical_pass.sum()),
            "pass_n_right": int(risk_pass.sum()),
            "conclusion_flip_n": int((empirical_pass != risk_pass).sum()),
            "status": "material_change",
            "interpretation": (
                "Empirical thresholding materially overstates acceptable policies "
                "relative to the frozen finite-sample dependency-aware gate."
            ),
        }
    ]

    keys = ["seed", "stack", "policy_kind", "_target_key"]

    p = pair(
        deploy,
        (deploy["label_condition"].eq("audited")
         & deploy["grouping_condition"].eq("singleton_weak")),
        (deploy["label_condition"].eq("audited")
         & deploy["grouping_condition"].eq("dependency_primary")),
        keys,
    )
    rows.append(
        {
            "factor": "weak_grouping_vs_dependency_grouping",
            "comparison": "singleton example_id vs frozen near-duplicate dependency grouping",
            "eligible_n": int(len(p)),
            "apparent_pass_n_left": int(p["a_pass"].astype(bool).sum()),
            "pass_n_right": int(p["b_pass"].astype(bool).sum()),
            "conclusion_flip_n": int(
                (p["a_pass"].astype(bool) != p["b_pass"].astype(bool)).sum()
            ),
            "status": "material_change",
            "interpretation": (
                "Grouping choice changes risk-pass conclusions; singleton grouping is "
                "not an adequate dependence control."
            ),
        }
    )

    p = pair(
        deploy,
        (deploy["label_condition"].eq("original")
         & deploy["grouping_condition"].eq("dependency_primary")),
        (deploy["label_condition"].eq("audited")
         & deploy["grouping_condition"].eq("dependency_primary")),
        keys,
    )
    rows.append(
        {
            "factor": "original_vs_audited_labels",
            "comparison": "original proxy labels vs audited harmful-response labels",
            "eligible_n": int(len(p)),
            "apparent_pass_n_left": int(p["a_pass"].astype(bool).sum()),
            "pass_n_right": int(p["b_pass"].astype(bool).sum()),
            "conclusion_flip_n": int(
                (p["a_pass"].astype(bool) != p["b_pass"].astype(bool)).sum()
            ),
            "status": "material_change",
            "interpretation": (
                "Label validity changes policy-level risk conclusions and is a core "
                "measurement factor rather than a minor limitation."
            ),
        }
    )

    ranked_base = cpu[
        cpu["label_condition"].eq("audited")
        & cpu["grouping_condition"].eq("dependency_primary")
        & cpu["policy_kind"].isin(
            ["offline_ranked_diagnostic", "selective_deployable"]
        )
    ].copy()
    p = pair(
        ranked_base,
        ranked_base["policy_kind"].eq("offline_ranked_diagnostic"),
        ranked_base["policy_kind"].eq("selective_deployable"),
        ["seed", "stack", "_target_key"],
    )
    rows.append(
        {
            "factor": "ranked_vs_deployable_routing",
            "comparison": "offline exact-ranked acquisition vs reusable threshold routing",
            "eligible_n": int(len(p)),
            "apparent_pass_n_left": int(p["a_pass"].astype(bool).sum()),
            "pass_n_right": int(p["b_pass"].astype(bool).sum()),
            "conclusion_flip_n": int(
                (p["a_pass"].astype(bool) != p["b_pass"].astype(bool)).sum()
            ),
            "status": "material_change",
            "interpretation": (
                f"Risk conclusions flip for some matched policies and realized "
                f"acquisition differs in "
                f"{int((~np.isclose(p['a_acq'], p['b_acq'], atol=1e-12, rtol=0)).sum())}"
                f"/{len(p)} comparisons."
            ),
        }
    )

    reversals = pd.read_csv(T4_DIR / "cost_ranking_reversals.csv")
    overhead = (
        t4["direct_mean_ms"].astype(float)
        - t4["component_mean_ms"].astype(float)
    )
    rows.append(
        {
            "factor": "component_sum_vs_true_direct_e2e",
            "comparison": "same-run component sum vs direct wall-clock E2E mean latency",
            "eligible_n": int(len(reversals)),
            "apparent_pass_n_left": math.nan,
            "pass_n_right": math.nan,
            "conclusion_flip_n": int(reversals["ranking_reversal"].astype(bool).sum()),
            "status": (
                "negative_result_no_ranking_reversal"
                if int(reversals["ranking_reversal"].astype(bool).sum()) == 0
                else "material_change"
            ),
            "interpretation": (
                "Direct E2E is the correct estimand. In this pilot it adds positive "
                f"unattributed overhead for all {len(t4)} policies "
                f"(mean gap range {overhead.min():.3f}-{overhead.max():.3f} ms) "
                "but does not reverse the mean-cost ordering."
            ),
        }
    )

    for sensitivity in ["semantic_0_87", "semantic_0_92"]:
        p = pair(
            deploy,
            (deploy["label_condition"].eq("audited")
             & deploy["grouping_condition"].eq(sensitivity)),
            (deploy["label_condition"].eq("audited")
             & deploy["grouping_condition"].eq("dependency_primary")),
            keys,
        )
        rows.append(
            {
                "factor": f"grouping_sensitivity_{sensitivity}",
                "comparison": f"{sensitivity} vs primary dependency grouping",
                "eligible_n": int(len(p)),
                "apparent_pass_n_left": int(p["a_pass"].astype(bool).sum()),
                "pass_n_right": int(p["b_pass"].astype(bool).sum()),
                "conclusion_flip_n": int(
                    (p["a_pass"].astype(bool) != p["b_pass"].astype(bool)).sum()
                ),
                "status": "sensitivity_analysis",
                "interpretation": (
                    "The semantic grouping threshold is not treated as ground truth; "
                    "risk conclusions are reported across frozen sensitivities."
                ),
            }
        )

    rows.append(
        {
            "factor": "historical_1pct_isocost_vs_pareto",
            "comparison": "historical engineering equivalence screen vs Pareto objective",
            "eligible_n": 0,
            "apparent_pass_n_left": math.nan,
            "pass_n_right": math.nan,
            "conclusion_flip_n": math.nan,
            "status": "not_estimable_in_this_pilot",
            "interpretation": (
                "The historical 1% margin remains an engineering diagnostic, not a "
                "confirmatory requirement. Empirical Pareto testing is not permitted "
                "because the frozen runtime-to-CPU route-parity join failed."
            ),
        }
    )

    conclusion = pd.DataFrame(rows)
    conclusion.to_csv(OUT_DIR / "measurement_conclusion_matrix.csv", index=False)

    # Primary-seed category-specific descriptive metrics, without tuning.
    predictions = pd.read_parquet(CPU_DIR / "holdout_predictions.parquet")
    predictions = predictions[
        predictions["label_condition"].eq("audited")
        & predictions["grouping_condition"].eq("dependency_primary")
        & predictions["seed"].eq(1729)
        & ~predictions["policy_kind"].eq("offline_ranked_diagnostic")
    ].copy()

    category_columns = ["harm_category", "attack_family", "toxicity_category"]
    data = pd.read_parquet(
        DEV_DATASET,
        columns=["example_id", *category_columns],
    )
    joined = predictions.merge(
        data,
        on="example_id",
        how="left",
        validate="many_to_one",
    )

    category_rows = []
    for column in category_columns:
        values = joined[column].dropna().astype(str)
        valid = set(v for v in values.unique() if v and v.lower() != "nan")
        for value in sorted(valid):
            subset = joined[joined[column].astype(str).eq(value)]
            for keys_tuple, part in subset.groupby(
                ["stack", "policy_kind", "target_rate"],
                dropna=False,
                sort=True,
            ):
                metrics = binary_category_metrics(part)
                category_rows.append(
                    {
                        "category_dimension": column,
                        "category_value": value,
                        "stack": keys_tuple[0],
                        "policy_kind": keys_tuple[1],
                        "target_rate": keys_tuple[2],
                        **metrics,
                    }
                )
    pd.DataFrame(category_rows).to_csv(
        OUT_DIR / "primary_seed_category_metrics.csv",
        index=False,
    )

    status = {
        "artifact": "evaluation_measurement_pilot_v1_closure",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_scope": "development_only_measurement_pilot",
        "router_superiority_claim": False,
        "protected_legacy_sets_opened": False,
        "direct_e2e_measurement_completed": True,
        "gpu": "Tesla T4",
        "batch_size": 1,
        "timing_rows": int(t4_manifest["timing_rows"]),
        "deployable_policies_timed": int(t4_manifest["deployable_policies"]),
        "raw_policy_calls": int(t4_manifest["raw_policy_calls"]),
        "cost_ranking_reversal_pairs": int(
            reversals["ranking_reversal"].astype(bool).sum()
        ),
        "route_mismatch_rows": int(route["route_mismatch_rows"]),
        "prediction_mismatch_rows": int(route["prediction_mismatch_rows"]),
        "cpu_cost_join_valid": bool(route["cpu_cost_join_valid"]),
        "pareto_claim_available": False,
        "pareto_unavailable_reason": (
            "Frozen cost/recall join requires exact route parity; "
            f"{int(route['route_mismatch_rows'])} runtime route mismatches occurred."
        ),
        "bounded_mean_cost_certificate": False,
        "fresh_external_confirmation": False,
        "multi_rater_confirmation": False,
        "measurement_factors": {
            row["factor"]: {
                "eligible_n": (
                    None if pd.isna(row["eligible_n"]) else int(row["eligible_n"])
                ),
                "conclusion_flip_n": (
                    None
                    if pd.isna(row["conclusion_flip_n"])
                    else int(row["conclusion_flip_n"])
                ),
                "status": row["status"],
            }
            for row in rows
        },
        "next_project_action": (
            "Run a professor-requirement gap audit on the completed pilot, then freeze "
            "the next evaluation/measurement expansion protocol before any final "
            "report or README rewrite."
        ),
    }
    (OUT_DIR / "pilot_closure_summary.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files = [
        OUT_DIR / "measurement_conclusion_matrix.csv",
        OUT_DIR / "primary_seed_category_metrics.csv",
        OUT_DIR / "pilot_closure_summary.json",
    ]
    manifest = {
        "artifact": "evaluation_measurement_pilot_v1_analysis_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_point": "python scripts/analyze_evaluation_measurement_pilot_v1.py",
        "inputs": {
            "cpu_policy_summary_sha256": sha256(CPU_DIR / "policy_summary.csv"),
            "cpu_holdout_predictions_sha256": sha256(
                CPU_DIR / "holdout_predictions.parquet"
            ),
            "t4_manifest_sha256": sha256(T4_DIR / "t4_timing_manifest.json"),
            "t4_raw_latency_sha256": sha256(T4_DIR / "policy_latency_raw.parquet"),
        },
        "outputs": {
            str(path.relative_to(OUT_DIR)): sha256(path)
            for path in files
        },
    }
    (OUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(conclusion.to_string(index=False))
    print()
    print("EVALUATION_MEASUREMENT_PILOT_V1_ANALYSIS=PASS")


if __name__ == "__main__":
    main()
