from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist

from monitor_fusion.external_validation.beta_binomial_profile import (
    BetaBinomialProfileNotEstimable,
)
from monitor_fusion.external_validation.cluster_ni_power import (
    _beta_binomial_counts,
    stable_seed,
)
from monitor_fusion.external_validation.constrained_null_mc import (
    one_sample_constrained_null_mc,
    two_sample_ni_constrained_null_mc,
)
from monitor_fusion.external_validation.robust_cluster_generators import (
    calibrate_logit_normal,
)

BASE_SEED = 20260913
OUTER = 3000
INNER = 999
MAX_WORKERS = min(3, max(1, (os.cpu_count() or 2) - 1))

REF_PATTERN = (15, 18, 20, 22, 25)
CMP_PATTERN = (15, 15, 20, 25, 25)
Y1_G_GRID = (120, 140, 160)
Y0_G_GRID = (80, 100, 120)
GENERATORS = ("beta_binomial", "two_point", "logit_normal")
PRIMARY_POWER_GENERATORS = {"beta_binomial", "logit_normal"}

OUT_DIR = Path("results")
PART_DIR = OUT_DIR / "external_validation_targeted_quota_confirmation_v1_parts"
CSV_PATH = OUT_DIR / "external_validation_targeted_quota_confirmation_v1.csv"
JSON_PATH = OUT_DIR / "external_validation_targeted_quota_confirmation_v1.json"


def sizes(g: int, pattern: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        [pattern[i % len(pattern)] for i in range(g)],
        dtype=np.int64,
    )


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k+1, n-k))
    return lo, hi


def draw(
    rng: np.random.Generator,
    p: float,
    rho: float,
    s: np.ndarray,
    generator: str,
    calibration=None,
) -> np.ndarray:
    if generator == "beta_binomial":
        return _beta_binomial_counts(
            marginal_probability=p,
            icc=rho,
            cluster_sizes=s,
            repetitions=1,
            rng=rng,
        )[0].astype(np.int64)

    if generator == "two_point":
        variance = rho * p * (1.0 - p)
        high = min(1.0, p + variance / p)
        prob_high = p / high
        cluster_prob = np.where(
            rng.random(len(s)) < prob_high,
            high,
            0.0,
        )
        return rng.binomial(s, cluster_prob).astype(np.int64)

    if generator == "logit_normal":
        cal = calibration or calibrate_logit_normal(
            marginal_probability=p,
            response_scale_icc=rho,
        )
        z = rng.normal(size=len(s))
        cluster_prob = expit(cal.intercept + cal.random_intercept_sd * z)
        return rng.binomial(s, cluster_prob).astype(np.int64)

    raise ValueError(generator)


def result_row(
    *,
    quota_class: str,
    kind: str,
    g: int,
    scenario: str,
    generator: str,
    nominal_or_target: float,
    success: int,
    estimable: int,
) -> dict:
    lo, hi = cp(success, OUTER)
    return {
        "quota_class": quota_class,
        "kind": kind,
        "G_per_cell": g,
        "nominal_rows_per_cell_at_mean20": 20 * g,
        "scenario": scenario,
        "generator": generator,
        "nominal_alpha_or_power_target": nominal_or_target,
        "success_rate": success / OUTER,
        "mc_cp_lower_95": lo,
        "mc_cp_upper_95": hi,
        "estimable_repetitions": estimable,
        "outer_repetitions": OUTER,
        "inner_repetitions": INNER,
    }


def run_two_sample(task: tuple) -> dict:
    (
        quota_class,
        kind,
        g,
        scenario,
        generator,
        ref_p,
        cmp_p,
        rho_r,
        rho_c,
        ref_pattern_name,
        cmp_pattern_name,
    ) = task

    ref_pattern = REF_PATTERN if ref_pattern_name == "ref" else CMP_PATTERN
    cmp_pattern = REF_PATTERN if cmp_pattern_name == "ref" else CMP_PATTERN
    sr = sizes(g, ref_pattern)
    sc = sizes(g, cmp_pattern)

    if len(sr) != len(sc):
        raise RuntimeError("relative NI requires exact equal cluster counts")

    cal_r = cal_c = None
    if generator == "logit_normal":
        cal_r = calibrate_logit_normal(
            marginal_probability=ref_p,
            response_scale_icc=rho_r,
        )
        cal_c = calibrate_logit_normal(
            marginal_probability=cmp_p,
            response_scale_icc=rho_c,
        )

    rng = np.random.default_rng(
        stable_seed(
            BASE_SEED,
            quota_class,
            kind,
            g,
            scenario,
            generator,
            "outer",
        )
    )
    success = 0
    estimable = 0

    for rep in range(OUTER):
        r = draw(rng, ref_p, rho_r, sr, generator, cal_r)
        c = draw(rng, cmp_p, rho_c, sc, generator, cal_c)
        inner_rng = np.random.default_rng(
            stable_seed(
                BASE_SEED,
                quota_class,
                kind,
                g,
                scenario,
                generator,
                rep,
                "inner",
            )
        )
        try:
            out = two_sample_ni_constrained_null_mc(
                reference_events=r,
                reference_sizes=sr,
                comparison_events=c,
                comparison_sizes=sc,
                margin=0.03,
                alpha=0.025,
                bootstrap_repetitions=INNER,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue

        estimable += 1
        success += int(out.reject_noninferiority_null)

    return result_row(
        quota_class=quota_class,
        kind=kind,
        g=g,
        scenario=scenario,
        generator=generator,
        nominal_or_target=(0.025 if kind == "typeI" else 0.80),
        success=success,
        estimable=estimable,
    )


def run_one_sample(task: tuple) -> dict:
    (
        quota_class,
        kind,
        g,
        scenario,
        generator,
        true_p,
        boundary,
        rho,
        alpha,
        pattern_name,
    ) = task

    pattern = REF_PATTERN if pattern_name == "ref" else CMP_PATTERN
    s = sizes(g, pattern)

    calibration = None
    if generator == "logit_normal":
        calibration = calibrate_logit_normal(
            marginal_probability=true_p,
            response_scale_icc=rho,
        )

    rng = np.random.default_rng(
        stable_seed(
            BASE_SEED,
            quota_class,
            kind,
            g,
            scenario,
            generator,
            "outer",
        )
    )
    success = 0
    estimable = 0

    for rep in range(OUTER):
        x = draw(rng, true_p, rho, s, generator, calibration)
        inner_rng = np.random.default_rng(
            stable_seed(
                BASE_SEED,
                quota_class,
                kind,
                g,
                scenario,
                generator,
                rep,
                "inner",
            )
        )
        try:
            out = one_sample_constrained_null_mc(
                events=x,
                sizes=s,
                boundary=boundary,
                alpha=alpha,
                bootstrap_repetitions=INNER,
                rng=inner_rng,
            )
        except BetaBinomialProfileNotEstimable:
            continue

        estimable += 1
        success += int(out.reject_below_boundary)

    return result_row(
        quota_class=quota_class,
        kind=kind,
        g=g,
        scenario=scenario,
        generator=generator,
        nominal_or_target=(alpha if kind == "typeI" else 0.80),
        success=success,
        estimable=estimable,
    )


def task_key(task: tuple) -> str:
    return "__".join(
        str(x).replace(".", "p")
        for x in task[:5]
    )


def save_part(key: str, row: dict) -> None:
    PART_DIR.mkdir(parents=True, exist_ok=True)
    (PART_DIR / f"{key}.json").write_text(
        json.dumps(row, indent=2) + "\n",
        encoding="utf-8",
    )


def load_part(key: str):
    p = PART_DIR / f"{key}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_tasks():
    tasks = []

    for g in Y1_G_GRID:
        for generator in GENERATORS:
            # Type-I: low-risk unequal ICC, unequal bounded size distributions.
            tasks.append((
                "two",
                (
                    "Y1",
                    "typeI",
                    g,
                    "relative_NI_lowrisk_boundary",
                    generator,
                    0.02,
                    0.05,
                    0.05,
                    0.10,
                    "ref",
                    "cmp",
                ),
            ))
            # Type-I: reverse ICC direction at a typical rate.
            tasks.append((
                "two",
                (
                    "Y1",
                    "typeI",
                    g,
                    "relative_NI_typical_reverseICC_boundary",
                    generator,
                    0.05,
                    0.08,
                    0.10,
                    0.03,
                    "cmp",
                    "ref",
                ),
            ))
            tasks.append((
                "one",
                (
                    "Y1",
                    "typeI",
                    g,
                    "absolute_FNR_boundary",
                    generator,
                    0.10,
                    0.10,
                    0.10,
                    0.025,
                    "cmp",
                ),
            ))

            # Primary Y1 planning-power tasks.
            tasks.append((
                "two",
                (
                    "Y1",
                    "power",
                    g,
                    "relative_NI_zero_diff_lowrisk",
                    generator,
                    0.02,
                    0.02,
                    0.05,
                    0.10,
                    "ref",
                    "cmp",
                ),
            ))
            tasks.append((
                "two",
                (
                    "Y1",
                    "power",
                    g,
                    "relative_NI_zero_diff_primary",
                    generator,
                    0.05,
                    0.05,
                    0.05,
                    0.10,
                    "ref",
                    "cmp",
                ),
            ))
            tasks.append((
                "one",
                (
                    "Y1",
                    "power",
                    g,
                    "absolute_FNR_true_0p05",
                    generator,
                    0.05,
                    0.10,
                    0.10,
                    0.025,
                    "cmp",
                ),
            ))

    for g in Y0_G_GRID:
        for generator in GENERATORS:
            tasks.append((
                "one",
                (
                    "Y0",
                    "typeI",
                    g,
                    "FPR_boundary",
                    generator,
                    0.05,
                    0.05,
                    0.10,
                    0.05,
                    "cmp",
                ),
            ))
            tasks.append((
                "one",
                (
                    "Y0",
                    "power",
                    g,
                    "FPR_true_0p025",
                    generator,
                    0.025,
                    0.05,
                    0.10,
                    0.05,
                    "cmp",
                ),
            ))

    return tasks


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    jobs = []

    for mode, task in build_tasks():
        key = task_key(task)
        cached = load_part(key)
        if cached is not None:
            rows.append(cached)
        else:
            jobs.append((mode, key, task))

    print("=== TARGETED HIGH-REP QUOTA CONFIRMATION ===")
    print(
        f"tasks total={len(rows)+len(jobs)} cached={len(rows)} "
        f"pending={len(jobs)} workers={MAX_WORKERS} "
        f"outer/task={OUTER} inner={INNER}"
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fmap = {}
            for mode, key, task in jobs:
                fn = run_two_sample if mode == "two" else run_one_sample
                future = ex.submit(fn, task)
                fmap[future] = (
                    key,
                    task[0],
                    task[1],
                    task[2],
                    task[3],
                    task[4],
                )

            for future in as_completed(fmap):
                key, quota_class, kind, g, scenario, generator = fmap[future]
                row = future.result()
                save_part(key, row)
                rows.append(row)
                print(
                    f"DONE {quota_class} {kind} G={g} {generator} {scenario}: "
                    f"{row['success_rate']:.4f} "
                    f"(MC95% {row['mc_cp_lower_95']:.4f}-"
                    f"{row['mc_cp_upper_95']:.4f})"
                )

    rows = sorted(
        rows,
        key=lambda r: (
            r["quota_class"],
            r["G_per_cell"],
            r["kind"],
            r["scenario"],
            r["generator"],
        ),
    )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(quota_class: str, grid: tuple[int, ...]):
        out = []
        for g in grid:
            sub = [
                r for r in rows
                if r["quota_class"] == quota_class
                and r["G_per_cell"] == g
            ]
            typei = [r for r in sub if r["kind"] == "typeI"]
            power_primary = [
                r for r in sub
                if r["kind"] == "power"
                and r["generator"] in PRIMARY_POWER_GENERATORS
            ]
            power_stress = [
                r for r in sub
                if r["kind"] == "power"
                and r["generator"] == "two_point"
            ]

            clear_typei = [
                r for r in typei
                if float(r["mc_cp_lower_95"])
                > float(r["nominal_alpha_or_power_target"])
            ]
            min_primary_lower = min(
                float(r["mc_cp_lower_95"]) for r in power_primary
            )
            min_primary_point = min(
                float(r["success_rate"]) for r in power_primary
            )
            min_stress_point = min(
                float(r["success_rate"]) for r in power_stress
            )

            out.append({
                "G_per_cell": g,
                "rows_per_cell_at_mean20": 20 * g,
                "typeI_clear_failure_count": len(clear_typei),
                "typeI_passes_final_gate": len(clear_typei) == 0,
                "minimum_primary_power_point": min_primary_point,
                "minimum_primary_power_mc95_lower": min_primary_lower,
                "minimum_two_point_power_stress_point": min_stress_point,
                "power_passes_final_gate": min_primary_lower >= 0.80,
                "final_candidate": (
                    len(clear_typei) == 0
                    and min_primary_lower >= 0.80
                ),
            })
        return out

    y1_summary = summarize("Y1", Y1_G_GRID)
    y0_summary = summarize("Y0", Y0_G_GRID)

    y1_choice = next((x for x in y1_summary if x["final_candidate"]), None)
    y0_choice = next((x for x in y0_summary if x["final_candidate"]), None)

    payload = {
        "confirmation_id": "external_validation_targeted_quota_confirmation_v1",
        "status": (
            "candidate_quotas_identified_pending_contract_freeze"
            if y1_choice is not None and y0_choice is not None
            else "no_joint_candidate_stop_for_supervisor_level_redesign"
        ),
        "outer_repetitions_per_task": OUTER,
        "inner_null_repetitions": INNER,
        "cluster_size_range_candidate": [15, 25],
        "reference_pattern": list(REF_PATTERN),
        "comparison_stress_pattern": list(CMP_PATTERN),
        "power_gate_primary_generators": sorted(PRIMARY_POWER_GENERATORS),
        "two_point_power_role": "stress sensitivity",
        "Y1_summary": y1_summary,
        "Y0_summary": y0_summary,
        "Y1_selected_candidate": y1_choice,
        "Y0_selected_candidate": y0_choice,
        "important": (
            "Even if candidates are identified, raw candidate caps and the "
            "operational human-author/model-batch source contracts still need "
            "to be frozen before W0."
        ),
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== FINAL CONFIRMATION SUMMARY ===")
    print("Y1:")
    for x in y1_summary:
        print(
            f"  G={x['G_per_cell']} rows={x['rows_per_cell_at_mean20']}: "
            f"typeI={'PASS' if x['typeI_passes_final_gate'] else 'FAIL'}; "
            f"min primary power={x['minimum_primary_power_point']:.3f}; "
            f"min lower95={x['minimum_primary_power_mc95_lower']:.3f}; "
            f"two-point stress min={x['minimum_two_point_power_stress_point']:.3f}; "
            f"candidate={'YES' if x['final_candidate'] else 'NO'}"
        )

    print("Y0:")
    for x in y0_summary:
        print(
            f"  G={x['G_per_cell']} rows={x['rows_per_cell_at_mean20']}: "
            f"typeI={'PASS' if x['typeI_passes_final_gate'] else 'FAIL'}; "
            f"min primary power={x['minimum_primary_power_point']:.3f}; "
            f"min lower95={x['minimum_primary_power_mc95_lower']:.3f}; "
            f"two-point stress min={x['minimum_two_point_power_stress_point']:.3f}; "
            f"candidate={'YES' if x['final_candidate'] else 'NO'}"
        )

    print()
    if y1_choice is None or y0_choice is None:
        print(
            "DECISION: STOP. No jointly valid Y1/Y0 candidate was confirmed. "
            "Do not continue estimator hunting; this requires supervisor-level "
            "design reconsideration."
        )
    else:
        print(
            "CANDIDATE Y1 QUOTA: "
            f"G={y1_choice['G_per_cell']} independent provenance clusters "
            f"per cell, {y1_choice['rows_per_cell_at_mean20']} eligible Y=1 "
            "dependency representatives per cell at the mean-20 pattern."
        )
        print(
            "CANDIDATE Y0 QUOTA: "
            f"G={y0_choice['G_per_cell']} independent provenance clusters "
            f"per cell, {y0_choice['rows_per_cell_at_mean20']} eligible Y=0 "
            "dependency representatives per cell at the mean-20 pattern."
        )
        print(
            "These are candidate statistical quotas, not W0 authorization. "
            "Next freeze the inference/design contracts and reconcile them with "
            "the human-author/model-generator-batch source contracts and raw "
            "candidate caps."
        )


if __name__ == "__main__":
    run()
