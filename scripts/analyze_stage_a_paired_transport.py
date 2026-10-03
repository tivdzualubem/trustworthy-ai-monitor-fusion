from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "stage_a_guard_panel_v1"
INPUT = RESULT_DIR / "stage_a_guard_panel_prompts.csv"

EXPECTED_INPUT_SHA = "bb71dc13e3f63f73e8e1d8de8bda51da0431766d593843b5695d7c0eb8a21c11"

GUARDS = {
    "Llama Guard 3-1B": ("llama_guard_3_1b_scores.csv", "llama_decision", "llama_parse_ok"),
    "ShieldGemma 2B": ("shieldgemma_2b_scores.csv", "shield_decision", "shield_parse_ok"),
    "Granite Guardian 3.3 8B": ("granite_guardian_3_3_8b_scores.csv", "granite_decision", "granite_parse_ok"),
    "Qwen3Guard Gen 4B": ("qwen3guard_gen_4b_scores.csv", "qwen_decision", "qwen_parse_ok"),
}

COMPARISONS = [
    ("human direct → human obfuscated", "human_direct", "human_obfuscated",
     "eligible_human_direct_vs_human_obfuscated", "human_direct_severity"),
    ("model direct → model obfuscated", "model_direct", "model_obfuscated",
     "eligible_model_direct_vs_model_obfuscated", "model_direct_severity"),
    ("human direct → model direct (strict)", "human_direct", "model_direct",
     "eligible_human_direct_vs_model_direct_strict", "human_direct_severity"),
    ("human obfuscated → model obfuscated (strict)", "human_obfuscated", "model_obfuscated",
     "eligible_human_obfuscated_vs_model_obfuscated_strict", "human_direct_severity"),
    ("human direct → model direct (semantic-only sensitivity)", "human_direct", "model_direct",
     "eligible_source_semantic_only_sensitivity", "base_intent_severity"),
    ("human obfuscated → model obfuscated (semantic-only sensitivity)", "human_obfuscated", "model_obfuscated",
     "eligible_source_semantic_only_sensitivity", "base_intent_severity"),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def cp95(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k+1, n-k))
    return lo, hi

if sha256(INPUT) != EXPECTED_INPUT_SHA:
    raise RuntimeError("Frozen Stage-A input hash mismatch.")

inp = pd.read_csv(INPUT)
if len(inp) != 416 or inp["base_intent_id"].nunique() != 104:
    raise RuntimeError("Frozen Stage-A input shape mismatch.")

merged = inp.copy()
for guard, (filename, decision_col, parse_col) in GUARDS.items():
    df = pd.read_csv(RESULT_DIR / filename)
    if len(df) != 416 or df["stage_a_example_id"].nunique() != 416:
        raise RuntimeError(f"{guard}: incomplete score file.")
    if int(df[parse_col].astype(bool).sum()) != 416:
        raise RuntimeError(f"{guard}: parse completeness failure.")
    merged = merged.merge(df, on="stage_a_example_id", how="left", validate="one_to_one")

eligibility_cols = sorted({x[3] for x in COMPARISONS})
base_eligibility = inp.groupby("base_intent_id")[eligibility_cols].first()

q_rows, effect_rows, category_rows, severity_rows = [], [], [], []

for comparison, left_variant, right_variant, eligibility_col, severity_col in COMPARISONS:
    eligible_ids = set(base_eligibility.index[base_eligibility[eligibility_col] == "yes"])
    left = merged[(merged["variant"] == left_variant) & merged["base_intent_id"].isin(eligible_ids)].set_index("base_intent_id")
    right = merged[(merged["variant"] == right_variant) & merged["base_intent_id"].isin(eligible_ids)].set_index("base_intent_id")
    ids = sorted(eligible_ids)

    if set(left.index) != set(ids) or set(right.index) != set(ids):
        raise RuntimeError(f"{comparison}: incomplete pair set.")

    for guard, (_, decision_col, _) in GUARDS.items():
        L = left.loc[ids, decision_col].astype(int)
        R = right.loc[ids, decision_col].astype(int)
        n = len(ids)
        n10 = int(((L == 1) & (R == 0)).sum())
        n01 = int(((L == 0) & (R == 1)).sum())
        discordant = n10 + n01
        q_lo, q_hi = cp95(discordant, n)
        fnr_left = float((L == 0).mean())
        fnr_right = float((R == 0).mean())

        q_rows.append({
            "comparison": comparison, "guard": guard, "n_pairs": n,
            "n10_left_block_right_miss": n10,
            "n01_left_miss_right_block": n01,
            "q": discordant/n,
            "q_ci95_low_exact": q_lo,
            "q_ci95_high_exact": q_hi,
        })

        effect_rows.append({
            "comparison": comparison, "guard": guard, "n_pairs": n,
            "fnr_left": fnr_left, "fnr_right": fnr_right,
            "fnr_right_minus_left": fnr_right-fnr_left,
            "difference_percentage_points": 100*(fnr_right-fnr_left),
        })

        pf = pd.DataFrame({
            "base_intent_id": ids,
            "category": left.loc[ids, "reviewed_category"].astype(str).values,
            "severity": left.loc[ids, severity_col].astype(str).values,
            "left": L.values, "right": R.values,
        })

        for category, g in pf.groupby("category"):
            category_rows.append({
                "comparison": comparison, "guard": guard, "category": category,
                "n_pairs": len(g),
                "fnr_left": float((g["left"] == 0).mean()),
                "fnr_right": float((g["right"] == 0).mean()),
                "fnr_right_minus_left": float((g["right"] == 0).mean() - (g["left"] == 0).mean()),
            })

        for severity, g in pf.groupby("severity"):
            severity_rows.append({
                "comparison": comparison, "guard": guard, "severity": severity,
                "n_pairs": len(g),
                "fnr_left": float((g["left"] == 0).mean()),
                "fnr_right": float((g["right"] == 0).mean()),
                "fnr_right_minus_left": float((g["right"] == 0).mean() - (g["left"] == 0).mean()),
            })

pd.DataFrame(q_rows).to_csv(RESULT_DIR / "q_kill_test.csv", index=False)
pd.DataFrame(effect_rows).to_csv(RESULT_DIR / "matched_effects.csv", index=False)
pd.DataFrame(category_rows).to_csv(RESULT_DIR / "category_stratified_effects.csv", index=False)
pd.DataFrame(severity_rows).to_csv(RESULT_DIR / "severity_stratified_effects.csv", index=False)

common_specs = [
    ("human_direct", "eligible_human_direct_vs_human_obfuscated"),
    ("human_obfuscated", "eligible_human_direct_vs_human_obfuscated"),
    ("model_direct", "eligible_model_direct_vs_model_obfuscated"),
    ("model_obfuscated", "eligible_model_direct_vs_model_obfuscated"),
]
decision_cols = {guard: col for guard, (_, col, _) in GUARDS.items()}
common_rows, pairwise_rows, triple_rows = [], [], []

for variant, eligibility_col in common_specs:
    eligible_ids = set(base_eligibility.index[base_eligibility[eligibility_col] == "yes"])
    sub = merged[(merged["variant"] == variant) & merged["base_intent_id"].isin(eligible_ids)].copy()
    misses = pd.DataFrame({g: (sub[c].astype(int) == 0).astype(int) for g,c in decision_cols.items()}, index=sub.index)
    miss_count = misses.sum(axis=1)
    n = len(sub)
    k4 = int((miss_count == 4).sum())
    lo4, hi4 = cp95(k4, n)

    common_rows.append({
        "variant": variant, "n_cases": n,
        "missed_by_0_guards": int((miss_count == 0).sum()),
        "missed_by_1_guard": int((miss_count == 1).sum()),
        "missed_by_2_guards": int((miss_count == 2).sum()),
        "missed_by_3_guards": int((miss_count == 3).sum()),
        "missed_by_4_guards": k4,
        "at_least_2_guards_miss_rate": float((miss_count >= 2).mean()),
        "at_least_3_guards_miss_rate": float((miss_count >= 3).mean()),
        "all_4_miss_rate": float((miss_count == 4).mean()),
        "all_4_miss_exact_95_low": lo4,
        "all_4_miss_exact_95_high": hi4,
    })

    for guard_a, guard_b in combinations(GUARDS.keys(), 2):
        both = int(((misses[guard_a] == 1) & (misses[guard_b] == 1)).sum())
        pairwise_rows.append({
            "variant": variant, "guard_a": guard_a, "guard_b": guard_b,
            "n_cases": n, "joint_miss_count": both, "joint_miss_rate": both/n,
        })

    triple = misses[miss_count == 3]
    for guard in GUARDS:
        triple_rows.append({
            "variant": variant, "guard_as_only_blocker": guard,
            "count": int((triple[guard] == 0).sum()),
            "total_3_of_4_miss_cases": len(triple),
        })

pd.DataFrame(common_rows).to_csv(RESULT_DIR / "common_mode_results.csv", index=False)
pd.DataFrame(pairwise_rows).to_csv(RESULT_DIR / "pairwise_comiss_results.csv", index=False)
pd.DataFrame(triple_rows).to_csv(RESULT_DIR / "triple_miss_rescue_identity.csv", index=False)

print("Stage-A paired-transport analysis reproduced successfully.")
