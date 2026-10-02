from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "results/stage_a_human_direct_authoring_v1/human_direct_authoring_worksheet.csv"
MAP = ROOT / "results/stage_a_human_direct_authoring_v1/human_direct_authoring_map.csv"
MODEL = ROOT / "results/stage_a_model_direct_generation_v1/model_direct_generations.csv"
OBF = ROOT / "results/stage_a_obfuscation_generation_v1/obfuscated_pairs.csv"
ELIG = ROOT / "results/stage_a_pair_eligibility_v1/pair_eligibility.csv"
SEM = ROOT / "results/stage_a_semantic_equivalence_review_v1/semantic_equivalence_worksheet.csv"
SEV = ROOT / "results/stage_a_severity_review_v1/severity_review_worksheet.csv"
CONTRACT = ROOT / "configs/stage_a_guard_panel_prompt_contract_v1.json"
OUT = ROOT / "results/stage_a_guard_panel_input_v1"


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    human = rows(HUMAN); mapping = rows(MAP); model = rows(MODEL); obf = rows(OBF)
    elig = rows(ELIG); sem = rows(SEM); sev = rows(SEV)
    assert all(len(x) == 104 for x in [human, mapping, model, obf, elig, sem, sev])

    h_by_author = {r["authoring_item_id"]: r for r in human}
    h_by_id = {m["base_intent_id"]: h_by_author[m["authoring_item_id"]] for m in mapping}
    m_by_id = {r["base_intent_id"]: r for r in model}
    o_by_id = {r["base_intent_id"]: r for r in obf}
    e_by_id = {r["base_intent_id"]: r for r in elig}
    s_by_id = {r["base_intent_id"]: r for r in sem}
    v_by_id = {r["base_intent_id"]: r for r in sev}
    ids = set(h_by_id)
    assert len(ids) == 104
    assert ids == set(m_by_id) == set(o_by_id) == set(e_by_id) == set(s_by_id) == set(v_by_id)

    out_rows = []
    for bid in sorted(ids):
        h, m, o, e, s, v = h_by_id[bid], m_by_id[bid], o_by_id[bid], e_by_id[bid], s_by_id[bid], v_by_id[bid]
        variants = [
            ("human_direct", "human", "direct", h["human_direct_text"]),
            ("model_direct", "model", "direct", m["model_direct_text"]),
            ("human_obfuscated", "human", "obfuscated", o["human_obfuscated_text"]),
            ("model_obfuscated", "model", "obfuscated", o["model_obfuscated_text"]),
        ]
        for variant, source, representation, prompt in variants:
            assert prompt.strip()
            out_rows.append({
                "stage_a_example_id": f"SA::{bid}::{variant}",
                "base_intent_id": bid,
                "reviewed_category": e["reviewed_category"],
                "obfuscation_transform_id": e["obfuscation_transform_id"],
                "variant": variant,
                "source": source,
                "representation": representation,
                "prompt_text": prompt,
                "semantic_status": e["semantic_status"],
                "severity_status": e["severity_status"],
                "base_intent_severity": v["base_intent_severity"],
                "human_direct_severity": v["human_direct_severity"],
                "model_direct_severity": v["model_direct_severity"],
                "eligible_human_direct_vs_human_obfuscated": e["eligible_human_direct_vs_human_obfuscated"],
                "eligible_model_direct_vs_model_obfuscated": e["eligible_model_direct_vs_model_obfuscated"],
                "eligible_human_direct_vs_model_direct_strict": e["eligible_human_direct_vs_model_direct_strict"],
                "eligible_human_obfuscated_vs_model_obfuscated_strict": e["eligible_human_obfuscated_vs_model_obfuscated_strict"],
                "eligible_source_semantic_only_sensitivity": e["eligible_source_semantic_only_sensitivity"],
            })

    assert len(out_rows) == 416
    assert len({r["stage_a_example_id"] for r in out_rows}) == 416
    vc = Counter(r["variant"] for r in out_rows)
    assert vc == Counter({"human_direct": 104, "model_direct": 104, "human_obfuscated": 104, "model_obfuscated": 104})

    OUT.mkdir(exist_ok=True)
    out_csv = OUT / "stage_a_guard_panel_prompts.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)

    manifest = {
        "artifact_id": "stage_a_guard_panel_input_v1",
        "status": "frozen_pre_preflight",
        "rows": 416,
        "base_intents": 104,
        "variant_counts": dict(sorted(vc.items())),
        "guard_count": 4,
        "planned_monitor_prompt_scores": 1664,
        "inputs": {p.name: sha(p) for p in [HUMAN, MAP, MODEL, OBF, ELIG, SEM, SEV, CONTRACT]},
        "output_csv_sha256": sha(out_csv),
        "guard_scoring_performed": False,
        "q_estimation_performed": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STAGE-A GUARD INPUT: PASS")
    print("rows=416 base_intents=104 variants=104 each")
    print("csv_sha256=", sha(out_csv))

if __name__ == "__main__":
    main()
