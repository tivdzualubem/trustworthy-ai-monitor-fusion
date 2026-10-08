#!/usr/bin/env python3
"""Prepare blind policy and transformation-review worksheets from frozen Stage-A inputs.

Does not read predictions, infer labels, or modify frozen inputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results" / "stage_a_readout_transfer_kill_v1"
PROBE = PREFIX / "readout_probe_prompts.csv"
ABLATION = PREFIX / "o3_ablation_prompts.csv"
OUTPUT = ROOT / "results" / "stage_a_task_label_validity_review_v1"

EXPECTED_PROBE_SHA = "e9d4c6047c19cc75c1d3b1badd6abc0fa6cf3c234507efd96353597dfd044a8b"
EXPECTED_O3_SHA = "a425e3cea9b097da4006fd1b9f90e202ef96a774efba39b74851111f4b2594de"
GUARDS = ["llama_guard_3_1b", "shieldgemma_2b",
          "granite_guardian_3_3_8b", "qwen3guard_gen_4b"]
CONDITIONS = ["O2", "base64_payload_only", "plain_wrapper", "encoded_wrapper"]

def rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(path, headers, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUTPUT)
    args = ap.parse_args()
    for path, digest in ((PROBE, EXPECTED_PROBE_SHA), (ABLATION, EXPECTED_O3_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen input SHA256 differs: {path}")

    probe = rows(PROBE)
    ablation = rows(ABLATION)
    assert len(probe) == 152 and len(ablation) == 304

    by_case = {}
    for p in probe:
        case = p["semantic_case_id"]
        key = p["representation"]
        if key in by_case.setdefault(case, {}):
            raise RuntimeError(f"Repeated case: {case}/{key}")
        by_case[case][key] = p
    if len(by_case) != 76:
        raise RuntimeError("Unexpected number of semantic cases")

    subs = str.maketrans({"a":"4","e":"3","i":"1","o":"0","s":"5",
                         "A":"4","E":"3","I":"1","O":"0","S":"5"})
    for case, p in by_case.items():
        if set(p) != {"direct", "O2"}:
            raise RuntimeError(f"Incomplete pair {case}")
        a,b = p["direct"],p["O2"]
        if a["prompt_text"].translate(subs) != b["prompt_text"]:
            raise RuntimeError(f"Mechanical O2 mismatch: {case}")
        if a["fold_id"] != b["fold_id"] or a["label_harmful"] != b["label_harmful"]:
            raise RuntimeError(f"Fold/label mismatch: {case}")

    o3_by_case = {}
    for p in ablation:
        d = o3_by_case.setdefault(p["semantic_case_id"], {})
        if p["condition"] in d:
            raise RuntimeError("Duplicate O3 condition")
        d[p["condition"]] = p
    if set(o3_by_case) != set(by_case):
        raise RuntimeError("O3/probe cases differ")

    # Reviewer input: omit case_type, proxy labels, categories, and predictions.
    policy = []
    transform = []
    for case in sorted(by_case):
        p = by_case[case]
        a = p["direct"]
        o = p["O2"]
        same = o3_by_case[case]
        if set(same) != {"direct_plain", "base64_payload_only", "plain_wrapper", "encoded_wrapper"}:
            raise RuntimeError(f"Incomplete ablation: {case}")
        if same["direct_plain"]["prompt_text"] != a["prompt_text"]:
            raise RuntimeError(f"Direct-text mismatch: {case}")
        for guard in GUARDS:
            for rep in ["direct", "O2"]:
                policy.append({
                    "review_id": f"{case}::{guard}::{rep}",
                    "semantic_case_id":case,"guard":guard,"representation":rep,
                    "prompt_text":p[rep]["prompt_text"],
                    "reviewer_id":"","policy_label":"","policy_guideline":"",
                    "label_rationale":"","review_status":"pending"
                })
        for cond in CONDITIONS:
            variant = (o["prompt_text"] if cond == "O2" else same[cond]["prompt_text"])
            transform.append({
                "review_id":f"{case}::{cond}",
                "semantic_case_id":case,"condition":cond,
                "direct_text":a["prompt_text"],"transformed_text":variant,
                "reviewer_id":"","meaning_preserved":"","label_preserved":"",
                "review_rationale":"","review_status":"pending"
            })

    if len(policy) != 608 or len(transform) != 304:
        raise RuntimeError("Review row counts incorrect")
    save(args.outdir/"blind_native_policy_review.csv",list(policy[0]),policy)
    save(args.outdir/"blind_transformation_review.csv",list(transform[0]),transform)
    print("policy_rows:",len(policy),"transformation_rows:",len(transform))
    print("inputs_verified: yes; original results unchanged: yes")
    print("output:",args.outdir)

if __name__ == "__main__":
    main()
