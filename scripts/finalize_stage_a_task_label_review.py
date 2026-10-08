#!/usr/bin/env python3
"""Finalize the researcher-approved Stage-A AI-assisted review without changing frozen results.

Reads the original blind worksheets and exact AI-first-pass drafts.
Copies human-signed-off new artifacts to Windows Downloads for easy upload.
Does not re-label cases, resolve uncertainties, or modify existing input/analysis files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "results" / "stage_a_task_label_validity_review_v1"
DATE = "2026-10-08"
DRAFTS = {
    "policy": {
        "filename": "policy_review_AI_DRAFT.csv",
        "sha256": "f0818dc94c8d59b004714e06d0621e71eb72bd48011434ee938161f2ff6af25b",
        "blank": "blind_native_policy_review.csv",
        "blank_sha256": "4f15be5d418682587e6d60bead3d3902b27eab262495094ea980dae303fb0bd8",
        "out": "policy_review_researcher_approved.csv",
        "count": 608,
        "editable": {"reviewer_id", "policy_label", "policy_guideline", "label_rationale", "review_status"},
    },
    "transformation": {
        "filename": "transformation_review_AI_DRAFT.csv",
        "sha256": "ae6f2089f2d968e3bf24d544d936f9c0434bfbbdf38d4b6af958441c6141113e",
        "blank": "blind_transformation_review.csv",
        "blank_sha256": "8fb6f11d03a9e86cc85396391d7d2eb3e9813d381b93f016543ee8ed432b6356",
        "out": "transformation_review_researcher_approved.csv",
        "count": 304,
        "editable": {"reviewer_id", "meaning_preserved", "label_preserved", "review_rationale", "review_status"},
    },
}

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def parse(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    with io.StringIO(data.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)

def encode(headers: list[str], data: list[dict[str, str]]) -> bytes:
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=headers, lineterminator="\n")
    w.writeheader()
    w.writerows(data)
    return s.getvalue().encode("utf-8")

def windows_downloads() -> Path:
    try:
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             '[Environment]::GetFolderPath("UserProfile")'],
            capture_output=True, text=True, check=True,
        )
        q = subprocess.run(
            ["wslpath", "-u", p.stdout.strip()],
            capture_output=True, text=True, check=True,
        )
        path = Path(q.stdout.strip()) / "Downloads"
        if path.is_dir():
            return path
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    alternatives = list(Path("/mnt/c/Users").glob("*/Downloads"))
    if len(alternatives) == 1:
        return alternatives[0]
    raise RuntimeError("Could not identify Windows Downloads directory.")

def draft_bytes(downloads: Path, spec: dict) -> bytes:
    candidates = [downloads / spec["filename"]]
    candidates.extend(sorted(downloads.glob(spec["filename"].replace(".csv", "*.csv"))))
    for path in dict.fromkeys(candidates):
        if path.is_file():
            body = path.read_bytes()
            if digest(body) == spec["sha256"]:
                print("Verified downloaded draft:", path.name)
                return body
    for path in sorted(downloads.glob("stage_a_phase2_review_drafts*.zip")):
        with zipfile.ZipFile(path) as z:
            for entry in z.namelist():
                if Path(entry).name == spec["filename"]:
                    body = z.read(entry)
                    if digest(body) == spec["sha256"]:
                        print("Verified draft from ZIP:", path.name)
                        return body
    raise RuntimeError(
        f"Exact {spec['filename']} not found in Windows Downloads. "
        "Download the original AI draft or its review ZIP without editing it."
    )

def finalize_one(downloads: Path, spec: dict) -> tuple[bytes, Counter]:
    blindpath = REVIEW_DIR / spec["blank"]
    if not blindpath.exists():
        raise RuntimeError(f"Missing original blind worksheet: {blindpath}")
    blank = blindpath.read_bytes()
    if digest(blank) != spec["blank_sha256"]:
        raise RuntimeError(f"Original blank worksheet has changed: {blindpath}")
    draft = draft_bytes(downloads, spec)
    header, base = parse(blank)
    draft_header, annotated = parse(draft)
    if header != draft_header or len(base) != spec["count"] or len(annotated) != spec["count"]:
        raise RuntimeError("Draft and blank CSV columns/count differ")

    ids = set()
    for original, reviewed in zip(base, annotated):
        if reviewed["review_id"] in ids:
            raise RuntimeError("Duplicate review ID")
        ids.add(reviewed["review_id"])
        for column in header:
            if column not in spec["editable"] and original[column] != reviewed[column]:
                raise RuntimeError(f"Unapproved input difference: {reviewed['review_id']} / {column}")
        if reviewed["reviewer_id"] != "AI_ASSISTED_FIRST_PASS_GPT6":
            raise RuntimeError("AI draft origin unexpectedly changed")
        if reviewed["review_status"] != "ai_draft_unverified":
            raise RuntimeError("Expected unchanged AI draft status")
        if not reviewed["review_id"] or not reviewed["blind_case_id"]:
            raise RuntimeError("Missing case identifier")

    if spec["count"] == 608:
        allowed = {"violation", "not_violation", "uncertain", "out_of_scope"}
        labels = Counter(row["policy_label"] for row in annotated)
        if set(labels) - allowed or any(not r["label_rationale"].strip() for r in annotated):
            raise RuntimeError("Unexpected policy labels or blank rationales")
    else:
        allowed = {"yes", "no", "uncertain"}
        labels = Counter(row["label_preserved"] for row in annotated)
        if any(r["meaning_preserved"] not in allowed or
               r["label_preserved"] not in allowed or
               not r["review_rationale"].strip() for r in annotated):
            raise RuntimeError("Unexpected transformation judgments or blank rationales")

    for row in annotated:
        row["review_status"] = "researcher_verified"
        # Preserve reviewer_id as AI first-pass provenance. Record human signoff separately.
        row["researcher_signoff"] = "approved_ai_assisted_first_pass"
        row["researcher_signoff_date"] = DATE
    return encode(draft_header + ["researcher_signoff", "researcher_signoff_date"], annotated), labels

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path)
    args = ap.parse_args()
    downloads = args.downloads or windows_downloads()

    outputs = {}
    counts = {}
    for label, spec in DRAFTS.items():
        body, summary = finalize_one(downloads, spec)
        outputs[spec["out"]] = body
        counts[label] = dict(summary)

    record = {
        "audit_id": "stage_a_phase2_researcher_approved_ai_assisted_v1",
        "signoff_date": DATE,
        "approval_basis": "User explicitly reported reviewing and approving AI-draft judgments in ChatGPT on 2026-10-08; no item-level corrections supplied.",
        "annotation_provenance": "GPT-6 AI-assisted first pass; researcher-reviewed and approved; not independent annotation.",
        "do_not_claim": ["independent multi-rater annotation", "inter-annotator agreement", "native-model semantic understanding", "response-level outcome labels", "confirmatory validation"],
        "policy_label_counts": counts["policy"],
        "transformation_label_preserved_counts": counts["transformation"],
        "original_ai_draft_sha256": {name: cfg["sha256"] for name, cfg in DRAFTS.items()},
        "original_blind_sha256": {name: cfg["blank_sha256"] for name, cfg in DRAFTS.items()},
        "approved_files_sha256": {name: digest(body) for name, body in outputs.items()},
        "frozen_experiment_modified": False,
        "frozen_readout_transfer_signal": "false for all four guards; unchanged",
        "interpretation": "Exploratory prompt-moderation policy review only. Uncertain/out-of-scope cases must not be coerced into binary labels or used as independently verified truth.",
    }
    outputs["researcher_approval_record.json"] = (
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    # Validate every input first; then write only NEW files.
    for name in outputs:
        target = REVIEW_DIR / name
        if target.exists() and target.read_bytes() != outputs[name]:
            raise RuntimeError(f"Existing final artifact differs; refusing overwrite: {target}")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in outputs.items():
        (REVIEW_DIR / name).write_bytes(body)
        shutil.copy2(REVIEW_DIR / name, downloads / name)

    print("SUCCESS: researcher-approved AI-assisted artifacts prepared.")
    print("Policy labels:", counts["policy"])
    print("Transformation label preservation:", counts["transformation"])
    print("Files copied to Windows Downloads:")
    for name in outputs:
        print(" ", downloads / name)
    print("Original frozen results: unchanged. Private review key: not included.")

if __name__ == "__main__":
    main()
