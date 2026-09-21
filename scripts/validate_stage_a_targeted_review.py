"""Prepare and validate a human-completed Stage-A worksheet; never classify text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/stage_a_targeted_review_handoff_v1.json")
ONTOLOGY = Path("configs/external_validation_common_ontology_v1.json")
CROSSWALK = Path("configs/stage_a_public_crosswalk_v1.json")
HANDOFF = ROOT / "results/stage_a_targeted_review_handoff_v1"
REVIEW_FIELDS = ("review_decision", "reviewed_category", "review_rationale",
                 "reviewer", "independent_intent_group")
REQUIRED = ("candidate_id", "source", "revision", "source_label", "text", "text_hash",
            "proposed_screen_category", "screening_reason") + REVIEW_FIELDS
CATEGORIES = tuple(f"C{i}" for i in range(1, 9))
PENDING = {
    "severity_review_status": "pending",
    "formulation_pair_generation_status": "pending",
    "semantic_equivalence_review_status": "pending",
}
GATES = {
    "pair_generation_authorized": False,
    "severity_review_authorized": False,
    "semantic_equivalence_review_authorized": False,
    "guard_scoring_authorized": False,
    "q_estimation_authorized": False,
    "w0_authorized": False,
    "fresh_confirmatory_scoring_authorized": False,
    "prospective_design_frozen": False,
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def order_key(group: str, seed: int) -> str:
    # Same seed/group hashing convention as the existing full-pool selector.
    return hashlib.sha256(normalized(f"{seed}:{group}").encode("utf-8")).hexdigest()


def read_table(path: Path) -> tuple[list[str], list[dict], bytes]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    columns = reader.fieldnames or []
    rows = list(reader)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("CSV headers must be nonempty and unique")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row: number of cells differs from the header")
    return columns, rows, raw


def table_hash(columns: list[str], rows: list[dict]) -> str:
    return canonical_hash({"columns": columns, "rows": [[r[c] for c in columns] for r in rows]})


def load_frozen(root: Path = ROOT) -> tuple[dict, list[str], list[dict], bytes]:
    contract = json.loads((root / CONTRACT).read_text(encoding="utf-8"))
    ontology = json.loads((root / ONTOLOGY).read_text(encoding="utf-8"))
    crosswalk = json.loads((root / CROSSWALK).read_text(encoding="utf-8"))
    if canonical_hash(ontology) != contract["ontology_json_sha256"]:
        raise ValueError("Frozen ontology fingerprint mismatch")
    if canonical_hash(crosswalk) != contract["crosswalk_json_sha256"]:
        raise ValueError("Frozen crosswalk fingerprint mismatch")
    if {c["id"] for c in ontology["categories"]} != set(CATEGORIES):
        raise ValueError("Expected frozen C1-C8 ontology")
    if (contract["queue_rows"], contract["selection_seed"], contract["selection_per_category"]) != (200, 20260921, 13):
        raise ValueError("Unexpected queue size, selection seed, or quota")
    if crosswalk["selection_seed"] != 20260921 or crosswalk["target_per_category"] != 13:
        raise ValueError("Selection settings differ from the existing Stage-A contract")
    columns, rows, raw = read_table(root / contract["queue_path"])
    if set(REQUIRED) - set(columns) or table_hash(columns, rows) != contract["queue_table_sha256"]:
        raise ValueError("Immutable screening queue schema or fingerprint mismatch")
    if len(rows) != 200 or len({r["candidate_id"] for r in rows}) != 200:
        raise ValueError("Expected 200 unique frozen candidate IDs")
    if Counter(r["proposed_screen_category"] for r in rows) != {c: 25 for c in CATEGORIES}:
        raise ValueError("Expected 25 frozen screening nominations per category")
    if any(r[field] for r in rows for field in REVIEW_FIELDS):
        raise ValueError("Immutable screening queue must contain no scientific review")
    if len({r["text_hash"] for r in rows}) != 200 or any(
        hashlib.sha256(normalized(r["text"]).encode("utf-8")).hexdigest() != r["text_hash"] for r in rows
    ):
        raise ValueError("Frozen text hashes must be correct and unique")
    return contract, columns, rows, raw


def prepare(worksheet: Path, root: Path = ROOT) -> None:
    _, _, _, raw = load_frozen(root)
    worksheet.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never erases an existing reviewer file or frozen input.
    with worksheet.open("xb") as f:
        f.write(raw)


def assess(columns: list[str], frozen: list[dict], review_columns: list[str],
           reviews: list[dict], seed: int = 20260921) -> tuple[dict, list[dict]]:
    errors, differences, included = [], [], []
    report = {"status": "blocked_invalid_worksheet", "errors": errors,
              "category_differences": differences, "accepted_counts_by_reviewed_category": {c: 0 for c in CATEGORIES},
              "category_has_at_least_13": {c: False for c in CATEGORIES}, "selected_count": 0,
              "worksheet_row_count": len(reviews), "outside_queue_decisions_created": 0,
              "scientific_review_performed_by_script": False, "downstream_gates": dict(GATES), **PENDING}
    if set(review_columns) != set(columns):
        errors.append("Worksheet columns must exactly match the frozen queue; only the five review fields may change")
    ids = [r.get("candidate_id") for r in reviews]
    if len(ids) != len(frozen) or len(ids) != len(set(ids)) or set(ids) != {r["candidate_id"] for r in frozen}:
        errors.append("Worksheet must contain each frozen candidate ID exactly once and no other IDs")
    if errors:
        return report, []
    reference = {r["candidate_id"]: r for r in frozen}
    for row in sorted(reviews, key=lambda r: r["candidate_id"]):
        cid = row["candidate_id"]
        base = reference[cid]
        valid = True
        for field in columns:
            if field not in REVIEW_FIELDS and row[field] != base[field]:
                errors.append(f"{cid}: immutable field changed: {field}")
                valid = False
        category = row["reviewed_category"]
        if category and category != base["proposed_screen_category"]:
            differences.append({"candidate_id": cid, "proposed_screen_category": base["proposed_screen_category"],
                                "reviewed_category": category, "review_decision": row["review_decision"]})
        if row["review_decision"] not in {"include", "exclude"}:
            errors.append(f"{cid}: review_decision must be exactly include or exclude")
            continue
        if row["review_decision"] == "exclude":
            continue
        if category not in CATEGORIES:
            errors.append(f"{cid}: included row requires exactly one valid C1-C8 reviewed_category")
            valid = False
        for field in ("review_rationale", "reviewer", "independent_intent_group"):
            if not row[field].strip():
                errors.append(f"{cid}: included row requires {field}")
                valid = False
        if base["source"] == "beavertails_evaluation" and category != "C5":
            errors.append(f"{cid}: BeaverTails other-primary-category cases must be excluded under the frozen crosswalk")
            valid = False
        if valid:
            included.append(row)
    groups = defaultdict(list)
    for row in included:
        groups[normalized(row["independent_intent_group"])].append(row["candidate_id"])
    duplicates = {key: value for key, value in groups.items() if len(value) > 1}
    for key, value in sorted(duplicates.items()):
        errors.append(f"Duplicate independent_intent_group {key!r}: {', '.join(value)}")
    eligible = [r for r in included if normalized(r["independent_intent_group"]) not in duplicates]
    counts = Counter(r["reviewed_category"] for r in eligible)
    report["accepted_counts_by_reviewed_category"] = {c: counts[c] for c in CATEGORIES}
    report["category_has_at_least_13"] = {c: counts[c] >= 13 for c in CATEGORIES}
    report["shortfall_by_category"] = {c: max(0, 13 - counts[c]) for c in CATEGORIES}
    report["submitted_include_count"] = sum(r["review_decision"] == "include" for r in reviews)
    report["submitted_exclude_count"] = sum(r["review_decision"] == "exclude" for r in reviews)
    if errors:
        return report, []
    if not all(report["category_has_at_least_13"].values()):
        report["status"] = "blocked_insufficient_category_coverage"
        return report, []
    selected = []
    for category in CATEGORIES:
        pool = sorted((r for r in eligible if r["reviewed_category"] == category),
                      key=lambda r: (order_key(r["independent_intent_group"], seed), r["candidate_id"]))
        selected.extend(dict(r, selection_order_key=order_key(r["independent_intent_group"], seed),
                             selection_status="pending_downstream_reviews", **PENDING, **GATES) for r in pool[:13])
    report.update(status="selected_pending_downstream_reviews", selected_count=len(selected),
                  selected_candidate_ids=[r["candidate_id"] for r in selected])
    return report, selected


def write_report(out: Path, report: dict) -> None:
    payloads = {
        "validation.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
        "validation.txt": "\n".join([
            f"Status: {report['status']}", f"Selected: {report['selected_count']}",
            f"Validation errors: {len(report.get('errors', []))}",
            f"Accepted counts: {report.get('accepted_counts_by_reviewed_category', {})}",
            f"At least 13/category: {report.get('category_has_at_least_13', {})}",
            f"Category differences recorded: {len(report.get('category_differences', []))}",
            "No decisions are created for the 699 candidates outside this queue.",
            *[f"{k}: {v}" for k, v in PENDING.items()], *[f"{k}: {v}" for k, v in GATES.items()],
            "See validation.json for errors and exact category-difference records.",
        ]) + "\n",
    }
    for name, content in payloads.items():
        path = out / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)


def validate(worksheet: Path, out: Path, root: Path = ROOT) -> int:
    selection = out / "selected_base_intents.csv"
    temp_selection = selection.with_suffix(".csv.tmp")
    # Protect the worksheet and immutable inputs even if CLI paths are mistaken.
    protected = {worksheet.resolve(), (root / CONTRACT).resolve(), (root / ONTOLOGY).resolve(),
                 (root / CROSSWALK).resolve(),
                 (root / "results/stage_a_targeted_review_v1/review_queue.csv").resolve()}
    outputs = [selection, temp_selection] + [out / name for name in
               ["validation.json", "validation.json.tmp", "validation.txt", "validation.txt.tmp"]]
    if any(path.resolve() in protected for path in outputs):
        raise ValueError("Output paths must not overwrite the worksheet or frozen inputs")
    out.mkdir(parents=True, exist_ok=True)
    # A failed rerun must never leave a previous successful selection available.
    selection.unlink(missing_ok=True)
    temp_selection.unlink(missing_ok=True)
    report = {"status": "blocked_validation_in_progress", "selected_count": 0, "errors": [],
              "downstream_gates": dict(GATES), "outside_queue_decisions_created": 0,
              "scientific_review_performed_by_script": False,
              "accepted_counts_by_reviewed_category": {c: 0 for c in CATEGORIES},
              "category_has_at_least_13": {c: False for c in CATEGORIES}, **PENDING}
    write_report(out, report)
    try:
        contract, columns, frozen, _ = load_frozen(root)
        review_columns, reviews, raw = read_table(worksheet)
        report, selected = assess(columns, frozen, review_columns, reviews, contract["selection_seed"])
        report.update(handoff_id=contract["handoff_id"], queue_table_sha256=contract["queue_table_sha256"],
                      worksheet_sha256=hashlib.sha256(raw).hexdigest(),
                      selection_seed=contract["selection_seed"], selection_per_category=13,
                      outside_queue_unreviewed_candidates=contract["outside_queue_unreviewed_candidates"])
        if selected:
            with temp_selection.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(selected[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(selected)
            report["selected_file_sha256"] = hashlib.sha256(temp_selection.read_bytes()).hexdigest()
            temp_selection.replace(selection)
        write_report(out, report)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        selection.unlink(missing_ok=True)
        temp_selection.unlink(missing_ok=True)
        report.update(status="blocked_invalid_input", selected_count=0, errors=[str(exc)])
        report.pop("selected_candidate_ids", None)
        report.pop("selected_file_sha256", None)
        write_report(out, report)
    return 0 if report["selected_count"] == 104 and not report["errors"] else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate"])
    parser.add_argument("--worksheet", type=Path, default=HANDOFF / "reviewer_worksheet.csv")
    parser.add_argument("--out-dir", type=Path, default=HANDOFF / "validation")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.worksheet)
            print(f"Created blank reviewer worksheet: {args.worksheet}")
        else:
            code = validate(args.worksheet, args.out_dir)
            print((args.out_dir / "validation.txt").read_text(encoding="utf-8"), end="")
            raise SystemExit(code)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        parser.exit(2, f"Blocked: {exc}\n")


if __name__ == "__main__":
    main()
