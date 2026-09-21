"""Synthetic fixtures only: these tests do not review any research candidate."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff", ROOT / "scripts/validate_stage_a_targeted_review.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.columns = list(h.REQUIRED) + ["source_license"]
        self.frozen = []
        for category in h.CATEGORIES:
            for i in range(25):
                text = f"synthetic fixture {category} {i}"
                row = dict.fromkeys(self.columns, "")
                row.update(candidate_id=f"fixture-{category}-{i:02d}", source="synthetic_test",
                           revision="fixture-revision", source_label="fixture-label", text=text,
                           text_hash=hashlib.sha256(h.normalized(text).encode()).hexdigest(),
                           proposed_screen_category=category, screening_reason="synthetic test only",
                           source_license="synthetic fixture")
                self.frozen.append(row)
        self.reviews = copy.deepcopy(self.frozen)
        for row in self.reviews:
            number = int(row["candidate_id"].rsplit("-", 1)[1])
            row["review_decision"] = "include" if number < 13 else "exclude"
            if number < 13:
                row.update(reviewed_category=row["proposed_screen_category"],
                           review_rationale="synthetic test rationale", reviewer="test fixture",
                           independent_intent_group="group-" + row["candidate_id"])

    def assess(self, rows=None, columns=None):
        return h.assess(self.columns, self.frozen, self.columns if columns is None else columns,
                        self.reviews if rows is None else rows)

    def fixture_root(self, directory):
        root = Path(directory)
        contract = json.loads((ROOT / h.CONTRACT).read_text())
        contract["queue_table_sha256"] = h.table_hash(self.columns, self.frozen)
        for relative in [h.CONTRACT, h.ONTOLOGY, h.CROSSWALK]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = contract if relative == h.CONTRACT else json.loads((ROOT / relative).read_text())
            path.write_text(json.dumps(data), encoding="utf-8")
        write_csv(root / contract["queue_path"], self.columns, self.frozen)
        return root, contract

    def test_altered_id_hash_text_and_provenance_rejected(self):
        for field in ["candidate_id", "text_hash", "text", "revision", "source_license",
                      "proposed_screen_category", "screening_reason"]:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.reviews)
                rows[0][field] += "changed"
                report, selected = self.assess(rows)
                self.assertTrue(report["errors"])
                self.assertEqual(selected, [])

    def test_missing_columns_and_included_review_fields_rejected(self):
        for field in h.REVIEW_FIELDS:
            report, selected = self.assess(columns=[c for c in self.columns if c != field])
            self.assertTrue(report["errors"])
            self.assertEqual(selected, [])
            rows = copy.deepcopy(self.reviews)
            rows[0][field] = ""
            report, selected = self.assess(rows)
            self.assertTrue(report["errors"])
            self.assertEqual(selected, [])

    def test_decision_and_category_values_are_strict(self):
        for field, value in [("review_decision", "maybe"), ("review_decision", " include "),
                             ("reviewed_category", "C9"), ("reviewed_category", "C1,C2")]:
            rows = copy.deepcopy(self.reviews)
            rows[0][field] = value
            report, selected = self.assess(rows)
            self.assertTrue(report["errors"])
            self.assertEqual(selected, [])

    def test_duplicate_normalized_group_rejected_globally(self):
        rows = copy.deepcopy(self.reviews)
        rows[25]["independent_intent_group"] = "  " + rows[0]["independent_intent_group"].upper() + "  "
        report, selected = self.assess(rows)
        self.assertTrue(any("Duplicate independent_intent_group" in e for e in report["errors"]))
        self.assertEqual(selected, [])

    def test_twelve_in_one_category_blocks_all_selection(self):
        rows = copy.deepcopy(self.reviews)
        rows[0]["review_decision"] = "exclude"
        report, selected = self.assess(rows)
        self.assertEqual(report["status"], "blocked_insufficient_category_coverage")
        self.assertEqual(report["accepted_counts_by_reviewed_category"]["C1"], 12)
        self.assertFalse(report["category_has_at_least_13"]["C1"])
        self.assertEqual(selected, [])

    def test_exactly_thirteen_each_produces_104_with_gates_closed(self):
        report, selected = self.assess()
        self.assertEqual(len(selected), 104)
        self.assertEqual(report["selected_count"], 104)
        self.assertEqual(report["accepted_counts_by_reviewed_category"], {c: 13 for c in h.CATEGORIES})
        self.assertEqual(len({r["independent_intent_group"] for r in selected}), 104)
        self.assertTrue(all(v is False for v in report["downstream_gates"].values()))
        for row in selected:
            self.assertTrue(all(row[key] is False for key in h.GATES))
            self.assertTrue(all(row[key] == "pending" for key in h.PENDING))

    def test_oversubscribed_selection_uses_group_seed_and_is_order_independent(self):
        rows = copy.deepcopy(self.reviews)
        for row in rows:
            row.update(review_decision="include", reviewed_category=row["proposed_screen_category"],
                       reviewer="fixture", review_rationale="fixture", independent_intent_group=row["candidate_id"])
        report, selected = self.assess(rows)
        self.assertEqual(len(selected), 104)
        for category in h.CATEGORIES:
            pool = [r for r in rows if r["reviewed_category"] == category]
            expected = sorted(pool, key=lambda r: h.order_key(r["independent_intent_group"], 20260921))[:13]
            self.assertEqual([r["candidate_id"] for r in selected if r["reviewed_category"] == category],
                             [r["candidate_id"] for r in expected])
        random.Random(71).shuffle(rows)
        self.assertEqual((report, selected), self.assess(rows))

    def test_category_differences_recorded_without_relabeling(self):
        rows = copy.deepcopy(self.reviews)
        rows[0]["reviewed_category"], rows[25]["reviewed_category"] = "C2", "C1"
        report, selected = self.assess(rows)
        self.assertEqual(len(report["category_differences"]), 2)
        first = next(r for r in selected if r["candidate_id"] == rows[0]["candidate_id"])
        self.assertEqual((first["proposed_screen_category"], first["reviewed_category"]), ("C1", "C2"))
        self.assertEqual(len(selected), 104)

    def test_beavertails_other_primary_category_cannot_be_included(self):
        self.frozen[100]["source"] = self.reviews[100]["source"] = "beavertails_evaluation"
        self.reviews[100]["reviewed_category"] = "C1"
        report, selected = self.assess()
        self.assertTrue(any("BeaverTails" in e for e in report["errors"]))
        self.assertEqual(len(report["category_differences"]), 1)
        self.assertEqual(selected, [])

    def test_extra_missing_or_duplicate_candidates_rejected(self):
        for rows in [self.reviews[:-1], self.reviews + [dict(self.reviews[0], candidate_id="outside-queue")],
                     self.reviews[:-1] + [self.reviews[0]]]:
            report, selected = self.assess(rows)
            self.assertTrue(report["errors"])
            self.assertEqual(report["outside_queue_decisions_created"], 0)
            self.assertEqual(selected, [])

    def test_real_frozen_queue_remains_blank_and_cannot_select(self):
        _, columns, rows, _ = h.load_frozen()
        report, selected = h.assess(columns, rows, columns, copy.deepcopy(rows))
        self.assertEqual(len(report["errors"]), 200)
        self.assertEqual(selected, [])
        self.assertTrue(all(r[field] == "" for r in rows for field in h.REVIEW_FIELDS))

    def test_prepare_is_exact_copy_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root, contract = self.fixture_root(directory)
            worksheet = root / "reviewer.csv"
            original = (root / contract["queue_path"]).read_bytes()
            h.prepare(worksheet, root)
            self.assertEqual(worksheet.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                h.prepare(worksheet, root)
            self.assertEqual((root / contract["queue_path"]).read_bytes(), original)

    def test_frozen_queue_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, contract = self.fixture_root(directory)
            changed = copy.deepcopy(self.frozen)
            changed[0]["text"] += " changed"
            write_csv(root / contract["queue_path"], self.columns, changed)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                h.load_frozen(root)

    def test_failed_rerun_removes_stale_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture_root(directory)
            worksheet, out = root / "reviewer.csv", root / "validation"
            write_csv(worksheet, self.columns, self.reviews)
            self.assertEqual(h.validate(worksheet, out, root), 0)
            selection = out / "selected_base_intents.csv"
            _, selected, _ = h.read_table(selection)
            self.assertEqual(len(selected), 104)
            self.assertTrue(all(r[k] == "False" for r in selected for k in h.GATES))
            self.reviews[0]["text_hash"] = "changed"
            write_csv(worksheet, self.columns, self.reviews)
            self.assertEqual(h.validate(worksheet, out, root), 2)
            self.assertFalse(selection.exists())
            self.assertEqual(json.loads((out / "validation.json").read_text())["selected_count"], 0)

    def test_missing_input_clears_stale_selection_and_reports_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture_root(directory)
            out = root / "validation"
            out.mkdir()
            (out / "selected_base_intents.csv").write_text("stale")
            self.assertEqual(h.validate(root / "missing.csv", out, root), 2)
            self.assertFalse((out / "selected_base_intents.csv").exists())
            report = json.loads((out / "validation.json").read_text())
            self.assertTrue(all(v is False for v in report["downstream_gates"].values()))

    def test_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture_root(directory)
            worksheet = root / "validation/selected_base_intents.csv"
            write_csv(worksheet, self.columns, self.reviews)
            before = worksheet.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                h.validate(worksheet, worksheet.parent, root)
            self.assertEqual(worksheet.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
