"""Screening invariants using synthetic text; no scientific reviews are made."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("queue_builder", ROOT / "scripts/build_stage_a_review_queue.py")
queue_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_builder)
CFG = json.loads(queue_builder.CONFIG.read_text(encoding="utf-8"))
CROSSWALK = json.loads(queue_builder.CROSSWALK.read_text(encoding="utf-8"))
SOURCES = json.loads(queue_builder.SOURCE_AUDIT.read_text(encoding="utf-8"))
ONTOLOGY = json.loads(queue_builder.ONTOLOGY.read_text(encoding="utf-8"))


def candidate(identifier="test:1", text="sample", label="others", source="wildguardtest"):
    source_key = "wildguardmix" if source == "wildguardtest" else "jailbreakbench"
    row = {"candidate_id": identifier, "text": text, "text_hash": queue_builder.text_key(text),
           "source": source, "source_label": label,
           "revision": SOURCES["sources"][source_key]["revision"],
           "extra_provenance": "preserve this exact field"}
    row.update({field: "" for field in queue_builder.REVIEW_FIELDS})
    return row


class TargetedQueueTests(unittest.TestCase):
    def test_config_rejects_missing_category_and_invalid_quota(self):
        queue_builder.validate_config(CFG, ONTOLOGY)
        invalid = copy.deepcopy(CFG)
        del invalid["rules"]["C3"]
        with self.assertRaises(ValueError):
            queue_builder.validate_config(invalid, ONTOLOGY)
        invalid = copy.deepcopy(CFG)
        invalid["target_queue_per_category"] = 26
        with self.assertRaises(ValueError):
            queue_builder.validate_config(invalid, ONTOLOGY)

    def test_drug_weapon_overlap_is_withheld_without_primary_assignment(self):
        row = candidate(text="weapons drugs")
        queue, audit = queue_builder.build_queue([row], CFG)
        self.assertEqual(queue, [])
        self.assertEqual(set(audit["overlap_details"][0]["evidence"]), {"C1", "C3"})
        self.assertEqual(audit["overlap_candidates_withheld"], 1)
        self.assertTrue(all(row[field] == "" for field in queue_builder.REVIEW_FIELDS))

    def test_generic_chemical_word_and_broad_illegal_label_do_not_nominate(self):
        row = candidate(text="chemical", label="fraud_assisting_illegal_activities")
        self.assertEqual(queue_builder.screen(row, CFG), {})

    def test_weapons_alone_never_nominate_C3(self):
        self.assertEqual(set(queue_builder.screen(candidate(text="weapons"), CFG)), {"C1"})

    def test_beavertails_only_nominates_C5_and_never_fills_review(self):
        row = candidate(text="weapons drugs malware")
        row["source"] = "beavertails_evaluation"
        row["source_label"] = "self_harm"
        queue, _ = queue_builder.build_queue([row], CFG)
        self.assertEqual(queue[0]["proposed_screen_category"], "C5")
        self.assertTrue(all(queue[0][field] == "" for field in queue_builder.REVIEW_FIELDS))

    def test_text_hash_and_all_provenance_are_preserved(self):
        row = candidate(text="  sample\n", label="cyberattack")
        queue, _ = queue_builder.build_queue([row], CFG)
        self.assertEqual({key: queue[0][key] for key in row}, row)

    def test_duplicate_text_is_removed_with_provenance_record(self):
        a = candidate(identifier="a", text="Sample  text")
        b = candidate(identifier="b", text="sample\ntext")
        rows, removed = queue_builder.validate_candidates([b, a], CROSSWALK, SOURCES)
        self.assertEqual(rows, [a])
        self.assertEqual(removed[0]["removed_candidate"], b)
        self.assertEqual(removed[0]["kept_candidate_id"], "a")

    def test_tampering_or_existing_review_is_rejected(self):
        for field, value in [("text_hash", "wrong"), ("revision", "unpinned"),
                             ("source", "confirmation_domain"), ("review_decision", "include"),
                             ("reviewed_category", "C3")]:
            with self.subTest(field=field):
                row = candidate()
                row[field] = value
                with self.assertRaises(ValueError):
                    queue_builder.validate_candidates([row], CROSSWALK, SOURCES)

    def test_beavertails_bad_category_or_license_is_rejected(self):
        bt = CROSSWALK["public_screening_sources"]["beavertails_evaluation"]
        row = candidate()
        row.update(source="beavertails_evaluation", revision=bt["revision"], source_label="self_harm",
                   source_repo_id=bt["repo_id"], source_split=bt["split"], source_data_file=bt["data_file"],
                   source_license=bt["license"])
        queue_builder.validate_candidates([row], CROSSWALK, SOURCES)
        for field, value in [("source_label", "drug_abuse,weapons,banned_substance"),
                             ("source_license", "incorrect")]:
            bad = dict(row, **{field: value})
            with self.assertRaises(ValueError):
                queue_builder.validate_candidates([bad], CROSSWALK, SOURCES)

    def test_shuffled_input_gives_same_queue_and_audit(self):
        rows = [candidate(identifier=f"test:{i}", text=f"sample {i}", label="cyberattack") for i in range(35)]
        first = queue_builder.build_queue(rows, CFG)
        random.Random(91).shuffle(rows)
        second = queue_builder.build_queue(rows, CFG)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 25)
        self.assertEqual(len({r["candidate_id"] for r in first[0]}), 25)

    def test_sparse_category_is_reported_without_padding(self):
        queue, audit = queue_builder.build_queue([candidate(text="opioids")], CFG)
        self.assertEqual(len(queue), 1)
        self.assertEqual(audit["by_category"]["C3"]["shortfall_to_20"], 19)
        self.assertIn("C3", audit["categories_below_20"])


if __name__ == "__main__":
    unittest.main()
