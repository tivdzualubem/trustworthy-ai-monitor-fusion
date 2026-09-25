"""Controlled extension regressions; completed other-category rows are test fixtures only."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('handoff_c5', ROOT / 'scripts/validate_stage_a_targeted_review.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
EXT = Path('configs/stage_a_c5_reserve_review_extension_v1.json')
WORKSHEET = Path('results/stage_a_targeted_review_handoff_v1/reviewer_worksheet.csv')


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


class C5ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract, self.columns, self.frozen, _ = h.load_frozen(ROOT)
        for relative in [h.CONTRACT, h.CROSSWALK, h.ONTOLOGY, h.RESERVE_AUDIT,
                         Path(self.contract['queue_path']), EXT, WORKSHEET, h.EXTENSION_WORKSHEET]:
            dst = self.root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dst)
        self.rows = h.read_table(self.root / WORKSHEET)[1]
        self.out = self.root / 'validation'

    def complete_other_categories(self):
        # Synthetic completion solely to exercise selection, never written to research artifacts.
        counts = {}
        for row in self.rows:
            category = row['proposed_screen_category']
            if category in {'C3', 'C5'}:
                continue
            n = counts.get(category, 0)
            counts[category] = n + 1
            row.update(review_decision='include' if n < 13 else 'exclude',
                       reviewed_category=category, reviewer='synthetic_test_only',
                       review_rationale='Synthetic completion for validator regression only.',
                       independent_intent_group='fixture-' + row['candidate_id'])
        self.save()

    def save(self):
        write_csv(self.root / WORKSHEET, self.columns, self.rows)

    def run_validation(self, extension=True):
        code = h.validate(self.root / WORKSHEET, self.out, self.root,
                          self.root / EXT if extension else None)
        return code, json.loads((self.out / 'validation.json').read_text())

    def test_original_fingerprint_and_approved_c3_preserved(self):
        self.assertEqual(self.contract['queue_table_sha256'],
                         'f1704482b35fbe6e5c79304733e948d2fbd2cefdf76a498a125d9a45c47b0937')
        self.assertEqual(h.table_hash(self.columns, self.frozen), self.contract['queue_table_sha256'])
        self.assertEqual(len(self.frozen), 200)
        self.assertEqual(h.table_hash(self.columns, [r for r in self.rows if r['proposed_screen_category']=='C3']),
                         'f417e31d44a442f36c4c4f6605c06b56801d35e70e849a0d85423782ac7ad155')
        extrows, _ = h.load_extension(self.root / EXT, self.root, self.contract, self.columns, self.frozen, self.rows)
        self.assertEqual(len(extrows), 9)
        self.assertTrue(all(r['reviewer']=='project_author' for r in extrows))

    def test_actual_handoff_has_16_c5_but_no_selection_or_downstream_gates(self):
        code, report = self.run_validation()
        self.assertEqual(code, 2)
        self.assertEqual(report['accepted_counts_by_reviewed_category']['C5'], 16)
        self.assertEqual(report['accepted_counts_by_reviewed_category']['C3'], 16)
        self.assertEqual(report['extension_reviewed_count'], 9)
        self.assertEqual(report['worksheet_row_count'], 200)
        self.assertEqual(report['combined_review_row_count'], 209)
        self.assertEqual(report['outside_queue_unreviewed_candidates'], 690)
        self.assertEqual(report['selected_count'], 0)
        self.assertFalse((self.out / 'selected_base_intents.csv').exists())
        self.assertEqual(report['downstream_gates'], h.GATES)
        for k, v in h.PENDING.items(): self.assertEqual(report[k], v)

    def test_extension_opt_in_and_independent_coverage_required(self):
        self.complete_other_categories()
        code, report = self.run_validation(False)
        self.assertEqual((code, report['accepted_counts_by_reviewed_category']['C5']), (2, 7))
        code, report = self.run_validation()
        self.assertEqual((code, report['selected_count']), (0, 104))
        selected = h.read_table(self.out / 'selected_base_intents.csv')[1]
        self.assertEqual(sum(r['reviewed_category']=='C5' for r in selected), 13)
        self.assertTrue(all(r[k]=='False' for r in selected for k in h.GATES))
        self.assertTrue(all(r[k]=='pending' for r in selected for k in h.PENDING))
        old = (self.out / 'selected_base_intents.csv').read_bytes()
        self.assertEqual(self.run_validation()[0], 0)
        self.assertEqual((self.out / 'selected_base_intents.csv').read_bytes(), old)
        included = [r for r in self.rows if r['proposed_screen_category']=='C5' and r['review_decision']=='include']
        for r in included[:4]: r['review_decision']='exclude'
        self.save()
        code, report = self.run_validation()
        self.assertEqual((code, report['accepted_counts_by_reviewed_category']['C5']), (2, 12))
        self.assertFalse((self.out / 'selected_base_intents.csv').exists())

    def test_known_duplicates_cannot_be_relabelled_as_independent(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['candidate_id'] in h.C5_EQUIVALENCES[0]:
                row.update(review_decision='include', independent_intent_group='fake-' + row['candidate_id'])
        report, selected = h.assess(self.columns, self.frozen, self.columns, rows)
        self.assertTrue(any('duplicate C5 intents' in e for e in report['errors']))
        self.assertLess(report['accepted_counts_by_reviewed_category']['C5'], 7)
        self.assertEqual(selected, [])

    def test_incomplete_missing_or_tampered_extension_clears_stale_selection(self):
        self.complete_other_categories()
        path = self.root / h.EXTENSION_WORKSHEET
        original = path.read_bytes()
        for change in ['missing', 'incomplete', 'text', 'decision', 'reviewer', 'group']:
            with self.subTest(change=change):
                path.write_bytes(original)
                self.assertEqual(self.run_validation()[0], 0)
                rows = h.read_table(path)[1]
                if change=='missing': path.unlink()
                else:
                    if change=='incomplete': rows.pop()
                    else:
                        field={'text':'text','decision':'review_decision','reviewer':'reviewer','group':'independent_intent_group'}[change]
                        rows[0][field]='altered'
                    write_csv(path,self.columns,rows)
                code, report = self.run_validation()
                self.assertEqual(code, 2)
                self.assertFalse((self.out / 'selected_base_intents.csv').exists())
                self.assertEqual(report['downstream_gates'], h.GATES)

    def test_reserve_membership_and_beavertails_restriction_beyond_table_hash(self):
        for change in ['nonreserve', 'other_category', 'revision']:
            with self.subTest(change=change):
                ext=json.loads((ROOT / EXT).read_text())
                rows=h.read_table(ROOT / h.EXTENSION_WORKSHEET)[1]
                if change=='nonreserve':
                    rows[0]['candidate_id']='beavertails_evaluation:test:0'
                    ext['approved_candidate_ids'][0]=rows[0]['candidate_id']
                elif change=='other_category': rows[0]['reviewed_category']='C1'
                else: rows[0]['revision']='unpinned'
                ext['approved_extension_table_sha256']=h.table_hash(self.columns,rows)
                (self.root / EXT).write_text(json.dumps(ext))
                write_csv(self.root / h.EXTENSION_WORKSHEET,self.columns,rows)
                with self.assertRaises(ValueError):
                    h.load_extension(self.root / EXT,self.root,self.contract,self.columns,self.frozen,self.rows)

    def test_changed_audit_or_c3_fails_closed(self):
        self.complete_other_categories()
        self.assertEqual(self.run_validation()[0],0)
        next(r for r in self.rows if r['proposed_screen_category']=='C3')['review_rationale']+=' changed'
        self.save()
        self.assertEqual(self.run_validation()[0],2)
        self.assertFalse((self.out/'selected_base_intents.csv').exists())
        audit=json.loads((self.root/h.RESERVE_AUDIT).read_text());audit['by_category']['C5']['reserve_candidate_ids'].pop()
        (self.root/h.RESERVE_AUDIT).write_text(json.dumps(audit))
        self.assertEqual(self.run_validation()[0],2)

    def test_contract_cannot_open_gates_or_change_seed(self):
        original=json.loads((self.root/EXT).read_text())
        for field,value in [('selection_seed',1),('reviewer','someone_else'),('severity_review_status','complete'),
                            ('downstream_gates',dict(h.GATES,w0_authorized=True))]:
            ext=copy.deepcopy(original);ext[field]=value
            (self.root/EXT).write_text(json.dumps(ext))
            self.assertEqual(self.run_validation()[0],2)


if __name__=='__main__': unittest.main()
