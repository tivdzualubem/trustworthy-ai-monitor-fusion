"""C8 and multi-extension selection regressions; mutations stay in temporary fixtures."""
import copy
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('handoff_multi', ROOT / 'scripts/validate_stage_a_targeted_review.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
C8 = Path('configs/stage_a_c8_reserve_review_extension_v1.json')
C8_ROWS = h.EXTENSION_SPECS['stage_a_c8_reserve_review_extension_v1'][2]
WORKSHEET = Path('results/stage_a_targeted_review_handoff_v1/reviewer_worksheet.csv')
EXPECTED = dict(zip(h.CATEGORIES, [17, 17, 16, 15, 16, 19, 18, 14]))
IDS = ['wildguard:357', 'jbb:76', 'wildguard:1477', 'wildguard:906']


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


class C8MultiExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract, self.columns, self.frozen, _ = h.load_frozen(ROOT)
        self.files = [h.CONTRACT, h.CROSSWALK, h.ONTOLOGY, h.RESERVE_AUDIT,
                      Path(self.contract['queue_path']), WORKSHEET, h.C5_CONTRACT,
                      h.EXTENSION_WORKSHEET, C8, C8_ROWS, h.CANDIDATE_SNAPSHOT, h.SOURCE_AUDIT]
        for relative in self.files:
            dst = self.root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dst)
        self.out = self.root / 'validation'

    def validate(self, extensions=None):
        paths = [h.C5_CONTRACT, C8] if extensions is None else extensions
        code = h.validate(self.root / WORKSHEET, self.out, self.root,
                          [self.root / p for p in paths])
        report = json.loads((self.out / 'validation.json').read_text())
        self.assertEqual(report['downstream_gates'], h.GATES)
        for k, v in h.PENDING.items(): self.assertEqual(report[k], v)
        return code, report

    def assert_blocked(self):
        code, report = self.validate()
        self.assertEqual(code, 2)
        self.assertEqual(report['selected_count'], 0)
        self.assertFalse((self.out / 'selected_base_intents.csv').exists())
        self.assertNotIn('selected_candidate_ids', report)
        return report

    def test_preserved_queue_completed_worksheet_and_c3(self):
        ext = json.loads((self.root / C8).read_text())
        queue = self.root / self.contract['queue_path']
        worksheet = self.root / WORKSHEET
        before = (queue.read_bytes(), worksheet.read_bytes(), (self.root / h.EXTENSION_WORKSHEET).read_bytes())
        self.assertEqual(h.table_hash(self.columns, self.frozen),
                         'f1704482b35fbe6e5c79304733e948d2fbd2cefdf76a498a125d9a45c47b0937')
        self.assertEqual(hashlib.sha256(before[0]).hexdigest(), ext['queue_file_sha256'])
        self.assertEqual(hashlib.sha256(before[1]).hexdigest(), ext['preserved_worksheet_file_sha256'])
        rows = h.read_table(worksheet)[1]
        self.assertEqual(len(rows), 200)
        self.assertTrue(all(r['review_decision'] in {'include', 'exclude'} for r in rows))
        self.assertEqual(h.table_hash(self.columns, [r for r in rows if r['proposed_screen_category']=='C3']),
                         'f417e31d44a442f36c4c4f6605c06b56801d35e70e849a0d85423782ac7ad155')
        self.assertEqual(self.validate()[0], 0)
        self.assertEqual(before, (queue.read_bytes(), worksheet.read_bytes(), (self.root / h.EXTENSION_WORKSHEET).read_bytes()))

    def test_exact_four_reserve_ids_and_pinned_provenance(self):
        rows = h.read_table(self.root / C8_ROWS)[1]
        self.assertEqual([r['candidate_id'] for r in rows], IDS)
        audit = json.loads((self.root / h.RESERVE_AUDIT).read_text())
        self.assertTrue(set(IDS) <= set(audit['by_category']['C8']['reserve_candidate_ids']))
        self.assertFalse(set(IDS) & {r['candidate_id'] for r in self.frozen})
        snapshot = {r['candidate_id']: r for r in h.read_table(self.root / h.CANDIDATE_SNAPSHOT)[1]}
        for row in rows:
            self.assertEqual(row['text'], snapshot[row['candidate_id']]['text'])
            revision = '886acc352a31533ffbcf4ef22c744658688086fc' if row['source']=='jailbreakbench' else 'd29c47f41c8b51348b5c8e8c81c039b3132b66d1'
            self.assertEqual(row['revision'], revision)
            self.assertEqual(row['reviewer'], 'project_author')
            self.assertIn('human formulation provenance not established', row['candidate_role'])
        self.assertEqual(len({r['independent_intent_group'] for r in rows}), 4)

    def test_c5_only_preserves_legacy_counts_and_blocks(self):
        code, report = self.validate([h.C5_CONTRACT])
        self.assertEqual(code, 2)
        self.assertEqual(report['accepted_counts_by_reviewed_category'], dict(EXPECTED, C8=10))
        self.assertEqual(report['status'], 'blocked_insufficient_category_coverage')
        self.assertEqual(report['extension_reviewed_count'], 9)
        # Legacy single-Path Python API remains supported.
        self.assertEqual(h.validate(self.root / WORKSHEET, self.out, self.root, self.root / h.C5_CONTRACT), 2)

    def test_c8_only_has_seven_c5_and_blocks(self):
        code, report = self.validate([C8])
        self.assertEqual(code, 2)
        self.assertEqual(report['accepted_counts_by_reviewed_category'], dict(EXPECTED, C5=7))
        self.assertEqual(report['selected_count'], 0)

    def test_both_select_104_deterministically_with_all_gates_closed(self):
        code, report = self.validate()
        self.assertEqual((code, report['selected_count']), (0, 104))
        self.assertEqual(report['status'], 'selected_pending_downstream_reviews')
        self.assertEqual(report['accepted_counts_by_reviewed_category'], EXPECTED)
        self.assertEqual(report['selection_seed'], 20260921)
        self.assertEqual(report['extension_reviewed_count'], 13)
        self.assertEqual(len(report['extensions']), 2)
        self.assertEqual(report['combined_review_row_count'], 213)
        self.assertEqual(report['outside_queue_unreviewed_candidates'], 686)
        selected_path = self.out / 'selected_base_intents.csv'
        first = selected_path.read_bytes()
        selected = h.read_table(selected_path)[1]
        self.assertEqual(Counter(r['reviewed_category'] for r in selected), {c: 13 for c in h.CATEGORIES})
        self.assertEqual(len({h.normalized(r['independent_intent_group']) for r in selected}), 104)
        self.assertTrue(all(r['selection_status']=='selected_pending_downstream_reviews' for r in selected))
        self.assertTrue(all(r[k]=='False' for r in selected for k in h.GATES))
        self.assertTrue(all(r[k]=='pending' for r in selected for k in h.PENDING))
        for paths in [[C8, h.C5_CONTRACT], [h.C5_CONTRACT, C8]]:
            self.assertEqual(self.validate(paths)[0], 0)
            self.assertEqual(selected_path.read_bytes(), first)

    def test_cli_repeated_contract_arguments(self):
        # CLI uses real read-only inputs and a temporary output directory.
        result = subprocess.run([sys.executable, str(ROOT/'scripts/validate_stage_a_targeted_review.py'),
                                 'validate', '--worksheet', str(ROOT/WORKSHEET), '--out-dir', str(self.out),
                                 '--extension-contract', str(ROOT/h.C5_CONTRACT),
                                 '--extension-contract', str(ROOT/C8)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.out/'validation.json').read_text())['selected_count'], 104)

    def test_missing_extra_incomplete_or_altered_c8_clears_stale_selection(self):
        path = self.root / C8_ROWS
        original = path.read_bytes()
        for change in ['missing', 'incomplete', 'extra', 'text', 'revision', 'source_repo_id',
                       'reviewed_category', 'reviewer', 'review_decision', 'independent_intent_group', 'candidate_id']:
            with self.subTest(change=change):
                path.write_bytes(original)
                self.assertEqual(self.validate()[0], 0)
                rows = h.read_table(path)[1]
                if change=='missing': path.unlink()
                else:
                    if change=='incomplete': rows.pop()
                    elif change=='extra': rows.append(dict(rows[0],candidate_id='wildguard:865'))
                    else: rows[0][change]='altered'
                    write_csv(path,self.columns,rows)
                self.assert_blocked()

    def test_missing_or_invalid_c5_clears_multi_selection(self):
        for relative in [h.C5_CONTRACT, h.EXTENSION_WORKSHEET]:
            path = self.root / relative
            original = path.read_bytes()
            for change in ['missing', 'invalid']:
                self.assertEqual(self.validate()[0],0)
                if change=='missing': path.unlink()
                else: path.write_text('invalid')
                self.assert_blocked()
                path.write_bytes(original)

    def test_nonreserve_category_and_provenance_checked_beyond_table_hash(self):
        original = json.loads((self.root/C8).read_text())
        original_rows = h.read_table(self.root/C8_ROWS)[1]
        for field,value in [('candidate_id','wildguard:170'),('reviewed_category','C1'),
                            ('revision','unpinned'),('source_repo_id','invented'),('review_decision','exclude'),
                            ('reviewer','other'),('independent_intent_group','invented')]:
            with self.subTest(field=field):
                ext=copy.deepcopy(original);rows=copy.deepcopy(original_rows)
                rows[0][field]=value
                if field=='candidate_id': ext['approved_candidate_ids'][0]=value
                ext['approved_extension_table_sha256']=h.table_hash(self.columns,rows)
                (self.root/C8).write_text(json.dumps(ext))
                write_csv(self.root/C8_ROWS,self.columns,rows)
                self.assert_blocked()

    def test_frozen_queue_and_completed_reviews_cannot_change(self):
        for relative,field in [(Path(self.contract['queue_path']),'text'),(WORKSHEET,'review_rationale')]:
            path=self.root/relative;original=path.read_bytes()
            self.assertEqual(self.validate()[0],0)
            columns,rows,_=h.read_table(path);rows[0][field]+=' changed'
            write_csv(path,columns,rows)
            self.assert_blocked();path.write_bytes(original)
        # Even line ending changes violate the separately pinned byte preservation.
        path=self.root/WORKSHEET;path.write_bytes(path.read_bytes().replace(b'\n',b'\r\n'))
        self.assert_blocked()

    def test_duplicate_or_relabelled_intents_cannot_inflate_counts(self):
        rows=h.read_table(self.root/WORKSHEET)[1]
        duplicate=next(r for r in rows if r['candidate_id']=='wildguard:1546')
        duplicate.update(review_decision='include',independent_intent_group='fake-distinct-group')
        write_csv(self.root/WORKSHEET,self.columns,rows)
        self.assert_blocked()
        report,selected=h.assess(self.columns,self.frozen,self.columns,rows)
        self.assertTrue(any('duplicate C5 intents' in e for e in report['errors']))
        self.assertFalse(selected)
        shutil.copyfile(ROOT/WORKSHEET,self.root/WORKSHEET)
        self.assertEqual(self.validate()[0],0)
        extrows=h.read_table(self.root/C8_ROWS)[1]
        extrows[0]['independent_intent_group']=extrows[1]['independent_intent_group']
        write_csv(self.root/C8_ROWS,self.columns,extrows)
        self.assert_blocked()

    def test_duplicate_extension_contract_is_rejected(self):
        self.assertEqual(self.validate()[0],0)
        code,report=self.validate([h.C5_CONTRACT,C8,C8])
        self.assertEqual(code,2)
        self.assertIn('Duplicate extension',report['errors'][0])
        self.assertFalse((self.out/'selected_base_intents.csv').exists())

    def test_source_snapshot_and_reserve_audit_integrity(self):
        for relative in [h.CANDIDATE_SNAPSHOT,h.SOURCE_AUDIT,h.RESERVE_AUDIT]:
            path=self.root/relative;original=path.read_bytes()
            self.assertEqual(self.validate()[0],0)
            path.write_text('invalid')
            self.assert_blocked();path.write_bytes(original)

    def test_gate_seed_quota_and_pending_contracts_fail_closed(self):
        original=json.loads((self.root/C8).read_text())
        changes=[('selection_seed',1),('selection_per_category',12)]
        changes += [(k,'complete') for k in h.PENDING]
        changes += [('downstream_gates',dict(h.GATES,**{k:True})) for k in h.GATES]
        for key,value in changes:
            with self.subTest(key=key,value=value):
                ext=copy.deepcopy(original);ext[key]=value
                (self.root/C8).write_text(json.dumps(ext))
                self.assert_blocked()


if __name__=='__main__': unittest.main()
