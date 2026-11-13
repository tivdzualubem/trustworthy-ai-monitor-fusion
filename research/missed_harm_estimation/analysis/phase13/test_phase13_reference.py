import sqlite3, tempfile, unittest
from pathlib import Path
import json
import phase13_reference_validity_gate as r

class ReadOnlyGate(unittest.TestCase):
 def test_gate_never_emits_private_text_or_case_ids(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);db=root/'cases_private.sqlite'
   with sqlite3.connect(db) as cx:
    cx.execute('CREATE TABLE cases(case_id TEXT, prompt TEXT, response TEXT, historical_y INTEGER)')
    cx.execute('CREATE TABLE ratings(case_id TEXT, rater TEXT, label TEXT, active_seconds REAL)')
    cx.executemany('INSERT INTO cases VALUES (?,?,?,?)',[(f'secret-{i}', 'SECRET_PROMPT','SECRET_RESPONSE',1) for i in range(2)])
    cx.executemany('INSERT INTO ratings VALUES(?,?,?,?)',[(f'secret-{i}','R1','harmful',35.) for i in range(2)])
   (root/'POLICY_FOR_REVIEWERS.json').write_text(json.dumps({'target_policy':{'system_policy':'secret policy text'},
       'ontology':{'categories':[{'id':f'C{i}'} for i in range(1,9)]}}))
   got=r.safe_gate(db)
   self.assertEqual(got['human_review_actions'],2)
   self.assertTrue(got['policy_snapshot']['ontology_has_C1_C8'])
   public=json.dumps(got)
   for x in ('SECRET_PROMPT','SECRET_RESPONSE','secret-0','secret policy text'):
    self.assertNotIn(x,public)
   self.assertEqual(got['verified_qualified_reference_truth'],'NOT_ESTABLISHED')

if __name__=='__main__':unittest.main()
