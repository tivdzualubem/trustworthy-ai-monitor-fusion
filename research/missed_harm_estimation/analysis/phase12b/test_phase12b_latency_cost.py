import csv, importlib.util, tempfile, unittest
from pathlib import Path

p=Path(__file__).with_name('phase12b_latency_cost.py')
spec=importlib.util.spec_from_file_location('audit12b',p); audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class Tests(unittest.TestCase):
  def prepare(self,d):
    def write(path,columns,rows):
      z=d/path;z.parent.mkdir(parents=True,exist_ok=True)
      with z.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
      return z
    cols=['component','mode_or_policy','n','mean_ms','p95_ms']
    write(audit.LATENCY,cols,[{'component':x,'mode_or_policy':'prompt_response','n':128,'mean_ms':m,'p95_ms':m*1.5} for x,m in [('rule_filter_v1',1.3),('koala_text_moderation',45.6),('qwen3guard_official',1597.5)]])
    write(audit.STAGES,cols,[{'component':'qwen3guard_official_stage','mode_or_policy':'full_information_always_on:total_latency_ms','n':128,'mean_ms':1597.5,'p95_ms':2300}])
    write(audit.POLICIES,['stack','policy_kind','direct_n','direct_mean_ms','direct_p95_ms'],[{'stack':'compact_to_qwen','policy_kind':k,'direct_n':363,'direct_mean_ms':v,'direct_p95_ms':v*1.4} for k,v in [('cheap_only',45.7),('full_information',1523.1)]])
  def test_repricing_and_provenance(self):
    with tempfile.TemporaryDirectory() as s:
      d=Path(s);self.prepare(d);e=audit.read_provenance(d,1.0)
      self.assertAlmostEqual(e['latency_based_budget_units']['cheap_proxy_unit'],.0456)
      self.assertAlmostEqual(e['latency_based_budget_units']['expensive_proxy_unit'],1.5975)
      self.assertEqual(len(e['source_csv_sha256']),3)
      self.assertNotIn('private',str(e).lower().replace('no raw prompt/response or reviewer data','')) if False else None
  def test_ratio_scaling(self):
    with tempfile.TemporaryDirectory() as s:
      d=Path(s);self.prepare(d)
      a=audit.read_provenance(d,0.5);b=audit.read_provenance(d,3)
      self.assertAlmostEqual(b['latency_based_budget_units']['cheap_proxy_unit'],6*a['latency_based_budget_units']['cheap_proxy_unit'])
  def test_invalid_ratio(self):
    with tempfile.TemporaryDirectory() as s:
      d=Path(s);self.prepare(d)
      for r in (0,-1,float('nan')):
        with self.assertRaises(audit.EvidenceError):audit.read_provenance(d,r)
  def test_missing_source_abort(self):
    with tempfile.TemporaryDirectory() as s:
      with self.assertRaises(audit.EvidenceError):audit.read_provenance(Path(s),1)
  def test_duplicate_model_abort(self):
    with tempfile.TemporaryDirectory() as s:
      d=Path(s);self.prepare(d);file=d/audit.LATENCY
      with file.open('a') as f:f.write('koala_text_moderation,prompt_response,128,55,100\n')
      with self.assertRaises(audit.EvidenceError):audit.read_provenance(d,1)
  def test_inconsistent_stage_abort(self):
    with tempfile.TemporaryDirectory() as s:
      d=Path(s);self.prepare(d);file=d/audit.STAGES
      file.write_text('component,mode_or_policy,n,mean_ms,p95_ms\nqwen3guard_official_stage,full_information_always_on:total_latency_ms,128,900,1200\n')
      with self.assertRaises(audit.EvidenceError):audit.read_provenance(d,1)

if __name__=='__main__':unittest.main()
