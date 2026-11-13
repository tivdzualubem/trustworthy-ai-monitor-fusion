import importlib.util,unittest,copy
from pathlib import Path
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('evidence',P/'phase10_evidence_gate.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
D=P/'inputs'
class GateTests(unittest.TestCase):
 def setUp(self):
  self.paths=[D/'phase8b_population_robustness_20261112.json',D/'phase9b_audit_20261112.json',D/'phase5b1_blocked_allowed_baselines_v1.zip',D/'phase5b2_stratppi_mean_development_v1.zip',D/'phase5b3_amppi_restricted_v1(1).zip']
  self.a=e.analyze(*self.paths)
 def test_reviewer_integrity(self):self.assertEqual(self.a['phase9b_real_review']['three_reviewers_primary_n'],288)
 def test_qwen_one_harmful_difference(self):self.assertEqual(self.a['phase5_old_labels']['qwen_effect_allowed_harm_reduction_cases'],1)
 def test_real_review_nonidentification(self):self.assertFalse(self.a['formal_status']['Y_identified_from_existing_reviewers'])
 def test_srs_stratppi_comparator(self):
  m=self.a['phase5_old_labels']['methods'];self.assertLess(m['allowed_only_srs']['A_RMSE'],m['stratppi_proportional_fixed1']['A_RMSE'])
 def test_synthetic_scenario_reversal(self):
  s=self.a['phase8b_synthetic']['development_footprint_1687'];self.assertLess(s['perfect_binary']['stratified_rmse'],s['perfect_binary']['majority_rmse']);self.assertLess(s['noisy_correlated']['majority_rmse'],s['noisy_correlated']['srs_rmse'])
 def test_amppi_budgets_checked(self):
  r={v['method']:v for v in self.a['phase5_old_labels']['amppi_restricted_substudy']};self.assertGreater(r['restricted_amppi_2predictor']['fraction_over_budget'],0.4)
 def test_human_timing(self):self.assertAlmostEqual(self.a['phase9b_real_review']['real_active_seconds'],15844.868,3)
 def test_no_guarantee(self):self.assertTrue(self.a['not_github_commit_ready'])
if __name__=='__main__':unittest.main()
