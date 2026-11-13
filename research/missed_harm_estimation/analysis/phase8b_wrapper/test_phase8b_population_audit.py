"""Pure aggregation tests; never open human-review SQLite or any source cases."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np

# Test the wrapper independent of actual phase8a; WSL invocation imports REAL phase8a.
stub=types.ModuleType('phase8_cost_matched')
stub.METHODS=('allowed_srs_single','allowed_proxy_stratified_equal','allowed_one_proxy_ppi_fixed_lambda1',
              'allowed_three_vote_majority','allowed_three_plus_r4_reserve_triage')
stub.SCENARIOS={'perfect_binary':{}}
sys.modules['phase8_cost_matched']=stub
spec=importlib.util.spec_from_file_location('phase8b',Path(__file__).parent/'phase8b_population_audit.py')
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class AuditTests(unittest.TestCase):
    def test_aggregate_rmse_and_budget(self):
        rows=[];diagnostics=[]
        for j in range(5):
            metrics={}
            for m,offset in zip(stub.METHODS,[0,.03,.02,.01,.04]):
                metrics[m]={
                    'rmse_midpoint_vs_latent_Y':.01+.001*j+offset,
                    'mean_ci_width':.05,
                    'ci_coverage_against_latent_Y_diagnostic_only':.9,
                    'ci_coverage_reviewer_completion_set':.99,
                    'mean_unique_responses':50.,'mean_reviewer_actions':50.,
                    'mean_spent_seconds':2200.,'max_spent_seconds':2399.,
                    'mean_reserved_seconds':2250.,'max_reserved_seconds':2399.,
                    'mean_point_identification_width':.001,
                }
            rows.append({'summary':{'perfect_binary|budget_2400s':metrics}})
            diagnostics.append({'proxy_positive_allowed':63+j,'proxy_negative_allowed':1000,
                                'harmful_allowed':34,'harmful_allowed_in_proxy_positive':34-j,
                                'harmful_allowed_in_proxy_negative':j})
        result=b.summarize_population_runs(rows,diagnostics,2400.,200,44)
        self.assertAlmostEqual(result['proxy_population_diagnostics']['fraction_of_populations_with_all_harm_in_proxy_positive'],.2)
        observed=result['summary']['perfect_binary|budget_2400s']['allowed_srs_single']['pooled_rmse_vs_latent_Y']
        wanted=np.sqrt(np.mean(np.square([.01+.001*j for j in range(5)])))
        self.assertAlmostEqual(observed,wanted)
        out={'geometries':{'phase6_5000':result}}
        self.assertTrue(b.validate_result(out,['phase6_5000'],2400.))
        bad=result['summary']['perfect_binary|budget_2400s']['allowed_srs_single']
        bad['max_spent_seconds']=2401.
        with self.assertRaises(ValueError):b.validate_result(out,['phase6_5000'],2400.)

if __name__=='__main__':unittest.main()
