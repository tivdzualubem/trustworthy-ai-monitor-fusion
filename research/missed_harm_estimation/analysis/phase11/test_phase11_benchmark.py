import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
import numpy as np
import phase11_benchmark as p

class Phase11Tests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(32)
        self.durations=np.array([18.,33.,41.,72.,115.])
        self.pop=p.make_population(20261113,1687,1557)

    def ledger(self,budget=4800,cheap=.08,exp=2.5):
        return p.Ledger(budget,self.durations,cheap,exp,np.random.default_rng(734))

    def test_population_truth_and_endpoints(self):
        self.assertEqual(self.pop.N,1687)
        self.assertEqual(int(self.pop.allowed.sum()),1557)
        self.assertEqual(int(self.pop.y[self.pop.allowed].sum()),34)
        self.assertTrue(np.array_equal(self.pop.z_expect('perfect'),self.pop.y))
        self.assertTrue(np.all((self.pop.z_expect('correlated_errors')>=0)&(self.pop.z_expect('correlated_errors')<=1)))

    def test_ledger_charges_all_sources_and_hard_cap(self):
        l=self.ledger(490)
        self.assertTrue(l.buy(p0=2,p1=3,human=True,calibration=True))
        self.assertEqual(l.cheap_calls,2)
        self.assertEqual(l.exp_calls,3)
        self.assertEqual(l.calibration_reviews,1)
        self.assertAlmostEqual(l.calibration_proxy_cost,7.66)
        self.assertAlmostEqual(l.proxy_cost,7.66)
        count=0
        while l.can(human=True):
            self.assertTrue(l.buy(human=True))
            count+=1
        self.assertGreater(count,0)
        self.assertLessEqual(l.spent,l.budget)
        self.assertFalse(l.buy(human=True))

    def test_no_zero_cost_gold_oracle(self):
        for name in p.METHODS:
            rng=np.random.default_rng(2001)
            l=self.ledger()
            res=p.METHOD_FUN[name](self.pop,'perfect',l,rng)
            self.assertTrue(res['valid'],name)
            self.assertGreater(l.review_sec,0,name)
            self.assertGreater(l.actions,0,name)
            self.assertLessEqual(l.spent,4800,name)
            if name in ('ppi_plus_tuned','stratppi_mean','multipppi_subset_gls','active_aipw_restricted'):
                self.assertGreater(l.calibration_reviews,0,name)
            if name not in ('whole_srs','allowed_srs'):
                self.assertGreater(l.cheap_calls,0,name)

    def test_pilot_lambda_no_tune_on_estimation_sample(self):
        l=self.ledger()
        r=p.ppi_plus(self.pop,'perfect',l,np.random.default_rng(99))
        self.assertEqual(r['ledger']['calibration_review_actions'],12)
        self.assertGreater(r['ledger']['estimation_review_actions'],10)
        self.assertTrue(-2<=r['lambda_tuned']<=3)

    def test_multi_optimizer_feasibility(self):
        rng=np.random.default_rng(4)
        S=p.covariance(rng.normal(size=(30,3)))
        u,flag=p.optimize_multi(S,[35.,36.,38.,40.,.08,2.5,2.58])
        self.assertTrue(np.all(u>=-1e-7));self.assertAlmostEqual(sum(u),1.,places=6)
        self.assertGreaterEqual(sum(u[:4]),.12-1e-6)
        self.assertTrue(math.isfinite(float(u[0])))

    def test_all_methods_valid_and_summary_paired(self):
        raw=[]
        for pp in range(2):
            pop=p.make_population(20261112+pp)
            for rr in range(2):
                for sc in p.SCENARIOS:
                    r=p.run_trial(pop,sc,4800,self.durations,.08,2.5,3982+100*pp+rr)
                    for x in r:x['population_index']=pp;x['rep_index']=rr
                    raw+=r
        self.assertEqual(len(raw),2*2*3*6)
        self.assertTrue(all(x['ledger']['hard_budget_ok'] for x in raw))
        self.assertTrue(all(x['valid'] for x in raw))
        summary,_=p.summarise(raw,2,2)
        for sc in p.SCENARIOS:
            x=summary[sc]['multipppi_subset_gls']
            self.assertEqual(x['paired_populations'],2)
            self.assertEqual(len(x['paired_population_bootstrap95_diff_Y']),2)

    def test_active_hansen_hurwitz_residual_identity(self):
        # E_p[(z_i - m_i)/(N p_i)] == population mean(z-m).
        z=np.array([.0,1.,.0,1.,1.]);m=np.array([.1,.7,.3,.8,.4])
        pr=np.array([.1,.2,.15,.25,.3]);N=len(z)
        exact=np.sum(pr*(z-m)/(N*pr))
        self.assertAlmostEqual(float(m.mean()+exact),z.mean(),places=12)

    def test_db_reads_only_durations(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'private.sqlite'
            with sqlite3.connect(f) as cx:
                cx.execute('CREATE TABLE ratings (rater TEXT, active_seconds REAL, prompt TEXT, response TEXT)')
                cx.executemany('INSERT INTO ratings VALUES (?, ?, ?, ?)',
                               [('R1',30.0,'secret prompt','secret response')]*336)
            times,source=p.read_review_durations(f)
            self.assertEqual(len(times),336)
            self.assertEqual(source,'OBSERVED_336_BROWSER_ACTIVE_SECONDS_RESAMPLED')
            self.assertTrue(np.all(times==30.))

    def test_expensive_routing_acquisition_is_charged(self):
        l=self.ledger()
        res=p.active_aipw(self.pop,'perfect',l,np.random.default_rng(123))
        self.assertTrue(res['valid'])
        self.assertEqual(res['expensive_routed_cases']+l.calibration_reviews,l.exp_calls)
        self.assertTrue(l.spent<=l.budget)

    def test_private_schema_not_emitted_in_aggregates(self):
        q=self.pop
        x=p.run_trial(q,'perfect',4800,self.durations,.08,2.5,43)
        for item in x:
            raw=json.dumps(item).lower()
            self.assertNotIn('case_id',raw)
            self.assertNotIn('prompt',raw)
            self.assertNotIn('response_text',raw)

if __name__=='__main__':unittest.main()
