import unittest
import numpy as np
import phase11_reference_unmodified as p
from phase14_certified_solver import repaired_optimize

class RestrictedOptimizerTests(unittest.TestCase):
    COST=np.array([47.16,47.2057,48.7576,48.8033,.0457,1.5976,1.6433])
    def test_synthetic_pilots(self):
        for seed in range(10):
            rng=np.random.default_rng(105+seed)
            X=rng.normal(size=(16,3));X[:,1]*=.15;X[:,2]*=.05
            S=p.covariance(X)
            r=repaired_optimize(S,self.COST)
            self.assertTrue(r['certified'],(seed,r['after_certificate']))
            self.assertLessEqual(r['after_certificate']['relative_gap_upper'],.005)
            self.assertAlmostEqual(sum(r['fractions']),1,places=7)
            self.assertGreaterEqual(sum(r['fractions'][:4]),.12-1e-8)
    def test_hard_proxies(self):
        rng=np.random.default_rng(87)
        for rho in [0.,.3,.9,.99]:
            n=16
            u=rng.standard_normal((n,3))
            u[:,2]=rho*u[:,1]+(1-rho*rho)**.5*u[:,2]
            r=repaired_optimize(p.covariance(u),self.COST)
            self.assertTrue(r['certified'],(rho,r['after_certificate']))
    def test_pilot_immutability(self):
        r=np.random.default_rng(23);S=p.covariance(r.normal(size=(25,3)));Q=S.copy()
        c=self.COST.copy();_ = repaired_optimize(S,c)
        np.testing.assert_array_equal(Q,S)
        np.testing.assert_array_equal(c,self.COST)
    def test_paired_benchmark_end_to_end(self):
        from phase14_targeted_paired import paired_run, summarize
        pop=p.make_population(42)
        old,new=paired_run(pop,'perfect',4800.,np.array([17.,29.,43.,61.,87.]),.04565729060156249,1.5975605696953123,12345,.005)
        self.assertTrue(old['valid'] and new['valid'])
        self.assertTrue(new['solver_audit']['success'])
        self.assertLessEqual(new['solver_audit']['certificate']['relative_gap_upper'],.005)
        self.assertLessEqual(new['ledger']['actual_total_equivalent_units'],4800+1e-7)
        out=summarize([{'scenario':'perfect','original':old,'repaired':new}])
        self.assertEqual(out['perfect']['paired_valid'],1)
    def test_input_validation(self):
        with self.assertRaises(ValueError):repaired_optimize(np.eye(3),[-1]*7)
        with self.assertRaises(ValueError):repaired_optimize(np.zeros((3,3)),self.COST)
        with self.assertRaises(ValueError):repaired_optimize(np.eye(3),self.COST,gold_min=1.5)
    def test_gradient_certificate_matches_numeric(self):
        import phase13_methods as m
        S=p.covariance(np.random.default_rng(5).normal(size=(24,3)))
        x=np.array([.18,.16,.15,.11,.15,.13,.12]);x/=x.sum()
        data=m.multi_optimality_certificate(S,self.COST,x)
        h=1e-6;d=np.zeros(7);d[0]=1;d[1]=-1
        f=m.multi_optimality_certificate(S,self.COST,x+h*d)['objective']
        self.assertAlmostEqual((f-data['objective'])/h,np.dot(data['gradient'],d),delta=.0003)

if __name__=='__main__':unittest.main()
