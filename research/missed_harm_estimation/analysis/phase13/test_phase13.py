import itertools, math, unittest
import numpy as np
import phase11_reference_unmodified as p
import phase13_methods as m
import phase13_targeted_interval_recheck as a

class Phase13Tests(unittest.TestCase):
 def test_feasible_clip(self):
  self.assertEqual(m.feasible_ci(-.5,.6,.30),(0.,.30))
  self.assertEqual(m.feasible_ci(.1,.4,.30),(.1,.30))
  self.assertIsNone(m.feasible_ci(.4,.9,.30))
  with self.assertRaises(ValueError):m.feasible_ci(.4,.1,.30)
 def test_clipping_preserves_pointwise_coverage_on_domain(self):
  rng=np.random.default_rng(3)
  for _ in range(500):
   q=rng.uniform(0,1);l=rng.uniform(-1,1);u=l+rng.uniform(0,2);truth=rng.uniform(0,q)
   cl=m.feasible_ci(l,u,q)
   self.assertEqual(l<=truth<=u,bool(cl and cl[0]-1e-12<=truth<=cl[1]+1e-12))
   if cl:self.assertLessEqual(cl[1]-cl[0],u-l+1e-10)
 def test_amppi_exact_unbiased_point_and_variance(self):
  z=np.array([1.,0.,1.]); pred=np.array([.6,.2,.4]);pi=np.array([.2,.8,.5]);q=.7
  es=0.;vs=0.;mom=0.;point_target=q*z.mean()
  for bits in itertools.product([0,1],repeat=3):
   b=np.array(bits,int);weight=float(np.prod(np.where(b,pi,1-pi)))
   z_part=np.where(b,z,np.nan)
   est,variance_est=m.amp_biased_predictor_unbiased_bernoulli(z_part,pred,pi,b,q)
   es+=weight*est;vs+=weight*variance_est;mom+=weight*(est-point_target)**2
  self.assertAlmostEqual(es,point_target,places=11)
  self.assertAlmostEqual(vs,m.expected_amp_variance(z,pred,pi,q),places=11)
  self.assertAlmostEqual(mom,vs,places=11)
 def test_amppi_invalid_probabilities(self):
  with self.assertRaises(ValueError):m.amp_biased_predictor_unbiased_bernoulli([1,0],[.2,.1],[0,1],[1,0])
 def test_multi_certificate_gradient(self):
  rng=np.random.default_rng(77); S=p.covariance(rng.normal(size=(50,3)))
  cost=np.array([45,46,47,48,.05,1.6,1.65]);u=np.array([.18,.16,.15,.11,.15,.13,.12]);u/=u.sum()
  c=m.multi_optimality_certificate(S,cost,u)
  eps=1e-6
  d=np.zeros(7);d[0]=1;d[1]=-1
  pval=m.multi_optimality_certificate(S,cost,u+eps*d)['objective']
  derivative=(pval-c['objective'])/eps
  self.assertAlmostEqual(derivative,c['gradient'][0]-c['gradient'][1],delta=.0001)
 def test_multi_certificate_upper_bound_optimizer(self):
  rng=np.random.default_rng(6);S=p.covariance(rng.normal(size=(50,3)))
  cost=np.array([45.,45.1,46.,47.,.05,1.6,1.65]);u,success=p.optimize_multi(S,cost)
  self.assertTrue(success)
  c=m.multi_optimality_certificate(S,cost,u)
  self.assertGreaterEqual(c['absolute_gap_upper'],-1e-9)
  self.assertLess(c['relative_gap_upper'],.02)
 def test_tuned_pilot_cost_invariant(self):
  pop=p.make_population(103);dur=np.array([18.,33.,41.,72.,115.]);seed=111
  original=p.run_trial(pop,'independent_errors',4800,dur,.04565729060156249,1.5975605696953123,seed)
  corrected=a.run_trial_with_interval_check(pop,'independent_errors',4800,dur,.04565729060156249,1.5975605696953123,seed)
  for x,y in zip(original,corrected):
   self.assertEqual(x['method'],y['method']);self.assertTrue(x['valid'] and y['valid'])
   self.assertAlmostEqual(x['estimate'],y['estimate'],places=12)
   self.assertEqual(x['ledger'],y['ledger'])
   self.assertEqual(x['ci_width'],y['ci_width'])
   self.assertLessEqual(y['feasible_interval_95']['width'],x['ci_width']+1e-10)
   self.assertEqual(x['covers_reviewer_Z'],y['feasible_interval_covers_Z'])
 def test_wald_is_not_finite_sample_certificate(self):
  x=m.asymptotic_wald(0,.02,.923)
  self.assertEqual(x[0],0.)
  self.assertGreater(x[1],0.)
 def test_false_proxies_are_not_real_cost_measurement(self):
  pop=p.make_population(44)
  self.assertTrue(np.all((pop.f0>=0)&(pop.f0<=1)))
  self.assertEqual(pop.N,1687)

if __name__=='__main__':unittest.main()
