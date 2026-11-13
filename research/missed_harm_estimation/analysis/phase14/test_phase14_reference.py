import unittest
from scipy.stats import hypergeom
from phase14_finite_population_interval import exact_allowed_srswor_interval as ci
class ReferenceProtocolTests(unittest.TestCase):
    def test_full_census_exact(self):
        for x in range(8):
            z=ci(20,7,7,x)
            self.assertEqual((z['K_min'],z['K_max']),(x,x))
    def test_full_uncertain(self):
        z=ci(19,13,13,0,13)
        self.assertEqual((z['K_min'],z['K_max']),(0,13))
    def test_bounds(self):
        z=ci(1687,1557,96,10,7)
        self.assertLessEqual(0,z['A_lower'])
        self.assertLessEqual(z['A_lower'],z['A_upper'])
        self.assertLessEqual(z['A_upper'],1557/1687)
    def test_exact_coverage_small_frames(self):
        for M in range(2,10):
            for n in range(1,M+1):
                for K in range(M+1):
                    cov=0
                    for x in range(n+1):
                        prob=hypergeom.pmf(x,M,K,n)
                        z=ci(M,M,n,x)
                        if z['K_min']<=K<=z['K_max']:cov+=prob
                    self.assertGreaterEqual(cov,.95-1e-9,(M,n,K,cov))
    def test_uncertain_expands_not_contracts(self):
        z0=ci(19,13,6,2)
        z1=ci(19,13,6,2,3)
        self.assertEqual(z0['K_min'],z1['K_min'])
        self.assertGreaterEqual(z1['K_max'],z0['K_max'])
    def test_invalid(self):
        with self.assertRaises(ValueError):ci(10,11,2,1)
        with self.assertRaises(ValueError):ci(10,9,3,2,2)
        with self.assertRaises(ValueError):ci(10,8,0,0)
if __name__=='__main__':unittest.main()
