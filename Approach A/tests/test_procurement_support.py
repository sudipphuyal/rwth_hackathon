import unittest
import numpy as np
import pandas as pd
from src.procurement_support import optimal_quantile,costs,frozen_quantiles,DEFAULT_SCENARIOS
from src.uncertainty_forecasting import interval

class ProcurementTests(unittest.TestCase):
    def test_cost_ratio_and_invalid_coefficients(self):
        self.assertEqual([optimal_quantile(s['C_under'],s['C_over']) for s in DEFAULT_SCENARIOS],[.5,2/3,.8,1/3])
        for a,b in [(0,0),(-1,2),(1,np.inf),(np.nan,1)]:
            with self.assertRaises(ValueError):optimal_quantile(a,b)
    def test_costs_and_missing_truth(self):
        z=costs([12,8,np.nan],[10,10,10],4,1)
        np.testing.assert_allclose(z.Shortage_kWh[:2],[2,0]);np.testing.assert_allclose(z.Surplus_kWh[:2],[0,2])
        np.testing.assert_allclose(z.NormalizedCost[:2],[8,2]);self.assertTrue(np.isnan(z.NormalizedCost.iloc[2]))
        with self.assertRaises(ValueError):costs([1],[-1],1,1)
    def test_arbitrary_quantiles_preserve_frozen_distribution(self):
        p=pd.DataFrame({'R':[-.5,-.1,.2,.5], 'ColdError':[-.1,0,.1,.2], 'Day':['a','b','c','d'], 'TemperatureBand':['cold']*4})
        r=pd.DataFrame({'ML':[10.], 'MLMatched':[8.], 'ColdCount':[4], 'TemperatureBand':['cold']})
        for full in [False,True]:
            got=frozen_quantiles(p,r,'ML',True,[.1,.5,.9],full=full)
            np.testing.assert_allclose(got,interval(p,r,'ML',True,full=full),rtol=0,atol=1e-12)
        requested=frozen_quantiles(p,r,'ML',False,[1/3,2/3,.8])
        np.testing.assert_allclose(requested[0],8+np.quantile(p.R*8,[1/3,2/3,.8]))
    def test_quantile_rule_compares_nearby_empirical_loss(self):
        demand=np.arange(101,dtype=float)
        for s in DEFAULT_SCENARIOS:
            p=optimal_quantile(s['C_under'],s['C_over']);q=np.quantile(demand,p)
            loss=lambda x:costs(demand,np.full(demand.shape,x),s['C_under'],s['C_over']).NormalizedCost.sum()
            self.assertLessEqual(loss(q),min(loss(q-1),loss(q+1))+1e-8)
    def test_daily_cost_cannot_replace_interval_settlement(self):
        intervalcost=costs([0,20],[10,10],1,1).NormalizedCost.sum()
        dailycost=costs([20],[20],1,1).NormalizedCost.sum()
        self.assertEqual(intervalcost,20);self.assertEqual(dailycost,0)

if __name__=='__main__':unittest.main()
