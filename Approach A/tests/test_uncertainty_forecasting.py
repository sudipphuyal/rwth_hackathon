import unittest
import numpy as np
import pandas as pd
from src.uncertainty_forecasting import interval,daily,conditions

class UncertaintyTests(unittest.TestCase):
    def test_nonnegative_and_cold_aligned_scenarios(self):
        p=pd.DataFrame({'R':[-2,0,1], 'ColdError':[-1,0,1], 'Day':['a','b','c'], 'TemperatureBand':['Cold']*3})
        r=pd.DataFrame({'ML':[10.], 'MLMatched':[8.], 'ColdCount':[4], 'TemperatureBand':['Cold']})
        v=interval(p,r,'ML',False,full=True)
        np.testing.assert_allclose(v,[[0,10,21.2]])
        self.assertTrue((np.diff(v,axis=1)>=0).all())
    def test_daily_preserves_error_correlation_and_rejects_missing_day(self):
        d=conditions(pd.DataFrame({'TargetTimestamp':pd.date_range('2022-01-01',periods=96,freq='15min',tz='UTC'), 'Temperature':0.,'ML':10.,'MLMatched':9.,'Actual':11.,'ColdCount':4.,'ObservedCount':100.,'ColdError':1.}))
        z=daily(d,'ML',True)
        self.assertEqual(z.Actual.iloc[0]-z.MLMatched.iloc[0],192)
        self.assertEqual(z.ColdError.iloc[0],96)
        d.loc[2,'Actual']=np.nan
        self.assertTrue(daily(d,'ML',True).empty)

if __name__=='__main__':unittest.main()
