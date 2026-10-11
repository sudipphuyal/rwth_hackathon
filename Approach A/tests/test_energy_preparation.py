import unittest
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.preprocessing import demand_features, issue_for_day
from src.preprocessing_batch import batch_household_features
from src.energy_features import add_physical_features,matrix,feature_groups,cold_features,deterministic_fit_sample
from src.energy_metrics import Scores,ValidationScores


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.config=PipelineConfig()
        idx=pd.date_range("2024-01-01","2024-04-03",freq="15min",tz="UTC")
        y=np.arange(len(idx))%97/100
        y[::117]=np.nan
        self.meter=pd.DataFrame({"Target":y,"TargetSource":np.where(np.isnan(y),"missing","measured")},index=idx)
        self.metadata=pd.Series({"Household_ID":"fixture","Weather_ID":"w"})

    def compare(self,meter,days):
        fast=batch_household_features(meter,self.metadata,days,self.config)
        for day in days:
            issue=issue_for_day(day,self.config)
            targets=pd.date_range(day,periods=96,freq="15min")
            slow=demand_features(meter.Target,targets,issue,self.config)
            current=fast.loc[fast.ForecastIssueTimestamp.eq(issue)].reset_index(drop=True)
            for c in slow:
                if c.endswith('Timestamp'):
                    pd.testing.assert_series_equal(pd.to_datetime(current[c],utc=True),pd.to_datetime(slow[c],utc=True),check_names=False)
                elif c=='ForecastRoute':self.assertEqual(current[c].tolist(),slow[c].tolist())
                else:np.testing.assert_allclose(current[c].to_numpy(dtype=float),slow[c].to_numpy(dtype=float),equal_nan=True,atol=1e-9,err_msg=c)

    def test_reference_equivalence_and_spring_dst(self):
        self.compare(self.meter,pd.date_range('2024-03-29','2024-04-01',freq='D',tz='UTC'))

    def test_gaps_staleness_and_cold_start(self):
        days=pd.date_range('2024-03-28','2024-03-30',freq='D',tz='UTC')
        missing=self.meter.drop(self.meter.loc['2024-02-10':'2024-03-29'].index)
        self.compare(missing,days)
        cold=self.meter.copy();cold.Target=np.nan;cold.TargetSource='missing'
        self.compare(cold,days)

    def test_future_poison_and_membership(self):
        day=pd.date_range('2024-03-31',periods=1,tz='UTC')
        a=batch_household_features(self.meter,self.metadata,day,self.config)
        cutoff=a.ForecastCutoffTimestamp.iloc[0]
        poison=self.meter.copy();poison.loc[poison.index>cutoff,'Target']=999999
        poison.loc[poison.index>cutoff,'TargetSource']='measured'
        b=batch_household_features(poison,self.metadata,day,self.config)
        exclude=['Target','TargetSource','TargetMissing','TargetExtremeReview','TargetNegative']
        pd.testing.assert_frame_equal(a.drop(columns=exclude),b.drop(columns=exclude))
        before=pd.date_range('2023-12-30',periods=1,tz='UTC')
        self.assertTrue(batch_household_features(self.meter,self.metadata,before,self.config).empty)

    def test_unknown_metadata_interactions_and_identity_exclusion(self):
        f=pd.DataFrame({'PredictedTemperature':[5.,5.,5.],'PredictedSunshine':[.5,.5,np.nan],
            'Survey_Building_LivingArea':[150.,200.,np.nan],'Survey_HeatPump_Installation_Type':['air-source','ground-source','Unknown'],
            'Installation_HasPVSystem':['True','False','Unknown'],'LocalHour':[12,20,12],'HourSin':[0.,1.,0.],
            'Survey_Building_Residents':[3.,2.,np.nan],'Survey_DHW_Production_ByHeatPump':['True','Unknown','Unknown'],
            'Survey_DHW_Production_ByElectricWaterHeater':['False','True','Unknown'],
            'Survey_Installation_HasElectricVehicle':['True','False','Unknown']})
        q=add_physical_features(f)
        self.assertEqual(q.SunshinePV.iloc[0],.5);self.assertEqual(q.SunshinePV.iloc[1],0)
        self.assertTrue(pd.isna(q.SunshinePV.iloc[2]));self.assertTrue(pd.isna(q.TemperatureAirSource.iloc[2]))
        self.assertTrue(pd.isna(q.ResidentsHeatPumpDHW.iloc[1]))
        self.assertEqual(matrix(q,['Installation_HasPVSystem'])[:,0].tolist(),[1.,0.,2.])
        self.assertFalse(any(c.startswith(('Recent','Previous','History','Gap','Historical')) for c in cold_features(q)))
        with self.assertRaises(ValueError):matrix(q,['Household_ID'])

    def test_sample_preserves_extreme_and_covers_every_slot(self):
        times=pd.date_range('2021-01-01',periods=8*96,freq='15min',tz='UTC')
        d=pd.DataFrame({'TargetTimestamp':times,'TargetExtremeReview':False})
        sample=deterministic_fit_sample(d)
        slots=d.loc[sample].TargetTimestamp.dt.hour*4+d.loc[sample].TargetTimestamp.dt.minute//15
        self.assertEqual(slots.nunique(),96)
        excluded=d.index[~sample][0];d.loc[excluded,'TargetExtremeReview']=True
        self.assertTrue(deterministic_fit_sample(d).iloc[excluded])

    def test_matched_support_not_full_ground_truth(self):
        d=pd.DataFrame({'TargetTimestamp':pd.to_datetime(['2024-01-01','2024-01-01','2024-01-01 00:15','2024-01-01 00:15'],format='mixed',utc=True),
                        'Target':[1.,np.nan,2.,3.]})
        p=np.array([2.,99.,3.,4.]);s=ValidationScores();s.update_day(d,p);r=s.result()
        self.assertEqual(r['Household']['n'],3)
        self.assertEqual(r['MatchedPortfolio']['MAE'],1.5)
        self.assertEqual(r['DailyEnergyMAE'],3)
        self.assertEqual(r['MatchedPortfolio']['ActualEnergy'],6)
        zero=Scores();zero.update([0.],[1.]);self.assertIsNone(zero.result()['WAPE'])
        empty=ValidationScores();empty.update_day(d.assign(Target=np.nan),p)
        self.assertIsNone(empty.result()['DailyEnergyMAE']);self.assertEqual(empty.result()['EvaluatedDays'],0)


if __name__=='__main__':unittest.main()
