"""Independent arithmetic and immutable-provenance checks, without selection."""
import json
import numpy as np
import pandas as pd
from .procurement_support import OUT,frozen_files,DEFAULT_SCENARIOS

def main():
    lock=json.loads((OUT/'decision_policy_lock.json').read_text())
    assert frozen_files()==lock['FrozenInputsSHA256']
    records=pd.read_parquet(OUT/'test_interval_policy_evaluation.parquet')
    daily=pd.read_csv(OUT/'test_daily_policy_evaluation.csv')
    metrics=pd.read_csv(OUT/'test_policy_metrics.csv')
    for level,frame,n in [('Quarter-hour',records,34944),('Daily',daily,364)]:
        for (scenario,method,policy),g in frame.groupby(['Scenario','Method','Policy']):
            s=next(s for s in lock['Scenarios'] if s['Scenario']==scenario)
            ok=g.Actual_kWh.notna();z=g.loc[ok];assert len(z)==n
            short=np.maximum(z.Actual_kWh-z.Procured_kWh,0);surplus=np.maximum(z.Procured_kWh-z.Actual_kWh,0)
            np.testing.assert_allclose(short,z.Shortage_kWh,atol=1e-9)
            np.testing.assert_allclose(surplus,z.Surplus_kWh,atol=1e-9)
            np.testing.assert_allclose(s['C_under']*short+s['C_over']*surplus,z.NormalizedCost,atol=1e-8)
            r=metrics.loc[(metrics.Scenario==scenario)&(metrics.Method==method)&(metrics.Policy==policy)&(metrics.Level==level)].iloc[0]
            np.testing.assert_allclose([z.NormalizedCost.sum(),short.sum(),surplus.sum()],[r.TotalNormalizedCost,r.TotalShortage_kWh,r.TotalSurplus_kWh],rtol=0,atol=1e-7)
    for method in ['ML','Blend']:
        f=pd.read_csv(OUT/f'2024-02-28_{method}_procurement_schedule.csv')
        assert len(f)==96 and f.TargetTimestamp.nunique()==96
        v=f[[name+'_kWh' for name in ['P10','P33','P50','P67','P80','P90']]].to_numpy()
        assert np.isfinite(v).all() and (v>=0).all() and (np.diff(v,axis=1)>=0).all()
        if lock['Scenarios']==[dict(s,OptimalQuantile=s['C_under']/(s['C_under']+s['C_over'])) for s in DEFAULT_SCENARIOS]:
            np.testing.assert_allclose(f.Q_A_kWh,f.P50_kWh);np.testing.assert_allclose(f.Q_B_kWh,f.P67_kWh)
            np.testing.assert_allclose(f.Q_C_kWh,f.P80_kWh);np.testing.assert_allclose(f.Q_D_kWh,f.P33_kWh)
    checks=json.loads((OUT/'validation_checks.json').read_text());checks['IndependentCostArithmeticPassed']=True;checks['QuantileOrderAndScenarioMappingPassed']=True
    (OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2));print(checks)

if __name__=='__main__':main()
