"""Post-freeze diagnostics only: no interval adjustment or model selection."""
import json
import numpy as np
import pandas as pd
from .uncertainty_forecasting import BASE,OUT,conditions,daily

def main():
    a=pd.read_parquet(OUT/'validation_portfolio.parquet')
    diagnostics=[]
    for m in ['ML','Blend']:
        p=a.loc[a.Actual.notna()].copy();cuts=np.quantile(p[m+'Matched'],[1/3,2/3]);p['DemandBand']=pd.cut(p[m+'Matched'],[0,*cuts,np.inf],labels=['Low','Medium','High'])
        for field in ['TemperatureBand','Season','DemandBand','Hour','Weekend']:
            for name,g in p.groupby(field,observed=True):
                err=g.Actual-g[m+'Matched'];diagnostics.append(dict(Method=m,Condition=field,Group=str(name),Days=g.Day.nunique(),N=len(g),MeanAbsoluteError_kWh=float(err.abs().mean()),MeanAbsoluteRelativeError=float((err/g[m+'Matched']).abs().mean()),BiasActualMinusPoint_kWh=float(err.mean()),HeatingExposure=float(np.maximum(15-g.Temperature,0).mean())))
    pd.DataFrame(diagnostics).to_csv(OUT/'validation_residual_diagnostics.csv',index=False)
    # Correct daily demand diagnostics using VALIDATION daily demand thresholds.
    g=pd.read_csv(OUT/'coverage_by_condition.csv');g=g.loc[~((g.Level=='Daily')&(g.Condition=='DemandBand'))]
    td=pd.read_csv(OUT/'test_daily_quantiles.csv');extra=[]
    for m,z in td.groupby('Method'):
        vd=daily(a,m,True);cuts=np.quantile(vd[m+'Matched'],[1/3,2/3]);z=z.copy();z['Band']=pd.cut(z[m+'Matched'],[0,*cuts,np.inf],labels=['Low','Medium','High'])
        for name,r in z.groupby('Band',observed=True):
            ok=r.Actual.notna();r=r.loc[ok];w=r.MatchedP90-r.MatchedP10;y=r.Actual
            extra.append(dict(Period='Test',Method=m,Level='Daily',Condition='DemandBand',Group=str(name),N=len(r),Coverage=float(((y>=r.MatchedP10)&(y<=r.MatchedP90)).mean()),Width=float(w.mean()),IntervalScore=float((w+10*np.maximum(r.MatchedP10-y,0)+10*np.maximum(y-r.MatchedP90,0)).mean())))
    pd.concat([g,pd.DataFrame(extra)],ignore_index=True).to_csv(OUT/'coverage_by_condition.csv',index=False)
    # Station residual summaries use validation only; not used to change calibration.
    sums={}
    for path in sorted((BASE/'evaluation/validation_residuals').glob('*.parquet')):
        r=pd.read_parquet(path,columns=['Household_ID','TargetTimestamp','Actual','Prediction'])
        f=pd.read_parquet(BASE/'daily/validation'/path.name,columns=['Household_ID','TargetTimestamp','Weather_ID'])
        z=r.merge(f,on=['Household_ID','TargetTimestamp'],validate='one_to_one').dropna(subset=['Actual'])
        for station,x in z.groupby('Weather_ID'):
            e=x.Actual-x.Prediction;previous=sums.setdefault(station,[0,0.,0.,0.]);previous[0]+=len(e);previous[1]+=e.abs().sum();previous[2]+=e.sum();previous[3]+=x.Actual.sum()
    pd.DataFrame([dict(Weather_ID=k,N=v[0],HouseholdMAE_kWh=v[1]/v[0],BiasActualMinusPoint_kWh=v[2]/v[0],WAPE=v[1]/v[3]) for k,v in sums.items()]).to_csv(OUT/'validation_station_diagnostics.csv',index=False)
    report=OUT/'UNCERTAINTY_REPORT.md';text=report.read_text();idx=text.index('## February 28')
    stats=pd.DataFrame(diagnostics);rows=stats.loc[(stats.Method=='ML')&stats.Condition.isin(['TemperatureBand','DemandBand'])]
    addition='Validation evidence (ML):\n\n| Condition | Group | Mean absolute error kWh | Mean absolute relative error |\n|---|---|---:|---:|\n'
    for r in rows.itertuples():addition+=f'| {r.Condition} | {r.Group} | {r.MeanAbsoluteError_kWh:.2f} | {r.MeanAbsoluteRelativeError:.1%} |\n'
    addition+='\nStation diagnostics are available in validation_station_diagnostics.csv; these household errors combine weather, household composition and metadata effects and cannot establish a causal station effect. No station-specific calibration was selected.\n\n'
    report.write_text(text[:idx]+addition+text[idx:])
    print('Validation weather/demand/station diagnostics saved; calibration untouched.')

if __name__=='__main__':main()
