"""Provisional asymmetric-loss decisions from frozen empirical demand forecasts.

No fitting, calibration selection or TEST-driven policy selection takes place.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .uncertainty_forecasting import BASE,OUT as UNCERTAINTY,conditions,daily,residual_table,quantiles

OUT=Path('outputs/procurement_support')
DEFAULT_SCENARIOS=[
    dict(Scenario='A',Description='Symmetric',C_under=1.,C_over=1.),
    dict(Scenario='B',Description='Shortage twice surplus',C_under=2.,C_over=1.),
    dict(Scenario='C',Description='Shortage four times surplus',C_under=4.,C_over=1.),
    dict(Scenario='D',Description='Surplus twice shortage',C_under=1.,C_over=2.),
]
PROBABILITIES=np.array([.1,1/3,.5,2/3,.8,.9])
Q_NAMES=['P10','P33','P50','P67','P80','P90']


def optimal_quantile(under,over):
    if not np.isfinite([under,over]).all() or under<0 or over<0 or under+over<=0:
        raise ValueError('Costs must be finite and nonnegative with a positive sum')
    return under/(under+over)


def costs(actual,quantity,under,over):
    optimal_quantile(under,over)
    actual=np.asarray(actual,dtype=float);quantity=np.asarray(quantity,dtype=float)
    if actual.shape!=quantity.shape:raise ValueError('Actual and quantity shapes differ')
    if (quantity[np.isfinite(quantity)]<0).any():raise ValueError('Negative procurement quantity')
    valid=np.isfinite(actual)&np.isfinite(quantity)
    shortage=np.maximum(actual-quantity,0);surplus=np.maximum(quantity-actual,0)
    return pd.DataFrame({'Actual_kWh':actual,'Procured_kWh':quantity,
        'Shortage_kWh':np.where(valid,shortage,np.nan),'Surplus_kWh':np.where(valid,surplus,np.nan),
        'NormalizedCost':np.where(valid,under*shortage+over*surplus,np.nan)})


def frozen_quantiles(pool,rows,method,conditional,probabilities,full=False):
    """Same Step 5 empirical distribution; only requested quantile levels differ."""
    probabilities=np.asarray(probabilities,dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any():
        raise ValueError('Quantile probabilities must be between zero and one')
    result=np.full((len(rows),len(probabilities)),np.nan)
    # Matched distribution is point*(1+R). Positive scaling preserves ranks.
    # Cache residual quantiles per frozen temperature group for efficiency.
    cache={}
    for i,row in enumerate(rows.itertuples(index=False)):
        point=getattr(row,method if full else method+'Matched')
        if not np.isfinite(point):continue
        z=quantiles(pool,row,conditional)
        if full:
            errors=z.R.to_numpy()*point+z.ColdError.fillna(0).to_numpy()*row.ColdCount
            result[i]=np.maximum(0,point+np.quantile(errors,probabilities))
        else:
            key=row.TemperatureBand if conditional else 'pooled'
            if key not in cache:cache[key]=np.quantile(z.R.to_numpy(),probabilities)
            result[i]=np.maximum(0,point+point*cache[key])
    return result


def frozen_files():
    paths=list((BASE/'models').glob('*.pkl'))
    paths+=list((BASE/'predictions').rglob('*'))
    paths+=list(UNCERTAINTY.rglob('*'))
    weather=Path('outputs/weather_forecasting_full_year')
    paths+=list((weather/'models').glob('*'))
    paths+=list(weather.glob('*lock*.json'))
    paths+=[BASE/'model_selection_lock.json',Path('src/config.py'),Path('src/uncertainty_forecasting.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}


def load_scenarios(path):
    scenarios=json.loads(Path(path).read_text()) if path else [dict(s) for s in DEFAULT_SCENARIOS]
    if not isinstance(scenarios,list) or not scenarios:raise ValueError('Scenario JSON must be a nonempty list')
    names=set()
    for s in scenarios:
        if not isinstance(s['Scenario'],str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',s['Scenario']):
            raise ValueError('Scenario identifier must start with a letter and contain only letters, digits or underscores')
        s.setdefault('Description','Supplied cost coefficients')
        if s['Scenario'] in names:raise ValueError('Duplicate scenario name')
        names.add(s['Scenario']);s['OptimalQuantile']=optimal_quantile(s['C_under'],s['C_over'])
    return scenarios


def metric(frame,**labels):
    valid=frame.NormalizedCost.notna();z=frame.loc[valid];n=len(z)
    return dict(**labels,N=n,TotalNormalizedCost=float(z.NormalizedCost.sum()),
        TotalShortage_kWh=float(z.Shortage_kWh.sum()),TotalSurplus_kWh=float(z.Surplus_kWh.sum()),
        ShortageObservations=int(z.Shortage_kWh.gt(0).sum()),ShortagePct=float(z.Shortage_kWh.gt(0).mean()*100),
        MeanAbsoluteProcurementDeviation_kWh=float((z.Shortage_kWh+z.Surplus_kWh).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios',help='JSON list with Scenario, Description, C_under, C_over')
    parser.add_argument('--cost-units',default='normalized hypothetical cost units per kWh')
    args=parser.parse_args();scenarios=load_scenarios(args.scenarios)
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    before=frozen_files()
    lock=json.loads((UNCERTAINTY/'calibration_lock.json').read_text())
    completion=json.loads((UNCERTAINTY/'test_evaluation_complete.json').read_text())
    assert hashlib.sha256((UNCERTAINTY/'calibration_lock.json').read_bytes()).hexdigest()==completion['calibration_lock_sha256']
    # Persist policies and provenance BEFORE reading any TEST costs/labels.
    decision_lock=dict(Scenarios=scenarios,CostUnits=args.cost_units,QuantileRule='C_under / (C_under + C_over)',
        TESTUsedForPolicies=False,FrozenCalibrationSHA256=completion['calibration_lock_sha256'],
        FrozenInputsSHA256=before,Status='PROVISIONAL PROCUREMENT DECISION-SUPPORT DEMONSTRATION')
    (OUT/'decision_policy_lock.json').write_text(json.dumps(decision_lock,indent=2))
    pd.DataFrame(scenarios).to_csv(OUT/'cost_scenarios.csv',index=False)
    probabilities=np.r_[PROBABILITIES,[s['OptimalQuantile'] for s in scenarios]]
    validation=pd.read_parquet(UNCERTAINTY/'validation_portfolio.parquet')
    pools={m:residual_table(validation,m) for m in ['ML','Blend']}
    daypools={m:residual_table(daily(validation,m,True),m) for m in pools}
    # Forecast-only February 28 schedules, with exact preserved P10/P50/P90.
    hh=pd.read_parquet(BASE/'predictions/2024-02-28_households.parquet',columns=['TargetTimestamp','PredictedTemperature','ForecastState'])
    context=hh.groupby('TargetTimestamp').agg(Temperature=('PredictedTemperature','mean'),ColdCount=('ForecastState',lambda x:x.eq('cold-start').sum())).reset_index()
    schedules=[];dailyrows=[];peakrows=[]
    for m in pools:
        old=pd.read_csv(UNCERTAINTY/f'2024-02-28_{m}_portfolio.csv');old.TargetTimestamp=pd.to_datetime(old.TargetTimestamp,utc=True)
        a=old.merge(context,on='TargetTimestamp',validate='one_to_one');a[m]=a.PointForecast_kWh;a=conditions(a)
        v=frozen_quantiles(pools[m],a,m,lock['selected_temperature_conditioning'][m],probabilities,full=True)
        np.testing.assert_allclose(v[:,[0,2,5]],old[['P10_kWh','P50_kWh','P90_kWh']],rtol=0,atol=1e-10)
        z=old.copy();z['LocalTimestamp']=z.TargetTimestamp.dt.tz_convert('Europe/Berlin')
        for j,k in enumerate(Q_NAMES):z[k+'_kWh']=v[:,j]
        for j,s in enumerate(scenarios):z['Q_'+s['Scenario']+'_kWh']=v[:,6+j]
        z['Method']=m;schedules.append(z)
        z.to_csv(OUT/f'2024-02-28_{m}_procurement_schedule.csv',index=False)
        dd=daily(a,m);dv=frozen_quantiles(daypools[m],dd,m,lock['selected_temperature_conditioning'][m],probabilities,full=True)[0]
        expected=completion['unseen'][m]
        np.testing.assert_allclose(dv[[0,2,5]]/1000,[expected['P10DailyEnergy_MWh'],expected['P50DailyEnergy_MWh'],expected['P90DailyEnergy_MWh']],rtol=0,atol=1e-10)
        for j,s in enumerate(scenarios):
            dailyrows.append(dict(Method=m,**s,PointDailyEnergy_MWh=float(a[m].sum()/1000),
                P10DailyEnergy_MWh=float(dv[0]/1000),P50DailyEnergy_MWh=float(dv[2]/1000),P90DailyEnergy_MWh=float(dv[5]/1000),
                SeparatelyCalibratedDailyProcurement_MWh=float(dv[6+j]/1000),
                SumIntervalProcurement_MWh=float(v[:,6+j].sum()/1000)))
    schedule=pd.concat(schedules,ignore_index=True)
    # All peak recommendations refer to the locked ML peak, including Blend.
    peak=completion['unseen']['ML']['PeakTimestamp']
    for r in schedule.loc[schedule.TargetTimestamp.eq(pd.Timestamp(peak))].itertuples(index=False):
        for s in scenarios:
            peakrows.append(dict(Method=r.Method,**s,TargetTimestamp=str(r.TargetTimestamp),LocalTimestamp=str(r.LocalTimestamp),
                PointForecast_kWh=r.PointForecast_kWh,P90_kWh=r.P90_kWh,Procurement_kWh=getattr(r,'Q_'+s['Scenario']+'_kWh')))
    pd.DataFrame(dailyrows).to_csv(OUT/'2024-02-28_daily_procurement.csv',index=False)
    pd.DataFrame(peakrows).to_csv(OUT/'2024-02-28_peak_procurement.csv',index=False)
    # TEST: frozen matched subset and existing point/quantile outputs. Full
    # portfolio truth is not invented. Simple persistence uses prior available
    # same-slot demand, with the frozen blend when the lag is missing.
    contextual=[];persistence=[]
    for i,path in enumerate(sorted((BASE/'predictions/test').glob('*.parquet'))):
        f=pd.read_parquet(path,columns=['Household_ID','TargetTimestamp','PredictedTemperature','ForecastState','BaselinePrediction'])
        contextual.append(f.groupby('TargetTimestamp').agg(Temperature=('PredictedTemperature','mean')).reset_index())
        lag=pd.read_parquet(BASE/'daily/test'/path.name,columns=['Household_ID','TargetTimestamp','PreviousAvailableSameInterval'])
        labels=pd.read_parquet(BASE/'evaluation/test_labels'/path.name,columns=['Household_ID','TargetTimestamp','Target'])
        f=f.merge(lag,on=['Household_ID','TargetTimestamp'],validate='one_to_one').merge(labels,on=['Household_ID','TargetTimestamp'],validate='one_to_one')
        f['Persistence']=f.PreviousAvailableSameInterval.fillna(f.BaselinePrediction)
        z=f.loc[f.Target.notna()].groupby('TargetTimestamp').agg(Persistence=('Persistence','sum'),ActualCheck=('Target','sum')).reset_index()
        persistence.append(z)
        if i%90==0:print('TEST policy evaluation inputs',i,flush=True)
    context=pd.concat(contextual,ignore_index=True);persist=pd.concat(persistence,ignore_index=True)
    saved=pd.read_parquet(UNCERTAINTY/'test_portfolio_quantiles.parquet')
    metrics=[];records=[];dailyrecords=[];value=[]
    for m in pools:
        a=saved.loc[saved.Method.eq(m)].copy().merge(context,on='TargetTimestamp',validate='one_to_one').merge(persist,on='TargetTimestamp',how='left',validate='one_to_one')
        np.testing.assert_allclose(a.Actual,a.ActualCheck,rtol=0,atol=1e-8,equal_nan=True)
        a=conditions(a);v=frozen_quantiles(pools[m],a,m,lock['selected_temperature_conditioning'][m],probabilities)
        np.testing.assert_allclose(v[:,[0,2,5]],a[['MatchedP10','MatchedP50','MatchedP90']],rtol=0,atol=1e-10,equal_nan=True)
        d=a.groupby('Day').agg(Actual=('Actual',lambda x:x.sum(min_count=96)),
            **{m+'Matched':(m+'Matched',lambda x:x.sum(min_count=96))},Temperature=('Temperature','mean'),
            TargetTimestamp=('TargetTimestamp','first'),Persistence=('Persistence',lambda x:x.sum(min_count=96))).reset_index()
        d=conditions(d);dv=frozen_quantiles(daypools[m],d,m,lock['selected_temperature_conditioning'][m],probabilities)
        stored_daily=pd.read_csv(UNCERTAINTY/'test_daily_quantiles.csv');stored_daily=stored_daily.loc[stored_daily.Method.eq(m)].set_index('Day').reindex(d.Day)
        np.testing.assert_allclose(dv[:,[0,2,5]],stored_daily[['MatchedP10','MatchedP50','MatchedP90']],rtol=0,atol=1e-8,equal_nan=True)
        for j,s in enumerate(scenarios):
            for level,rows,qu in [('Quarter-hour',a,v),('Daily',d,dv)]:
                policies={'PointForecast':rows[m+'Matched'].to_numpy(),'P50':qu[:,2],'CostOptimalQuantile':qu[:,6+j]}
                if m=='ML':policies['Persistence']=rows.Persistence.to_numpy()
                evaluated={}
                for policy,q in policies.items():
                    e=costs(rows.Actual.to_numpy(),q,s['C_under'],s['C_over']);e['Day']=rows.Day.to_numpy()
                    e['Method']='Persistence' if policy=='Persistence' else m;e['Policy']=policy;e['Scenario']=s['Scenario'];e['Level']=level
                    if level=='Quarter-hour':e['TargetTimestamp']=rows.TargetTimestamp.to_numpy();records.append(e)
                    else:dailyrecords.append(e)
                    met=metric(e,Method=e.Method.iloc[0],Policy=policy,Scenario=s['Scenario'],Level=level,OptimalQuantile=s['OptimalQuantile'])
                    metrics.append(met);evaluated[policy]=met['TotalNormalizedCost']
                value.append(dict(Method=m,Scenario=s['Scenario'],Level=level,PointCost=evaluated['PointForecast'],QuantileCost=evaluated['CostOptimalQuantile'],
                    RelativeCostReductionPct=100*(1-evaluated['CostOptimalQuantile']/evaluated['PointForecast'])))
    metrics=pd.DataFrame(metrics);value=pd.DataFrame(value)
    metrics.to_csv(OUT/'test_policy_metrics.csv',index=False);value.to_csv(OUT/'value_of_uncertainty.csv',index=False)
    pd.concat(records,ignore_index=True).to_parquet(OUT/'test_interval_policy_evaluation.parquet',index=False)
    pd.concat(dailyrecords,ignore_index=True).to_csv(OUT/'test_daily_policy_evaluation.csv',index=False)
    assert before==frozen_files(),'A frozen upstream artifact changed'
    checks=dict(FrozenInputsUnchanged=True,ExistingQuantilesReproduced=True,TestUsedForPolicySelection=False,
        ScoredTestIntervals=34944,ScoredTestDays=364,UnseenIntervalsPerMethod=96,Status='PROVISIONAL',
        MLDailyPointUnchanged_MWh=completion['unseen']['ML']['PointDailyEnergy_MWh'])
    (OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2))
    report(scenarios,pd.DataFrame(dailyrows),pd.DataFrame(peakrows),metrics,value,args.cost_units)
    print(pd.DataFrame(dailyrows)[['Method','Scenario','SeparatelyCalibratedDailyProcurement_MWh','SumIntervalProcurement_MWh']].to_string(index=False))
    print(value.to_string(index=False))


def report(scenarios,day,peak,metrics,value,units):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=pd.read_csv(OUT/'2024-02-28_ML_procurement_schedule.csv');x=pd.to_datetime(f.TargetTimestamp,utc=True).dt.tz_convert('Europe/Berlin')
    # Use the first predefined asymmetric shortage scenario; never select by TEST cost.
    scenario=next((s for s in scenarios if s['C_under']>s['C_over']),scenarios[0]);q='Q_'+scenario['Scenario']+'_kWh'
    fig,ax=plt.subplots(figsize=(10,4.5));ax.fill_between(x,f.P10_kWh,f.P90_kWh,alpha=.22,label='Frozen P10–P90')
    ax.plot(x,f.PointForecast_kWh,label='Locked ML point');ax.plot(x,f[q],label=f"Scenario {scenario['Scenario']}: Q at {scenario['OptimalQuantile']:.1%}")
    ax.set_ylabel('kWh / 15-minute interval');ax.set_xlabel('Europe/Berlin time; UTC delivery day');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'figures/2024-02-28_procurement.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,len(scenarios),figsize=(3.2*len(scenarios),6.3),squeeze=False)
    for row,m in enumerate(['ML','Blend']):
        for col,s in enumerate(scenarios):
            z=metrics.loc[(metrics.Method==m)&(metrics.Level=='Quarter-hour')&(metrics.Scenario==s['Scenario'])].set_index('Policy').reindex(['PointForecast','P50','CostOptimalQuantile'])
            ax=axes[row,col];ax.bar(['Point','P50','q*'],z.TotalNormalizedCost/1000,color=['#567899','#92b4c7','#e39246']);ax.set_title(f"{m} — {s['Scenario']} ({s['C_under']:g}:{s['C_over']:g})");ax.set_ylim(0,max(z.TotalNormalizedCost)/1000*1.18)
            if col==0:ax.set_ylabel('Normalized imbalance cost (thousands)')
    fig.suptitle('Frozen TEST policies — illustrative cost coefficients',y=.99);fig.tight_layout();fig.savefig(OUT/'figures/test_policy_costs.png',dpi=180);plt.close(fig)
    lines=['# Step 6 — provisional procurement decision support','',
        '**PROVISIONAL PROCUREMENT DECISION-SUPPORT DEMONSTRATION. No production procurement approval.**','',
        'For shortage demand D>Q, loss is C_under × (D−Q); for surplus Q>D, loss is C_over × (Q−D). Minimizing expected linear imbalance loss gives q*=C_under/(C_under+C_over), hence Q=F⁻¹(q*). Atoms can admit multiple minimizers; empirical sample quantiles use the same NumPy linear convention as Step 5. With discrete calibration samples, linear interpolation need not minimize raw finite-sample empirical loss exactly; it preserves the frozen quantile convention. This rule assumes demand uncertainty independent of the procurement action and constant marginal costs, without a separately modelled purchase-price term or constraints.','',
        f'Cost units: **{units}**. Defaults are illustrative normalized coefficients, not actual E.ON prices or monetary savings. A JSON scenario list and `--cost-units` allow later supplied cost coefficients to replace them. No tariffs were fetched or inferred.','',
        '| Scenario | Under | Over | q* |','|---|---:|---:|---:|']
    for s in scenarios:lines.append(f"| {s['Scenario']} — {s['Description']} | {s['C_under']:g} | {s['C_over']:g} | {s['OptimalQuantile']:.6f} |")
    lines+=['','The exact fractions 1/3 and 2/3 are used despite the P33/P67 column labels. Equal costs call for median demand; more expensive shortage calls for a higher quantile, and more expensive surplus for a lower quantile. P90 is not automatically the safe or optimal purchase: it corresponds to a 9:1 shortage/surplus cost ratio under this loss.','',
        'Requested quantiles come directly from Step 5 validation-locked empirical residual distributions. ML retains its locked temperature-band conditioning; Blend retains pooled calibration. Full forecasts keep the same aligned targetless-household proxy addition. Complete daily residual sums provide separate daily quantiles. Existing Point/P10/P50/P90 are reproduced numerically and preserved; no models, features, splits or uncertainty intervals are modified. The policy/provenance lock is written before TEST evaluation.','',
        '## Frozen TEST economic evaluation','',
        'Scoring covers 34,944 supported quarter-hours and 364 daily observations. Actual demand and each policy are restricted to the same retrospectively measured household subset. Full portfolio ground truth is incomplete, so these costs are not verified full-portfolio costs. The zero-support day is excluded, not treated as zero demand. One daily observation sums only matched measurements at each interval; the household subset can change within a day.','',
        'The persistence benchmark uses previous-available same-interval demand for each household, with its frozen Blend point as fallback where the lag is missing. This is an explicitly fallback-completed persistence benchmark, not an additional trained model.','',
        '| Scenario | Method | Policy | Total normalized cost | Shortage MWh | Surplus MWh | Shortage % | Mean deviation kWh |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in metrics.loc[metrics.Level=='Quarter-hour'].itertuples():lines.append(f'| {r.Scenario} | {r.Method} | {r.Policy} | {r.TotalNormalizedCost:,.1f} | {r.TotalShortage_kWh/1000:.2f} | {r.TotalSurplus_kWh/1000:.2f} | {r.ShortagePct:.1f} | {r.MeanAbsoluteProcurementDeviation_kWh:.2f} |')
    lines+=['','Daily metrics use separately calibrated daily procurement and **net daily** shortage/surplus. They are not the sum of interval imbalance costs: surplus at one time cannot offset shortage at another in interval settlement. Full observation counts, shortage counts, daily costs and deviations are in test_policy_metrics.csv; per-observation evaluations are also saved.','',
        '| Scenario | Method | Interval cost reduction vs point | Daily cost reduction vs point |','|---|---|---:|---:|']
    for (scenario,m),g in value.groupby(['Scenario','Method']):
        r=g.set_index('Level').RelativeCostReductionPct;lines.append(f"| {scenario} | {m} | {r['Quarter-hour']:.2f}% | {r['Daily']:.2f}% |")
    lines+=['','Negative reduction means quantile procurement cost more than point procurement on this TEST set. These are retrospective normalized results; no scenario, quantile or forecasting method was selected or changed from them. Expected-loss optimality under a forecast distribution does not guarantee lower realized TEST costs when calibration undercovers or demand shifts.','',
        '## February 28, 2024 — true unseen demonstration','',
        'Locked ML daily point: 13.014909 MWh; P10/P50/P90: 10.470 / 12.381 / 14.681 MWh. Locked Blend daily point: 12.714074 MWh; P10/P50/P90: 10.959 / 12.641 / 14.444 MWh. No actual demand is available for this day; realized costs cannot be claimed.','',
        '| Scenario | q* | ML daily Q MWh | Blend daily Q MWh | ML sum interval Q MWh | Blend sum interval Q MWh |','|---|---:|---:|---:|---:|---:|']
    for s in scenarios:
        z=day.loc[day.Scenario.eq(s['Scenario'])].set_index('Method');lines.append(f"| {s['Scenario']} | {s['OptimalQuantile']:.6f} | {z.loc['ML','SeparatelyCalibratedDailyProcurement_MWh']:.3f} | {z.loc['Blend','SeparatelyCalibratedDailyProcurement_MWh']:.3f} | {z.loc['ML','SumIntervalProcurement_MWh']:.3f} | {z.loc['Blend','SumIntervalProcurement_MWh']:.3f} |")
    lines+=['','The daily purchase and sum of interval quantities are distinct hypothetical decisions. Quantiles of sums generally differ from sums of marginal quantiles because errors correlate over time. The daily residual distribution is calibrated independently. They are not a jointly reconciled execution schedule: choose the appropriate settlement objective, and add reconciliation/constraints before attempting both simultaneously.','',
        'The locked ML peak is **2024-02-28 02:00 Europe/Berlin (01:00 UTC)**: point 172.94 kWh/15-min and P90 221.24 kWh/15-min. All following policies refer to that same timestamp, including Blend. No kWh value is presented as power; average kW would be four times interval kWh.','',
        '| Scenario | ML peak procurement kWh/15-min | Blend at ML peak kWh/15-min |','|---|---:|---:|']
    for s in scenarios:
        z=peak.loc[peak.Scenario.eq(s['Scenario'])].set_index('Method');lines.append(f"| {s['Scenario']} | {z.loc['ML','Procurement_kWh']:.2f} | {z.loc['Blend','Procurement_kWh']:.2f} |")
    lines+=['','![Unseen-day procurement](figures/2024-02-28_procurement.png)','![TEST policy cost comparison](figures/test_policy_costs.png)','',
        '## Outputs and limitations','',
        '96-row schedules: 2024-02-28_ML_procurement_schedule.csv and 2024-02-28_Blend_procurement_schedule.csv. Daily decisions: 2024-02-28_daily_procurement.csv. Peak decisions: 2024-02-28_peak_procurement.csv. Evaluation: test_policy_metrics.csv, value_of_uncertainty.csv, test_interval_policy_evaluation.parquet and test_daily_policy_evaluation.csv. Locks/checks: decision_policy_lock.json and validation_checks.json. Figures: figures/2024-02-28_procurement.png and figures/test_policy_costs.png. Source: src/procurement_support.py.','',
        'Quarter-hour P10–P90 TEST coverage is 79.0% for ML and 77.0% for Blend; daily coverage is 73.4% and 74.5%, below nominal 80%. Spring coverage is weaker. Intervals were not widened. Four targetless households have provisional uncertainty; their common-shock proxy averaging may understate idiosyncratic variation. Actual full-portfolio ground truth is incomplete. No market liquidity, bid blocks, gate closure, transaction fees, intraday correction or market-price forecasting is modelled. The 96 intervals represent a UTC delivery day, displayed in Europe/Berlin local time. No orders are transmitted.','',
        'Confirmed: frozen-artifact hashes are unchanged and Step 5 quantiles are reproduced. Exploratory evidence: the TEST cost comparisons describe this historical matched subset. Modelling hypothesis: the empirical distribution and illustrative cost asymmetry approximate tomorrow’s demand and loss sufficiently for a demonstration; they do not establish deployable procurement optimality.','',
        'The completed forecast → uncertainty → asymmetric-cost policy pipeline is ready to present at the hackathon with these limitations prominently shown. It is a procurement decision-support prototype, not a production trading system.','',
        'READY FOR HACKATHON PRESENTATION: YES']
    (OUT/'PROCUREMENT_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
