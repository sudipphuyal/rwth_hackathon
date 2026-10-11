"""Validation-only empirical calibration around immutable point forecasts.

Portfolio residuals and cold proxy errors are aligned by timestamp. Daily
quantiles use sums of complete historical day blocks, never sums of quantiles.
"""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .energy_model import completed_baselines,apply_blend

BASE=Path('outputs/energy_forecasting')
OUT=Path('outputs/uncertainty_forecasting')
METHODS={'ML':'Prediction','Blend':'BaselinePrediction'}

def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'models').glob('locked*.pkl')}

def conditions(d):
    t=pd.to_datetime(d.TargetTimestamp,utc=True).dt.tz_convert('Europe/Berlin')
    d=d.copy();d['Hour']=t.dt.hour;d['Weekend']=t.dt.weekday.ge(5)
    d['Season']=t.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
    d['TemperatureBand']=pd.cut(d.Temperature,[-np.inf,5,15,np.inf],labels=['Cold (<5C)','Mild (5-15C)','Warm (>=15C)'],right=False).astype(str)
    d['Day']=pd.to_datetime(d.TargetTimestamp,utc=True).dt.strftime('%Y-%m-%d')
    return d

def aggregate(f,actual=None,cold_proxy=None):
    q=f.copy()
    if actual is not None:q['Actual']=actual
    if cold_proxy is not None:q['ProxyError']=q.Actual-cold_proxy;q['ProxyPoint']=cold_proxy
    q['Cold']=q.ForecastState.eq('cold-start')
    q['Temperature']=q.PredictedTemperature
    g=q.groupby('TargetTimestamp',sort=True)
    out=g.agg(Temperature=('Temperature','mean'),ColdCount=('Cold','sum'),ActiveCount=('Household_ID','size'))
    for m,p in METHODS.items():
        out[m]=g[p].sum()
        if actual is not None:
            valid=q.Actual.notna();z=q.loc[valid].groupby('TargetTimestamp')
            out[m+'Matched']=z[p].sum().reindex(out.index)
    if actual is not None:
        valid=q.Actual.notna();z=q.loc[valid].groupby('TargetTimestamp')
        out['Actual']=z.Actual.sum().reindex(out.index);out['ObservedCount']=z.size().reindex(out.index).fillna(0)
    if cold_proxy is not None:
        z=q.loc[q.Actual.notna() & q.ForecastState.eq('established')].groupby('TargetTimestamp')
        # Aligned proxy means retain shared weather/time shocks with portfolio errors.
        out['ColdError']=z.ProxyError.mean().reindex(out.index)
    out=out.reset_index();return conditions(out)

def residual_table(a,m):
    z=a.loc[a.Actual.notna() & a[m+'Matched'].gt(0)].copy()
    z['R']=(z.Actual-z[m+'Matched'])/z[m+'Matched']
    # Per hypothetical targetless household residual, aligned with the SAME block.
    return z

def quantiles(pool,row,conditional):
    z=pool
    if conditional:
        candidate=z.loc[z.TemperatureBand.eq(row.TemperatureBand)]
        if candidate.Day.nunique()>=30:z=candidate
    return z

def interval(pool,rows,m,conditional,full=False,daily=False):
    result=[]
    for row in rows.itertuples(index=False):
        z=quantiles(pool,row,conditional)
        point=getattr(row,m if full else m+'Matched')
        errors=z.R.to_numpy()*point
        if full:
            errors=errors+z.ColdError.fillna(0).to_numpy()*row.ColdCount
        v=np.maximum(0,point+np.quantile(errors,[.1,.5,.9]))
        result.append(v)
    return np.asarray(result)

def daily(a,m,training=False):
    rows=[]
    for day,g in a.groupby('Day',sort=True):
        # Whole days only; gaps are never converted to zero actuals.
        if len(g)!=96 or (training and (g.Actual.isna().any() or g.ObservedCount.le(0).any())):continue
        r=g.iloc[0].to_dict();r.update(Day=day,Temperature=g.Temperature.mean(),ColdCount=g.ColdCount.mean())
        r[m]=g[m].sum();r[m+'Matched']=g[m+'Matched'].sum(min_count=96) if m+'Matched' in g else np.nan
        if 'Actual' in g:r['Actual']=g.Actual.sum(min_count=96)
        if 'ColdError' in g:r['ColdError']=g.ColdError.sum(min_count=96)
        rows.append(r)
    if not rows:return pd.DataFrame()
    return conditions(pd.DataFrame(rows))

def scores(rows,v):
    actual=rows.Actual.to_numpy();ok=np.isfinite(actual)&np.isfinite(v).all(axis=1)
    if not ok.any():return dict(N=0,Coverage=float('nan'),Width=float('nan'),IntervalScore=float('nan'))
    y=actual[ok];w=v[ok,2]-v[ok,0]
    return dict(N=int(ok.sum()),Coverage=float(np.mean((y>=v[ok,0])&(y<=v[ok,2]))),Width=float(w.mean()),
        IntervalScore=float(np.mean(w+10*np.maximum(v[ok,0]-y,0)+10*np.maximum(y-v[ok,2],0))))

def grouped(rows,v,period,method,level):
    z=rows.copy();z['Low']=v[:,0];z['Median']=v[:,1];z['High']=v[:,2]
    z['DemandBand']=pd.cut(z[method+'Matched'],[0,*DEMAND_CUTS[method],np.inf],labels=['Low','Medium','High']).astype(str)
    result=[]
    for col in ['Season','TemperatureBand','DemandBand','Hour','Weekend']:
        for name,g in z.groupby(col,observed=True):
            s=scores(g,g[['Low','Median','High']].to_numpy());
            if s['N']:result.append(dict(Period=period,Method=method,Level=level,Condition=col,Group=str(name),**s))
    return result

DEMAND_CUTS={}

def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    if (OUT/'test_evaluation_complete.json').exists():raise RuntimeError('Frozen evaluation already exists; no adaptive rerun allowed')
    before=hashes()
    with open(BASE/'models/validation_only_cold.pkl','rb') as stream:cold=pickle.load(stream)
    tables=[];samples=[]
    cache=OUT/'validation_portfolio.parquet';samplecache=OUT/'validation_household_sample.parquet'
    if cache.exists():tables=[pd.read_parquet(cache)];samples=[pd.read_parquet(samplecache)]
    else:
        for i,p in enumerate(sorted((BASE/'evaluation/validation_residuals').glob('*.parquet'))):
            r=pd.read_parquet(p);f=pd.read_parquet(BASE/'daily/validation'/p.name)
            f=f.merge(r[['Household_ID','TargetTimestamp','Prediction','BaselinePrediction','Actual']],on=['Household_ID','TargetTimestamp'],how='inner',validate='one_to_one')
            with threadpool_limits(limits=4):proxy=cold.predict(f)
            tables.append(aggregate(f,f.Actual,proxy))
            s=f.iloc[::16][['TargetTimestamp','ForecastState','Prediction','BaselinePrediction','Actual','PredictedTemperature']].copy()
            s['ColdPrediction']=proxy[::16];s['Day']=s.TargetTimestamp.dt.strftime('%Y-%m-%d');samples.append(s)
            if i%40==0:print('Validation proxy residuals',i,flush=True)
        pd.concat(tables,ignore_index=True).to_parquet(cache,index=False);pd.concat(samples,ignore_index=True).to_parquet(samplecache,index=False)
    a=pd.concat(tables,ignore_index=True);sample=pd.concat(samples,ignore_index=True)
    # Each assessed day uses ONLY full residual days ending before its D-1 issue.
    # First 60 days warm up. Candidate choices use prequential validation scores.
    comparisons={};chosen={};val_outputs={};groups=[]
    for m in METHODS:
        pool=residual_table(a,m);DEMAND_CUTS[m]=np.quantile(pool[m+'Matched'],[1/3,2/3]).tolist()
        vals={False:[],True:[]};targets=[]
        for day,g in pool.groupby('Day',sort=True):
            past=pool.loc[pool.Day<str((pd.Timestamp(day)-pd.Timedelta(days=1)).date())]
            if past.Day.nunique()<60:continue
            targets.append(g)
            for cond in vals:vals[cond].append(interval(past,g,m,cond))
        target=pd.concat(targets,ignore_index=True)
        comparisons[m]={str(c):scores(target,np.concatenate(v)) for c,v in vals.items()}
        # Proper interval score includes both width and penalties for missed outcomes.
        cond=min(vals,key=lambda c:comparisons[m][str(c)]['IntervalScore']);chosen[m]=cond
        v=np.concatenate(vals[cond]);val_outputs[m]=(target,v);groups+=grouped(target,v,'Validation',m,'Quarter-hour')
    hq={};bins=[0,.1,.3,.6,1,np.inf]
    sample['Band']=pd.cut(sample.Prediction,bins,labels=False,include_lowest=True)
    for state in ['established','sparse_history','cold-start']:
        for band in range(5):
            if state=='cold-start':
                s=sample.loc[sample.ForecastState.eq('established')].copy();pred=s.ColdPrediction
                s=s.loc[pd.cut(pred,bins,labels=False,include_lowest=True).eq(band)];err=s.Actual-s.ColdPrediction
            else:
                s=sample.loc[sample.ForecastState.eq('sparse/new' if state=='sparse_history' else state)&sample.Band.eq(band)];err=s.Actual-s.Prediction
            if s.Day.nunique()<30 or err.notna().sum()<500:
                s=sample.loc[sample.ForecastState.eq('established')];err=s.Actual-(s.ColdPrediction if state=='cold-start' else s.Prediction)
            hq[state,str(band)]=np.nanquantile(err,[.1,.5,.9])
    lock=dict(method='Empirical relative portfolio residuals; optional temperature bands with >=30 distinct days; aligned cold proxy additions; whole-day residuals for daily totals',
              selected_temperature_conditioning=chosen,prequential_validation=comparisons,warmup_days=60,minimum_conditional_days=30,
              demand_cuts=DEMAND_CUTS,models_before=before,test_used_for_selection=False,
              residual_source='TRAIN-only models on 2022-02-28 to 2023-02-27 validation; last day embargoed at 09:30 UTC',
              last_full_day='2023-02-26',household_sample_stride=16)
    lock['household_residual_quantiles']={f'{s}:{b}':q.tolist() for (s,b),q in hq.items()}
    (OUT/'calibration_lock.json').write_text(json.dumps(lock,indent=2));lockhash=hashlib.sha256((OUT/'calibration_lock.json').read_bytes()).hexdigest()
    # Freeze historical blocks, then read TEST outcomes for the first time in this step.
    pools={m:residual_table(a,m) for m in METHODS}
    daypools={m:residual_table(daily(a,m,True),m) for m in METHODS}
    test=[]
    for p in sorted((BASE/'predictions/test').glob('*.parquet')):
        f=pd.read_parquet(p);labels=pd.read_parquet(BASE/'evaluation/test_labels'/p.name)
        f=f.merge(labels[['Household_ID','TargetTimestamp','Target']],on=['Household_ID','TargetTimestamp'],validate='one_to_one')
        test.append(aggregate(f,f.Target))
    t=pd.concat(test,ignore_index=True);metrics={};outputs=[];dailyoutputs=[]
    for m in METHODS:
        v=interval(pools[m],t,m,chosen[m]);full=interval(pools[m],t,m,chosen[m],full=True)
        metrics[m]={'Validation':scores(*val_outputs[m]),'Test':scores(t,v)}
        groups+=grouped(t,v,'Test',m,'Quarter-hour')
        z=t[['TargetTimestamp','Day','Actual','ObservedCount','ActiveCount',m,m+'Matched']].copy();z['Method']=m
        for j,k in enumerate(['P10','P50','P90']):z[k]=full[:,j];z['Matched'+k]=v[:,j]
        outputs.append(z)
        d=daily(t,m);dv=interval(daypools[m],d,m,chosen[m]);dfull=interval(daypools[m],d,m,chosen[m],full=True)
        metrics[m]['TestDaily']=scores(d,dv)
        # Independent prequential DAILY validation, matching selected interval method.
        dp=daypools[m];dt=[];vv=[]
        for day,g in dp.groupby('Day'):
            past=dp.loc[dp.Day<str((pd.Timestamp(day)-pd.Timedelta(days=1)).date())]
            if past.Day.nunique()<60:continue
            dt.append(g);vv.append(interval(past,g,m,chosen[m]))
        metrics[m]['ValidationDaily']=scores(pd.concat(dt),np.concatenate(vv))
        groups+=grouped(d,dv,'Test',m,'Daily')
        z=d[['Day','Actual',m,m+'Matched']].copy();z['Method']=m
        for j,k in enumerate(['P10','P50','P90']):z[k]=dfull[:,j];z['Matched'+k]=dv[:,j]
        dailyoutputs.append(z)
    pd.concat(outputs).to_parquet(OUT/'test_portfolio_quantiles.parquet',index=False)
    pd.concat(dailyoutputs).to_csv(OUT/'test_daily_quantiles.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'coverage_by_condition.csv',index=False)
    # Forecast-only unseen day. Reuse saved locked ML predictions verbatim.
    f=pd.read_parquet(BASE/'daily/unseen_day/2024-02-28.parquet')
    saved=pd.read_parquet(BASE/'predictions/2024-02-28_households.parquet')
    f=f.merge(saved[['Household_ID','TargetTimestamp','Prediction']],on=['Household_ID','TargetTimestamp'],validate='one_to_one')
    with open(BASE/'models/locked_cold.pkl','rb') as stream:lockedcold=pickle.load(stream)
    with threadpool_limits(limits=4):fallback=lockedcold.predict(f)
    f['BaselinePrediction']=apply_blend(completed_baselines(f,fallback),{'A':.75,'B':.25})
    tomorrow=aggregate(f);summaries={}
    for m in METHODS:
        v=interval(pools[m],tomorrow,m,chosen[m],full=True)
        z=tomorrow[['TargetTimestamp']].copy();z['PointForecast_kWh']=tomorrow[m]
        for j,k in enumerate(['P10','P50','P90']):z[k+'_kWh']=v[:,j]
        z.to_csv(OUT/f'2024-02-28_{m}_portfolio.csv',index=False)
        d=daily(tomorrow,m);dv=interval(daypools[m],d,m,chosen[m],full=True)[0]/1000
        peak=int(tomorrow[m].to_numpy().argmax())
        summaries[m]=dict(PointDailyEnergy_MWh=float(tomorrow[m].sum()/1000),P10DailyEnergy_MWh=float(dv[0]),P50DailyEnergy_MWh=float(dv[1]),P90DailyEnergy_MWh=float(dv[2]),
            PeakTimestamp=str(tomorrow.TargetTimestamp.iloc[peak]),PeakPoint_kWh=float(tomorrow[m].iloc[peak]),PeakP90_kWh=float(v[peak,2]),PeakWidth_kWh=float(v[peak,2]-v[peak,0]))
    assert abs(summaries['ML']['PointDailyEnergy_MWh']-13.014908665757588)<1e-10
    # Individual marginal intervals: empirical state/point bands. Cold proxy residuals
    # use established households with their personal history withheld by saved model.
    h=f[['Household_ID','TargetTimestamp','ForecastState','Prediction']].copy();v=[]
    for row in h.itertuples():
        band=int(pd.cut([row.Prediction],bins,labels=False,include_lowest=True)[0]);state=row.ForecastState
        if state not in ['established','cold-start']:state='sparse_history'
        q=hq[state,str(band)].copy()
        if state!='established':
            ref=hq['established',str(band)];q[0]=min(q[0],ref[0]);q[2]=max(q[2],ref[2])
        v.append(np.maximum(0,row.Prediction+q))
    v=np.asarray(v)
    for j,k in enumerate(['P10','P50','P90']):h[k]=v[:,j]
    h.rename(columns={'Prediction':'PointForecast_kWh'}).to_parquet(OUT/'2024-02-28_household_quantiles.parquet',index=False)
    (OUT/'household_calibration.json').write_text(json.dumps({f'{s}:{b}':q.tolist() for (s,b),q in hq.items()},indent=2))
    assert before==hashes();assert lockhash==hashlib.sha256((OUT/'calibration_lock.json').read_bytes()).hexdigest()
    result=dict(metrics=metrics,unseen=summaries,locked_model_hashes_unchanged=True,calibration_lock_sha256=lockhash,test_used_for_calibration=False)
    (OUT/'test_evaluation_complete.json').write_text(json.dumps(result,indent=2))
    plots_report(result,a)
    print(json.dumps(result,indent=2),flush=True)


def plots_report(result,a):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=pd.read_csv(OUT/'coverage_by_condition.csv')
    z=groups.loc[(groups.Period=='Test')&(groups.Level=='Quarter-hour')&(groups.Condition=='Season')]
    fig,ax=plt.subplots(figsize=(8,4))
    for m,g in z.groupby('Method'):ax.plot(g.Group,g.Coverage*100,'o-',label=m)
    ax.axhline(80,color='black',ls='--',label='80% target');ax.set_ylabel('Matched portfolio coverage (%)');ax.set_ylim(0,100);ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/test_coverage.png',dpi=170);plt.close(fig)
    z=pd.read_csv(OUT/'2024-02-28_ML_portfolio.csv');x=pd.to_datetime(z.TargetTimestamp,utc=True).dt.tz_convert('Europe/Berlin')
    fig,ax=plt.subplots(figsize=(10,4));ax.fill_between(x,z.P10_kWh,z.P90_kWh,alpha=.25,label='P10–P90');ax.plot(x,z.PointForecast_kWh,label='Locked point');ax.plot(x,z.P50_kWh,label='Calibrated P50');ax.set_ylabel('Portfolio kWh / 15 minutes');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'figures/2024-02-28_uncertainty.png',dpi=170);plt.close(fig)
    diagnostic=[]
    for col in ['Season','TemperatureBand','Hour','Weekend']:
        for name,g in residual_table(a,'ML').groupby(col,observed=True):diagnostic.append(dict(Condition=col,Group=str(name),Days=g.Day.nunique(),MeanAbsoluteError_kWh=float((g.Actual-g.MLMatched).abs().mean()),MeanAbsoluteRelativeError=float(g.R.abs().mean()),MeanTemperature=float(g.Temperature.mean()),MeanHeatingExposure=float(np.maximum(15-g.Temperature,0).mean())))
    pd.DataFrame(diagnostic).to_csv(OUT/'validation_residual_diagnostics.csv',index=False)
    lines=['# Step 5 — frozen empirical uncertainty','',
      'Both locked point forecasts are preserved. No electricity or weather model was trained, retuned or replaced. TEST outcomes were read only after the validation calibration lock was saved. PointForecast remains separate from the residual-corrected P50.','',
      'Calibration uses TRAIN-only model predictions on the annual validation period. Each prequential validation day uses residual days strictly before D−1 (conservative relative to the 11:00 issue). The first 60 days are warm-up. Temperature conditioning (<5°C, 5–15°C, ≥15°C) requires 30 distinct days and otherwise falls back to pooled residuals; the candidate with lower prequential interval score is selected separately for ML and blend. Selection targets interval quality, not a new point-model benchmark.','',
      'Portfolio residuals are normalized by matched forecast demand, so widths scale with forecast demand. This is an empirical scaling hypothesis rather than a formal conformal guarantee. Daily quantiles use complete 96-interval historical daily residual sums: they are not sums of interval or household quantiles. Shared weather errors are already present in these aggregate residuals.','',
      'Full-portfolio scenarios add four targetless-household proxy errors using the mean history-withheld established-household residual at the same historical timestamp/day as the portfolio residual. This preserves common time/weather shocks, but averaging understates idiosyncratic variation for only four homes; full-portfolio bounds are provisional. Direct ground truth for these four homes does not exist. TEST coverage therefore refers only to the retrospective matched measured subset, not the full portfolio.','',
      '| Method | Validation interval coverage | TEST interval coverage | TEST mean width kWh | Validation daily coverage | TEST daily coverage | TEST daily width MWh |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in result['metrics'].items():lines.append(f"| {m} | {r['Validation']['Coverage']:.1%} | {r['Test']['Coverage']:.1%} | {r['Test']['Width']:.2f} | {r['ValidationDaily']['Coverage']:.1%} | {r['TestDaily']['Coverage']:.1%} | {r['TestDaily']['Width']/1000:.3f} |")
    lines+=['','Validation assessment covers the post-warm-up period, rather than all 365 days; spring coverage is limited. Both final test and validation zero-support days are excluded from scoring, and the partial embargoed validation day is excluded from daily calibration. Missing measurements remain missing. Matched support changes over time, and applying relative residuals to a growing roster assumes aggregate error scaling remains transferable.','',
      'Condition-level results are in coverage_by_condition.csv (season, temperature, demand thirds learned from validation, hour and weekday/weekend). Heating exposure is deterministically tied to predicted temperature; it is not independent evidence. Validation residual diagnostics report absolute and relative error by conditions. Station mixtures change with roster, but there are too few independent annual/station exposure changes to justify an additional station-specific calibrator. Predicted temperature summarizes the mixture; it does not eliminate station assignment or weather forecast uncertainty.','',
      'Individual unseen-day marginal intervals use state and demand bands from a deterministic every-sixteenth-row validation sample. Targetless households use cold proxy residuals from established households predicted without personal consumption history. Sparse and cold residual limits are widened to at least the established limits in the same point band before nonnegative clipping. These individual intervals have not received a separate out-of-time coverage benchmark and should be treated as provisional diagnostics. They are never summed to produce portfolio quantiles.','',
      '## February 28, 2024','', '| Method | Point MWh | P10 MWh | P50 MWh | P90 MWh |','|---|---:|---:|---:|---:|']
    for m,s in result['unseen'].items():lines.append(f"| {m} | {s['PointDailyEnergy_MWh']:.6f} | {s['P10DailyEnergy_MWh']:.3f} | {s['P50DailyEnergy_MWh']:.3f} | {s['P90DailyEnergy_MWh']:.3f} |")
    s=result['unseen']['ML'];lines +=['',f"ML peak: {s['PeakTimestamp']} (02:00 Europe/Berlin), point {s['PeakPoint_kWh']:.2f} kWh, P90 {s['PeakP90_kWh']:.2f} kWh, P10–P90 width {s['PeakWidth_kWh']:.2f} kWh. These are interval energy values, not kW; multiply by four for average power.",'',
      'The intervals represent demand uncertainty, not market-price scenarios. A simple decision layer can use P50 as a central demand quantity and consider quantities toward P90 when shortage imbalance costs dominate surplus costs. P50 itself is not inherently a lower-risk purchase; the appropriate quantile depends on the cost asymmetry. No purchasing optimization was implemented.','',
      'Outputs: 2024-02-28_ML_portfolio.csv and 2024-02-28_Blend_portfolio.csv (96 rows each); test_daily_quantiles.csv; test_portfolio_quantiles.parquet; 2024-02-28_household_quantiles.parquet; calibration_lock.json; household_calibration.json; test_evaluation_complete.json; coverage_by_condition.csv and validation_residual_diagnostics.csv. The JSON contains daily quantiles and peak statistics.','',
      '![TEST seasonal coverage](figures/test_coverage.png)','![February 28 uncertainty](figures/2024-02-28_uncertainty.png)','',
      'Limitations: only one annual residual history, temporal dependence, roster/missing-support changes, unverified targetless demand, possible outdated metadata, no station-mixture-specific calibration, and no guarantee of simultaneous coverage across all 96 intervals. Daily coverage is separately measured. TEST shortfalls are reported without any recalibration.','',
      'READY FOR PROCUREMENT DECISION STEP: NO']
    (OUT/'UNCERTAINTY_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
