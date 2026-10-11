"""Train/validate/lock pooled demand models; then daily frozen-model test."""
import gc
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .config import PipelineConfig,ChronologicalSplit,WEATHER_OUTPUT_DIR
from .preprocessing import issue_for_day,cutoff_for_issue
from .energy_features import matrix,deterministic_fit_sample
from .energy_model import EnergyModel,PARAMETERS,completed_baselines,apply_blend
from .energy_metrics import ValidationScores,Scores,selection_key,subgroup_labels

OUT=Path("outputs/energy_forecasting")
KEYS=["Household_ID","TargetTimestamp","ForecastIssueTimestamp","ForecastCutoffTimestamp"]
REPORT_FIELDS=["ForecastState","Installation_HasPVSystem","Survey_HeatPump_Installation_Type",
    "Survey_Installation_HasElectricVehicle","Survey_DHW_Production_ByHeatPump",
    "Survey_DHW_Production_ByElectricWaterHeater","Survey_DHW_Production_BySolar",
    "Survey_Building_LivingArea","PredictedTemperature","PredictedSunshineMissing","Weather_ID",
    "FirstObservedTimestamp","HistoryAgeDaysAtForecast"]


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=str,allow_nan=False))


def load_fit(periods,cutoff,columns):
    xs,ys,normal,extreme=[],[],[],[];source_max=None;households=set()
    required=list(dict.fromkeys(columns+['Target','TargetSource','TargetTimestamp','TargetExtremeReview','ForecastState','Household_ID']))
    for period in periods:
        for path in sorted((OUT/'data'/period).glob('*.parquet')):
            d=pq.ParquetFile(path).read(columns=required).to_pandas()
            keep=d.Target.notna()&d.TargetSource.isin(['measured','reconstructed'])&d.TargetTimestamp.le(cutoff)&deterministic_fit_sample(d)
            q=d.loc[keep]
            if q.empty:continue
            xs.append(matrix(q,columns));ys.append(q.Target.to_numpy(dtype=float))
            normal.append(q.ForecastState.eq('established').to_numpy());extreme.append(q.TargetExtremeReview.to_numpy())
            source_max=max(source_max,q.TargetTimestamp.max()) if source_max is not None else q.TargetTimestamp.max()
            households.update(q.Household_ID.unique())
    x=np.concatenate(xs);y=np.concatenate(ys);n=np.concatenate(normal);e=np.concatenate(extreme)
    del xs,ys,normal,extreme;gc.collect()
    return x,y,n,e,source_max,households


def predict_matrix(model,x,all_columns):
    indices=[all_columns.index(c) for c in model.columns]
    return np.maximum(0.,model.model.predict(x[:,indices]))


def val_days(columns,cutoff):
    required=list(dict.fromkeys(columns+KEYS+REPORT_FIELDS+['Target','TargetSource','TargetExtremeReview']))
    for path in sorted((OUT/'daily'/'validation').glob('*.parquet')):
        d=pq.ParquetFile(path).read(columns=required).to_pandas()
        d=d.loc[d.TargetTimestamp.le(cutoff)]
        if not d.empty:yield d


def evaluate_cold(models,columns,cutoff):
    result={name:ValidationScores() for name in models}
    for d in val_days(columns,cutoff):
        x=matrix(d,columns)
        for name,m in models.items():result[name].update_day(d,predict_matrix(m,x,columns))
    return {name:s.result() for name,s in result.items()}


def evaluate_candidates(models,cold,columns,cutoff):
    scores={name:ValidationScores() for name in models}
    for name in ['A','B','C']:scores['baseline_'+name]=ValidationScores()
    blends={}
    for left,right in [('A','B'),('A','C'),('B','C')]:
        for weight in [.25,.5,.75]:
            name=f'blend_{left}_{right}_{weight}'
            blends[name]={left:weight,right:1-weight};scores[name]=ValidationScores()
    raw_avail={b:0 for b in ['A','B','C']};total=0
    for d in val_days(columns,cutoff):
        x=matrix(d,columns);fallback=predict_matrix(cold,x,columns)
        bs=completed_baselines(d,fallback)
        total+=len(d)
        for b,col in [('A','PreviousAvailableSameInterval'),('B','PreviousWeekSameLocalInterval'),('C','RecentSameSlot28dMedian')]:
            raw_avail[b]+=int(d[col].notna().sum());scores['baseline_'+b].update_day(d,bs[b])
        for name,weights in blends.items():scores[name].update_day(d,apply_blend(bs,weights))
        is_cold=d.ForecastState.eq('cold-start').to_numpy()
        for name,m in models.items():
            p=predict_matrix(m,x,columns);p[is_cold]=fallback[is_cold];scores[name].update_day(d,p)
    result={name:s.result() for name,s in scores.items()}
    return result,blends,{b:raw_avail[b]/total for b in raw_avail}


def flatten(results,stage):
    records=[]
    for name,r in results.items():
        records.append({'Stage':stage,'Candidate':name,'HouseholdMAE':r['Household']['MAE'],
          'HouseholdRMSE':r['Household']['RMSE'],'HouseholdWAPE':r['Household']['WAPE'],
          'HouseholdBias':r['Household']['Bias'],'PortfolioMAE':r['MatchedPortfolio']['MAE'],
          'PortfolioRMSE':r['MatchedPortfolio']['RMSE'],'PortfolioWAPE':r['MatchedPortfolio']['WAPE'],
          'PortfolioBias':r['MatchedPortfolio']['Bias'],'DailyEnergyMAE':r['DailyEnergyMAE']})
    return records


def test_walk_forward(primary,cold,baseline,columns,training_households):
    selected_scores=ValidationScores();baseline_scores=ValidationScores();households={};groups={};portfolio=[]
    state_counts=[];fallback_proxy=ValidationScores()
    required=list(dict.fromkeys(columns+KEYS+REPORT_FIELDS+['HistoryObservedIntervals','DemandHistorySourceMaxTimestamp',
        'LagAvailableSourceMaxTimestamp','LagWeekSourceMaxTimestamp','SameSlotSourceMaxTimestamp',
        'PredictedWeatherHistoryCutoffTimestamp','WeatherModelFitSourceMaxTimestamp']))
    weights_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'models').glob('locked_*.pkl')}
    prediction_dir=OUT/'predictions'/'test';prediction_dir.mkdir(parents=True,exist_ok=True)
    for i,path in enumerate(sorted((OUT/'daily'/'test').glob('*.parquet'))):
        # First read only model inputs. Predict BEFORE reading this day's labels.
        pf=pq.ParquetFile(path);d=pf.read(columns=required).to_pandas()
        x=matrix(d,columns);fallback=predict_matrix(cold,x,columns)
        prediction=predict_matrix(primary,x,columns)
        mask=d.ForecastState.eq('cold-start').to_numpy();prediction[mask]=fallback[mask]
        bs=completed_baselines(d,fallback);bp=apply_blend(bs,baseline)
        if not np.isfinite(prediction).all():raise AssertionError('Missing full-portfolio forecast')
        for c in ['DemandHistorySourceMaxTimestamp','LagAvailableSourceMaxTimestamp','LagWeekSourceMaxTimestamp','SameSlotSourceMaxTimestamp']:
            if (d[c].notna()&(d[c]>d.ForecastCutoffTimestamp)).any():raise AssertionError('Future demand input')
        if (d.WeatherModelFitSourceMaxTimestamp>d.PredictedWeatherHistoryCutoffTimestamp).any():raise AssertionError('Future weather fit')
        if (d.FirstObservedTimestamp>d.ForecastCutoffTimestamp).any():raise AssertionError('Household forecast before membership')
        # Only now reveal actual D consumption, solely for evaluation.
        labels=pf.read(columns=['Target','TargetSource','TargetExtremeReview']).to_pandas()
        evaluation=d.assign(Target=labels.Target.to_numpy())
        selected_scores.update_day(evaluation,prediction);baseline_scores.update_day(evaluation,bp)
        fallback_proxy.update_day(evaluation,fallback)
        good=labels.Target.notna().to_numpy();actual=labels.Target.to_numpy(dtype=float)
        q=pd.DataFrame({'TargetTimestamp':d.TargetTimestamp,'FullForecast':prediction,
            'MatchedPrediction':np.where(good,prediction,0),'MatchedActual':np.where(good,actual,0),
            'MatchedBaseline':np.where(good,bp,0),'ActiveCount':1,'ObservedCount':good.astype(int)})
        sums=q.groupby('TargetTimestamp').sum().reset_index()
        no_support=sums.ObservedCount.eq(0)
        sums.loc[no_support,['MatchedPrediction','MatchedActual','MatchedBaseline']]=np.nan
        sums['FullActualKnown']=sums.ObservedCount.eq(sums.ActiveCount)
        sums['FullActual']=sums.MatchedActual.where(sums.FullActualKnown)
        portfolio.append(sums)
        states=d.groupby('ForecastState').Household_ID.nunique().to_dict()
        state_counts.append({'DeliveryDate':str(d.TargetTimestamp.iloc[0].date()),**states,
                             'ActiveHouseholds':d.Household_ID.nunique(),
                             'UnseenTrainingHouseholds':d.loc[~d.Household_ID.isin(training_households),'Household_ID'].nunique()})
        for h,idx in d.groupby('Household_ID').groups.items():
            households.setdefault(h,Scores()).update(actual[idx],prediction[idx])
        for group,values in subgroup_labels(d).items():
            values=np.asarray(values)
            for label in pd.unique(values):
                idx=values==label;groups.setdefault((group,str(label)),Scores()).update(actual[idx],prediction[idx])
        out=d[KEYS+REPORT_FIELDS].copy();out['Prediction']=prediction;out['BaselinePrediction']=bp
        out['UsedColdFallback']=mask
        out.to_parquet(prediction_dir/path.name,index=False,compression='zstd')
        # Actuals stay in separate evaluation partitions.
        evdir=OUT/'evaluation'/'test_labels';evdir.mkdir(parents=True,exist_ok=True)
        labelled=d[KEYS].copy();labelled['Target']=actual;labelled['TargetSource']=labels.TargetSource
        labelled['TargetExtremeReview']=labels.TargetExtremeReview
        labelled.to_parquet(evdir/path.name,index=False,compression='zstd')
        if i%30==0:print(f'Frozen daily test: {i+1}/365 days',flush=True)
    for p in (OUT/'models').glob('locked_*.pkl'):
        if weights_hash[p.name]!=hashlib.sha256(p.read_bytes()).hexdigest():raise AssertionError('Model weights changed during test')
    ports=pd.concat(portfolio,ignore_index=True)
    ports.to_parquet(OUT/'predictions'/'test_portfolio.parquet',index=False)
    daily=ports.groupby(ports.TargetTimestamp.dt.floor('D')).agg(MatchedActual=('MatchedActual',lambda x:x.sum(min_count=1)),
        MatchedPrediction=('MatchedPrediction',lambda x:x.sum(min_count=1)),FullForecast=('FullForecast','sum'),MatchedBaseline=('MatchedBaseline',lambda x:x.sum(min_count=1)),
        ActiveHouseholds=('ActiveCount','max'),MinObservedHouseholds=('ObservedCount','min'))
    daily['AbsoluteEnergyError']=(daily.MatchedPrediction-daily.MatchedActual).abs()
    daily['BaselineAbsoluteEnergyError']=(daily.MatchedBaseline-daily.MatchedActual).abs()
    daily.to_csv(OUT/'evaluation'/'test_daily_energy.csv')
    pd.DataFrame(state_counts).to_csv(OUT/'evaluation'/'active_households_by_day.csv',index=False)
    pd.DataFrame([{'Household_ID':h,**s.result()} for h,s in households.items()]).to_csv(OUT/'evaluation'/'household_metrics.csv',index=False)
    pd.DataFrame([{'Group':g,'Value':v,**s.result()} for (g,v),s in groups.items()]).to_csv(OUT/'evaluation'/'subgroup_metrics.csv',index=False)
    return {'selected':selected_scores.result(),'baseline':baseline_scores.result(),'cold_proxy_all_history_withheld':fallback_proxy.result(),
            'frozen_weights_sha256':weights_hash,'days':len(portfolio),'matched_support_only':True,
            'full_actual_unknown_intervals':int((~ports.FullActualKnown).sum())}


def unseen_forecast(primary,cold,columns):
    path=OUT/'daily'/'unseen_day'/'2024-02-28.parquet'
    required=list(dict.fromkeys(columns+KEYS+REPORT_FIELDS+['DemandHistorySourceMaxTimestamp','WeatherModelFitSourceMaxTimestamp','PredictedWeatherHistoryCutoffTimestamp']))
    d=pq.ParquetFile(path).read(columns=required).to_pandas()
    x=matrix(d,columns);p=predict_matrix(primary,x,columns);fallback=predict_matrix(cold,x,columns)
    mask=d.ForecastState.eq('cold-start').to_numpy();p[mask]=fallback[mask]
    output=d[KEYS+REPORT_FIELDS].copy();output['Prediction']=p;output['UsedColdFallback']=mask
    output.to_parquet(OUT/'predictions'/'2024-02-28_households.parquet',index=False)
    aggregate=output.groupby('TargetTimestamp').Prediction.sum().rename('PredictedPortfolioKWh').reset_index()
    aggregate['AveragePowerKW']=4*aggregate.PredictedPortfolioKWh
    aggregate.to_csv(OUT/'predictions'/'2024-02-28_portfolio.csv',index=False)
    contributions=[]
    for group in ['Installation_HasPVSystem','Survey_HeatPump_Installation_Type','ForecastState']:
        for label,g in output.groupby(group):contributions.append({'Group':group,'Value':label,'Households':g.Household_ID.nunique(),'DailyEnergyKWh':g.Prediction.sum()})
    pd.DataFrame(contributions).to_csv(OUT/'predictions'/'2024-02-28_group_contributions.csv',index=False)
    peak=aggregate.loc[aggregate.PredictedPortfolioKWh.idxmax()]
    return {'day':'2024-02-28','active_households':d.Household_ID.nunique(),'rows':len(d),
            'total_daily_kwh':float(p.sum()),'peak_interval_kwh':float(peak.PredictedPortfolioKWh),
            'peak_average_kw':float(peak.AveragePowerKW),'peak_timestamp_utc':str(peak.TargetTimestamp),
            'peak_timestamp_local':str(peak.TargetTimestamp.tz_convert('Europe/Berlin')),
            'actual_demand_available':False,'energy_model_retrained_after_test':False,
            'forecast_states':d.groupby('ForecastState').Household_ID.nunique().to_dict()}


def main():
    if not (OUT/'materialization_complete.json').exists():raise RuntimeError('Run materialize_energy_data first')
    if (OUT/'test_evaluation_complete.json').exists():raise RuntimeError('Test already evaluated; refusing adaptive rerun')
    for name in ['models','evaluation','predictions']:(OUT/name).mkdir(exist_ok=True)
    config,split=PipelineConfig(),ChronologicalSplit()
    contracts=json.loads((OUT/'input_feature_groups.json').read_text());groups=contracts['groups'];all_columns=groups['G']
    cold_cols=contracts['cold_start']
    union=list(dict.fromkeys(all_columns+cold_cols))
    train_cap=cutoff_for_issue(issue_for_day(pd.Timestamp(split.validation_start),config),config)
    val_cap=cutoff_for_issue(issue_for_day(pd.Timestamp(split.test_start),config),config)
    x,y,normal,extreme,source,train_hh=load_fit(['train'],train_cap,union)
    save('fit_training_summary.json',{'rows':len(y),'normal_rows':int(normal.sum()),'extreme_rows_retained':int(extreme.sum()),
        'households':len(train_hh),'fit_max_label_timestamp':source,'sample':'deterministic rotating 1/8 plus every flagged extreme'})
    print(f'Training sample {len(y):,} rows; {len(train_hh)} households',flush=True)
    records=[]
    with threadpool_limits(limits=4):
        # Cold mapping learns from established households, with personal history absent.
        ci=[union.index(c) for c in cold_cols];cold_models={}
        for loss in ['squared_error','absolute_error']:
            cold_models[loss]=EnergyModel(cold_cols,loss).fit(x[normal][:,ci],y[normal],source)
        cold_results=evaluate_cold(cold_models,union,val_cap);records+=flatten(cold_results,'cold_proxy_validation')
        cold_loss=min(cold_results,key=lambda k:selection_key(cold_results[k]));cold=cold_models[cold_loss]
        save('cold_validation.json',{'results':cold_results,'chosen_loss':cold_loss,'history_features_withheld':True})
        print('Cold-start loss selected:',cold_loss,flush=True)
        # Full required-feature model compares losses; baseline evaluation is included.
        gindices=[union.index(c) for c in all_columns]
        full={loss:EnergyModel(all_columns,loss).fit(x[:,gindices],y,source) for loss in ['squared_error','absolute_error']}
        full_results,blends,coverage=evaluate_candidates(full,cold,union,val_cap)
        records+=flatten(full_results,'loss_and_baselines_validation')
        loss=min(full,key=lambda k:selection_key(full_results[k]));chosen_full=full[loss]
        baseline_names=[n for n in full_results if n.startswith(('baseline_','blend_'))]
        best_baseline=min(baseline_names,key=lambda k:selection_key(full_results[k]))
        baseline=blends[best_baseline] if best_baseline in blends else {best_baseline[-1]:1.}
        # Ordered ablations A..G use the selected loss, no test data.
        ablations={'G':chosen_full}
        for group in ['A','B','C','D','E','F']:
            indices=[union.index(c) for c in groups[group]]
            ablations[group]=EnergyModel(groups[group],loss).fit(x[:,indices],y,source)
            print('Fitted feature ablation',group,flush=True)
        # Optional-group decision candidates retain the required 7d behaviour core.
        behaviour=[c for c in groups['G'] if c not in groups['F']]
        optional={name:list(dict.fromkeys(groups[name]+behaviour)) for name in ['D','E']}
        for name,cols in optional.items():
            ablations[name+'_plus_recent']=EnergyModel(cols,loss).fit(x[:,[union.index(c) for c in cols]],y,source)
        sensitivity=EnergyModel(all_columns,loss).fit(x[~extreme][:,gindices],y[~extreme],source)
        evaluation_models={**ablations,'G_extreme_excluded_diagnostic':sensitivity}
        ar,_,_=evaluate_candidates(evaluation_models,cold,union,val_cap);records+=flatten(ar,'ablation_validation')
        primary_names=['G','D_plus_recent','E_plus_recent']
        chosen_name=min(primary_names,key=lambda k:selection_key(ar[k]));chosen_cols=ablations[chosen_name].columns
        # Outlier removal is diagnostic only: the PRIMARY always retains genuine extremes.
        lock={'split':vars(split),'loss':loss,'parameters':PARAMETERS,'selected_feature_set':chosen_name,
              'input_columns':chosen_cols,'cold_input_columns':cold_cols,'cold_loss':cold_loss,
              'baseline_name':best_baseline,'baseline_weights':baseline,'baseline_raw_availability':coverage,
              'selection_criterion':'matched-support portfolio WAPE; household MAE tie-break',
              'primary_outlier_policy':'retain measured extremes unchanged; robust leaf size/L2; loss chosen on validation',
              'outlier_diagnostic_validation':ar['G_extreme_excluded_diagnostic'],
              'heating_base_c':config.heating_base_c,'validation_label_max':val_cap,
              'fit_label_max_before_validation':source,'Household_ID_is_feature':False,
              'train_fit_thinning':'deterministic 1/8 rotating slots plus all extreme flags',
              'weather_selection_lock_sha256':hashlib.sha256((WEATHER_OUTPUT_DIR/'model_selection_lock.json').read_bytes()).hexdigest()}
        save('model_selection_lock.json',lock)
        pd.DataFrame(records).to_csv(OUT/'evaluation'/'validation_comparison.csv',index=False)
        save('validation_results.json',{'full':full_results,'ablations':ar,'selected':ar[chosen_name],
                                      'cold_proxy':cold_results[cold_loss],'best_baseline':full_results[best_baseline]})
        print('Locked energy recipe:',loss,chosen_name,'baseline',best_baseline,flush=True)
        del x,y,normal,extreme,full,cold_models,ablations,evaluation_models,sensitivity;gc.collect()
        # Refit only train+validation labels already available at FIRST test issue.
        refit_cols=list(dict.fromkeys(chosen_cols+cold_cols))
        x,y,normal,extreme,source,refit_hh=load_fit(['train','validation'],val_cap,refit_cols)
        primary=EnergyModel(chosen_cols,loss).fit(x[:,[refit_cols.index(c) for c in chosen_cols]],y,source)
        cold=EnergyModel(cold_cols,cold_loss).fit(x[normal][:,[refit_cols.index(c) for c in cold_cols]],y[normal],source)
        save('refit_summary.json',{'rows':len(y),'normal_rows':int(normal.sum()),'extreme_rows_retained':int(extreme.sum()),
            'fit_max_label_timestamp':source,'training_households':len(train_hh),'train_validation_households':len(refit_hh)})
        for name,model in [('locked_primary.pkl',primary),('locked_cold.pkl',cold)]:
            with open(OUT/'models'/name,'wb') as f:pickle.dump(model,f)
        del x,y,normal,extreme;gc.collect()
        # Explicit chronological day loop; no .fit occurs inside or after it.
        test=test_walk_forward(primary,cold,baseline,refit_cols,train_hh)
        save('test_evaluation_complete.json',test)
        demonstration=unseen_forecast(primary,cold,refit_cols)
        save('2024-02-28_summary.json',demonstration)
        print(json.dumps(demonstration,indent=2),flush=True)


if __name__=='__main__':main()
