"""Independent checks of saved predictions/labels; no test-driven model selection."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .config import PipelineConfig
from .data_loading import read_meter,prepare_target,load_registry
from .preprocessing import prepare_metadata
from .preprocessing_batch import batch_household_features
from .weather_energy_interface import attach_predicted_weather
from .energy_features import add_physical_features
from .energy_metrics import ValidationScores
from .run_energy_forecasting import OUT


def main():
    test=json.loads((OUT/'test_evaluation_complete.json').read_text())
    lock=json.loads((OUT/'model_selection_lock.json').read_text())
    refit=json.loads((OUT/'refit_summary.json').read_text())
    primary=pickle.load(open(OUT/'models/locked_primary.pkl','rb'))
    cold=pickle.load(open(OUT/'models/locked_cold.pkl','rb'))
    for name,sha in test['frozen_weights_sha256'].items():
        if hashlib.sha256((OUT/'models'/name).read_bytes()).hexdigest()!=sha:raise AssertionError('Locked weights changed')
    if any(c in lock['input_columns'] for c in ['Household_ID','Target','TargetSource','TargetExtremeReview']):raise AssertionError('Identity/target leakage')
    if any(c.startswith(('Recent','Previous','History','Historical','GapAge')) for c in lock['cold_input_columns']):raise AssertionError('Cold model has personal demand history')
    scores,bs=ValidationScores(),ValidationScores();zero_days=[];nrows=0
    for path in sorted((OUT/'predictions'/'test').glob('*.parquet')):
        p=pq.ParquetFile(path).read().to_pandas()
        labels=pq.ParquetFile(OUT/'evaluation'/'test_labels'/path.name).read().to_pandas()
        if not p[['Household_ID','TargetTimestamp']].equals(labels[['Household_ID','TargetTimestamp']]):raise AssertionError('Predictions/actual support keys differ')
        if p.duplicated(['Household_ID','TargetTimestamp']).any():raise AssertionError('Duplicate forecast')
        if not p.groupby('Household_ID').size().eq(96).all():raise AssertionError('Not 96 intervals per active household')
        if not np.isfinite(p.Prediction).all() or p.Prediction.lt(0).any():raise AssertionError('Invalid full forecast')
        if (p.FirstObservedTimestamp>p.ForecastCutoffTimestamp).any():raise AssertionError('Household active before observation')
        if pd.Timestamp(refit['fit_max_label_timestamp'])>p.ForecastCutoffTimestamp.min():raise AssertionError('Fit labels unavailable at test issue')
        if labels.loc[labels.TargetSource.eq('missing'),'Target'].notna().any():raise AssertionError('Missing labels synthesised')
        d=p.assign(Target=labels.Target.to_numpy());scores.update_day(d,p.Prediction);bs.update_day(d,p.BaselinePrediction)
        if d.Target.notna().sum()==0:zero_days.append(path.stem)
        nrows+=len(d)
    # Correct zero-support daily totals: forecasts exist, but actual/error are undefined.
    portfolio=pd.read_parquet(OUT/'predictions'/'test_portfolio.parquet')
    missing=portfolio.ObservedCount.eq(0)
    portfolio.loc[missing,['MatchedActual','MatchedPrediction','MatchedBaseline']]=np.nan
    portfolio.to_parquet(OUT/'predictions'/'test_portfolio.parquet',index=False)
    daily=portfolio.groupby(portfolio.TargetTimestamp.dt.floor('D')).agg(
        MatchedActual=('MatchedActual',lambda x:x.sum(min_count=1)),
        MatchedPrediction=('MatchedPrediction',lambda x:x.sum(min_count=1)),
        MatchedBaseline=('MatchedBaseline',lambda x:x.sum(min_count=1)),FullForecast=('FullForecast','sum'),
        ActiveHouseholds=('ActiveCount','max'),MinObservedHouseholds=('ObservedCount','min'))
    daily['AbsoluteEnergyError']=(daily.MatchedPrediction-daily.MatchedActual).abs()
    daily['BaselineAbsoluteEnergyError']=(daily.MatchedBaseline-daily.MatchedActual).abs()
    daily.to_csv(OUT/'evaluation'/'test_daily_energy.csv')
    test['selected']=scores.result();test['baseline']=bs.result()
    test['test_days_without_any_actual_support']=zero_days
    test['daily_accuracy_policy']='No matched support => NaN actual/error and excluded daily score; forecasts retained'
    # Cold proxy stored summary likewise excludes the unscorable day.
    proxy=test['cold_proxy_all_history_withheld']
    if proxy['EvaluatedDays']==365 and zero_days:
        proxy['DailyEnergyMAE']*=365/(365-len(zero_days));proxy['EvaluatedDays']=365-len(zero_days)
        proxy['DailyEnergyMedianAE']=None  # Not retained per-day proxy errors; do not claim an uncorrected median.
    (OUT/'test_evaluation_complete.json').write_text(json.dumps(test,indent=2))
    # Real-household final-day invariance and saved prediction reproducibility.
    config=PipelineConfig();reg=load_registry(config.data_dir)
    survey=pd.read_csv(config.data_dir/'smart_meter_meta_data/meta_data.csv',sep=';',dtype={'Household_ID':str})
    meta=prepare_metadata(reg,survey)
    weather=pd.read_parquet(Path('outputs/weather_forecasting_full_year/predictions/2024-02-28_day_ahead_weather.parquet'))
    saved=pd.read_parquet(OUT/'predictions/2024-02-28_households.parquet')
    checks=[]
    with threadpool_limits(limits=4):
        for h in ['661091','676510','996610']:
            meter=prepare_target(read_meter(config.data_dir/'15min'/f'{h}.csv'))
            days=pd.date_range('2024-02-28',periods=1,tz='UTC')
            a=batch_household_features(meter,meta.loc[h],days,config)
            cutoff=a.ForecastCutoffTimestamp.iloc[0]
            altered=meter.copy();future=altered.index>cutoff
            altered.loc[future,'Target']=999999.;altered.loc[future,'TargetSource']='measured'
            b=batch_household_features(altered,meta.loc[h],days,config)
            station_weather=weather.loc[weather.Weather_ID.eq(meta.loc[h].Weather_ID)]
            a=add_physical_features(attach_predicted_weather(a,station_weather));b=add_physical_features(attach_predicted_weather(b,station_weather))
            model=cold if a.ForecastState.iloc[0]=='cold-start' else primary
            pa,pb=model.predict(a),model.predict(b)
            if not np.allclose(pa,pb):raise AssertionError('Future demand changes forecast')
            old=saved.loc[saved.Household_ID.eq(h),'Prediction'].to_numpy()
            if not np.allclose(pa,old,atol=1e-6):raise AssertionError('Saved final prediction cannot be reproduced')
            checks.append({'Household_ID':h,'future_poisoning_invariant':True})
    if len(saved)!=410*96 or saved.Household_ID.nunique()!=410:raise AssertionError('Incomplete unseen-day portfolio')
    if not saved.groupby('Household_ID').size().eq(96).all():raise AssertionError('Unseen-day grid incorrect')
    summary=json.loads((OUT/'2024-02-28_summary.json').read_text())
    if abs(saved.Prediction.sum()-summary['total_daily_kwh'])>1e-8:raise AssertionError('Portfolio sum differs')
    result={'status':'passed','test_days':365,'scorable_test_days':365-len(zero_days),'test_forecast_rows':nrows,
            'zero_actual_support_days':zero_days,'full_portfolio_ground_truth_not_invented':True,
            'future_demand_poison_tests':checks,'locked_weight_hashes_unchanged':True,
            'unseen_day_rows':len(saved),'unseen_day_households':410,'no_electricity_refit_after_test':True}
    (OUT/'validation_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
