"""Replay the locked recipe on TRAIN only to save out-of-time validation residuals.

Analysis copies only: never touches the locked train+validation/test weights.
No re-selection and no test targets are read by this command.
"""
import json
import pickle
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from .config import PipelineConfig,ChronologicalSplit
from .preprocessing import issue_for_day,cutoff_for_issue
from .run_energy_forecasting import OUT,load_fit,val_days,predict_matrix
from .energy_model import EnergyModel,completed_baselines,apply_blend
from .energy_metrics import ValidationScores
from .energy_features import matrix


def main():
    lock=json.loads((OUT/'model_selection_lock.json').read_text())
    cols=list(dict.fromkeys(lock['input_columns']+lock['cold_input_columns']))
    config,split=PipelineConfig(),ChronologicalSplit()
    cutoff=cutoff_for_issue(issue_for_day(pd.Timestamp(split.validation_start),config),config)
    cap=cutoff_for_issue(issue_for_day(pd.Timestamp(split.test_start),config),config)
    x,y,normal,extreme,source,_=load_fit(['train'],cutoff,cols)
    with threadpool_limits(limits=4):
        primary=EnergyModel(lock['input_columns'],lock['loss']).fit(x[:,[cols.index(c) for c in lock['input_columns']]],y,source)
        cold=EnergyModel(lock['cold_input_columns'],lock['cold_loss']).fit(x[normal][:,[cols.index(c) for c in lock['cold_input_columns']]],y[normal],source)
        for name,m in [('validation_only_primary.pkl',primary),('validation_only_cold.pkl',cold)]:
            with open(OUT/'models'/name,'wb') as f:pickle.dump(m,f)
        a,b=ValidationScores(),ValidationScores()
        dest=OUT/'predictions'/'validation';dest.mkdir(exist_ok=True)
        eval_dest=OUT/'evaluation'/'validation_residuals';eval_dest.mkdir(exist_ok=True)
        zero_support=[]
        for d in val_days(cols,cap):
            xm=matrix(d,cols)
            fallback=predict_matrix(cold,xm,cols);prediction=predict_matrix(primary,xm,cols)
            mask=d.ForecastState.eq('cold-start').to_numpy();prediction[mask]=fallback[mask]
            baseline=apply_blend(completed_baselines(d,fallback),lock['baseline_weights'])
            a.update_day(d,prediction);b.update_day(d,baseline)
            if not d.Target.notna().any():zero_support.append(str(d.TargetTimestamp.iloc[0].date()))
            output=d[['Household_ID','TargetTimestamp','ForecastIssueTimestamp','ForecastState']].copy()
            output['Prediction']=prediction;output['BaselinePrediction']=baseline
            # Evaluation residuals explicitly separated from deployable forecast columns.
            output['Actual']=d.Target.to_numpy();output['Residual']=output.Actual-output.Prediction
            output['TargetSource']=d.TargetSource.to_numpy()
            name=f'{d.TargetTimestamp.iloc[0].date()}.parquet'
            output.to_parquet(eval_dest/name,index=False)
            output.drop(columns=['Actual','Residual','TargetSource']).to_parquet(dest/name,index=False)
        results={'selected':a.result(),'baseline':b.result(),'no_support_days':zero_support,
                 'training_only_fit_source_max':str(source),'validation_label_cap':str(cap),
                 'locked_test_weights_unchanged':True,'no_test_targets_used':True}
        original=json.loads((OUT/'validation_results.json').read_text())
        for view in ['Household','MatchedPortfolio']:
            for metric in ['MAE','RMSE','WAPE','Bias']:
                if abs(results['selected'][view][metric]-original['selected'][view][metric])>1e-8:
                    raise AssertionError('Validation recipe replay differs from original selected prediction')
        (OUT/'validation_residual_summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2))


if __name__=='__main__':main()
