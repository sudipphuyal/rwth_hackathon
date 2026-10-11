"""Reporting only: fixed dates, saved forecasts/labels, identical measured support."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path('outputs/energy_forecasting')
DATES=['2023-03-15','2023-09-05','2024-02-27']


def main():
    portfolio=pd.read_parquet(BASE/'predictions/test_portfolio.parquet')
    rows=[];checks=[]
    available=sorted(p.stem for p in (BASE/'predictions/test').glob('*.parquet'))
    for requested in DATES:
        for date in [d for d in available if d>=requested]:
            f=pd.read_parquet(BASE/f'predictions/test/{date}.parquet')
            labels=pd.read_parquet(BASE/f'evaluation/test_labels/{date}.parquet')
            z=f.merge(labels[['Household_ID','TargetTimestamp','Target']],on=['Household_ID','TargetTimestamp'],validate='one_to_one')
            z=z.loc[z.Target.notna()]
            if len(z):break
        else:raise ValueError(f'No following measured support for {requested}')
        g=z.groupby('TargetTimestamp').agg(Actual=('Target','sum'),ML=('Prediction','sum'),Blend=('BaselinePrediction','sum'),Households=('Household_ID','nunique'))
        saved=portfolio.loc[portfolio.TargetTimestamp.isin(g.index)].set_index('TargetTimestamp')
        np.testing.assert_allclose(g[['Actual','ML','Blend']],saved.loc[g.index,['MatchedActual','MatchedPrediction','MatchedBaseline']],rtol=0,atol=1e-9)
        actual=g.Actual.sum()/1000;ml=g.ML.sum()/1000;blend=g.Blend.sum()/1000
        rows.append(dict(DeliveryDate=date,ObservedHouseholds=z.Household_ID.nunique(),ActualEnergy_MWh=actual,MLForecastEnergy_MWh=ml,
            MLError_MWh=abs(ml-actual),MLSignedErrorPercent=100*(ml-actual)/actual if actual>0 else np.nan,
            BlendForecastEnergy_MWh=blend,BlendError_MWh=abs(blend-actual)))
        peak=g.loc[g.ML.idxmax()]
        checks.append(dict(RequestedDate=requested,DeliveryDate=date,Substitution=date!=requested,ScoreableIntervals=len(g),
            ObservedHouseholdsDefinition='Distinct households with at least one measured target that day',
            MinObservedHouseholdsPerInterval=int(g.Households.min()),MaxObservedHouseholdsPerInterval=int(g.Households.max()),
            PeakMatchedMLTimestampUTC=str(g.ML.idxmax()),PeakMatchedML_kWhPer15min=float(peak.ML),
            ActualAtMatchedMLPeak_kWhPer15min=float(peak.Actual),MatchedPortfolioCrossCheckPassed=True))
    pd.DataFrame(rows).to_csv('representative_test_days.csv',index=False)
    Path('representative_test_days_checks.json').write_text(json.dumps(dict(DatesPredefined=DATES,Checks=checks,SignedErrorDefinition='100 * (forecast - actual) / actual',
        MissingTargetsNotZero=True,ModelsNotRun=True),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
