"""Independent output checks; no calibration, prediction fitting or selection."""
import json,hashlib
import numpy as np
import pandas as pd
from .uncertainty_forecasting import OUT,BASE,hashes

def main():
    lock=json.loads((OUT/'calibration_lock.json').read_text());result=json.loads((OUT/'test_evaluation_complete.json').read_text())
    assert hashes()==lock['models_before']
    assert hashlib.sha256((OUT/'calibration_lock.json').read_bytes()).hexdigest()==result['calibration_lock_sha256']
    for m in ['ML','Blend']:
        f=pd.read_csv(OUT/f'2024-02-28_{m}_portfolio.csv');assert len(f)==96
        assert pd.to_datetime(f.TargetTimestamp,utc=True).nunique()==96
        v=f[['P10_kWh','P50_kWh','P90_kWh']].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (np.diff(v,axis=1)>=0).all()
    old=pd.read_csv(BASE/'predictions/2024-02-28_portfolio.csv');new=pd.read_csv(OUT/'2024-02-28_ML_portfolio.csv')
    np.testing.assert_allclose(old.PredictedPortfolioKWh,new.PointForecast_kWh,rtol=0,atol=1e-10)
    q=pd.read_parquet(OUT/'test_portfolio_quantiles.parquet')
    assert len(q)==35040*2
    for m,g in q.groupby('Method'):
        v=g[['P10','P50','P90']].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (np.diff(v,axis=1)>=0).all()
        ok=g.Actual.notna();assert ok.sum()==34944
        observed=((g.loc[ok,'Actual']>=g.loc[ok,'MatchedP10'])&(g.loc[ok,'Actual']<=g.loc[ok,'MatchedP90'])).mean()
        assert abs(observed-result['metrics'][m]['Test']['Coverage'])<1e-12
    h=pd.read_parquet(OUT/'2024-02-28_household_quantiles.parquet');assert len(h)==39360 and h.Household_ID.nunique()==410
    assert (np.diff(h[['P10','P50','P90']].to_numpy(),axis=1)>=0).all()
    checks=dict(passed=True,locked_models_unchanged=True,calibration_lock_unchanged=True,unseen_point_exactly_preserved=True,
        test_scored_intervals_per_method=34944,unseen_households=410,unseen_intervals=96)
    (OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2));print(checks)

if __name__=='__main__':main()
