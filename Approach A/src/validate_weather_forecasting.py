"""Validate locked weather artifacts; never retrain or reselect against test."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import PipelineConfig, ChronologicalSplit, WEATHER_OUTPUT_DIR
from .data_loading import load_weather
from .weather_forecasting import VARIABLES, station_features
from .weather_energy_interface import attach_predicted_weather


def main():
    out = WEATHER_OUTPUT_DIR
    config, split = PipelineConfig(), ChronologicalSplit()
    done = json.loads((out / "test_evaluation_completed.json").read_text())
    lock = json.loads((out / "model_selection_lock.json").read_text())
    sha = hashlib.sha256((out / "model_selection_lock.json").read_bytes()).hexdigest()
    if done["lock_sha256"] != sha:
        raise AssertionError("Weather selection lock changed after test evaluation")
    table = pd.read_parquet(out / "predictions/historical_day_ahead_weather.parquet")
    future = pd.read_parquet(out / "predictions/2024-02-28_day_ahead_weather.parquet")
    from .weather_forecasting import split_masks, season_of
    masks = split_masks(table, split)
    season_coverage = {}
    for period, mask in masks.items():
        if not table.loc[mask, "Period"].eq(period).all():
            raise AssertionError("Forecast period labels differ from frozen split")
        if period in ["validation", "test"]:
            dates = table.loc[mask, "DeliveryDate"].drop_duplicates().sort_values()
            if len(dates) != 365 or dates.dt.month.nunique() != 12:
                raise AssertionError("Holdout does not span a complete year")
            season_coverage[period] = season_of(dates).value_counts().to_dict()
    for frame in [table, future]:
        if frame.duplicated(["Weather_ID","TargetWeatherTimestamp"]).any():
            raise AssertionError("Duplicate weather target keys")
        if not frame.groupby(["Weather_ID","DeliveryDate"]).size().eq(24).all():
            raise AssertionError("Not 24 hourly forecasts per station-day")
        if frame.Weather_ID.nunique()!=8:
            raise AssertionError("Station lost")
        if (frame.HistorySourceMaxTimestamp > frame.WeatherHistoryCutoffTimestamp).any():
            raise AssertionError("After-cutoff observations used")
        if (frame.WeatherModelFitSourceMaxTimestamp > frame.WeatherHistoryCutoffTimestamp).any():
            raise AssertionError("Model fit used unavailable labels")
        if any(c.startswith("Actual") for c in frame):
            raise AssertionError("Actual weather mixed into deployed forecasts")
        for c in ["PredictedTemperature","PredictedSunshine","PredictedHumidity"]:
            if not frame[c+"Missing"].eq(frame[c].isna()).all():
                raise AssertionError("Missing indicator mismatch")
        for c,lo,hi in [("PredictedSunshine",0,1),("PredictedHumidity",0,100)]:
            if ((frame[c]<lo)|(frame[c]>hi)).any():
                raise AssertionError("Physical bound violation")
        missing_stations=frame.Weather_ID.isin(["HbsbG","ceOxS","sV3mR"])
        if not frame.loc[missing_stations,"PredictedSunshineMissing"].all():
            raise AssertionError("Invented sunshine for unavailable stations")
    # The Step 2 table is read unchanged; the adapter only returns a derived copy.
    energy = pd.read_parquet("outputs/preprocessing/modelling/household=661091/part.parquet")
    joined=attach_predicted_weather(energy,table)
    if len(joined)!=len(energy) or joined.PredictedTemperatureMissing.any():
        raise AssertionError("Assigned-station energy/weather join failed")
    # Real-station poisoning includes the target day and later D-1 observations.
    stations=load_weather(config.data_dir)
    with open(out / "models/production_model.pkl","rb") as f: recipe=pickle.load(f)
    contract=json.loads((out / "weather_feature_contract.json").read_text())
    tests=[]
    with threadpool_limits(limits=2):
        for name in ["8jB","HbsbG"]:
            station=stations[name]
            day=pd.date_range(split.unseen_demand_day,periods=1,tz="UTC")
            a=station_features(station,day,contract["station_codes"][name],name,config)
            cut=a.WeatherHistoryCutoffTimestamp.iloc[0]
            poisoned=station.copy()
            poisoned.loc[poisoned.index>cut,list(VARIABLES)]=999999.
            b=station_features(poisoned,day,contract["station_codes"][name],name,config)
            pd.testing.assert_frame_equal(a,b)
            for c in VARIABLES:
                ap=recipe.candidates(a,c)[lock["selected_methods"][c]]
                bp=recipe.candidates(b,c)[lock["selected_methods"][c]]
                if not np.allclose(ap,bp,equal_nan=True):
                    raise AssertionError("Future weather changes forecast")
                saved=future.loc[future.Weather_ID.eq(name),VARIABLES[c]].to_numpy()
                if not np.allclose(ap,saved,equal_nan=True):
                    raise AssertionError("Saved next-day forecast not reproducible")
            tests.append({"station":name,"future_poisoning_invariant":True})
    # Sunshine daylight assessment uses TRAINED dark-hour mask, never target actuals to classify night.
    with open(out / "models/locked_test_model.pkl","rb") as f: test_recipe=pickle.load(f)
    ev=pd.read_parquet(out / "evaluation/test_actuals_and_candidates.parquet")
    from .weather_forecasting import metrics
    sunshine=[]
    for name,g in ev.groupby("Weather_ID"):
        keys=pd.MultiIndex.from_arrays([g.Weather_ID,g.TargetWeatherTimestamp.dt.month,g.TargetUTCHour],
                                       names=["Weather_ID","TargetMonth","TargetUTCHour"])
        darkness=test_recipe.dark["Sunshine_duration_hourly"].reindex(keys).astype("boolean").fillna(False).to_numpy(dtype=bool)
        p=g["Sunshine_duration_hourly__"+lock["selected_methods"]["Sunshine_duration_hourly"]]
        for group,mask in [("empirical_dark_hours",darkness),("other_hours",~darkness)]:
            m=metrics(g.Actual_Sunshine_duration_hourly.to_numpy()[mask],p.to_numpy()[mask])
            m.update(Weather_ID=name,HourClass=group)
            sunshine.append(m)
    pd.DataFrame(sunshine).to_csv(out / "evaluation/sunshine_dark_other_hours.csv",index=False)
    result={"status":"passed","historical_forecast_rows":len(table),"next_day_rows":len(future),
            "stations":8,"24_hour_grid":True,"after_cutoff_source_violations":0,
            "out_of_time_fit_violations":0,"actual_weather_in_deployable_table":False,
            "unavailable_sunshine_preserved":True,"unchanged_lock_sha256":sha,
            "assigned_station_energy_join_rows":len(joined),"real_station_future_poison_tests":tests,
            "full_year_season_delivery_days":season_coverage,
            "test_evaluated_only_after_selection":True}
    (out / "validation_checks.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":main()
