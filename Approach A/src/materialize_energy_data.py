"""Materialise Step 2 features efficiently; reuse ONLY locked predicted weather."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import PipelineConfig,ChronologicalSplit,WEATHER_OUTPUT_DIR
from .data_loading import read_meter,prepare_target,load_registry
from .preprocessing import prepare_metadata
from .preprocessing_batch import batch_household_features
from .weather_energy_interface import attach_predicted_weather
from .energy_features import add_physical_features,feature_groups,cold_features

OUT=Path("outputs/energy_forecasting")


def main():
    config,split=PipelineConfig(),ChronologicalSplit()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"materialization_complete.json").exists():
        print("Existing complete energy feature materialisation retained");return
    registry=load_registry(config.data_dir)
    survey=pd.read_csv(config.data_dir/"smart_meter_meta_data/meta_data.csv",sep=";",dtype={"Household_ID":str})
    meta=prepare_metadata(registry,survey)
    forecasts=pd.read_parquet(WEATHER_OUTPUT_DIR/"predictions/historical_day_ahead_weather.parquet")
    forecasts=pd.concat([forecasts,pd.read_parquet(WEATHER_OUTPUT_DIR/"predictions/2024-02-28_day_ahead_weather.parquet")],ignore_index=True)
    days=pd.date_range(split.train_start,split.unseen_demand_day,freq="D",tz="UTC")
    manifest=[];schema=None
    periods={"train":(split.train_start,split.train_end),"validation":(split.validation_start,split.validation_end),
             "test":(split.test_start,split.test_end),"unseen_day":(split.unseen_demand_day,split.unseen_demand_day)}
    for i,h in enumerate(registry.Household_ID):
        meter=prepare_target(read_meter(config.data_dir/"15min"/f"{h}.csv"))
        d=batch_household_features(meter,meta.loc[h],days,config)
        station=meta.loc[h].Weather_ID
        d=attach_predicted_weather(d,forecasts.loc[forecasts.Weather_ID.eq(station)])
        d=add_physical_features(d,config.heating_base_c)
        if schema is None:
            schema=list(d.columns)
            (OUT/"input_feature_groups.json").write_text(json.dumps({"groups":feature_groups(d),"cold_start":cold_features(d),
                "heating_base_c":config.heating_base_c,"Household_ID_is_input":False,
                "weather_inputs":["PredictedTemperature","PredictedSunshine","PredictedHumidity","PredictedTemperatureMissing","PredictedSunshineMissing","PredictedHumidityMissing"],
                "metadata_assumption":config.metadata_assumption,"categorical_unknown":"explicit category 2, distinct from False 0",
                "train_fit_thinning":"1/8 rotating quarter-hour/hour phase, all extremes forced retained; full validation/test resolution"},indent=2))
        elif list(d.columns)!=schema:raise AssertionError("Energy schemas differ")
        # Compact continuous features; retain labels in original float64 precision.
        for c in d.select_dtypes("float64"):
            if c!="Target":d[c]=d[c].astype("float32")
        for period,(start,end) in periods.items():
            mask=d.TargetTimestamp.between(pd.Timestamp(start,tz="UTC"),pd.Timestamp(end,tz="UTC")+pd.Timedelta(days=1)-pd.Timedelta(minutes=15))
            q=d.loc[mask]
            if q.empty:continue
            folder=OUT/"data"/period;folder.mkdir(parents=True,exist_ok=True)
            q.to_parquet(folder/f"{h}.parquet",index=False,compression="zstd",row_group_size=96*7)
            manifest.append({"Household_ID":h,"Period":period,"Rows":len(q),"ObservedTargets":int(q.Target.notna().sum()),
                             "FirstObservedTimestamp":str(meter.index.min()),"FirstForecastDate":str(q.TargetTimestamp.min().date()),
                             "LastForecastDate":str(q.TargetTimestamp.max().date())})
        if i%20==0:print(f"Materialised {i+1}/410 households",flush=True)
    pd.DataFrame(manifest).to_csv(OUT/"materialization_manifest.csv",index=False)
    # Daily partitions make literal daily walk-forward prediction fast and keep
    # label revelation separate from model-input reads. Memory bounded to a month.
    for period in ["validation","test","unseen_day"]:
        paths=sorted((OUT/"data"/period).glob("*.parquet"))
        dataset=ds.dataset([str(p) for p in paths],format="parquet")
        start,end=periods[period]
        first=pd.Timestamp(start,tz="UTC").replace(day=1)
        months=pd.date_range(first,pd.Timestamp(end,tz="UTC"),freq="MS")
        dest=OUT/"daily"/period;dest.mkdir(parents=True,exist_ok=True)
        for month in months:
            stop=month+pd.DateOffset(months=1)
            filtered=dataset.to_table(filter=(ds.field("TargetTimestamp")>=month)&(ds.field("TargetTimestamp")<stop)).to_pandas()
            for day,g in filtered.groupby(filtered.TargetTimestamp.dt.floor("D")):
                g.to_parquet(dest/f"{day.date()}.parquet",index=False,compression="zstd",row_group_size=8192)
            del filtered
        print(f"Daily {period} partitions ready",flush=True)
    counts=pd.DataFrame(manifest).groupby("Period")[["Rows","ObservedTargets"]].sum().to_dict()
    (OUT/"materialization_complete.json").write_text(json.dumps({"period_counts":counts,"schema":schema,
        "active_rule":"First supplied interval is available by issue (Timestamp+30min <= issue); no departure dates assumed",
        "split":vars(split),"weather_source":str(WEATHER_OUTPUT_DIR),"raw_data_modified":False},indent=2))
    print(json.dumps(counts,indent=2),flush=True)


if __name__=="__main__":main()
