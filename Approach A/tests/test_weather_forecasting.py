import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.config import PipelineConfig, ChronologicalSplit
from src.preprocessing import issue_for_day
from src.weather_forecasting import (VARIABLES, station_features, input_columns,
    assert_weather_features, weather_cutoff, split_masks, WeatherRecipe, deployable_table)


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig()
        index = pd.date_range("2023-01-01", "2024-04-03", freq="h", tz="UTC")
        self.station = pd.DataFrame({
            "Temperature_avg_hourly": np.sin(index.hour/24*2*np.pi)+index.dayofyear/100,
            "Humidity_avg_hourly": np.full(len(index), 70.),
            "Sunshine_duration_hourly": np.where((index.hour>=8)&(index.hour<17), .5, 0.)}, index=index)

    def build(self, station=None, start="2024-03-31", end="2024-03-31"):
        return station_features(self.station if station is None else station,
            pd.date_range(start,end,freq="D",tz="UTC"), 0, "fixture", self.config)

    def test_future_poison_and_truncate_all_feature_values(self):
        day = pd.Timestamp("2024-03-31",tz="UTC")
        cutoff = weather_cutoff(issue_for_day(day,self.config),self.config)
        f = self.build()
        future = self.station.copy()
        future.loc[future.index>cutoff,:] = 999999.
        pd.testing.assert_frame_equal(f, self.build(future))
        pd.testing.assert_frame_equal(f, self.build(self.station.loc[:cutoff]))

    def test_noon_later_hours_are_not_used(self):
        f = self.build(start="2024-02-28",end="2024-02-28")
        morning = f.loc[f.TargetUTCHour.eq(7)].iloc[0]
        evening = f.loc[f.TargetUTCHour.eq(20)].iloc[0]
        self.assertEqual(morning.Temperature_avg_hourly_SameHourSourceMaxTimestamp,
                         pd.Timestamp("2024-02-27 07:00",tz="UTC"))
        self.assertEqual(evening.Temperature_avg_hourly_SameHourSourceMaxTimestamp,
                         pd.Timestamp("2024-02-26 20:00",tz="UTC"))

    def test_keyed_missing_week_lag(self):
        missing = pd.Timestamp("2024-03-24 16:00",tz="UTC")
        s = self.station.copy()
        s.loc[missing,"Temperature_avg_hourly"] = np.nan
        f=self.build(s)
        self.assertTrue(pd.isna(f.loc[f.TargetUTCHour.eq(16),"Temperature_avg_hourly_week"].iloc[0]))

    def test_dst_leads_and_24_utc_hours(self):
        f=self.build(start="2024-03-30",end="2024-04-01")
        self.assertTrue(f.groupby("DeliveryDate").size().eq(24).all())
        self.assertEqual(f.loc[f.DeliveryDate.eq(pd.Timestamp("2024-04-01",tz="UTC")),"ForecastLeadHours"].min(),15)
        self.assertEqual(f.loc[f.DeliveryDate.eq(pd.Timestamp("2024-03-31",tz="UTC")),"ForecastLeadHours"].min(),14)
        self.assertEqual(f.TargetWeatherTimestamp.nunique(),72)

    def test_global_splits_disjoint_and_adjacent(self):
        dates=pd.date_range("2019-05-20","2024-02-27",freq="D",tz="UTC")
        f=pd.DataFrame({"DeliveryDate":dates});m=split_masks(f)
        self.assertTrue(sum(v.astype(int) for v in m.values()).eq(1).all())
        self.assertEqual({k:int(v.sum()) for k,v in m.items()},dict(train=1015,validation=365,test=365))
        # Each holdout must contain all calendar months and meteorological seasons.
        for period in ['validation','test']:
            self.assertEqual(len(set(dates[m[period]].month)),12)
        for preceding,following in [("train","validation"),("validation","test")]:
            self.assertLess(dates[m[preceding]].max(),dates[m[following]].min())

    def test_boundary_embargo_fit_before_first_issue(self):
        first=issue_for_day(pd.Timestamp(ChronologicalSplit().validation_start),self.config)
        cutoff=weather_cutoff(first,self.config)
        self.assertEqual(cutoff,pd.Timestamp("2022-02-27 08:00",tz="UTC"))
        self.assertGreater(pd.Timestamp("2022-02-27 23:00",tz="UTC"),cutoff)

    def test_feature_guard_and_column_contract(self):
        f=self.build(); columns=input_columns(f)
        self.assertEqual(len(columns),55)
        self.assertFalse(any("Timestamp" in c for c in columns))
        self.assertFalse(any(c.startswith("Actual") for c in columns))
        f.loc[0,"HistorySourceMaxTimestamp"]=f.ForecastIssueTimestamp.iloc[0]
        with self.assertRaises(AssertionError):assert_weather_features(f)

    def test_sunshine_unavailable_and_darkness_training_only(self):
        station=self.station.copy()
        station.Sunshine_duration_hourly=np.nan
        missing=self.build(station)
        self.assertTrue(missing.Sunshine_duration_hourly_available_hour.isna().all())
        self.assertEqual(str(missing.Sunshine_duration_hourly_SameHourSourceMaxTimestamp.dtype),"datetime64[ns, UTC]")
        f=self.build(start="2024-01-01",end="2024-01-30")
        labels=pd.DataFrame({c:np.ones(len(f)) for c in VARIABLES},index=f.index)
        labels.Sunshine_duration_hourly=np.where(f.TargetUTCHour.between(8,16),.5,0.)
        with patch.dict("src.weather_forecasting.MODEL_PARAMETERS", {"max_iter":3}):
            recipe=WeatherRecipe(input_columns(f)).fit(f,labels)
        target=self.build(start="2024-02-28",end="2024-02-28")
        candidates=recipe.candidates(target,"Sunshine_duration_hourly")
        for y in candidates.values():self.assertTrue(((y>=0)&(y<=1)).all())
        unknown=target.copy();unknown.Weather_ID="unavailable_station"
        for y in recipe.candidates(unknown,"Sunshine_duration_hourly").values():
            self.assertTrue(np.isnan(y).all())
        prediction={c:recipe.candidates(target,c)["hist_gradient_boosting"] for c in VARIABLES}
        table=deployable_table(target,prediction,"test",recipe,{c:"hist_gradient_boosting" for c in VARIABLES})
        self.assertFalse(any("Actual" in c for c in table))
        self.assertEqual(len(table),24)


if __name__ == "__main__":unittest.main()
