import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_loading import prepare_target, read_meter, WEATHER_COLUMNS
from src.feature_availability import LABEL_COLUMNS, assert_cutoff, make_contract
from src.preprocessing import (calendar_features, cutoff_for_issue, issue_for_day,
                               modelling_day, nullable_boolean, prepare_metadata)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig()
        dates = pd.date_range("2024-02-01", "2024-04-02", freq="15min", tz="UTC")
        self.meter = pd.DataFrame({"Target": np.arange(len(dates)) % 96 / 100,
                                  "TargetSource": "measured"}, index=dates)
        hours = pd.date_range("2024-02-01", "2024-04-02", freq="h", tz="UTC")
        self.weather = pd.DataFrame({c: np.ones(len(hours)) for c in WEATHER_COLUMNS}, index=hours)
        self.metadata = pd.Series({"Household_ID": "fixture", "Weather_ID": "station",
                                   "MetadataSnapshotAssumedKnown": True})

    def test_future_mutation_cannot_change_features(self):
        day = pd.Timestamp("2024-03-31", tz="UTC")
        issue = issue_for_day(day, self.config)
        cutoff = cutoff_for_issue(issue, self.config)
        baseline = modelling_day(self.meter, self.metadata, self.weather, day, self.config)
        altered = self.meter.copy()
        altered.loc[altered.index > cutoff, "Target"] = 99999
        weather = self.weather.copy()
        wc = issue - pd.Timedelta(hours=2)
        weather.loc[weather.index > wc, :] = 99999
        result = modelling_day(altered, self.metadata, weather, day, self.config)
        safe_columns = [c for c in baseline if c not in LABEL_COLUMNS]
        pd.testing.assert_frame_equal(baseline[safe_columns], result[safe_columns])
        self.assertFalse(baseline.Target.equals(result.Target))

    def test_truncated_history_yields_same_features(self):
        day = pd.Timestamp("2024-03-31", tz="UTC")
        issue = issue_for_day(day, self.config)
        full = modelling_day(self.meter, self.metadata, self.weather, day, self.config)
        cut = modelling_day(self.meter.loc[:cutoff_for_issue(issue, self.config)], self.metadata,
                            self.weather.loc[:issue-pd.Timedelta(hours=2)], day, self.config)
        pd.testing.assert_frame_equal(full.drop(columns=LABEL_COLUMNS), cut.drop(columns=LABEL_COLUMNS))
        self.assertTrue(cut.Target.isna().all())

    def test_previous_day_late_morning_lags_are_asof_available(self):
        day = pd.Timestamp("2024-03-10", tz="UTC")
        frame = modelling_day(self.meter, self.metadata, self.weather, day, self.config)
        local_source = frame.LagAvailableSourceMaxTimestamp.dt.tz_convert(self.config.behavioral_timezone)
        # Morning target uses issue-day morning; evening uses an older day.
        morning = frame.LocalHour.eq(8)
        evening = frame.LocalHour.eq(20)
        self.assertTrue((local_source[morning].dt.day == 9).all())
        self.assertTrue((local_source[evening].dt.day == 8).all())
        assert_cutoff(frame)

    def test_gap_is_not_row_shift(self):
        day = pd.Timestamp("2024-03-10", tz="UTC")
        missing = pd.Timestamp("2024-03-03 08:00", tz=self.config.behavioral_timezone).tz_convert("UTC")
        frame = modelling_day(self.meter.drop(index=missing), self.metadata, self.weather, day, self.config)
        target = pd.Timestamp("2024-03-10 08:00", tz=self.config.behavioral_timezone).tz_convert("UTC")
        row = frame.loc[frame.TargetTimestamp.eq(target)].iloc[0]
        self.assertTrue(pd.isna(row.PreviousWeekSameLocalInterval))
        self.assertLess(row.Recent28dCoverage, 1)

    def test_reconstruction_hierarchy(self):
        d = pd.DataFrame({"kWh_received_Total": [1., np.nan, np.nan, 0.],
                          "kWh_received_HeatPump": [.4, .2, .2, 0.],
                          "kWh_received_Other": [.6, .3, np.nan, 0.]})
        result = prepare_target(d)
        self.assertEqual(result.TargetSource.tolist(), ["measured", "reconstructed", "missing", "measured"])
        self.assertAlmostEqual(result.Target.iloc[1], .5)
        self.assertTrue(pd.isna(result.Target.iloc[2]))
        invalid_components = pd.DataFrame({"kWh_received_Total": [np.nan, np.nan],
            "kWh_received_HeatPump": [np.inf, -.2], "kWh_received_Other": [.3, .3]})
        self.assertTrue(prepare_target(invalid_components).Target.isna().all())
        d.loc[0, "kWh_received_Other"] = 1
        with self.assertRaises(ValueError):
            prepare_target(d)

    def test_metadata_unknown_coexistence(self):
        from src.preprocessing import META_CATEGORICAL, META_NUMERIC
        r = pd.DataFrame({"Household_ID": ["a", "b"], "Weather_ID": ["w", "w"],
                          "Installation_HasPVSystem": [False, np.nan]})
        s = pd.DataFrame({"Household_ID": ["a"], **{c: [np.nan] for c in META_CATEGORICAL + META_NUMERIC}})
        s["Survey_HeatDistribution_System_FloorHeating"] = True
        s["Survey_HeatDistribution_System_Radiator"] = True
        m = prepare_metadata(r, s)
        self.assertEqual(m.loc["a", "Installation_HasPVSystem"], "False")
        self.assertEqual(m.loc["b", "Installation_HasPVSystem"], "Unknown")
        self.assertEqual(m.loc["a", "Survey_DHW_Production_ByHeatPump"], "Unknown")
        self.assertEqual(m.loc["a", "Survey_HeatDistribution_System_Radiator"], "True")
        self.assertEqual(m.loc["a", "Survey_HeatDistribution_System_FloorHeating"], "True")
        self.assertTrue(m.loc["b", "Survey_Building_LivingArea_missing"])

    def test_dst_utc_delivery_and_local_calendar(self):
        for date, hour_counts in [("2024-03-31", (0, 4)), ("2024-10-27", (8, 4))]:
            targets = pd.date_range(date, periods=96, freq="15min", tz="UTC")
            f = calendar_features(targets, issue_for_day(targets[0], self.config), self.config)
            self.assertEqual(len(f), 96)
            self.assertEqual(f.TargetTimestamp.nunique(), 96)
            self.assertEqual(sum(f.LocalHour == 2), hour_counts[0])
        for date, expected in [("2024-03-31", 92), ("2024-10-27", 100)]:
            start = pd.Timestamp(date, tz=self.config.behavioral_timezone)
            end = start + pd.DateOffset(days=1)
            self.assertEqual(len(pd.date_range(start, end, freq="15min", inclusive="left")), expected)

    def test_cold_start_and_absent_sunshine(self):
        day = pd.Timestamp("2024-03-10", tz="UTC")
        weather = self.weather.copy()
        weather.Sunshine_duration_hourly = np.nan
        frame = modelling_day(self.meter.iloc[:0], self.metadata, weather, day, self.config)
        self.assertTrue(frame.ForecastRoute.eq("cold_start").all())
        self.assertTrue(frame.Target.isna().all())
        self.assertTrue(frame.sunshine_missing.all())
        self.assertTrue(frame.Historical24h_Sunshine_duration_hourly.isna().all())
        self.assertTrue(frame.Forecast_Temperature_avg_hourly.isna().all())
        contract = make_contract(frame)
        self.assertTrue(set(LABEL_COLUMNS).isdisjoint(contract["candidate_features"]))
        normal = modelling_day(self.meter, self.metadata, self.weather, day, self.config)
        self.assertEqual(list(normal.columns), list(frame.columns))

    def test_lineage_guard_rejects_late_source(self):
        frame = modelling_day(self.meter, self.metadata, self.weather, "2024-03-10", self.config)
        frame.loc[0, "LagAvailableSourceMaxTimestamp"] = frame.ForecastIssueTimestamp.iloc[0]
        with self.assertRaises(AssertionError):
            assert_cutoff(frame)

    def test_column_name_loading_and_duplicate_detection(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "1.csv"
            data = pd.DataFrame({"kWh_received_Other": [.6], "Timestamp": ["2024-01-01 00:00+00:00"],
                "Group": ["control"], "Household_ID": [1], "kWh_received_Total": [1],
                "AffectsTimePoint": ["unknown"], "kWh_received_HeatPump": [.4]})
            data.to_csv(p, sep=";", index=False)
            self.assertEqual(read_meter(p).kWh_received_Total.iloc[0], 1)
            pd.concat([data, data]).to_csv(p, sep=";", index=False)
            with self.assertRaises(ValueError):
                read_meter(p)


if __name__ == "__main__":
    unittest.main()
