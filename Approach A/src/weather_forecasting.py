"""Station-level, causal features and simple direct hourly weather models.

This module never imports household demand labels or target-day observations
when constructing inputs. All delivery grids follow the Step 2 UTC convention.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ChronologicalSplit, PipelineConfig
from .preprocessing import issue_for_day

VARIABLES = {
    "Temperature_avg_hourly": "PredictedTemperature",
    "Sunshine_duration_hourly": "PredictedSunshine",
    "Humidity_avg_hourly": "PredictedHumidity",
}
BASELINES = ["available_same_hour", "previous_week", "same_hour_7d_mean",
             "same_hour_7d_median", "climatology"]
MODEL_PARAMETERS = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
                        min_samples_leaf=40, l2_regularization=1.0,
                        early_stopping=False, random_state=42)


def weather_cutoff(issue, config):
    return issue - pd.Timedelta(minutes=config.weather_interval_minutes + config.weather_latency_minutes)


def split_masks(frame, split=ChronologicalSplit()):
    dates = frame.DeliveryDate
    return {name: dates.between(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
            for name, start, end in [
                ("train", split.train_start, split.train_end),
                ("validation", split.validation_start, split.validation_end),
                ("test", split.test_start, split.test_end)]}


def station_features(station, days, station_code, station_id, config=PipelineConfig()):
    """Vectorised trailing windows and timestamp-keyed UTC same-hour lookups.

    Rolling operations are backward-only. Source timestamps/ages travel with
    each forecast; no interpolation or fitting against target weather occurs.
    """
    targets = pd.date_range(days.min(), days.max() + pd.Timedelta(days=1),
                            freq="h", inclusive="left", tz="UTC")
    dates = targets.floor("D")
    issues = pd.DatetimeIndex([issue_for_day(d, config) for d in days]).repeat(24)
    cutoffs = weather_cutoff(issues, config)
    local = targets.tz_convert(config.behavioral_timezone)
    doy = targets.dayofyear
    yearlength = np.where(targets.is_leap_year, 366, 365)
    f = pd.DataFrame({"Weather_ID": station_id, "DeliveryDate": dates,
        "TargetWeatherTimestamp": targets, "ForecastIssueTimestamp": issues,
        "WeatherHistoryCutoffTimestamp": cutoffs,
        "HistorySourceMaxTimestamp": cutoffs.where(cutoffs >= station.index.min(), pd.NaT),
        "ForecastLeadHours": (targets - issues).total_seconds() / 3600,
        "StationCode": station_code, "TargetUTCHour": targets.hour,
        "TargetLocalHour": local.hour, "TargetMonth": targets.month,
        "TargetDayOfYear": doy,
        "HourSin": np.sin(2*np.pi*targets.hour/24), "HourCos": np.cos(2*np.pi*targets.hour/24),
        "YearSin": np.sin(2*np.pi*(doy-1)/yearlength), "YearCos": np.cos(2*np.pi*(doy-1)/yearlength)})
    hour_keys = cutoffs.floor("D") + pd.to_timedelta(targets.hour, unit="h")
    hour_keys = hour_keys.where(hour_keys <= cutoffs, hour_keys - pd.Timedelta(days=1))
    week_keys = targets - pd.Timedelta(days=7)
    for c in VARIABLES:
        s = station[c]
        latest = s.ffill()
        sources = pd.Series(station.index, index=station.index).where(s.notna()).ffill()
        f[c + "_latest"] = latest.reindex(cutoffs).to_numpy()
        last_ts = pd.DatetimeIndex(sources.reindex(cutoffs))
        f[c + "_age_hours"] = (cutoffs - last_ts).total_seconds()/3600
        for hours in [6, 12, 24]:
            roll = s.rolling(f"{hours}h", min_periods=1)
            f[c + f"_mean_{hours}h"] = roll.mean().reindex(cutoffs).to_numpy()
            f[c + f"_coverage_{hours}h"] = roll.count().reindex(cutoffs).to_numpy()/hours
        f[c + "_std_24h"] = s.rolling("24h", min_periods=2).std().reindex(cutoffs).to_numpy()
        f[c + "_change_6h"] = f[c + "_latest"].to_numpy() - latest.reindex(cutoffs-pd.Timedelta(hours=6)).to_numpy()
        f[c + "_week"] = s.reindex(week_keys).to_numpy()
        f[c + "_week"] = f[c + "_week"].where(week_keys <= cutoffs)
        same, means, medians = np.full(len(f), np.nan), np.full(len(f), np.nan), np.full(len(f), np.nan)
        same_sources = pd.Series(pd.NaT, index=f.index, dtype="datetime64[ns, UTC]")
        for hour in range(24):
            mask = targets.hour == hour
            hourly = s.loc[s.index.hour == hour]
            keys = hour_keys[mask]
            same[mask] = hourly.ffill().reindex(keys).to_numpy()
            means[mask] = hourly.rolling("7D", min_periods=1).mean().reindex(keys).to_numpy()
            medians[mask] = hourly.rolling("7D", min_periods=1).median().reindex(keys).to_numpy()
            source = pd.Series(hourly.index, index=hourly.index).where(hourly.notna()).ffill()
            same_sources.loc[mask] = source.reindex(keys).array
        f[c + "_available_hour"] = same
        f[c + "_same_hour_mean_7d"] = means
        f[c + "_same_hour_median_7d"] = medians
        f[c + "_SameHourSourceMaxTimestamp"] = pd.to_datetime(same_sources, utc=True)
        f[c + "_available_hour_age"] = (targets - pd.DatetimeIndex(pd.to_datetime(same_sources, utc=True))).total_seconds()/3600
    assert_weather_features(f)
    return f


def assert_weather_features(frame):
    if frame.duplicated(["Weather_ID", "TargetWeatherTimestamp"]).any():
        raise AssertionError("Duplicate station-target key")
    for c in ["HistorySourceMaxTimestamp"] + [c for c in frame if c.endswith("SameHourSourceMaxTimestamp")]:
        if (frame[c].notna() & (frame[c] > frame.WeatherHistoryCutoffTimestamp)).any():
            raise AssertionError(f"Future weather feature: {c}")
    if (frame.ForecastIssueTimestamp >= frame.TargetWeatherTimestamp).any():
        raise AssertionError("Forecast issue does not precede target")


def input_columns(frame):
    return ["StationCode", "TargetUTCHour", "TargetLocalHour", "TargetMonth", "TargetDayOfYear",
            "ForecastLeadHours", "HourSin", "HourCos", "YearSin", "YearCos"] + [
                c for c in frame if any(c.startswith(v + "_") for v in VARIABLES)
                and not c.endswith("SourceMaxTimestamp")]


def evaluation_labels(stations, frame):
    """Labels are deliberately constructed through a separate API, not feature joins."""
    labels = frame[["Weather_ID", "TargetWeatherTimestamp"]].copy()
    for c in VARIABLES:
        labels[c] = np.nan
    for station, indices in frame.groupby("Weather_ID").groups.items():
        targets = pd.DatetimeIndex(frame.loc[indices, "TargetWeatherTimestamp"])
        for c in VARIABLES:
            labels.loc[indices, c] = stations[station][c].reindex(targets).to_numpy()
    return labels


class WeatherRecipe:
    """One pooled station-aware HGB per variable, plus training-only climatology."""
    def __init__(self, columns):
        self.columns = columns
        self.models, self.climate, self.hours, self.global_means, self.dark = {}, {}, {}, {}, {}
        self.supported_stations = {}
        self.fit_source_max = pd.NaT

    def fit(self, frame, labels, variable_subset=None):
        self.fit_source_max = frame.TargetWeatherTimestamp.max()
        for c in (variable_subset or VARIABLES):
            good = labels[c].notna()
            if good.sum() < 100:
                raise ValueError(f"Insufficient fit labels for {c}")
            table = frame.loc[good, ["Weather_ID", "TargetMonth", "TargetUTCHour"]].copy()
            table["value"] = labels.loc[good, c].to_numpy()
            self.supported_stations[c] = set(table.Weather_ID)
            self.climate[c] = table.groupby(["Weather_ID", "TargetMonth", "TargetUTCHour"]).value.mean()
            self.hours[c] = table.groupby(["Weather_ID", "TargetUTCHour"]).value.mean()
            self.global_means[c] = table.groupby("Weather_ID").value.mean()
            if c == "Sunshine_duration_hourly":
                grouped = table.groupby(["Weather_ID", "TargetMonth", "TargetUTCHour"]).value
                self.dark[c] = (grouped.max().le(0) & grouped.count().ge(20))
            model = HistGradientBoostingRegressor(categorical_features=[0], **MODEL_PARAMETERS)
            model.fit(frame.loc[good, self.columns], labels.loc[good, c])
            self.models[c] = model
        return self

    def climatology(self, frame, c):
        keys = pd.MultiIndex.from_frame(frame[["Weather_ID", "TargetMonth", "TargetUTCHour"]])
        y = pd.Series(self.climate[c].reindex(keys).to_numpy(), index=frame.index)
        hk = pd.MultiIndex.from_frame(frame[["Weather_ID", "TargetUTCHour"]])
        y = y.fillna(pd.Series(self.hours[c].reindex(hk).to_numpy(), index=frame.index))
        return y.fillna(frame.Weather_ID.map(self.global_means[c])).to_numpy()

    def postprocess(self, predictions, frame, c):
        y = np.asarray(predictions).copy()
        known = frame.Weather_ID.isin(self.supported_stations[c]).to_numpy()
        y[~known] = np.nan
        if c == "Humidity_avg_hourly":
            y = np.clip(y, 0, 100)
        elif c == "Sunshine_duration_hourly":
            y = np.clip(y, 0, 1)
            keys = pd.MultiIndex.from_frame(frame[["Weather_ID", "TargetMonth", "TargetUTCHour"]])
            darkness = self.dark[c].reindex(keys).astype("boolean").fillna(False).to_numpy(dtype=bool)
            y[darkness & known] = 0
        return y

    def candidates(self, frame, c):
        raw = {"available_same_hour": frame[c+"_available_hour"].to_numpy(),
               "previous_week": frame[c+"_week"].to_numpy(),
               "same_hour_7d_mean": frame[c+"_same_hour_mean_7d"].to_numpy(),
               "same_hour_7d_median": frame[c+"_same_hour_median_7d"].to_numpy(),
               "climatology": self.climatology(frame, c),
               "hist_gradient_boosting": self.models[c].predict(frame[self.columns])}
        return {name: self.postprocess(values, frame, c) for name, values in raw.items()}


def metrics(actual, predicted):
    a, p = np.asarray(actual), np.asarray(predicted)
    good = np.isfinite(a) & np.isfinite(p)
    error = p[good] - a[good]
    return {"n": int(good.sum()), "actual_n": int(np.isfinite(a).sum()),
            "prediction_coverage": float(good.sum()/np.isfinite(a).sum()) if np.isfinite(a).any() else 0,
            "MAE": float(np.abs(error).mean()) if len(error) else None,
            "RMSE": float(np.sqrt(np.mean(error**2))) if len(error) else None,
            "Bias": float(error.mean()) if len(error) else None}


def season_of(timestamp):
    month = timestamp.dt.month
    return month.map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
                      6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})


def deployable_table(frame, predicted, split_name, recipe, selected):
    d = frame[["ForecastIssueTimestamp", "DeliveryDate", "TargetWeatherTimestamp", "Weather_ID",
               "ForecastLeadHours", "WeatherHistoryCutoffTimestamp", "HistorySourceMaxTimestamp"]].copy()
    d["Period"] = split_name
    d["WeatherModelFitSourceMaxTimestamp"] = recipe.fit_source_max if recipe else pd.NaT
    d["WeatherModelFitSourceMaxTimestamp"] = pd.to_datetime(d.WeatherModelFitSourceMaxTimestamp, utc=True)
    for c, output in VARIABLES.items():
        d[output] = predicted[c]
        d[output + "Missing"] = ~np.isfinite(predicted[c])
        d[output + "Method"] = selected[c]
    if recipe and (d.WeatherModelFitSourceMaxTimestamp > d.WeatherHistoryCutoffTimestamp).any():
        raise AssertionError("Model trained with weather not yet available at forecast issue")
    return d
