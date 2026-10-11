"""Join station forecasts to EXISTING energy feature rows; no demand modelling."""
import pandas as pd

PREDICTED_COLUMNS = ["PredictedTemperature", "PredictedSunshine", "PredictedHumidity",
                     "PredictedTemperatureMissing", "PredictedSunshineMissing", "PredictedHumidityMissing"]


def attach_predicted_weather(energy_rows: pd.DataFrame, forecasts: pd.DataFrame):
    """Return a new frame. Do not mutate/rebuild Step 2 outputs or import actual weather.

    Exact issue + assigned station + floor-hour keys; repeated hourly sunshine
    retains hours-per-hour units (not a quarter-hour total). Unmatched keys fail
    so coverage cannot silently degrade to zeros.
    """
    keys = ["ForecastIssueTimestamp", "Weather_ID", "TargetWeatherTimestamp"]
    if forecasts.duplicated(keys).any():
        raise ValueError("Duplicate issued station weather forecast")
    allowed = keys + PREDICTED_COLUMNS + ["WeatherHistoryCutoffTimestamp", "WeatherModelFitSourceMaxTimestamp"]
    right = forecasts[allowed].rename(columns={"WeatherHistoryCutoffTimestamp":"PredictedWeatherHistoryCutoffTimestamp"})
    left = energy_rows.copy()
    if set(PREDICTED_COLUMNS) & set(left):
        raise ValueError("Predicted weather already attached")
    left["TargetWeatherTimestamp"] = left.TargetTimestamp.dt.floor("h")
    result = left.merge(right, on=keys, how="left", validate="many_to_one", indicator="_weather_join")
    if result._weather_join.ne("both").any():
        raise ValueError("Missing issued weather keys for some household intervals")
    if (result.PredictedWeatherHistoryCutoffTimestamp > result.ForecastIssueTimestamp).any():
        raise ValueError("Predicted weather used future history")
    if (result.WeatherModelFitSourceMaxTimestamp.notna() &
        (result.WeatherModelFitSourceMaxTimestamp > result.PredictedWeatherHistoryCutoffTimestamp)).any():
        raise ValueError("Weather model fit includes labels unavailable at issue")
    return result.drop(columns="_weather_join")
