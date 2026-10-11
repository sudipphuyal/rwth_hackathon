"""One-household-at-a-time, cutoff-safe day-ahead table construction."""
import numpy as np
import pandas as pd

from .data_loading import WEATHER_COLUMNS
from .feature_availability import assert_cutoff
from .portfolio import history_quality

META_NUMERIC = ["Survey_Building_LivingArea", "Survey_Building_Residents"]
META_CATEGORICAL = ["Survey_Building_Type", "Survey_HeatPump_Installation_Type",
    "Survey_HeatDistribution_System_FloorHeating", "Survey_HeatDistribution_System_Radiator",
    "Survey_DHW_Production_ByHeatPump", "Survey_DHW_Production_ByElectricWaterHeater",
    "Survey_DHW_Production_BySolar", "Survey_Installation_HasDryer",
    "Survey_Installation_HasFreezer", "Survey_Installation_HasElectricVehicle"]


def nullable_boolean(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"
    v = str(value).lower()
    if v in {"true", "1"}:
        return "True"
    if v in {"false", "0"}:
        return "False"
    raise ValueError(f"Unrecognised boolean: {value}")


def prepare_metadata(registry, survey):
    if survey.Household_ID.duplicated().any():
        raise ValueError("Duplicate survey keys")
    d = registry[["Household_ID", "Weather_ID", "Installation_HasPVSystem"]].merge(
        survey[["Household_ID"] + META_NUMERIC + META_CATEGORICAL],
        on="Household_ID", how="left", validate="one_to_one")
    for c in META_NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="raise")
        d[c + "_missing"] = d[c].isna()
    for c in META_CATEGORICAL + ["Installation_HasPVSystem"]:
        if c in ["Survey_Building_Type", "Survey_HeatPump_Installation_Type"]:
            d[c] = d[c].fillna("Unknown")
        else:
            d[c] = d[c].map(nullable_boolean)
        d[c + "_missing"] = d[c].eq("Unknown")
    # Keep both heat distribution flags; no invented exclusive category.
    d["MetadataSnapshotAssumedKnown"] = True
    return d.set_index("Household_ID", drop=False)


def issue_for_day(day, config):
    day = pd.Timestamp(day)
    date = day.date() - pd.Timedelta(days=1)
    # Late morning is unambiguous even on DST transition days.
    return pd.Timestamp(f"{date} {config.issue_hour_local:02d}:00:00",
                        tz=config.behavioral_timezone).tz_convert("UTC")


def cutoff_for_issue(issue, config):
    # Meter timestamps assumed to label interval START; conservative latency.
    return issue - pd.Timedelta(minutes=config.meter_interval_minutes + config.meter_latency_minutes)


def calendar_features(targets, issue, config):
    local = targets.tz_convert(config.behavioral_timezone)
    minute = local.hour * 60 + local.minute
    doy = local.dayofyear
    year_length = np.where(local.is_leap_year, 366, 365)
    return pd.DataFrame({
        "TargetTimestamp": targets, "ForecastIssueTimestamp": issue,
        "ForecastCutoffTimestamp": cutoff_for_issue(issue, config),
        "LocalQuarterHourSlot": minute // 15, "LocalHour": local.hour,
        "LocalWeekday": local.weekday, "LocalWeekend": local.weekday >= 5,
        "LocalMonth": local.month, "LocalDayOfYear": doy,
        "HourSin": np.sin(2 * np.pi * minute / 1440),
        "HourCos": np.cos(2 * np.pi * minute / 1440),
        "DayOfYearSin": np.sin(2 * np.pi * (doy - 1) / year_length),
        "DayOfYearCos": np.cos(2 * np.pi * (doy - 1) / year_length),
        "LocalUTCOffsetMinutes": [int(t.utcoffset().total_seconds() / 60) for t in local],
        "ForecastLeadHours": (targets - issue).total_seconds() / 3600,
    })


def demand_features(series, targets, issue, config):
    cutoff = cutoff_for_issue(issue, config)
    history = series.loc[series.index <= cutoff]
    valid = history.dropna()
    features = history_quality(history, cutoff, config)
    # Every issue-level demand aggregate/quality field reads only this bounded history.
    features["DemandHistorySourceMaxTimestamp"] = history.index.max() if len(history) else pd.NaT
    features["HistoricalHouseholdBaseline"] = valid.mean()
    windows = {}
    for days in [7, 28]:
        w = valid.loc[valid.index > cutoff - pd.Timedelta(days=days)]
        windows[days] = w
        features[f"Recent{days}dMean"] = w.mean()
        features[f"Recent{days}dMedian"] = w.median()
    recent = windows[7]
    hours = recent.index.tz_convert(config.behavioral_timezone).hour
    for name, mask in {"Night": hours < 6, "Day": (hours >= 6) & (hours < 18),
                       "Evening": hours >= 18}.items():
        features[f"Recent7d{name}Mean"] = recent.loc[mask].mean()
        features[f"Recent7d{name}Count"] = int(mask.sum())
    features["Recent7dPeak"] = recent.max()
    features["Recent7dP95"] = recent.quantile(.95) if len(recent) else np.nan
    preceding = windows[28].loc[windows[28].index <= cutoff - pd.Timedelta(days=7)]
    base = preceding.mean()
    features["RecentChangeRatio"] = recent.mean() / base if pd.notna(base) and base > 0 else np.nan
    features["Recent7dZeroFraction"] = recent.eq(0).mean() if len(recent) else np.nan
    f = pd.DataFrame([features] * len(targets))
    f["TargetTimestamp"] = targets
    # Group by local calendar slot after imposing issue cutoff, never by row offsets.
    same_slot_history = valid.loc[valid.index > cutoff - pd.Timedelta(days=config.same_slot_lookback_days)]
    local = same_slot_history.index.tz_convert(config.behavioral_timezone)
    slot = local.hour * 4 + local.minute // 15
    past = pd.DataFrame({"value": same_slot_history.to_numpy(), "slot": slot,
                         "date": local.date, "utc": same_slot_history.index})
    grouped = {s: g for s, g in past.groupby("slot")}
    target_local = targets.tz_convert(config.behavioral_timezone)
    previous, lag_t, week, week_t, median, count, median_t = [], [], [], [], [], [], []
    for t in target_local:
        g = grouped.get(t.hour * 4 + t.minute // 15)
        if g is None or g.empty:
            previous.append(np.nan); lag_t.append(pd.NaT); week.append(np.nan)
            week_t.append(pd.NaT); median.append(np.nan); count.append(0); median_t.append(pd.NaT)
            continue
        # Most recent available matching wall-clock interval in the last 28 days.
        previous.append(g.value.iloc[-1]); lag_t.append(g.utc.iloc[-1])
        wanted = t.date() - pd.Timedelta(days=7)
        wg = g[g.date == wanted]
        # A duplicated DST wall-clock hour is aggregated by median, not arbitrarily joined.
        week.append(wg.value.median()); week_t.append(wg.utc.max())
        median.append(g.value.median()); count.append(len(g)); median_t.append(g.utc.max())
    f["PreviousAvailableSameInterval"] = previous
    f["LagAvailableSourceMaxTimestamp"] = pd.to_datetime(lag_t, utc=True)
    f["PreviousAvailableSameIntervalAgeHours"] = (
        pd.Series(targets) - f.LagAvailableSourceMaxTimestamp).dt.total_seconds() / 3600
    f["PreviousWeekSameLocalInterval"] = week
    f["LagWeekSourceMaxTimestamp"] = pd.to_datetime(week_t, utc=True)
    f["RecentSameSlot28dMedian"] = median
    f["RecentSameSlot28dCount"] = count
    f["SameSlotSourceMaxTimestamp"] = pd.to_datetime(median_t, utc=True)
    return f.drop(columns="TargetTimestamp")


def weather_features(station, targets, issue, config):
    # Observation timestamp assumed hourly interval start. One-hour latency is explicit.
    cutoff = issue - pd.Timedelta(minutes=config.weather_interval_minutes + config.weather_latency_minutes)
    h = station.loc[station.index <= cutoff]
    recent = h.loc[h.index > cutoff - pd.Timedelta(hours=24)]
    vals = {}
    vals["HistoricalWeatherCutoffTimestamp"] = cutoff
    for c in WEATHER_COLUMNS:
        # Rain/sunshine remain hourly totals; their 24h summary is additive.
        if c in ["Precipitation_total_hourly", "Sunshine_duration_hourly"]:
            vals["Historical24h_" + c] = recent[c].sum(min_count=1)
        else:
            vals["Historical24h_" + c] = recent[c].mean()
        vals["Historical24h_" + c + "_coverage"] = recent[c].count() / 24
        vals["Forecast_" + c] = np.nan  # Never populate with realised target-day weather.
        vals["Forecast_" + c + "_missing"] = True
    vals["HistoricalWeatherSourceMaxTimestamp"] = h.index.max() if len(h) else pd.NaT
    vals["HistoricalHeatingExposure24h"] = (config.heating_base_c - recent.Temperature_avg_hourly).clip(lower=0).mean()
    vals["ForecastHeatingExposure"] = np.nan
    vals["HeatingExposureBaseC"] = config.heating_base_c
    # Absence determined from bounded station history, not future station availability.
    vals["sunshine_missing"] = bool(h.Sunshine_duration_hourly.notna().sum() == 0)
    return pd.DataFrame([vals] * len(targets))


def oracle_weather(station, targets, household_id, issue):
    """Quarantined realised target-hour covariates. Not imported by default modelling code."""
    aligned = station.reindex(targets.floor("h"))[WEATHER_COLUMNS].reset_index(drop=True)
    aligned = aligned.add_prefix("OracleObserved_")
    aligned.insert(0, "Household_ID", household_id)
    aligned.insert(1, "TargetTimestamp", targets)
    aligned.insert(2, "ForecastIssueTimestamp", issue)
    aligned["OracleWeatherObservationTimestamp"] = targets.floor("h")
    aligned["OracleSunshineMissing"] = aligned.OracleObserved_Sunshine_duration_hourly.isna()
    # Hourly totals repeated as HOURLY covariates, not quarter-hour energy/weather totals.
    return aligned


def modelling_day(meter, metadata, station, day, config):
    day = pd.Timestamp(day)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    targets = pd.date_range(day.normalize(), periods=96, freq="15min")
    issue = issue_for_day(day, config)
    f = calendar_features(targets, issue, config)
    f.insert(0, "Household_ID", metadata.Household_ID)
    demand = demand_features(meter.Target, targets, issue, config)
    weather = weather_features(station, targets, issue, config)
    f = pd.concat([f, demand, weather], axis=1)
    for c, value in metadata.items():
        if c != "Household_ID":
            f[c] = value
    # Labels are attached only after all historical features have been produced.
    labels = meter.reindex(targets)
    f["Target"] = labels.Target.to_numpy()
    f["TargetSource"] = labels.TargetSource.fillna("missing").to_numpy()
    f["TargetMissing"] = f.Target.isna()
    f["TargetNegative"] = f.Target.lt(0)
    f["TargetExtremeReview"] = f.Target.gt(20)
    # Feature schema must not depend on which household/day happens to be processed.
    optional_numeric = ["HistoricalHouseholdBaseline", "HistoryMissingRate", "GapAgeHours",
        "Recent7dMean", "Recent7dMedian", "Recent28dMean", "Recent28dMedian",
        "Recent7dNightMean", "Recent7dDayMean", "Recent7dEveningMean", "Recent7dPeak",
        "Recent7dP95", "RecentChangeRatio", "Recent7dZeroFraction",
        "PreviousAvailableSameInterval", "PreviousAvailableSameIntervalAgeHours",
        "PreviousWeekSameLocalInterval", "RecentSameSlot28dMedian",
        "HistoricalHeatingExposure24h", "ForecastHeatingExposure"]
    optional_numeric += ["Historical24h_" + c for c in WEATHER_COLUMNS]
    for c in optional_numeric:
        f[c + "_missing"] = f[c].isna()
    # Nullable lineage is always UTC, even for a household with no history at all.
    for c in f:
        if c.endswith("SourceMaxTimestamp"):
            f[c] = pd.to_datetime(f[c], utc=True)
    assert_cutoff(f)
    return f
