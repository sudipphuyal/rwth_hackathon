import pandas as pd


def history_quality(history: pd.Series, cutoff: pd.Timestamp, config) -> dict:
    """Denominator uses elapsed grid since first known source row, never future file end."""
    if history.empty:
        return dict(HistoryObservedIntervals=0, HistoryExpectedIntervals=0,
                    HistoryMissingRate=float("nan"), HistoryValidDays=0,
                    GapAgeHours=float("nan"), Recent7dCoverage=0.0,
                    Recent28dCoverage=0.0, ForecastRoute="cold_start")
    expected = len(pd.date_range(history.index.min(), cutoff, freq="15min"))
    valid = history.dropna()
    daily = valid.groupby(valid.index.floor("D")).count()
    # Complete UTC days only. The partial cutoff day never counts as a valid day.
    valid_days = int(((daily >= 90) & (daily.index + pd.Timedelta(days=1) <= cutoff)).sum())
    age = (cutoff - valid.index.max()).total_seconds() / 3600 if len(valid) else float("nan")
    cover = {}
    for days in [7, 28]:
        n = int(valid.loc[(valid.index > cutoff - pd.Timedelta(days=days)) & (valid.index <= cutoff)].count())
        cover[days] = n / (days * 96)
    route = "cold_start" if not len(valid) else "sparse_history"
    if (valid_days >= config.normal_min_valid_days
            and cover[28] >= config.normal_min_recent_28_coverage
            and age <= config.normal_max_gap_hours):
        route = "normal"
    return dict(HistoryObservedIntervals=len(valid), HistoryExpectedIntervals=expected,
                HistoryMissingRate=1 - len(valid) / expected if expected else float("nan"),
                HistoryValidDays=valid_days, GapAgeHours=age,
                Recent7dCoverage=cover[7], Recent28dCoverage=cover[28], ForecastRoute=route)
