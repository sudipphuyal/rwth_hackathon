"""Batched execution of the existing Step 2 demand/calendar feature definitions.

The reference implementation remains preprocessing.demand_features. This backend
uses the same windows, slot/date joins and cutoff convention, avoiding repeated
Python scans of full household history for each day. Equivalence is tested.
"""
import numpy as np
import pandas as pd

from .preprocessing import calendar_features, issue_for_day, cutoff_for_issue

NS_DAY = 86_400_000_000_000


def batch_household_features(meter, metadata, days, config):
    issues_day = pd.DatetimeIndex([issue_for_day(d, config) for d in days])
    cutoffs_day = cutoff_for_issue(issues_day, config)
    first = meter.index.min()
    active = cutoffs_day >= first
    days, issues_day, cutoffs_day = days[active], issues_day[active], cutoffs_day[active]
    if not len(days):
        return pd.DataFrame()
    targets = pd.date_range(days.min(), days.max()+pd.Timedelta(days=1), freq="15min", inclusive="left")
    issues, cutoffs = issues_day.repeat(96), cutoffs_day.repeat(96)
    f = calendar_features(targets, issues, config)
    f.insert(0, "Household_ID", metadata.Household_ID)
    dense_idx = pd.date_range(first, max(meter.index.max(), targets.max()), freq="15min")
    y = meter.Target.reindex(dense_idx)
    daily = pd.DataFrame(index=cutoffs_day)
    def at_cutoff(s):
        return s.reindex(cutoffs_day).to_numpy()
    count = y.notna().astype(int).cumsum()
    daily["HistoryObservedIntervals"] = at_cutoff(count)
    daily["HistoryExpectedIntervals"] = ((cutoffs_day-first).total_seconds()/900+1).astype(int)
    daily["HistoryMissingRate"] = 1-daily.HistoryObservedIntervals/daily.HistoryExpectedIntervals
    daily["HistoricalHouseholdBaseline"] = at_cutoff(y.fillna(0).cumsum()/count.replace(0,np.nan))
    valid_source = pd.Series(dense_idx,index=dense_idx).where(y.notna()).ffill()
    latest = pd.DatetimeIndex(valid_source.reindex(cutoffs_day))
    daily["GapAgeHours"] = (cutoffs_day-latest).total_seconds()/3600
    good_days = y.resample("D").count().ge(90).astype(int).cumsum()
    good_days.index += pd.Timedelta(days=1)
    daily["HistoryValidDays"] = good_days.reindex(cutoffs_day,method="ffill").fillna(0).to_numpy(dtype=int)
    rolling = {}
    for length in [7,28]:
        w = y.rolling(f"{length}D",min_periods=1)
        rolling[length] = w
        daily[f"Recent{length}dMean"] = at_cutoff(w.mean())
        daily[f"Recent{length}dMedian"] = at_cutoff(w.median())
        daily[f"Recent{length}dCoverage"] = at_cutoff(w.count())/(length*96)
    local = dense_idx.tz_convert(config.behavioral_timezone)
    hours = local.hour
    for name,mask in [("Night",hours<6),("Day",(hours>=6)&(hours<18)),("Evening",hours>=18)]:
        w=y.where(mask).rolling("7D",min_periods=1)
        daily[f"Recent7d{name}Mean"] = at_cutoff(w.mean())
        daily[f"Recent7d{name}Count"] = np.nan_to_num(at_cutoff(w.count())).astype(int)
    daily["Recent7dPeak"] = at_cutoff(rolling[7].max())
    daily["Recent7dP95"] = at_cutoff(rolling[7].quantile(.95))
    sum7, n7 = at_cutoff(rolling[7].sum()), at_cutoff(rolling[7].count())
    sum28, n28 = at_cutoff(rolling[28].sum()), at_cutoff(rolling[28].count())
    prior=np.divide(np.nan_to_num(sum28)-np.nan_to_num(sum7),n28-n7,
                    out=np.full(len(days),np.nan),where=(n28-n7)>0)
    daily["RecentChangeRatio"] = np.divide(daily.Recent7dMean,prior,
                       out=np.full(len(days),np.nan),where=prior>0)
    zeros=y.eq(0).astype(float).where(y.notna())
    daily["Recent7dZeroFraction"] = at_cutoff(zeros.rolling("7D",min_periods=1).mean())
    raw_sources=pd.Series(meter.index,index=meter.index).reindex(dense_idx).ffill()
    daily["DemandHistorySourceMaxTimestamp"] = raw_sources.reindex(cutoffs_day).array
    established=(daily.HistoryValidDays>=config.normal_min_valid_days)&(daily.Recent28dCoverage>=config.normal_min_recent_28_coverage)&(daily.GapAgeHours<=config.normal_max_gap_hours)
    daily["ForecastRoute"] = np.where(daily.HistoryObservedIntervals.eq(0),"cold_start",np.where(established,"normal","sparse_history"))
    # Virtual cutoff rows ensure rolling medians end at the actual cutoff, even
    # across long gaps and DST repeated/absent wall-clock slots.
    slot=local.hour*4+local.minute//15
    lookup={name:np.full((len(days),96),np.nan) for name in ["value","median","count"]}
    src=np.full((len(days),96),np.datetime64("NaT"),dtype="datetime64[ns]")
    for s in range(96):
        ys=y.loc[slot==s]
        expanded=ys.reindex(ys.index.union(cutoffs_day)).sort_index()
        source=pd.Series(expanded.index,index=expanded.index).where(expanded.notna()).ffill()
        recent=expanded.rolling(f"{config.same_slot_lookback_days}D",min_periods=1)
        last=source.reindex(cutoffs_day)
        usable=last.notna()&(last>cutoffs_day-pd.Timedelta(days=config.same_slot_lookback_days))
        lookup["value"][:,s]=expanded.ffill().reindex(cutoffs_day).where(usable).to_numpy()
        lookup["median"][:,s]=recent.median().reindex(cutoffs_day).to_numpy()
        lookup["count"][:,s]=np.nan_to_num(recent.count().reindex(cutoffs_day).to_numpy())
        src[:,s]=last.where(usable).to_numpy(dtype="datetime64[ns]")
    for c in daily:
        f[c]=daily[c].repeat(96).reset_index(drop=True)
    row_day=np.repeat(np.arange(len(days)),96)
    target_slots=f.LocalQuarterHourSlot.to_numpy(dtype=int)
    f["PreviousAvailableSameInterval"] = lookup["value"][row_day,target_slots]
    f["RecentSameSlot28dMedian"] = lookup["median"][row_day,target_slots]
    f["RecentSameSlot28dCount"] = lookup["count"][row_day,target_slots].astype(int)
    f["LagAvailableSourceMaxTimestamp"] = pd.to_datetime(src[row_day,target_slots],utc=True)
    f["SameSlotSourceMaxTimestamp"] = f.LagAvailableSourceMaxTimestamp
    f["PreviousAvailableSameIntervalAgeHours"] = (f.TargetTimestamp-f.LagAvailableSourceMaxTimestamp).dt.total_seconds()/3600
    ordinal=local.tz_localize(None).normalize().asi8//NS_DAY
    valid=y.notna()
    keyed=pd.DataFrame({"key":ordinal[valid]*96+slot[valid],"value":y.loc[valid].to_numpy(),
                       "source":dense_idx[valid]})
    week=keyed.groupby("key").agg(value=("value","median"),source=("source","max"))
    target_local=targets.tz_convert(config.behavioral_timezone)
    wanted=(target_local.tz_localize(None).normalize().asi8//NS_DAY-7)*96+target_slots
    found=week.reindex(wanted)
    f["PreviousWeekSameLocalInterval"]=found.value.to_numpy()
    f["LagWeekSourceMaxTimestamp"]=pd.to_datetime(found.source.to_numpy(),utc=True)
    for c,value in metadata.items():
        if c!="Household_ID":f[c]=value
    labels=meter.reindex(targets)
    f["Target"]=labels.Target.to_numpy()
    f["TargetSource"]=labels.TargetSource.fillna("missing").to_numpy()
    f["TargetMissing"]=f.Target.isna()
    f["TargetNegative"]=f.Target.lt(0)
    f["TargetExtremeReview"]=f.Target.gt(20)
    f["FirstObservedTimestamp"]=first
    f["HistoryAgeDaysAtForecast"]=(f.ForecastIssueTimestamp-first).dt.total_seconds()/86400
    f["ActiveAtIssue"]=True
    f["ForecastState"]=f.ForecastRoute.map({"normal":"established","sparse_history":"sparse/new","cold_start":"cold-start"})
    numeric_missing=[c for c in f if pd.api.types.is_numeric_dtype(f[c]) and not pd.api.types.is_bool_dtype(f[c])
                     and c not in ["Target"] and not c.endswith("_missing")]
    flags={c+"_missing":f[c].isna() for c in numeric_missing if c+"_missing" not in f}
    f=pd.concat([f,pd.DataFrame(flags,index=f.index)],axis=1)
    for c in f:
        if c.endswith("SourceMaxTimestamp") and (f[c].notna()&(f[c]>f.ForecastCutoffTimestamp)).any():
            raise AssertionError(f"Demand feature after cutoff: {c}")
    return f
