"""Explicit energy input whitelist and a small physical interaction set."""
import numpy as np
import pandas as pd

CALENDAR=["LocalQuarterHourSlot","LocalHour","LocalWeekday","LocalWeekend","LocalMonth",
          "LocalDayOfYear","HourSin","HourCos","DayOfYearSin","DayOfYearCos","ForecastLeadHours","LocalUTCOffsetMinutes"]
QUALITY=["HistoryObservedIntervals","HistoryExpectedIntervals","HistoryValidDays","HistoryMissingRate",
         "HistoryAgeDaysAtForecast","GapAgeHours","Recent7dCoverage","Recent28dCoverage"]
CORE=["PreviousAvailableSameInterval","PreviousAvailableSameIntervalAgeHours","PreviousWeekSameLocalInterval",
      "RecentSameSlot28dMedian","RecentSameSlot28dCount","Recent28dMean","Recent28dMedian","HistoricalHouseholdBaseline"]
BEHAVIOUR=["Recent7dMean","Recent7dMedian","Recent7dNightMean","Recent7dDayMean","Recent7dEveningMean",
           "Recent7dPeak","Recent7dP95","RecentChangeRatio","Recent7dZeroFraction"]
WEATHER=["PredictedTemperature","PredictedSunshine","PredictedHumidity","PredictedTemperatureMissing",
         "PredictedSunshineMissing","PredictedHumidityMissing","HeatingExposure","Weather_ID"]
BUILDING=["Survey_Building_LivingArea","Survey_Building_Residents","TemperatureLivingArea","HeatingExposureLivingArea"]
THERMAL=["Survey_HeatPump_Installation_Type","Survey_HeatDistribution_System_FloorHeating",
         "Survey_HeatDistribution_System_Radiator","Survey_DHW_Production_ByHeatPump",
         "Survey_DHW_Production_ByElectricWaterHeater","Survey_DHW_Production_BySolar",
         "TemperatureAirSource","TemperatureGroundSource","HeatingExposureAirSource","HeatingExposureGroundSource",
         "ResidentsHeatPumpDHW","ResidentsElectricDHW"]
PV=["Installation_HasPVSystem","SunshinePV","SunshinePVDaytime","SunshinePVHourSin"]
SECONDARY=["Survey_Installation_HasElectricVehicle","Survey_Installation_HasDryer",
           "Survey_Installation_HasFreezer","EVNightEvening"]
CATEGORIES={c:{"False":0,"True":1,"Unknown":2} for c in THERMAL+PV+SECONDARY if c.startswith(("Survey_","Installation_"))}
CATEGORIES["Survey_HeatPump_Installation_Type"]={"air-source":0,"ground-source":1,"Unknown":2}
CATEGORIES["Weather_ID"]={v:i for i,v in enumerate(["8jB","HbsbG","Hg","MqO","ceOxS","sV3mR","wDD","z6I"])}


def indicator(series,positive="True",negative="False"):
    return series.map({positive:1.,negative:0.})


def add_physical_features(d,base_c=15.):
    d=d.copy()
    heat=(base_c-d.PredictedTemperature).clip(lower=0)
    area=d.Survey_Building_LivingArea
    air=indicator(d.Survey_HeatPump_Installation_Type,"air-source","ground-source")
    ground=indicator(d.Survey_HeatPump_Installation_Type,"ground-source","air-source")
    pv=indicator(d.Installation_HasPVSystem)
    d["HeatingExposure"]=heat
    d["TemperatureLivingArea"]=d.PredictedTemperature*area
    d["HeatingExposureLivingArea"]=heat*area
    for name,flag in [("AirSource",air),("GroundSource",ground)]:
        d["Temperature"+name]=d.PredictedTemperature*flag
        d["HeatingExposure"+name]=heat*flag
    d["SunshinePV"]=d.PredictedSunshine*pv
    d["SunshinePVDaytime"]=d.SunshinePV*d.LocalHour.between(6,17)
    d["SunshinePVHourSin"]=d.SunshinePV*d.HourSin
    d["ResidentsHeatPumpDHW"]=d.Survey_Building_Residents*indicator(d.Survey_DHW_Production_ByHeatPump)
    d["ResidentsElectricDHW"]=d.Survey_Building_Residents*indicator(d.Survey_DHW_Production_ByElectricWaterHeater)
    d["EVNightEvening"]=indicator(d.Survey_Installation_HasElectricVehicle)*((d.LocalHour<6)|(d.LocalHour>=18))
    return d


def feature_groups(frame):
    groups={};current=CALENDAR+CORE+QUALITY
    additions={"A":[],"B":WEATHER,"C":BUILDING,"D":THERMAL,"E":PV,"F":SECONDARY,"G":BEHAVIOUR}
    for name,extra in additions.items():
        current=list(dict.fromkeys(current+extra))
        flags=[c+"_missing" for c in current if c+"_missing" in frame]
        groups[name]=list(dict.fromkeys(current+flags))
    return groups


def cold_features(frame):
    columns=list(dict.fromkeys(CALENDAR+WEATHER+BUILDING+THERMAL+PV+SECONDARY))
    return columns+[c+"_missing" for c in columns if c+"_missing" in frame]


def matrix(frame,columns):
    # Do not infer ordinal identity or ever include target/lineage columns.
    if any(c in columns for c in ["Household_ID","Target","TargetSource","TargetExtremeReview"]):
        raise ValueError("Label/household-ID requested as an energy input")
    out=np.empty((len(frame),len(columns)),dtype=np.float32)
    for j,c in enumerate(columns):
        if c in CATEGORIES:
            encoded=frame[c].map(CATEGORIES[c])
            if encoded.isna().any():raise ValueError(f"Unrecognised category in {c}")
            out[:,j]=encoded.to_numpy(dtype=np.float32)
        else:
            out[:,j]=frame[c].to_numpy(dtype=np.float32)
    if np.isinf(out).any():raise ValueError("Infinite energy feature")
    return out


def categorical_indices(columns):
    return [i for i,c in enumerate(columns) if c in CATEGORIES]


def deterministic_fit_sample(d):
    """One quarter-hour in alternating hours; cycles across all 96 slots in 8 days.

    Computational thinning within TRAIN only, not a random temporal split.
    Every flagged extreme is forced into the fitted sample unchanged.
    """
    ordinal=d.TargetTimestamp.dt.floor("D").astype("int64")//86_400_000_000_000
    cycle=ordinal%8
    keep=(d.TargetTimestamp.dt.hour%2==cycle//4)&(d.TargetTimestamp.dt.minute//15==cycle%4)
    return keep|d.TargetExtremeReview
