"""Explicit role and temporal lineage contract; labels and oracle weather are separate."""
import pandas as pd

IDENTIFIER_COLUMNS = ["Household_ID", "TargetTimestamp", "ForecastIssueTimestamp",
                      "ForecastCutoffTimestamp", "HistoricalWeatherCutoffTimestamp"]
LABEL_COLUMNS = ["Target", "TargetSource", "TargetMissing", "TargetNegative", "TargetExtremeReview"]


def assert_cutoff(frame: pd.DataFrame):
    for c in frame.columns:
        if c.endswith("SourceMaxTimestamp"):
            if (frame[c].notna() & (frame[c] > frame.ForecastIssueTimestamp)).any():
                raise AssertionError(f"Feature available after issue: {c}")
            if c.startswith("Demand") or c.startswith("Lag") or c.startswith("SameSlot"):
                if (frame[c].notna() & (frame[c] > frame.ForecastCutoffTimestamp)).any():
                    raise AssertionError(f"Demand after meter cutoff: {c}")
            if c.startswith("HistoricalWeather"):
                if (frame[c].notna() & (frame[c] > frame.HistoricalWeatherCutoffTimestamp)).any():
                    raise AssertionError("Weather observation after availability cutoff")
    if (frame.ForecastIssueTimestamp >= frame.TargetTimestamp).any():
        raise AssertionError("Issue must precede every target interval")
    if frame.duplicated(["Household_ID", "TargetTimestamp"]).any():
        raise AssertionError("Duplicate modelling target keys")
    if not frame.loc[frame.TargetSource.eq("missing"), "Target"].isna().all():
        raise AssertionError("Missing labels were fabricated")
    if frame.loc[frame.TargetSource.ne("missing"), "Target"].isna().any():
        raise AssertionError("Label provenance/value mismatch")


def make_contract(frame: pd.DataFrame) -> dict:
    lineage = [c for c in frame if c.endswith("SourceMaxTimestamp")]
    diagnostics = [c for c in frame if c.startswith("Metadata") or c in ["ForecastRoute", "HeatingExposureBaseC"]]
    pending = [c for c in frame if c.startswith("Forecast_") or c.startswith("ForecastHeatingExposure")]
    excluded = set(IDENTIFIER_COLUMNS + LABEL_COLUMNS + lineage + diagnostics + pending)
    features = [c for c in frame if c not in excluded]
    # Household_ID is a category/join key, never a numerical ordering.
    return {
        "identifiers": IDENTIFIER_COLUMNS,
        "categorical_household_key": "Household_ID",
        "labels_never_inputs": LABEL_COLUMNS,
        "lineage_never_inputs": lineage,
        "diagnostics_never_inputs": diagnostics,
        "reserved_forecast_weather_not_yet_available": pending,
        "candidate_features": features,
        "metadata_asof_assumption": "Fixed undated snapshot assumed known at every issue; not historically verified",
        "deployable_forecast_weather": "Missing placeholders; no issued forecast archive supplied",
        "oracle_weather": "Stored in separate directory; never joined by default",
    }
