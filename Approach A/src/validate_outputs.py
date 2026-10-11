"""Validate generated partitions independently, plus real-household future poisoning."""
import argparse
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from .config import PipelineConfig
from .data_loading import load_registry, load_weather, prepare_target, read_meter
from .feature_availability import LABEL_COLUMNS, assert_cutoff
from .preprocessing import cutoff_for_issue, issue_for_day, modelling_day, prepare_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preprocessing"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = PipelineConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    run = json.loads((args.output_dir / "audit/run_summary.json").read_text())
    # Validate against the configuration that actually produced these files.
    if run["config"] != config.serializable():
        raise AssertionError("Generated output configuration differs from current validation configuration")
    registry = load_registry(args.data_dir)
    metadata = pd.read_parquet(args.output_dir / "metadata.parquet")
    stations = load_weather(args.data_dir)
    counts = dict(measured=0, reconstructed=0, missing=0)
    households, rows, oracle_missing, future_placeholders = set(), 0, 0, 0
    feature_columns = None
    for p in sorted((args.output_dir / "modelling").glob("household=*/part.parquet")):
        # Read physical file columns without inferred hive partition aliases.
        f = pq.ParquetFile(p).read().to_pandas()
        assert_cutoff(f)
        households.update(f.Household_ID.unique())
        rows += len(f)
        if feature_columns is None:
            feature_columns = list(f.columns)
        elif feature_columns != list(f.columns):
            raise AssertionError("Partition schemas differ")
        if not f.groupby(["Household_ID", "ForecastIssueTimestamp"]).size().eq(96).all():
            raise AssertionError("Delivery day does not contain exactly 96 intervals")
        for c in f:
            if c.startswith("Forecast_") and not c.endswith("_missing"):
                if not f[c].isna().all():
                    raise AssertionError("Forecast placeholder populated without forecast source")
                future_placeholders += int(f[c].isna().sum())
        for k, n in f.TargetSource.value_counts().items():
            counts[k] += int(n)
        oracle = pq.ParquetFile(args.output_dir / "oracle_weather" / p.parent.name / p.name).read().to_pandas()
        if not f[["Household_ID", "TargetTimestamp", "ForecastIssueTimestamp"]].equals(
            oracle[["Household_ID", "TargetTimestamp", "ForecastIssueTimestamp"]]):
            raise AssertionError("Oracle/deployable target keys differ")
        station = stations[f.Weather_ID.iloc[0]]
        expected = station.reindex(pd.DatetimeIndex(f.TargetTimestamp).floor("h")).Temperature_avg_hourly.to_numpy()
        actual = oracle.OracleObserved_Temperature_avg_hourly.to_numpy()
        import numpy as np
        if not np.allclose(expected, actual, equal_nan=True):
            raise AssertionError("Oracle station/target-hour join incorrect")
        oracle_missing += int(oracle.OracleSunshineMissing.sum())
        local = f.TargetTimestamp.dt.tz_convert(config.behavioral_timezone)
        if not f.LocalQuarterHourSlot.eq(local.dt.hour * 4 + local.dt.minute // 15).all():
            raise AssertionError("Local calendar does not match target UTC conversion")
    if households != set(registry.Household_ID) or rows != run["materialized_rows"]:
        raise AssertionError("Portfolio lost households or rows")
    bool_columns = [c for c in metadata if c.startswith(("Survey_HeatDistribution", "Survey_DHW", "Survey_Installation"))
                    and not c.endswith("_missing")] + ["Installation_HasPVSystem"]
    for c in bool_columns:
        if not set(metadata[c]).issubset({"True", "False", "Unknown"}):
            raise AssertionError(f"Unknown category lost: {c}")
    # Three actual households: treatment, control/spike, and no-Total cold start.
    meta = prepare_metadata(registry, pd.read_csv(args.data_dir / "smart_meter_meta_data/meta_data.csv",
                                                sep=";", dtype={"Household_ID": str}))
    tests = []
    for household in ["661091", "676510", "996610"]:
        meter = prepare_target(read_meter(args.data_dir / "15min" / f"{household}.csv"))
        day = pd.Timestamp("2024-02-27", tz="UTC")
        issue = issue_for_day(day, config)
        station = stations[meta.loc[household].Weather_ID]
        baseline = modelling_day(meter, meta.loc[household], station, day, config)
        poison = meter.copy()
        future = poison.index > cutoff_for_issue(issue, config)
        poison.loc[future, "Target"] = 999999.
        poison.loc[future, "TargetSource"] = "measured"  # Consistent test labels; never exported.
        poison_weather = station.copy()
        wc = issue - pd.Timedelta(minutes=config.weather_interval_minutes + config.weather_latency_minutes)
        for c in ["Temperature_avg_hourly", "Humidity_avg_hourly", "Sunshine_duration_hourly"]:
            poison_weather.loc[poison_weather.index > wc, c] = 999999.
        result = modelling_day(poison, meta.loc[household], poison_weather, day, config)
        features = [c for c in baseline if c not in LABEL_COLUMNS]
        pd.testing.assert_frame_equal(baseline[features], result[features])
        tests.append({"Household_ID": household, "future_poisoning_changed_features": False})
    result = {"status": "passed", "partition_count": len(households), "validated_rows": rows,
              "selected_slice_target_counts": counts,
              "oracle_sunshine_missing_rows": oracle_missing,
              "missing_issued_weather_placeholder_cells": future_placeholders,
              "real_household_future_poison_tests": tests,
              "checks": ["cutoff_lineage", "target_provenance", "full_portfolio", "96_intervals_per_issue",
                         "uniform_schema", "oracle_join", "local_calendar", "unknown_metadata",
                         "issued_weather_not_fabricated", "future_poisoning"]}
    (args.output_dir / "audit/output_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
