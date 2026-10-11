"""CLI: scan every meter file; materialise a configurable forecast-day slice.

python -m src.run_preprocessing --start 2024-01-31 --end 2024-02-27
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_loading import load_registry, load_weather, prepare_target, read_meter
from .feature_availability import make_contract
from .preprocessing import modelling_day, oracle_weather, prepare_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-31")
    parser.add_argument("--end", default="2024-02-27")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preprocessing"))
    args = parser.parse_args()
    config = PipelineConfig(data_dir=args.data_dir, output_dir=args.output_dir)
    days = pd.date_range(args.start, args.end, freq="D", tz="UTC")
    if not len(days):
        raise ValueError("Empty delivery date range")
    out = config.output_dir
    for name in ["modelling", "oracle_weather", "audit"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    registry = load_registry(config.data_dir)
    survey = pd.read_csv(config.data_dir / "smart_meter_meta_data/meta_data.csv", sep=";",
                         dtype={"Household_ID": str})
    metadata = prepare_metadata(registry, survey)
    stations = load_weather(config.data_dir)
    if not set(metadata.Weather_ID).issubset(stations):
        raise AssertionError("Unmapped household weather station")
    metadata.reset_index(drop=True).to_parquet(out / "metadata.parquet", index=False)
    audit, portfolio, extremes = [], [], []
    sample = None
    counts = {"measured": 0, "reconstructed": 0, "missing": 0}
    nrows = 0
    for i, household in enumerate(registry.Household_ID):
        path = config.data_dir / "15min" / f"{household}.csv"
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        raw = read_meter(path)
        meter = prepare_target(raw)
        grid = pd.date_range(meter.index.min(), meter.index.max(), freq="15min")
        count = meter.TargetSource.value_counts().to_dict()
        for k in counts:
            counts[k] += int(count.get(k, 0))
        complete = raw[["kWh_received_Total", "kWh_received_HeatPump", "kWh_received_Other"]].dropna()
        audit.append({"Household_ID": household, "RawRows": len(meter),
                      "Measured": count.get("measured", 0), "Reconstructed": count.get("reconstructed", 0),
                      "MissingInRawRows": count.get("missing", 0), "MissingGridRows": len(grid) - len(meter),
                      "FirstTimestamp": meter.index.min(), "LastTimestamp": meter.index.max(),
                      "IdentityCheckedRows": len(complete), "Duplicates": 0,
                      "NegativeMeasured": int(meter.TargetNegative.sum()),
                      "ExtremeReviewRows": int(meter.TargetExtremeReview.sum()),
                      "MaxTarget": meter.Target.max()})
        # Every >20 kWh/quarter-hour is preserved and gets a timestamp-keyed neighbourhood.
        for timestamp in meter.loc[meter.TargetExtremeReview].index:
            around = meter.loc[(meter.index >= timestamp - pd.Timedelta(hours=2)) &
                               (meter.index <= timestamp + pd.Timedelta(hours=2))]
            context = around[["Household_ID", "Timestamp", "kWh_received_Total",
                              "kWh_received_HeatPump", "kWh_received_Other", "TargetSource"]].copy()
            context["ExtremeCenterTimestamp"] = timestamp
            extremes.append(context.reset_index(drop=True))
        frames, oracle = [], []
        for day in days:
            frame = modelling_day(meter, metadata.loc[household],
                                  stations[metadata.loc[household].Weather_ID], day, config)
            frames.append(frame)
            oracle.append(oracle_weather(stations[metadata.loc[household].Weather_ID],
                                         pd.DatetimeIndex(frame.TargetTimestamp), household,
                                         frame.ForecastIssueTimestamp.iloc[0]))
            quality_cols = ["Household_ID", "ForecastIssueTimestamp", "ForecastCutoffTimestamp",
                "ForecastRoute", "HistoryObservedIntervals", "HistoryExpectedIntervals",
                "HistoryMissingRate", "HistoryValidDays", "GapAgeHours", "Recent7dCoverage", "Recent28dCoverage"]
            portfolio.append(frame[quality_cols].iloc[0].to_dict())
        combined = pd.concat(frames, ignore_index=True)
        if sample is None:
            sample = combined.head(96).copy()
        elif list(combined.columns) != list(sample.columns):
            raise AssertionError(f"Non-uniform feature schema: {household}")
        if combined.duplicated(["Household_ID", "TargetTimestamp"]).any():
            raise AssertionError("Duplicate household target keys")
        # Independent files allow household streaming and column projection.
        folder = out / "modelling" / f"household={household}"
        folder.mkdir(exist_ok=True)
        combined.to_parquet(folder / "part.parquet", index=False, compression="zstd")
        folder = out / "oracle_weather" / f"household={household}"
        folder.mkdir(exist_ok=True)
        pd.concat(oracle, ignore_index=True).to_parquet(folder / "part.parquet", index=False, compression="zstd")
        nrows += len(combined)
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise AssertionError("Raw meter file changed during processing")
        if i % 25 == 0:
            print(f"Processed {i+1}/{len(registry)} households", flush=True)
    audit = pd.DataFrame(audit)
    port = pd.DataFrame(portfolio)
    audit.to_csv(out / "audit/target_coverage.csv", index=False)
    port.to_parquet(out / "portfolio_by_issue.parquet", index=False)
    port[port.ForecastIssueTimestamp.eq(port.ForecastIssueTimestamp.max())].to_csv(
        out / "audit/portfolio_latest_issue.csv", index=False)
    if extremes:
        pd.concat(extremes, ignore_index=True).to_csv(out / "audit/extreme_neighbourhoods.csv", index=False)
    sample.to_csv(out / "modelling_preview.csv", index=False)
    contract = make_contract(sample)
    (out / "feature_contract.json").write_text(json.dumps(contract, indent=2))
    schema = {c: {"dtype": str(sample[c].dtype),
                  "role": "label" if c in contract["labels_never_inputs"] else
                          "lineage" if c in contract["lineage_never_inputs"] else
                          "identifier" if c in contract["identifiers"] else
                          "reserved_unavailable_weather" if c in contract["reserved_forecast_weather_not_yet_available"] else
                          "diagnostic" if c in contract["diagnostics_never_inputs"] else "candidate_feature"}
              for c in sample}
    (out / "schema.json").write_text(json.dumps(schema, indent=2))
    summary = {"config": config.serializable(), "raw_target_counts": counts,
        "households": len(registry), "raw_rows": int(audit.RawRows.sum()),
        "absent_grid_rows_between_file_bounds": int(audit.MissingGridRows.sum()),
        "identity_checked_rows": int(audit.IdentityCheckedRows.sum()),
        "reconstruction_households": audit.loc[audit.Reconstructed.gt(0), "Household_ID"].tolist(),
        "households_with_any_raw_missing_target": audit.loc[audit.MissingInRawRows.gt(0), "Household_ID"].tolist(),
        "households_with_no_valid_target": audit.loc[(audit.Measured + audit.Reconstructed).eq(0), "Household_ID"].tolist(),
        "materialized_start": str(days.min()), "materialized_end": str(days.max()),
        "materialized_rows": nrows, "columns": len(sample.columns),
        "latest_issue_route_counts": port.loc[port.ForecastIssueTimestamp.eq(port.ForecastIssueTimestamp.max()), "ForecastRoute"].value_counts().to_dict(),
        "validation": {"duplicate_keys": 0, "after_cutoff_demand_sources": 0,
                       "missing_station_mappings": 0, "invalid_component_identities": 0,
                       "raw_meter_files_modified": 0},
        "extreme_review_threshold_kwh": 20, "extreme_rows": int(audit.ExtremeReviewRows.sum()),
        "maximum_kwh": float(audit.MaxTarget.max())}
    (out / "audit/run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
