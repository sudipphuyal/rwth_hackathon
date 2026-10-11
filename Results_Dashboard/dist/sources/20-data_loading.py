from pathlib import Path
import json
import numpy as np
import pandas as pd

METER_COLUMNS = ["Household_ID", "Group", "AffectsTimePoint", "Timestamp",
                 "kWh_received_Total", "kWh_received_HeatPump", "kWh_received_Other"]
WEATHER_COLUMNS = ["Temperature_avg_hourly", "Humidity_avg_hourly",
                   "Precipitation_total_hourly", "Sunshine_duration_hourly",
                   "WindSpeed_hourly", "DewPoint_hourly"]


def read_meter(path: Path) -> pd.DataFrame:
    """Read by name. Raw files are never written to."""
    d = pd.read_csv(path, sep=";", usecols=METER_COLUMNS)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], utc=True, errors="raise")
    d["Household_ID"] = d["Household_ID"].astype(str)
    if set(d.Household_ID) != {path.stem}:
        raise ValueError(f"Household/file mismatch: {path}")
    if d.duplicated(["Household_ID", "Timestamp"]).any():
        raise ValueError(f"Duplicate meter keys: {path}")
    if ((d.Timestamp.dt.minute % 15 != 0) | (d.Timestamp.dt.second != 0)).any():
        raise ValueError(f"Off-grid meter timestamp: {path}")
    for c in METER_COLUMNS[-3:]:
        d[c] = pd.to_numeric(d[c], errors="raise")
        if np.isinf(d[c]).any():
            raise ValueError(f"Infinite meter value: {path}, {c}")
    return d.sort_values("Timestamp").set_index("Timestamp", drop=False)


def prepare_target(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    total, hp, other = (d[c] for c in METER_COLUMNS[-3:])
    complete = total.notna() & hp.notna() & other.notna()
    if not np.allclose(total[complete], (hp + other)[complete], atol=1e-6, rtol=0):
        raise ValueError("Total/component identity fails on complete rows")
    reconstruct = (total.isna() & hp.notna() & other.notna() & np.isfinite(hp)
                   & np.isfinite(other) & (hp >= 0) & (other >= 0))
    d["Target"] = total.where(~reconstruct, hp + other)
    d["TargetSource"] = np.select([total.notna(), reconstruct],
                                  ["measured", "reconstructed"], default="missing")
    d["target_source"] = d.TargetSource  # Explicit requested provenance flag.
    d["TargetMissing"] = d.Target.isna()
    d["TargetNegative"] = d.Target.lt(0)
    # A review flag, not an exclusion rule; 20 kWh/15min corresponds to 80 kW.
    d["TargetExtremeReview"] = d.Target.gt(20)
    return d


def load_registry(data_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(data_dir / "smart_meter_meta_data/households.csv", sep=";",
                    dtype={"Household_ID": str, "Weather_ID": str})
    if d.Household_ID.duplicated().any():
        raise ValueError("Duplicate household registry keys")
    actual = {p.stem for p in (data_dir / "15min").glob("*.csv")}
    if actual != set(d.Household_ID):
        raise ValueError("Registry and household files differ")
    return d


def load_weather(data_dir: Path) -> dict[str, pd.DataFrame]:
    weather = {}
    for p in sorted((data_dir / "weather_data_hourly").glob("*.csv")):
        d = pd.read_csv(p, sep=";", usecols=["Weather_ID", "Timestamp"] + WEATHER_COLUMNS,
                        dtype={"Weather_ID": str})
        d.Timestamp = pd.to_datetime(d.Timestamp, utc=True)
        if d.Timestamp.duplicated().any() or set(d.Weather_ID) != {p.stem}:
            raise ValueError(f"Invalid weather keys: {p}")
        weather[p.stem] = d.sort_values("Timestamp").set_index("Timestamp", drop=False)
    return weather


def iter_modelling_tables(output_dir: Path, include_labels: bool = False):
    """Stream an explicit feature whitelist; oracle weather is never loaded here.

    Label inclusion is opt-in for later training/evaluation preparation. No synthetic
    labels are created and users must mask TargetSource == 'missing' when fitting.
    """
    import pyarrow.parquet as pq
    contract = json.loads((output_dir / "feature_contract.json").read_text())
    columns = contract["identifiers"] + contract["candidate_features"]
    if include_labels:
        columns += contract["labels_never_inputs"]
    columns = list(dict.fromkeys(columns))
    for path in sorted((output_dir / "modelling").glob("household=*/part.parquet")):
        yield pq.ParquetFile(path).read(columns=columns).to_pandas()
