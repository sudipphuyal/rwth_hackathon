from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs/preprocessing")
    # Locality is not documented. This is a benchmark assumption, not a discovery.
    behavioral_timezone: str = "Europe/Berlin"
    issue_hour_local: int = 11
    delivery_timezone: str = "UTC"  # Exactly 96 intervals, including DST dates.
    meter_interval_minutes: int = 15
    meter_latency_minutes: int = 15
    weather_interval_minutes: int = 60
    weather_latency_minutes: int = 60
    heating_base_c: float = 15.0  # Candidate exposure base, NOT official SIA/US HDD.
    same_slot_lookback_days: int = 28
    normal_min_valid_days: int = 28
    normal_min_recent_28_coverage: float = 0.8
    normal_max_gap_hours: float = 48.0
    # Undated metadata cannot establish genuine historical as-of availability.
    metadata_assumption: str = "fixed_snapshot_assumed_available_before_all_benchmark_issues"

    def __post_init__(self):
        if self.delivery_timezone != "UTC" or self.meter_interval_minutes != 15:
            raise ValueError("This pipeline's 96-interval delivery grid requires UTC and 15-minute intervals")
        if not 0 <= self.issue_hour_local <= 23:
            raise ValueError("Issue hour must be 0..23")
        if self.meter_latency_minutes < 0 or self.weather_latency_minutes < 0:
            raise ValueError("Reporting latency must be nonnegative")

    def serializable(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class ChronologicalSplit:
    """GLOBAL UTC delivery-date split, fixed before any weather model evaluation.

    Actual meter bounds: 2019-05-20..2024-02-27 (1745 days). At the user's
    explicit seasonal-coverage request, reserve the latest complete year
    for test, the preceding complete year for validation, and older dates
    for training. The earlier 70/10/20 experiment is preserved separately.
    Forecast-availability embargoes are applied INSIDE these periods, not
    by moving the split dates in response to model/test performance.
    """
    train_start: str = "2019-05-20"
    train_end: str = "2022-02-27"
    validation_start: str = "2022-02-28"
    validation_end: str = "2023-02-27"
    test_start: str = "2023-02-28"
    test_end: str = "2024-02-27"
    unseen_demand_day: str = "2024-02-28"


# Original outputs/weather_forecasting is retained; no test locks are deleted.
WEATHER_OUTPUT_DIR = Path("outputs/weather_forecasting_full_year")
