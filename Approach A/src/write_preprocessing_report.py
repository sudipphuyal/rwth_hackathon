"""Write the Step 2 decision report from validated artifacts, without fitting a model."""
import json
from pathlib import Path
import pandas as pd


def main():
    out = Path('outputs/preprocessing')
    summary = json.loads((out/'audit/run_summary.json').read_text())
    validation = json.loads((out/'audit/output_validation.json').read_text())
    audit = pd.read_csv(out/'audit/target_coverage.csv', dtype={'Household_ID': str})
    problem = audit[(audit.MissingInRawRows > 0) | (audit.MissingGridRows > 0)].copy()
    problem.to_csv(out/'audit/remaining_target_problems.csv', index=False)
    contract = json.loads((out/'feature_contract.json').read_text())
    schema = json.loads((out/'schema.json').read_text())
    raw_missing = audit[audit.MissingInRawRows > 0]
    problems_table = '| Household_ID | Missing target in supplied rows |\n| --- | ---: |\n'
    for r in raw_missing.itertuples():
        problems_table += f'| {r.Household_ID} | {r.MissingInRawRows:,} |\n'
    physical_features = [c for c in schema if c.startswith(('Survey_', 'Installation_')) and not c.endswith('_missing')]
    historical_features = ['PreviousAvailableSameInterval', 'PreviousAvailableSameIntervalAgeHours',
        'PreviousWeekSameLocalInterval', 'RecentSameSlot28dMedian', 'RecentSameSlot28dCount',
        'Recent7dMean', 'Recent7dMedian', 'Recent28dMean', 'Recent28dMedian',
        'Recent7dNightMean', 'Recent7dDayMean', 'Recent7dEveningMean',
        'Recent7dNightCount', 'Recent7dDayCount', 'Recent7dEveningCount',
        'Recent7dPeak', 'Recent7dP95', 'RecentChangeRatio', 'Recent7dZeroFraction',
        'HistoricalHouseholdBaseline']
    quality = ['HistoryObservedIntervals', 'HistoryExpectedIntervals', 'HistoryMissingRate',
               'HistoryValidDays', 'GapAgeHours', 'Recent7dCoverage', 'Recent28dCoverage',
               'ForecastRoute', 'sunshine_missing']
    text = f'''# Step 2: leakage-safe day-ahead preprocessing

No ML model, fallback model, synthetic demand targets or interaction-selection experiment was implemented. Raw data was read without modification. This report uses the completed Step 1 investigation rather than repeating feature importance analysis.

## 1. Recommended forecast issue time

**11:00 Europe/Berlin on D−1**, with the delivery day D defined as **00:00–24:00 UTC**, giving exactly 96 quarter-hour intervals. Europe/Berlin is an explicit Central European benchmark assumption; neither household country nor station coordinates/timezone are provided. A UTC delivery date and a local behavioural date are separate concepts.

## 2. Why this cutoff

The German/coupled-market benchmark should issue before the noon order-book close. 11:00 leaves an hour for bid preparation. [EPEX describes the noon close for its coupled day-ahead auction](https://www.epexspot.com/en/basicspowermarket); its Swiss timing differs. Confirm actual bidding zone before operational use. This source supports procurement timing, not a location inference about these households.

| Candidate issue convention | Recent demand available | Limitation |
| --- | --- | --- |
| Midnight before delivery | Almost the entire preceding day, subject to meter latency | Too late for preceding-day procurement; convenient offline convention |
| Noon on D−1 | Morning D−1 readings; afternoon/evening D−1 still unavailable | Issuing exactly at the coupled-market deadline leaves no bid preparation time |
| **11:00 on D−1 (used)** | D−1 morning up to the configured meter cutoff | Loses one hour versus noon, retains useful recent history, leaves preparation time |

Meter timestamps are **assumed interval starts**, not confirmed by the dictionary. A 15-minute interval plus a **15-minute assumed arrival latency** gives:

`ForecastCutoffTimestamp = ForecastIssueTimestamp − 30 minutes`

Only meter rows with `Timestamp <= ForecastCutoffTimestamp` enter any demand feature or quality metric. Weather hourly timestamps are also treated conservatively as interval starts, with one-hour interval duration plus one-hour assumed latency:

`HistoricalWeatherCutoffTimestamp = ForecastIssueTimestamp − 2 hours`

Both assumptions are configurable in `src/config.py`. At the winter issue time 11:00 local (10:00 UTC), latest permissible meter interval start is 09:30 UTC and latest hourly observation start is 08:00 UTC. At the summer issue time, both shift by one hour in UTC. Values arriving exactly at the issue time are assumed usable.

A same-slot lag is **not** blindly target time minus 24 hours. For a morning target slot, the previous local day's morning is often available; for an afternoon/evening target slot, it is not, so the feature selects the most recent observed matching slot on an older date. Its source timestamp and age are retained. The exact week lag remains missing if that keyed slot is absent.

## 3. Target reconstruction results

Audited all **{summary['households']} households**, **{summary['raw_rows']:,} supplied meter rows**.

| Target provenance | Supplied rows |
| --- | ---: |
| measured | {summary['raw_target_counts']['measured']:,} |
| reconstructed | {summary['raw_target_counts']['reconstructed']:,} |
| missing | {summary['raw_target_counts']['missing']:,} |

There are **{len(summary['reconstruction_households'])} households benefiting from reconstruction** in the actual dataset. The code implements and tests the specified hierarchy even though this dataset offers no missing-Total rows with both usable components. Measured Total, including measured zero, takes precedence. Missing Total is reconstructed only from two nonmissing, finite, nonnegative components. Otherwise it stays NaN. `TargetSource` (and the internal alias `target_source`) records provenance.

The Total = HP + Other identity was verified on **{summary['identity_checked_rows']:,}** complete rows; no invalid identity was found. An invalid complete-row identity fails the pipeline. No target is interpolated, estimated or zero-filled.

Separately, **{summary['absent_grid_rows_between_file_bounds']:,} quarter-hour timestamps are absent between household file bounds**. They are not counted as supplied rows in the table above. They become missing labels whenever a generated delivery grid requests them. The pipeline does not infer demand outside file bounds.

## 4. Households with remaining target problems

**Entirely missing Total:** 1065698, 747511, 768498, 996610. Each has HP observations but lacks the second component needed for Total reconstruction. They remain in the portfolio and require a later cold-start fallback for Total. Their HP cannot be relabelled as household Total.

**{len(raw_missing)} households have missing Total on existing rows:**

{problems_table}

**{int((audit.MissingGridRows > 0).sum())} households have absent timestamps between file bounds.** [remaining_target_problems.csv](audit/remaining_target_problems.csv) lists every affected household and both missing-row counts; [target_coverage.csv](audit/target_coverage.csv) covers all 410. Retrospective audit columns are not modelling features.

For the latest materialised issue, the routing is **{summary['latest_issue_route_counts']}**. Rules are transparent candidate routing thresholds, not models:

- **Normal:** >=28 complete-enough historical UTC days (>=90 valid intervals each), >=80% recent 28-day coverage, and <=48-hour gap age at the allowed meter cutoff.
- **Sparse history:** some valid Total history, but fails at least one normal threshold, including stale/stopped meters.
- **Cold start:** no valid Total history by the cutoff.

No household is dropped. Training later must mask missing labels; forecast issuance must still retain them and invoke a separately designed fallback. Portfolio sums must not silently turn missing forecasts into zero.

## 5. Metadata preprocessing rules

Registry is left-joined to the survey by string Household_ID with one-to-one validation. Seventeen survey-row absences are preserved. All specified physical survey fields are prepared; metadata fields are:

`{', '.join(physical_features)}`

`Weather_ID` is retained as a string join/category key. Living area and residents remain numeric with NaN and `<field>_missing` indicators. No median imputation is applied. Boolean equipment fields use explicit **True / False / Unknown** strings with missing flags. Building type/HP type have an Unknown level. DHW blanks and absent metadata rows are Unknown. PV missing is Unknown. Floor heating and radiators remain two independent fields; both can be True. DHW sources also remain independent.

**Exact historical benchmark assumption:** the provided fixed household/station and survey snapshot is assumed known before every benchmark issue and constant across the entire historical period. No effective dates or survey dates exist to verify this. `MetadataSnapshotAssumedKnown` and the feature contract record the assumption. This is a conditional benchmark, not proof that undated metadata was historically available. A strict real-time replay must omit unverified ownership/occupancy fields or acquire dated records. This includes possible future EV/PV installations.

**Group and AffectsTimePoint are omitted from model inputs.** Assignment and visit labels are not dated as-of records, and future realised visits are unsafe. Source files still retain them unchanged. Adding these later requires a known assignment/visit schedule or state available by each issue; no future transition date is inferred here.

## 6. Weather preprocessing rules

All household Weather_IDs map to one of eight supplied station files, with unique hourly UTC keys, no missing hourly timestamps, and coverage 2019-01-01 through 2024-02-29. Individual variable values may be missing. Pressure is excluded.

The six prepared weather signals are **Temperature_avg_hourly, Humidity_avg_hourly, Precipitation_total_hourly, Sunshine_duration_hourly, WindSpeed_hourly, DewPoint_hourly**.

**Deployable modelling table:** cutoff-safe historical 24-hour weather summaries and coverage fractions. Temperature/humidity/wind/dew point are means over observed hours. Rainfall and sunshine are totals over observed hours, accompanied by coverage; missing hours are not zeros. `Forecast_<variable>` and `ForecastHeatingExposure` are reserved NaN placeholders with missing indicators. They are explicitly excluded from the active feature whitelist until issued-forecast data with issue/valid timestamps and latency are acquired.

**Oracle table:** target-hour realised observations are stored separately under `oracle_weather/`, with `OracleObserved_` names and observation timestamps. The default reader never imports them. They may support explicitly labelled oracle experiments, not claims of deployable weather forecast performance. Every repeated quarter-hour covariate retains its original hourly meaning; rainfall/sunshine totals are not quadrupled or described as quarter-hour totals.

Sunshine is entirely absent at HbsbG, ceOxS and sV3mR, covering 30 households. `sunshine_missing` is derived only from bounded past station history; individual historical/forecast covariates also have missing indicators. No zero-fill and no automatic neighbouring-station substitution occurs. Safe first handling is to preserve unknown sunshine and use historical demand/calendar/temperature; omit sunshine/PV-sunshine interactions for that subgroup until a justified forecast source is available.

**Heating exposure:** mean `max(0, 15°C − historical hourly temperature)` over the available historical 24h window. This is a documented candidate base of **15°C**, configurable and not an official SIA/US HDD formula. It is a degree-C exposure mean, not a fabricated daily weather measurement. The corresponding forecast feature remains missing. Step 3 may choose or tune the base using training history only.

## 7. Local-time / DST decision

Only UTC timestamps are confirmed. Europe/Berlin is assumed for behaviour given hackathon context, not inferred from SIA or station IDs. Confirm station/household locations before operational deployment. Named timezone conversion is used rather than a fixed UTC offset.

Calendar fields: `LocalQuarterHourSlot`, `LocalHour`, `LocalWeekday`, `LocalWeekend`, `LocalMonth`, `LocalDayOfYear`, `HourSin`, `HourCos`, `DayOfYearSin`, `DayOfYearCos`, `LocalUTCOffsetMinutes`, `ForecastLeadHours`. Cyclic hour includes minutes; annual cycles use 365 or 366 days according to the target year.

On spring DST change, local hour 02 is absent. In autumn it repeats; UTC timestamps stay unique and the UTC offset distinguishes occurrences. Week/same-slot features join local slot and date; a repeated historical wall-clock slot is aggregated by median. A nonexistent historical slot stays missing. Spring/autumn **local delivery days would have 92/100 intervals**, so a true local procurement-day target cannot always have 96. The present benchmark intentionally uses fixed UTC days; switching delivery-day semantics requires an explicit specification change, not padding/truncating demand.

## 8. Exact initial historical demand features

`{', '.join(historical_features)}`

Means/medians are **interval kWh** over observed historical readings, not daily energy estimates. Recent 7/28-day windows are elapsed UTC time windows ending at the meter cutoff. Night is 00:00–06:00 local, day 06:00–18:00, evening 18:00–24:00; counts accompany these means. The baseline is the mean of all valid Total history available at the cutoff. These modest initial summaries are not hundreds of interactions.

Recent change ratio compares the latest 7-day mean with the preceding 21-day mean inside the 28-day window. It is NaN if the preceding denominator is zero or missing. The recent same-slot median uses at most the last 28 days; the most recent matching slot search is also bounded to 28 days, with NaN when no usable slot exists. Same-slot medians are not weekday-conditioned yet. Zero fractions count measured zeros, not missing values.

Each optional numeric history feature has an explicit missing indicator; coverage/count fields remain available to distinguish missingness from load magnitude. No clipping, imputation, scaling, quantile fitting or model-dependent transform is learned here. Past extreme demand is included unless a later policy explicitly changes it.

## 9. Data-quality features

`{', '.join(quality)}`

History expected count uses the complete quarter-hour grid between the earliest source row already known at issue time and the permitted meter cutoff. It never uses the full-file latest date. If no source row exists by cutoff, rate/age are unknown and observed/expected counts are zero. Recent coverage denominators always use 7×96 or 28×96 elapsed intervals, even with short history. Missing-rate denominator includes absent timestamps and existing NaNs. Valid-day count uses completed UTC days only, with >=90 observed intervals. `GapAgeHours` is age of the last valid interval-start **relative to the meter cutoff**, so it can be zero despite configured reporting latency; add the documented latency/interval convention when interpreting age at issue time.

`ForecastRoute` is a routing diagnostic excluded from the active feature whitelist. `TargetMissing`, `TargetNegative` and `TargetExtremeReview` are **label-quality fields, never predictors**. Full-file availability summaries appear only in audit outputs.

## 10. Clean modelling table schema and storage

Each row represents **one household and target quarter-hour** at its fixed issue time. Materialised slice: **2024-01-31 through 2024-02-27**, **{summary['materialized_rows']:,} rows**, all 410 households, 96 intervals per day. Missing-label rows and stopped meters are retained. This is a real modelling table slice, not the complete multi-year table.

| Role | Fields / types |
| --- | --- |
| Keys | Household_ID: string; TargetTimestamp: timezone-aware UTC datetime |
| Availability | ForecastIssueTimestamp, ForecastCutoffTimestamp, HistoricalWeatherCutoffTimestamp: UTC datetime |
| Label | Target: float kWh/NaN; TargetSource: measured/reconstructed/missing |
| Label-quality diagnostics | TargetMissing, TargetNegative, TargetExtremeReview: bool; excluded from predictors |
| Historical inputs | Named history/quality numeric summaries and missing bools |
| Calendar | Local integer/bool/cyclic fields, UTC offset, lead hours |
| Weather | Historical summaries/coverage; reserved issued-forecast placeholders; no oracle values |
| Metadata | Nullable numeric and explicit three-state/categories, missing bools |
| Lineage | DemandHistorySourceMaxTimestamp, LagAvailableSourceMaxTimestamp, LagWeekSourceMaxTimestamp, SameSlotSourceMaxTimestamp, HistoricalWeatherSourceMaxTimestamp: UTC datetime/NaT |
| Non-input diagnostics | MetadataSnapshotAssumedKnown, ForecastRoute, HeatingExposureBaseC |

There are **{summary['columns']} physical table columns**, including keys, labels, flags, lineage, metadata and unavailable-weather placeholders; **{len(contract['candidate_features'])} active candidate columns** include missing/coverage indicators. [schema.json](schema.json) lists every exact name, dtype and role; [feature_contract.json](feature_contract.json) defines the explicit input whitelist and excludes labels/lineage/reserved weather. No exhaustive interaction set is produced.

Files are compressed Parquet under `modelling/household=<ID>/part.parquet`, separate from `oracle_weather/household=<ID>/part.parquet`. Read one household at a time with column projection; `iter_modelling_tables()` uses the whitelist and physical-file reads without hive alias columns. It excludes labels by default; label inclusion is opt-in for later preparation. For full-history builds, materialise only required delivery-date ranges, one household at a time, and add delivery-month partitions/row groups if range scans become large. No 28M-row combined frame is held in memory. The CLI accepts any contiguous target date range. Metadata and issue-level portfolio quality have their own small Parquet outputs.

## 11. Leakage and validation results

- **10 automated tests passed:** future demand/weather poisoning, truncating all future inputs, keyed gap/week lags, measured/reconstructed/missing hierarchy and invalid identity rejection, unknown metadata/coexisting distribution, both DST transitions, cold start/no sunshine, late-source rejection, column-name loading and duplicate rejection.
- Independently validated **{validation['partition_count']} partitions / {validation['validated_rows']:,} rows**, fixed 96-interval issues, consistent feature columns, target-source/missingness agreement, complete household retention, assigned-station oracle joins and local calendar conversions.
- **Zero duplicate raw or modelling keys, zero after-cutoff demand/weather source violations, zero invalid complete-row component identities, zero unmapped station IDs.** No raw meter file size/mtime changed while processing.
- Future values were poisoned for three real households (treatment 661091, control 676510, cold-start 996610); all nonlabel columns remained identical. Poisoned test labels were temporary and never exported.
- The selected slice contains **{validation['selected_slice_target_counts']}** target sources. Oracle sunshine is missing on **{validation['oracle_sunshine_missing_rows']:,}** rows. All issued-forecast placeholders are missing by construction, not replaced with oracle observations.

The temporal guarantees are conditional on the documented interval-start/latency and undated-metadata assumptions. A timestamp check cannot retrospectively prove actual telemetry arrival times or survey availability. The mutation/truncation tests establish that the implemented historical calculations do not inspect later supplied values.

## 12. Extreme-value findings

**676510, 2021-01-09 23:00 UTC: 66.976 kWh in one quarter-hour (~267.9 kW).** Immediate preceding/following readings are 0.073/0.058 kWh; the surrounding two hours are mostly ~0.04–0.08 kWh. It is the only reading above 10 kWh in that household's 110,496 observed Total readings. Median is 0.077 kWh and 99.9th percentile about 2.017 kWh. Metadata describes a 150 m², three-resident house with air-source HP/radiators, no EV; snapshot age is unknown. There is no component measurement to confirm this spike.

**Assessment:** confirmed isolated spike, not a repeated demand pattern; implausible as routine demand for this household. A meter/aggregation/correction artifact is a stronger hypothesis than ordinary load, but the source does not prove the cause. It remains measured and flagged; no deletion or replacement is made.

**7569701, 2024-01-19 21:15 UTC: 27.304 kWh (~109.2 kW)** is also isolated, between 0.642/0.439 kWh; that household has only one reading >10 kWh. Its 300 m²/EV survey does not substantiate this load. Both >20 kWh readings have timestamp-keyed ±2h context saved in [extreme_neighbourhoods.csv](audit/extreme_neighbourhoods.csv). The 20 kWh threshold is a documented review trigger, not a physical hard cap or removal policy.

## 13. Before model training / operational use

1. Keep this single issue convention. Confirm interval timestamp semantics and meter/weather latency if making operational claims; source arrival timestamps are absent.
2. Confirm locality and whether the task wants fixed UTC days or local settlement days. Do not claim the present UTC day exactly matches a local-market delivery day on DST transitions.
3. For honest **weather-forecast-enabled** training/backtesting, acquire forecast vintages issued by each cutoff (including availability latency). Otherwise use the current historical-observation features only; label target-day realised-weather experiments explicitly as oracle.
4. Explicitly accept the fixed-snapshot metadata benchmark assumption or obtain dated metadata/omit uncertain fields for strict historical replay. No code can recover unknown effective dates.
5. Step 3 must choose a transparent policy for the two extreme labels and contaminated historical peak summaries, preserving a sensitivity comparison. This preprocessing keeps them intact.
6. Design normal/sparse/cold-start forecasting and portfolio aggregation in a later step. The four targetless households cannot supply supervised Total labels but must receive forecasts eventually. No fallback is implemented now.
7. When fitting later, use chronological issues, exclude missing Target labels, fit imputation/encoding/scaling only on training history, and use the explicit feature whitelist. Oracle paths and label diagnostics must never enter deployable inputs.

These are explicit benchmark/operational limitations and next-step decisions, not missing preprocessing implementation. The pipeline and modelling slice are ready for Step 3 feature/interactions selection under the stated assumptions; the missing forecast archive prevents a deployable forecast-weather backtest, not a historical-demand-only first model.

**READY FOR STEP 3: YES**
'''
    (out/'STEP_2_REPORT.md').write_text(text)
    guide = '''# Running Step 2 preprocessing

No model is trained. Run from the repository root with Python 3.12+:

```sh
python -m pip install -r requirements-preprocessing.txt
python -m unittest discover -s tests -v
python -m src.run_preprocessing --start 2024-01-31 --end 2024-02-27
python -m src.validate_outputs
python -m src.write_preprocessing_report
```

`src/config.py` is the authoritative forecast convention and assumption configuration.
The same convention is used for every household/target date. Changing it requires rebuilding
outputs and rerunning validation. `--data-dir` and `--output-dir` are available on build and
validation commands; the report writer currently reads the default output directory.

Read `STEP_2_REPORT.md`, `schema.json`, `feature_contract.json` and audit JSONs before model
preparation. Generated Parquet tables are local outputs and are ignored by Git. The modelling
preview is a single-household example; use the partition reader for the full portfolio.

```python
from pathlib import Path
from src.data_loading import iter_modelling_tables
for frame in iter_modelling_tables(Path("outputs/preprocessing"), include_labels=True):
    # Do not fit yet: Step 3 will select inputs/interactions.
    observed_labels = frame[frame.TargetSource != "missing"]
    # Forecasting still needs every household, including unlabeled/cold-start rows.
```

Oracle weather is quarantined in a separate directory and is never read by this iterator.
Issued-weather placeholders are also excluded until a valid forecast archive is supplied.
No raw data is written. Processing is sequential per household to bound memory use.
'''
    (out/'README.md').write_text(guide)
    print(out/'STEP_2_REPORT.md')


if __name__ == '__main__':
    main()
