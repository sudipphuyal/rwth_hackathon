# Final hackathon submission checklist

## Required organizer deliverables

- [x] Complete final implementation: all `src/*.py`, `src/__init__.py`, and `tests/*.py` exist in the working directory. Forecasting, calibration and procurement code is not confined to notebooks or temporary outputs.
- [x] `ONE_PAGE_DESCRIPTION.md`: 498 words, including title, team roster, headings and both compact results tables (whitespace count excluding Markdown syntax/table separator).
- [x] `ONE_PAGE_DESCRIPTION.pdf`: exactly one normal A4 page; rendered and visually checked for clipping, table alignment and complete content.
- [x] Team names confirmed directly by the user: **Sudip Phuyal; Manila Bhandari; Rujjal Sada; Mahsa Azarnia; Vignesh Pechiappan Ayyathurai**. The organizer package author in `pyproject.toml` is not substituted for the team roster.
- [x] README gives the five-stage run order, exact CLIs, input-data requirements and a separate saved-output review route.
- [x] Required dependencies are specified: `requirements-reproduction.txt` records the actual frozen experiment versions; existing `requirements-preprocessing.txt` / `requirements-weather.txt` describe supported ranges. `requirements-submission.txt` covers the optional PDF builder.
- [x] Paths for the final pipeline are relative to the repository root. Earlier `/tmp` installations supplied dependencies only; they contain no essential forecasting implementation. Python 3.12.14 was used; run from the root with the original `data/` layout.
- [x] Frozen reports, results, selection locks, model weights, residual calibration and presentation figures remain available. No retraining, retuning, recalibration or changes to prior reported results were performed.

## Review these generated outputs

| Purpose | Location |
|---|---|
| Short judge-facing description | `ONE_PAGE_DESCRIPTION.md`, `ONE_PAGE_DESCRIPTION.pdf` |
| Weather evidence and figures | `outputs/weather_forecasting_full_year/WEATHER_REPORT.md`, `figures/` |
| Energy evidence and ablations | `outputs/energy_forecasting/ENERGY_REPORT.md`, `figures/` |
| Frozen calibration and coverage | `outputs/uncertainty_forecasting/UNCERTAINTY_REPORT.md`, `calibration_lock.json`, `test_evaluation_complete.json` |
| Procurement comparison and limitations | `outputs/procurement_support/PROCUREMENT_REPORT.md`, `test_policy_metrics.csv`, `value_of_uncertainty.csv` |
| Unseen-day 96-interval schedules | `outputs/procurement_support/2024-02-28_ML_procurement_schedule.csv`, `2024-02-28_Blend_procurement_schedule.csv` |
| Separate daily and peak decisions | `outputs/procurement_support/2024-02-28_daily_procurement.csv`, `2024-02-28_peak_procurement.csv` |
| Presentation figures | `outputs/procurement_support/figures/2024-02-28_procurement.png`, `test_policy_costs.png` |
| Submission checks and exact values | `SUBMISSION_VERIFICATION.json` |

The procurement figures were rendered and visually reviewed in Step 6. The one-page PDF has been separately rendered and checked in this submission step. The latest final deck and presenter notes are in the repository's `Presentation/` folder.

## Packaging — action at upload

**Published package:** implementation, frozen models, required replay inputs and saved outputs are tracked on `main`. The root `PROJECT_MANIFEST.json` lists the contribution files and hashes. The dashboard and final presentation are included alongside Approach A and Approach B.

Include the one-pager files, this checklist, `SUBMISSION_VERIFICATION.json`, README, all of `src/` and `tests/`, every requirements file, `pyproject.toml`, `poetry.lock`, `.gitignore`, and the judge-facing reports/JSON summaries/CSV decisions/figures above. Preserve `outputs/energy_forecasting/models/`, `outputs/weather_forecasting_full_year/models/`, all model/calibration locks and completion records. Preserve the input dataset or arrange access to the identical organizer-supplied `data/` files; source-only reproduction needs that data.

For replay of the existing frozen procurement experiment rather than rebuilding from raw data, retain the full saved weather/energy predictions, energy TEST daily feature tables and TEST labels, uncertainty validation portfolio residual cache and quantile files. `decision_policy_lock.json` lists every frozen input fingerprint checked by Step 6. A compact source-and-results upload can omit recomputable caches listed below, but it will not support all frozen replay/verification commands without regenerating those caches in a separate environment.

The current workspace is approximately **5.9 GB**; generated energy outputs are approximately **3.8 GB**, and raw data approximately **1.6 GB**. Confirm the organizer's upload-size limit before choosing a full archive versus a compact code-and-results submission. No archive was fabricated with silently omitted inputs.

## Files that can be excluded from an archive

No source data or reproducibility artifacts were deleted. These are recommendations for archive packaging:

- `.git/`, `.DS_Store`, `**/__pycache__/`, `*.pyc`, virtual environments and local dependency caches.
- Empty `test_file`, and optional exploratory `analysis/daily_exploration.csv` (approximately 69 MB). Retain the analysis source and smaller assessment reports if including the exploratory audit.
- `outputs/preprocessing/oracle_weather/` (approximately 20 MB): explicitly hindsight-only diagnostics, not operational forecast inputs.
- `outputs/weather_forecasting/` (approximately 20 MB): earlier weather experiment. The final pipeline uses `outputs/weather_forecasting_full_year/`.
- `outputs/energy_forecasting/data/` (approximately 2.8 GB): recomputable household feature partitions. Keep their generation code and materialization manifest. Omitting them prevents frozen replay commands that specifically read these partitions until restored/recomputed separately.
- `outputs/preprocessing/modelling/` (approximately 83 MB): regenerable reference preview; preserve its report/contracts/audits.
- `outputs/energy_forecasting/daily/` (approximately 423 MB) and detailed evaluation Parquet files are **not automatically dispensable**: some frozen validation/procurement replay commands consume them. Exclude only for a documented compact source/results package, not a complete frozen-replay archive.

The revised one-pager also includes the three predefined TEST dates; exact matched-support daily values are in `representative_test_days.csv`, with source/support checks in `representative_test_days_checks.json`. Include both files in the submission. The reporting-only CLI is `python -m src.representative_test_days`; it does not run models.

## Numerical consistency and limits

`SUBMISSION_VERIFICATION.json` derives the headline values from frozen weather/energy/uncertainty/procurement files, checks the expected rounding in the one-pager and README, verifies the team names and A4 page count, and confirms unchanged SHA-256 fingerprints of 857 prior output/model files. No headline discrepancies were found. The published forecast/decision results are preserved exactly; rounded numbers are for presentation only.

The daily P80 is a separately calibrated daily decision, not a sum of quarter-hour P80 quantities. All historical procurement costs are normalized hypothetical units on matched measured support. Daily uncertainty undercovers, full-portfolio truth is incomplete and four targetless homes have provisional uncertainty. This is a hackathon proof of concept, not a production trading system.

**Submission documents and team roster are complete.** A working-directory submission is ready; Git-only packaging and organizer size limits must be handled as described above. The published package preserves the team roster.
