# Step 4 energy forecasting

Run from the repository root with Python 3.12+ and the existing dependencies:

```sh
python -m pip install -r requirements-weather.txt
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 python -m src.materialize_energy_data
OMP_NUM_THREADS=4 python -m src.run_energy_forecasting
OMP_NUM_THREADS=4 python -m src.validate_energy_forecasting
OMP_NUM_THREADS=4 python -m src.validation_energy_predictions
python -m src.write_energy_report
```

Existing materialisation is retained when complete. The forecasting CLI refuses to rerun
selection/test once test_evaluation_complete.json exists. Do not delete locks to tune on test.
Validation residual replay trains ANALYSIS COPIES using TRAIN only and the already-selected
recipe; it reads no test targets and never changes locked test/demonstration weights.

Read ENERGY_REPORT.md, model_selection_lock.json, materialization_complete.json and
validation_checks.json. Output directory is outputs/energy_forecasting. Generated datasets,
prediction Parquet files and trusted local model pickles are ignored by Git but remain on disk.

No household ID is an input. All target labels must be genuinely measured/reconstructed.
No actual target-day weather is joined. Activity starts only after the first supplied record
is available at issue; stopped meters are retained absent an authoritative departure roster.
No quantiles or procurement optimisation are implemented in Step 4.

## Step 5 — uncertainty around locked forecasts

Run `python -m src.uncertainty_forecasting` with the same Python dependencies as
Step 4. This loads saved models for predictions only, builds out-of-time
validation residual calibration, saves its lock, and evaluates the saved TEST
forecasts once. The completion marker refuses an adaptive second evaluation.
See `../uncertainty_forecasting/UNCERTAINTY_REPORT.md` and the 96-row unseen-day
CSV files. Full portfolio coverage cannot be verified for targetless households.
