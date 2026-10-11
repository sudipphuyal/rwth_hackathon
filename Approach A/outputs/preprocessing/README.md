# Running Step 2 preprocessing

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
