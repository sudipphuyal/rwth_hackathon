# Weather forecasting module

Run from the repository root with Python 3.12+ and repository dependencies:

```sh
python -m pip install -r requirements-weather.txt
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
OMP_NUM_THREADS=2 python -m src.run_weather_forecasting
OMP_NUM_THREADS=2 python -m src.validate_weather_forecasting
python -m src.write_weather_report
```

The completed revised full-year test run is protected by test_evaluation_completed.json.
Outputs are under outputs/weather_forecasting_full_year; the original experiment remains
under outputs/weather_forecasting and is superseded for energy inputs. The training
CLI refuses adaptive reruns after test evaluation; validation/report commands may inspect
saved results without refitting/reselecting. Do not delete the lock/marker to tune on test.
Fixed global dates are in src/config.py and splits.json. Model parameters/selection rule
are in weather_forecasting.py and the selection lock. Use original trusted local pickles
with the recorded compatible scikit-learn dependency.

Read WEATHER_REPORT.md, weather_feature_contract.json and validation_checks.json before
the energy step. Predictions and actuals are in separate directories. Energy inputs must
use predictions/historical_day_ahead_weather.parquet (train is out-of-time) and the final
2024-02-28 file. The adapter src/weather_energy_interface.py joins only predicted columns.
It does not regenerate the existing energy preprocessing or fit a demand model.
