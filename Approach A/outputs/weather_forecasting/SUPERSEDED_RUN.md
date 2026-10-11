# Preserved first weather experiment

The original 70/10/20 calendar split (2019-05-20–2022-09-30 train,
2022-10-01–2023-03-31 validation, 2023-04-01–2024-02-27 test) was superseded
by the user's explicit request for full-year seasonal holdouts.

Original split/selection locks, models, metrics, validation checks, and forecasts
remain here unchanged. The active experiment and energy-input forecasts are in
../weather_forecasting_full_year/. No original test marker or lock was deleted.
The revised test overlaps previously inspected results; the active report records
this limitation. The existing causal feature table was reused with the same
forecast convention and model feature definitions.
