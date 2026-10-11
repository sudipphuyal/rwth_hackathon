# Bringing the Heat — Weather-Aware Day-Ahead Energy Forecasting and Procurement Decision Support

**Team members:** Sudip Phuyal, Manila Bhandari, Rujjal Sada, Mahsa Azarnia, Vignesh Pechiappan Ayyathurai

## Challenge

Forecast next-day electricity grid import for approximately 410 heat-pump households at 15-minute resolution. Demand reflects recent behaviour, heating requirements, weather, PV self-consumption, buildings and EV/appliance peaks. Unequal histories and incomplete measurements complicate forecasting; the target is grid import, not gross load or PV generation.

## Solution

Historical weather first produces day-ahead temperature, sunshine and humidity forecasts. These join meter history, household metadata and calendar variables in one pooled scikit-learn **HistGradientBoostingRegressor**, a global gradient-boosted decision-tree regression model. Inputs emphasize seven-day behaviour, previous-week matching intervals, 28-day profiles and heat-pump/building/domestic-hot-water characteristics. New/sparse households remain in the global pipeline; four targetless homes receive history-free cold-start estimates, and missing metadata stays unknown.

Out-of-time residuals provide P10/P50/P90 uncertainty. Aggregate residuals and whole-day errors preserve correlation rather than summing household quantiles. The procurement rule **q* = C_under / (C_under + C_over)** chooses demand quantities by shortage/surplus cost asymmetry. Under illustrative 4:1 costs, P80 reduced retrospective normalized TEST interval imbalance cost by approximately **24%** versus the ML point forecast; these are not euros or actual E.ON prices.

## Experimental design

**TRAIN:** 2019-05-20 → 2022-02-27; **VALIDATION:** 2022-02-28 → 2023-02-27; **TEST:** 2023-02-28 → 2024-02-27.

Chronological splits avoid random mixing and cover full seasonal cycles in both holdouts. Selection uses training/validation only; models and calibration stay frozen on TEST. Recent TEST consumption becomes historical input when available, but targets never retrain models. Tomorrow’s actual weather is never used; weather is forecast from past observations with explicit availability cutoffs.

## Results

| Frozen full-year TEST headline | Result |
|---|---:|
| Temperature forecast MAE | 2.50 °C |
| Portfolio WAPE: ML / persistence blend | 14.29% / 14.47% |
| Household MAE / RMSE improvement versus blend | 7.7% / 16.1% |
| ML P10–P90 coverage; nominal 80% | 79.0% |

ML improves quarter-hour/household forecasting; the 75% previous-available / 25% previous-week blend remains competitive for daily energy. Validation WAPE falls from **16.14%** (history/calendar) to **15.32%** (+ predicted weather) and **14.13%** (+ metadata/recent behaviour), supporting predictive value rather than causality.

| TEST day | Actual MWh | ML MWh | ML abs. error MWh | Blend MWh |
|---|---:|---:|---:|---:|
| 2023-03-15 | 13.704 | 11.379 | 2.325 | 10.172 |
| 2023-09-05 | 5.135 | 6.048 | 0.914 | 5.914 |
| 2024-02-27 | 11.344 | 12.104 | 0.760 | 11.541 |

These dates were predefined, not selected for favourable errors. Daily examples use the same measured household support for actual and predicted demand; missing household demand is not treated as zero. They illustrate, rather than replace, full-year metrics.

## Limitations

Weather forecasting is statistical, not professional numerical weather prediction; incomplete targets/metadata leave four homes with provisional uncertainty. Daily coverage undercovers at approximately 73–75%; illustrative prices and absent live-market constraints make this a reproducible proof-of-concept decision-support system, not a production trading system.

## Day-ahead demonstration

**TRUE UNSEEN DAY — 2024-02-28:** ML point **13.015 MWh**; daily **P10 10.470**, **P50 12.381**, **P90 14.681 MWh**. Illustrative 4:1 shortage/surplus costs give **P80 procurement 14.011 MWh**. No actual smart-meter demand exists for this date in the supplied dataset: this is a genuine unseen next-day forecast, excluded from accuracy calculations.
