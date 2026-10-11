# Slide evidence and sources

## Slide 1: Day Ahead Residential Energy Forecasting

The wording comes directly from the final report, not a new project name.



- [Final report](01-Final-Report.docx): The first two paragraphs provide the exact project title and subtitle. The forecasting-strategy section defines the two approaches.

## Slide 2: Two approaches to the forecasting task





- [Final report](01-Final-Report.docx): The two forecasting strategies and final output specification.

## Slide 3: Approach A: data and information considered

These are supplied-data descriptions and saved audit findings. The feature discussion explains modelling motivation, not causal effects or measured importance.

The target is received Total interval energy, interpreted as grid import. Without a metering diagram, gross consumption after PV self-consumption cannot be independently resolved. We have no measured PV production or EV charging schedule. Household and station location/timezone are not confirmed by the supplied metadata.

- Input assessment (historical source not included in this repository): Scope and confirmed structure describe meter coverage, eight stations and available versus absent measurements.
- [Preprocessing report](02-STEP_2_REPORT.md#L31): The audit reports 410 households and 28,180,608 supplied meter rows. Missing-household and survey/weather sections explain the gaps.

## Slide 4: Approach A: cleaning and missing data

Cleaning is traced to the source functions. Missingness and reconstruction counts are saved audit outputs. The distinction is important: the code supports reconstruction, but it reconstructed no labels in these data.

The 517,536 missing supplied Total values and 1,543,680 absent timestamps are distinct audit quantities. Numeric metadata is not median-imputed; equipment blanks remain Unknown. The fixed, undated survey snapshot is assumed known before every benchmark issue, which is a benchmark assumption rather than verified historical availability. No label is generated merely to fill a gap. The four targetless homes remain unscoreable: their true accuracy is unknown. A history-withheld proxy test is indirect evidence only.

- [Meter loading and target preparation](20-data_loading.py#L13): read_meter validates keys/timestamps/numbers. prepare_target checks Total = HeatPump + Other, records missingness and flags extremes.
- [Metadata preprocessing](21-preprocessing.py#L26): prepare_metadata performs the registry left join, keeps numeric NaN and encodes True/False/Unknown.
- [Preprocessing results](02-STEP_2_REPORT.md#L31): Measured/reconstructed/missing counts confirm that reconstruction actually filled zero target rows.
- [Cold-start evaluation](04-ENERGY_REPORT.md#L313): The history-free model generates predictions for the four homes without Total labels.

## Slide 5: Approach A: features and demand model

The configuration comes from the saved selection lock. Feature inclusion reflects validation selection, not an interpretation of the test curves. The model did not use future observed weather.

Squared error was retained on validation because absolute-error loss lowered individual MAE but substantially underpredicted aggregate demand. The practical target includes portfolio volume. Large measured labels remained in fit/scoring because a review flag alone did not prove a meter fault. Regularization and large leaves control model complexity. PV/EV omission from the main model is a validation feature-group finding, not evidence of no physical effect or no future value. Recent demand can already capture some ownership-related behaviour. The cold-start model has no personal demand history, so it retains those metadata groups.

- [Locked model configuration](03-model_selection_lock.json#L11): parameters, loss, selected_feature_set, input_columns and cold_input_columns specify the model actually selected.
- [Feature and loss evidence](04-ENERGY_REPORT.md#L161): Sections 4–7 explain squared-error selection, measured-extreme policy and the validated feature groups.

## Slide 6: Approach A: training, validation and test

The date boundaries come from the saved split. Validation selects the model, train+validation refit provides the test weights, and the test evaluates those frozen weights. Recent past test-day data can update features without updating weights.

A uses 11:00 Europe/Berlin on the previous day and latest meter interval starts at issue minus 30 minutes. This models a 15-minute interval duration plus a 15-minute arrival assumption. Weather hourly starts end at issue minus two hours. UTC target grids have 96 slots, while local calendar features use a named DST-aware timezone. UTC delivery-day semantics and Europe/Berlin behaviour are explicit benchmark assumptions. Boundary labels unavailable before the next issue are embargoed. The test is held out from energy-model fitting/selection; earlier exploratory analysis had inspected raw historical data, so it should not be described as completely unseen raw data.

- [Locked chronological split](03-model_selection_lock.json#L2): split gives inclusive train, validation, test and unseen-day dates.
- [Refit summary](45-refit_summary.json): Records the train+validation refit, before the frozen test.
- [Frozen test protocol](04-ENERGY_REPORT.md#L217): Section 8 describes fixed weights, daily rolling inputs and prediction before target-based scoring.
- [Issue-time preprocessing](02-STEP_2_REPORT.md#L5): Issue time and meter interval/arrival assumptions define the input cutoff.

## Slide 7: Approach A: actual versus predicted test demand

Daily energy = sum of matched quarter-hour kWh ÷ 1,000. Household improvement = 100 × (baseline error − A error) ÷ baseline error. MAE: 100 × (0.187212204 − 0.172859662) ÷ 0.187212204 = 7.666456%. RMSE: 100 × (0.339287631 − 0.284705460) ÷ 0.339287631 = 16.087286%. Portfolio WAPE = 100 × sum(|prediction − actual|) ÷ sum(actual). The 14.29% and 14.47% refer to quarter-hour portfolio errors, while 1.148 and 1.109 MWh are means of absolute daily-total errors.

The benchmark is exactly 75% previous-available matching local-slot demand plus 25% previous-week matching local-slot demand, selected on validation. It is not Approach B Ridge. The final report calls it persistence, but the saved lock gives this blend. Portfolio WAPE decreases by 0.175 percentage points (about 1.21% relative), while daily-total MAE increases about 3.56%. Only the same measured household/interval support enters each pair of scores. The observed count changes by date. The no-support day 29 October is omitted and never zero-filled.

- [Saved daily test evaluation](07-test_daily_energy.csv#L1): The plotted series are MatchedActual and MatchedPrediction, divided by 1,000. No-support actuals remain missing.
- [Frozen quarter-hour portfolio predictions](08-test_portfolio.parquet): MatchedActual and MatchedPrediction sum only the same observed households at each timestamp.
- [Full-year error metrics](05-summary_metrics.csv#L1): Select Period=Test. selected means Approach A; baseline is the saved lag blend. Scope distinguishes Household and MatchedPortfolio.
- [Saved improvement percentages](06-improvements_vs_baseline.csv#L1): Household MAE and RMSE rows provide 7.666456% and 16.087286% improvements.
- [Exact benchmark definition](03-model_selection_lock.json#L199): baseline_name=blend_A_B_0.75 and weights A=0.75, B=0.25 identify the comparison benchmark.

## Slide 8: Approach A: variation within the test day

Daily signed error = 100 × (forecast total − actual total) ÷ actual total. March: 100 × (11.378833240 − 13.703601) ÷ 13.703601 = −16.964649%. September: 100 × (6.048369062 − 5.134834) ÷ 5.134834 = +17.790937%. The curves use all 96 intervals, without smoothing. Each pair has identical observed households.

The dates were predefined in the final report and are not selected for favourable visual performance. Both curves use the identical observed households within each day. The two charts have different vertical ranges so label reading matters. A positive daily total error can still contain negative interval errors. We plot interval energy in kWh/15 min, not instantaneous power.

- [Frozen test portfolio file](08-test_portfolio.parquet): Filter TargetTimestamp to 2023-03-15 and 2023-09-05. Plot all 96 MatchedActual and MatchedPrediction rows for each date.
- [Historical comparison values](14-COMMON_SUPPORT_COMPARISON.csv#L1): A_OwnSupport columns verify the measured household counts, totals, signed errors and interval MAE.

## Slide 9: Approach B: cleaning and missing data

The handling comes from B’s supplied source code. Dropping unusable observations or incomplete training rows differs from dropping a household: all files stay in the final forecasting portfolio.

B removes unusable observations and incomplete feature/target rows, but retains every household file in the final portfolio. Its supplied source does not interpolate missing training labels or clip measured outliers. Forecasts and fallback outputs alone are clipped nonnegative. For completely unobserved Total demand, portfolio imputation provides a prediction without measured validation of that household.

- [Approach B source: loading](17-SummerSchool_410_Household_Forecast.py#L231): dropna removes invalid timestamp/target rows. The following sort and drop_duplicates clean the observed records.
- [Approach B source: feature grid](17-SummerSchool_410_Household_Forecast.py#L439): create_ridge_dataset reindexes the 15-minute grid and drops rows missing a required feature or target.
- [Approach B source: fallbacks](17-SummerSchool_410_Household_Forecast.py#L1365): fallback_forecast tries week, day, historical slot mean, then the portfolio reference.
- [Updated B description](18-SummerSchool_OnePage_Description_410_Households_Updated_Results.docx): Reports 406 files with usable consumption and four needing complete imputation.

## Slide 10: Approach B: training and uncertainty

The training/scaling procedure and conformal rule come from B’s source. Its nominal 80% intervals are household-quarter-hour intervals for Ridge forecasts, not a demonstrated full-portfolio daily interval.

The calibration window is excluded from Ridge and scaler training. Ridge uses at least 1,344 complete pre-calibration rows, all 1,344 complete rows in the preceding 14 calendar days, and complete future day/week lags. B uses the entire preceding UTC day through 23:45 UTC, while A issues earlier. Conformal intervals are nominal marginal household-quarter-hour intervals, not a portfolio daily or simultaneous guarantee. Temporal dependence and distribution shift can undermine nominal coverage. No empirical B interval-coverage evaluation was supplied.

- [Approach B settings](17-SummerSchool_410_Household_Forecast.py#L115): CALIBRATION_DAYS=14, TARGET_COVERAGE=0.80 and RIDGE_ALPHA=1.0 define the procedure.
- [Approach B training](17-SummerSchool_410_Household_Forecast.py#L1029): train_ridge_for_date pools eligible rows and fits StandardScaler + Ridge. The six FEATURES are lag_1d, lag_7d and cyclic time/weekday inputs.
- [Approach B calibration quantile](17-SummerSchool_410_Household_Forecast.py#L1279): Computes the finite-sample quantile level and uses the higher absolute-residual quantile.

## Slide 11: Reported historical results for both approaches

Daily MWh = summed quarter-hour kWh ÷ 1,000. Signed error = 100 × (forecast − actual) ÷ actual. Absolute daily error = |forecast − actual|. A numbers are reconstructed from frozen outputs. B numbers come from supplied historical figures adopted in the final report. Different household supports mean these are descriptive results, not a controlled ranking.

A historical results come from frozen saved predictions. B historical rows are supplied report values using 294/374/371 measured homes. The updated B document instead lists 368/381/374 complete actual profiles and declines full-410 metrics. The current B script computes historical metrics only if every one of the 410 actual profiles exists, so it does not reproduce the published subset rows. There is no saved B household prediction export for common-support verification. The relationship between these historical artifacts is not independently established. Historical supports and issue cutoffs differ. Present the rows as descriptive reported results, not a controlled model ranking.

- [Final combined report](01-Final-Report.docx): Representative Day Ahead Tests is the source for the published A and B rows.
- [Machine-readable historical comparison](14-COMMON_SUPPORT_COMPARISON.csv#L1): A_OwnSupport and B_Report columns retain each method’s own observed supports and published totals.
- [Comparison provenance](15-FINAL_COMPARISON_NOTES.md#L1): The opening paragraphs identify A frozen-file reconstruction and user-supplied B historical figures, and explain why the supports cannot rank the models.
- [B full-profile scoring guard](17-SummerSchool_410_Household_Forecast.py#L2173): The current script scores historical portfolios only if all 410 actual profiles exist.

## Slide 12: Final day-ahead forecast: 28 February 2024

A: sum(Prediction) ÷ 1,000 = 13.014908666 MWh, rounded to 13.015. B: 12,011.58 kWh ÷ 1,000 = 12.01158 MWh, rounded to 12.012. B routing: 372 + 11 + 23 + 4 = 410. A’s daily uncertainty comes from a separate calibrated daily residual distribution, not sums of marginal interval bounds.

Both totals cover 410 homes and all 96 forecast slots. A uses the same locked weights as its test. B point/routing figures are report-stated; B numeric household exports were not supplied. The old 372-only B total is superseded. A daily point 13.015 differs from its median 12.381 MWh because residual calibration and the model point are distinct estimates. Full-410 daily P10–P90 is provisional, especially for the four homes with no Total actuals. No target-day actuals are supplied, so neither forecast can be scored for this date.

- [Final forecast comparison](16-FEB28_FORECAST_COMPARISON.csv#L1): DailyPoint_MWh and DailyP10/P50/P90_MWh supply the final totals and daily uncertainty. ActualAvailable=False.
- [A final household forecasts](https://github.com/sudipphuyal/rwth_hackathon/blob/main/Approach%20A/outputs/energy_forecasting/predictions/2024-02-28_households.csv#L1): Sum all 410 × 96 Prediction values and divide by 1,000 for A’s point total.
- [Updated B description](18-SummerSchool_OnePage_Description_410_Households_Updated_Results.docx): The final target and method breakdown report 12,011.58 kWh, 372 Ridge, 11 previous-week, 23 historical and four portfolio forecasts.
- [Final report](01-Final-Report.docx): Final Day Ahead Forecast confirms all-410 coverage and absence of actual demand.

## Slide 13: Results and remaining limitations

Coverage = 100 × number of actual values inside P10–P90 ÷ number of scored values. A quarter-hour coverage is 100 × 0.78957760989 = 78.957761%, shown as 79.0%. Daily coverage is 100 × 0.73351648352 = 73.351648%, shown as 73.4%.

A household gains do not establish that A beats B or that every aggregate metric improves. Daily error and interval coverage are practical limitations. Further validation needs B saved predictions, matched issue times/supports, dated metadata, fresh uncertainty calibration and actual imbalance settlement costs. No production procurement claim or real monetary saving follows from the demonstration.

- [A error metrics](05-summary_metrics.csv#L1): Annual error scores show the modest portfolio improvement.
- [Uncertainty test results](12-test_evaluation_complete.json#L1): metrics.ML.Test.Coverage and metrics.ML.TestDaily.Coverage yield 79.0% and 73.4%.
- [Final report limitations](01-Final-Report.docx): Lists different historical supports/cutoffs and uncertainty scope.
- [Procurement limitations](29-PROCUREMENT_REPORT.md#L1): The demonstration uses hypothetical normalized coefficients and incomplete full-portfolio ground truth.

## Slide 14: Supporting detail: Approach A weather stage

For each weather variable, test MAE = mean(|forecast − observed|) on observed common support. The slide rounds the saved test rows. Weather predictions, rather than tomorrow’s realised observations, are then inputs to demand forecasting.

A forecasts temperature and humidity with HGB, while validation selected an available same-hour baseline for sunshine. Absent sunshine at three stations remains missing. Weather bounds/night treatment use historical training-only rules. Expanding out-of-time weather predictions provide energy-training inputs; validation/test use appropriately frozen models. This avoids using target-day realised weather as a deployable covariate.

- [Weather model selection](11-WEATHER_REPORT.md#L49): The locked selection uses HGB for temperature/humidity and available_same_hour for sunshine.
- [Weather held-out metrics](10-selected_performance.csv#L1): Use Period=test rows. MAE values are 2.504354°C, 0.166881 hours/hour and 8.431613 percentage points.
- [Weather-to-energy integration](04-ENERGY_REPORT.md#L7): Explains out-of-time training weather forecasts and locked test-weather inputs.

## Slide 15: Supporting detail: Approach A feature study

The bars are saved validation WAPE values: 16.1447, 15.3213, 15.2821, 15.2232, 14.1276 and 14.1423%. They compare feature groups in one configuration. They do not isolate causal effects, and the recent-plus-PV value is slightly worse than recent behaviour without explicit PV.

The feature bars are validation results for one model configuration and support set, not causal importance estimates. History means history plus calendar/quality inputs. Later bars add groups cumulatively. Recent + PV compares explicit PV to the recent-behaviour design, rather than all PV/EV effects. The 75/25 lag blend achieved lower validation portfolio WAPE (13.2024%) than the selected ML (14.1276%). Test results did not reselect features or weights.

- [Feature-group comparison](09-validation_comparison.csv#L1): Match groups A, B, C, D, D_plus_recent and E_plus_recent to the six plotted feature sets.
- [Validation ablation table](04-ENERGY_REPORT.md#L195): Sections 7 and 6 describe the values and which inputs enter each group.

## Slide 16: Supporting detail: 27 February test profile

For every interval, residual = predicted − actual kWh. Positive means overprediction. Daily error = 100 × (12.104080562 − 11.344351) ÷ 11.344351 = +6.696986%, shown as +6.70%. Portfolio interval MAE = mean(|residual|) = 10.112742 kWh.

Residual means forecast minus actual. Positive values denote surplus relative to demand under point procurement, while negative values denote shortage. The daily energy error is computed from summed forecasts/actuals and differs from the mean absolute interval error. All 96 slots on this date use the same 374 observed homes.

- [Frozen test portfolio file](08-test_portfolio.parquet): Filter 2024-02-27, compare MatchedPrediction to MatchedActual, and subtract for the residual.
- [Historical comparison](14-COMMON_SUPPORT_COMPARISON.csv#L1): The final row gives A’s 374-home totals, +6.696986% error and interval MAE 10.112742 kWh.

## Slide 17: Supporting detail: Approach B eligibility

1,344 = 14 days × 96 intervals. Complete training rows can span more than 14 calendar days. Calibration requires the complete last 14 days. If eligibility fails, the fallback routes supply forecasts rather than fabricated training labels.

1,344 equals 14 × 96. Pre-calibration training rows need not be a contiguous fortnight, but all 14 immediately preceding calibration days must be complete. Fallbacks run in week/day/historical/portfolio order; zero previous-day routes occurred in the final reported breakdown. The portfolio reference is the mean forecast across 372 Ridge households. Tree/clustering alternatives are described in the B report, but their experiment artifacts were not supplied and do not justify new measured performance claims.

- [B eligibility logic](17-SummerSchool_410_Household_Forecast.py#L439): create_ridge_dataset requires >=1,344 complete training rows, exactly 1,344 calibration rows and complete day/week future lags.
- [B fallback logic](17-SummerSchool_410_Household_Forecast.py#L1365): Specifies the fallback hierarchy.
- [B final routing report](18-SummerSchool_OnePage_Description_410_Households_Updated_Results.docx): Provides the final 372 + 11 + 23 + 4 household breakdown.

## Slide 18: Supporting detail: uncertainty by approach

A coverage comes from the saved uncertainty test results, calculated as in slide 12. A derives daily bounds from daily residual sums because quantiles do not generally add. B uses q = quantile of pooled absolute calibration errors at min(1, ceil((n+1) × 0.80)/n), with the higher quantile convention. This is a nominal marginal rule, with temporal dependence and distribution-shift limitations.

A actual test coverage 79.0% quarter-hour / 73.4% daily refers to retrospective matched measured support. Its full-410 interval relies on provisional targetless-household proxy residuals. Nominal coverage is not measured coverage. Neither model has a simultaneous 96-interval coverage guarantee. Daily quantiles do not equal the sum of interval quantiles. B intervals concern Ridge households only and no full-portfolio daily bounds or empirical coverage results were supplied.

- [A uncertainty method](13-UNCERTAINTY_REPORT.md#L1): Explains validation residuals, temperature conditioning, matched support and separately calibrated daily quantiles.
- [A uncertainty results](12-test_evaluation_complete.json#L1): metrics.ML.Test and metrics.ML.TestDaily provide test coverage.
- [B conformal calculation](17-SummerSchool_410_Household_Forecast.py#L1279): Pooled absolute calibration errors supply the marginal Ridge household quantile.

## Slide 19: Approach A: purchase quantities and penalty ratios

Quantile probability = shortage penalty / (shortage penalty + surplus penalty). Each purchase total is SumIntervalProcurement_MWh for Method=ML. Each percentage is RelativeCostReductionPct for Method=ML and Level=Quarter-hour. Final-day volumes and historical test penalties describe different periods.

The challenge expressly permits assumed costs/prices (Approach A/README.md, line 147). There are no supplied or fetched historical tariffs, bills or settlement-price series in this evaluation. We use constant hypothetical penalty coefficients, not a price-prediction model. The 24.09% is quarter-hour imbalance penalty reduction on 34,944 supported intervals over 28 Feb 2023–27 Feb 2024, comparing A P80 to A point-forecast procurement. It is not A versus B, a reduction in physical demand, or a saving in euros. Method ML means Approach A; file Scenario A/B/C/D are cost scenarios. Procurement Blend means A’s lag baseline, not Approach B.

For coefficients 4 and 1, q=4/(4+1)=0.8. At each interval, shortage=max(actual−quantity,0) and surplus=max(quantity−actual,0). Loss=4×shortage+1×surplus. Buying P80 reduces expensive shortages and increases cheaper surplus. Point totals: shortage 260,961.327280 kWh, surplus 274,758.930694 kWh, loss 1,318,604.239813 normalized units. P80 totals: shortage 103,875.960108 kWh, surplus 585,422.556618 kWh, loss 1,000,926.397052 units. Reduction=100×(1,318,604.239813−1,000,926.397052)/1,318,604.239813=24.091978%.

The rates are sensitivity scenarios, not estimated historical cost ratios. Equal 1:1 penalties give −0.88%, so the policy does not improve every assumed objective. The 26.22% separately calibrated daily net-cost result differs from the 24.09% interval result. No test-based policy retuning occurred. Expected-loss optimality under the forecast distribution does not guarantee realized loss improvement.

For the final day, the separately calibrated 4:1 daily recommendation is 14.011 MWh, while the sum of interval P80 quantities is 14.998 MWh. They use different settlement objectives and are not a reconciled execution schedule. Actual demand is absent, so realized final-day cost cannot be verified. The prototype excludes purchase-price terms, time-varying prices, transaction costs, bid constraints, liquidity and real execution. These must be added before operational use.

- [Challenge allows assumed costs](46-README.md#L147): The challenge explicitly permits assumed costs/prices for business comparisons.
- [Scenario definitions](22-cost_scenarios.csv#L1): C_under, C_over and OptimalQuantile define the four hypothetical scenarios.
- [Saved cost-reduction percentages](26-value_of_uncertainty.csv#L1): Select Method=ML and Level=Quarter-hour. The A–D Scenario rows give PointCost, QuantileCost and RelativeCostReductionPct.
- [Underlying test policy metrics](27-test_policy_metrics.csv#L1): Select Method=ML, Level=Quarter-hour and Policy=PointForecast or CostOptimalQuantile to retrieve costs, shortages and surpluses.
- [Cost and quantile functions](32-procurement_support.py#L25): optimal_quantile implements the cost-ratio quantile. costs implements shortage, surplus and normalized loss.
- [Percentage calculation](32-procurement_support.py#L190): RelativeCostReductionPct = 100 × (1 − CostOptimalQuantile cost / PointForecast cost).
- [Procurement explanation and limits](29-PROCUREMENT_REPORT.md#L1): Defines normalized hypothetical cost units, the 34,944 supported intervals and the limits of the retrospective evaluation.
- [Final-day daily and interval recommendations](23-2024-02-28_daily_procurement.csv#L1): Distinguishes SeparatelyCalibratedDailyProcurement_MWh from SumIntervalProcurement_MWh.

## Slide 20: 28 February: forecast and purchase recommendation

Point = 13.014908666 MWh. Sum of 96 P80 interval quantities = 14.997672233 MWh. Historical reduction = 24.091978%. The separately calibrated daily P80 is 14.011264223 MWh and belongs to a different settlement objective.

This is a conditional demonstration using the existing Approach A model, not a test-selected model or a universal recommendation. The 4:1 ratio is an illustrative sensitivity case. Actual tariffs are absent. Expected linear imbalance-loss optimality under the estimated forecast distribution does not establish optimal real electricity expenditure or guarantee realized savings. Historical scoring uses matched measured household subsets. Four targetless homes have provisional uncertainty. The model omits purchase-price terms, fees, bid constraints and execution. The delivery grid is a 96-interval UTC day.

- [Saved final-day demand and purchase totals](23-2024-02-28_daily_procurement.csv#L4): Method=ML, Scenario=C gives the 13.014909 MWh point and 14.997672 MWh sum of interval P80 purchases.
- [Editable recommendation curve data](24-2024-02-28_ML_procurement_schedule.csv#L1): The 96 PointForecast_kWh and Q_C_kWh rows provide both curves, rounded to six decimals for embedded Excel compatibility. Original Q_C_kWh sums to the displayed purchase total.
- [Historical simulated penalty reduction](26-value_of_uncertainty.csv#L6): ML, C, Quarter-hour gives 1,318,604.239813 point units and 1,000,926.397052 quantile units, a 24.091978% reduction.
- [Cost assumptions, frozen calibration and limits](29-PROCUREMENT_REPORT.md#L5): Describes expected linear loss, illustrative coefficients, locked validation calibration, observed test support and absent final-day actuals.

## Slide 21: Thank You







## Supporting procurement quantities from the complete CSV

ML means Approach A. Blend is its lag benchmark and is not Approach B. All quantities below are MWh for 28 February 2024. Daily purchases use a separately calibrated net daily objective. Interval purchases sum 96 separately chosen quantities. These columns represent different settlement objectives and are not a reconciled schedule.

| Model | Shortage : surplus ratio | Percentile | Point forecast | Daily purchase | Sum of interval purchases |
|---|---|---|---:|---:|---:|
| Approach A (ML) | 1:1 | P50 | 13.015 | 12.381 | 13.047 |
| Approach A (ML) | 2:1 | P66.7 | 13.015 | 13.195 | 13.971 |
| Approach A (ML) | 4:1 | P80 | 13.015 | 14.011 | 14.998 |
| Approach A (ML) | 1:2 | P33.3 | 13.015 | 11.679 | 12.104 |
| Lag blend | 1:1 | P50 | 12.714 | 12.641 | 12.609 |
| Lag blend | 2:1 | P66.7 | 12.714 | 13.171 | 13.462 |
| Lag blend | 4:1 | P80 | 12.714 | 13.759 | 14.375 |
| Lag blend | 1:2 | P33.3 | 12.714 | 12.169 | 11.812 |

Source: [Final-day procurement CSV](23-2024-02-28_daily_procurement.csv)

The 4:1 daily policy has 26.22% lower historical net daily penalty. The 4:1 interval policy has 24.09% lower historical interval penalty. Each compares Approach A’s quantile policy with its own point forecast under the same assumptions. Actual electricity prices and final-day measurements are unavailable.
