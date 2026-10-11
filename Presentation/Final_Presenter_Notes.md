# Day Ahead Residential Energy Forecasting

Presenter notes from the final 21-slide presentation.

## Slide 1: Day Ahead Residential Energy Forecasting

Our project is Day Ahead Residential Energy Forecasting. We developed two approaches to predict next-day demand across 410 households. The delivery day uses UTC

## Slide 2: Two approaches to the forecasting task

We tried to solve the same forecasting problem with two approaches.

Approach A uses predicted weather, recent demand and household information.

Approach B uses recurring consumption patterns through a simpler Ridge

regression model. Both produce household and portfolio forecasts. I will explain Approach A first, then Approach B, before comparing the results.

## Slide 3: Approach A: data and information considered

Approach A uses meter history, eight weather stations and household surveys to capture demand patterns and heating context. The target is received electricity in kilowatt-hours, abbreviated kWh. Four homes lack usable Total demand history. Direct photovoltaic generation, charging schedules and actual procurement prices were unavailable.

## Slide 4: Approach A: cleaning and missing data

Approach A checks timestamps, household identifiers and energy consistency. Measured zeros and extreme readings remain. Missing demand stays missing. No totals were reconstructed. Unknown survey fields remain unknown. Homes without usable history receive separate forecasts. UTC means Coordinated Universal Time.

## Slide 5: Approach A: features and demand model

A combines predicted weather, recent demand and heating information. Histogram-based gradient boosting regression combines small decision trees to learn nonlinear patterns. Squared error emphasises larger mistakes. L-two regularization adds a penalty to discourage overly large tree-leaf predictions. Photovoltaic, or PV, and electric vehicle, or EV, ownership remain in the model for homes without history.

## Slide 6: Approach A: training, validation and test

We trained chronologically, selected the configuration on validation, then refitted and froze weights before testing. Forecasts use information available at eleven in the morning, Berlin time, on the preceding day. Recent observations update inputs without retraining.

## Slide 7: Approach A: actual versus predicted test demand

The lines compare daily energy on identical measured households. Household mean absolute error, or MAE, improves 7.7 percent. Root mean squared error, or RMSE, improves 16.1 percent. Portfolio weighted absolute percentage error, or WAPE, improves slightly. Daily-total error worsens. The benchmark blends 75 percent previous available same-time demand with 25 percent previous-week demand. MWh means megawatt-hour.

## Slide 8: Approach A: variation within the test day

These curves show errors hidden by daily totals: March is underpredicted. September is overpredicted. kWh means kilowatt-hour, measured here per fifteen minutes. MWh means megawatt-hour. UTC means Coordinated Universal Time. Both curves use identical measured households.

## Slide 9: Approach B: cleaning and missing data

B removes unusable observations and duplicate timestamps, then creates a regular quarter-hour grid in Coordinated Universal Time. Gaps remain missing. Complete rows train the model. Other homes receive week, day, historical-profile or portfolio fallbacks.

## Slide 10: Approach B: training and uncertainty

B standardises six demand-lag and calendar features. Ridge regression is a linear model that restrains large coefficients. Alpha, set to one, controls that penalty. Fourteen days calibrate nominal eighty-percent household intervals, separately from training. It uses the complete preceding UTC, or Coordinated Universal Time, day.

## Slide 11: Reported historical results for both approaches

These three-date results include B's supplied report values. Error means signed daily percentage error. Different measured households and information cutoffs prevent an overall ranking. MWh means megawatt-hour.

## Slide 12: Final day-ahead forecast: 28 February 2024

Both final forecasts cover 410 homes: A predicts 13.015 megawatt-hours. B predicts 12.012. MWh means megawatt-hour. A's provisional P10-to-P90 range spans estimated demand's tenth to ninetieth percentiles. No final-day actuals were supplied, so accuracy is unknown.

## Slide 13: Results and remaining limitations

A improves household errors. B offers simplicity. A's uncertainty intervals contain measured demand in 79.0 percent of quarter-hours and 73.4 percent of days, below the eighty-percent target. We need aligned evaluations and better calibration. Procurement reductions use assumed penalties, not verified monetary savings.

## Slide 14: Supporting detail: Approach A weather stage

Approach A forecasts weather before forecasting energy demand. Temperature and humidity use histogram-based gradient boosting regression, which combines decision trees to learn nonlinear relationships. Validation selected the most recent available matching-hour observation for sunshine. MAE means mean absolute error: the average size of a prediction's error, regardless of direction. Temperature test error is 2.50 degrees Celsius. The temperature unit is degrees Celsius. Sunshine error is 0.167 hours of sunshine per hour, and humidity error is 8.43 percentage points. Weather inputs for energy training were produced using only observations available before each historical forecast. Hourly weather observations end two hours before issue, and absent sunshine remains unknown. Tomorrow's realised weather is never supplied as a model input.

## Slide 15: Supporting detail: Approach A feature study

This validation study compares successive information groups on the same measured households. WAPE means weighted absolute percentage error: total absolute error divided by total actual demand, expressed as a percentage. Lower values are better. Adding predicted weather, building information and heating context reduces the error. Heating context includes heat pumps and domestic hot water, often abbreviated DHW. Recent demand behaviour gives a further improvement. PV means photovoltaic. Explicit photovoltaic ownership does not improve this selected configuration. This does not show that solar equipment has no physical effect: existing demand history may already capture some of that behaviour. These are predictive comparisons, not causal energy savings. The lag-blend benchmark actually performs better on validation, so it remains a serious reference rather than an easy comparison.

## Slide 16: Supporting detail: 27 February test profile

On 27 February, Approach A predicts 12.104 megawatt-hours against 11.344 observed across the same 374 measured households. MWh means megawatt-hour. The signed daily error is positive 6.70 percent, indicating overprediction overall. The upper graph shows energy during each fifteen-minute interval in kWh, meaning kilowatt-hours. The lower graph shows forecast minus actual energy: positive values mean overprediction. Negative values mean underprediction. Their changing signs show why a daily total cannot describe every interval's error. UTC means Coordinated Universal Time, used for the hour axis. Both graphs use all 96 intervals from the frozen test predictions. These forecasts were not adjusted afterwards to make the curves agree.

## Slide 17: Supporting detail: Approach B eligibility

Ridge is regularized linear regression: it fits a linear demand relationship while discouraging large coefficients. A household needs at least 1,344 complete training rows before calibration, equivalent to fourteen days of quarter-hour observations, although those rows need not be consecutive. The immediately preceding fourteen calibration days must all be complete, and every forecast interval needs both its day and week demand lags. A household failing these checks moves through the fallback hierarchy: previous week, previous day, historical average by quarter-hour, then the portfolio reference. That reference is the mean forecast profile of the 372 Ridge households. The final day uses 372 Ridge forecasts, eleven previous-week profiles, twenty-three historical profiles and four portfolio references. No household uses the previous-day fallback. All time slots follow UTC, meaning Coordinated Universal Time. Predictions for the four homes without usable actual demand cannot be directly validated.

## Slide 18: Supporting detail: uncertainty by approach

The two uncertainty methods apply to different targets. Approach A uses validation errors to estimate portfolio demand percentiles. P10 is the tenth percentile, P50 is the fiftieth percentile or median, and P90 is the ninetieth percentile. Under the estimated distribution, ten, fifty and ninety percent of demand outcomes respectively fall at or below these values. The P10-to-P90 range targets eighty-percent coverage. Daily percentiles are calibrated from daily errors separately. Adding quarter-hour percentiles does not produce the daily percentile. On measured test data, A's ranges contain actual demand in 79.0 percent of quarter-hours and 73.4 percent of days. Full-portfolio ranges are provisional, particularly for four homes without measured demand. Approach B uses the preceding fourteen days to calibrate nominal eighty-percent intervals for Ridge household quarter-hours. These do not establish a daily portfolio range. Changing demand patterns and time dependence can reduce coverage, and neither approach guarantees simultaneous coverage of every household and interval.

## Slide 19: Approach A: purchase quantities and penalty ratios

The ratio compares the assumed penalty per missing kilowatt-hour with the penalty per extra kilowatt-hour. A shortage means buying less than actual demand. A surplus means buying more.

One to one means shortage and surplus cost the same per kilowatt-hour. Two to one means shortage costs twice as much. Four to one means shortage costs four times as much. One to two means surplus costs twice as much. These are illustrative assumptions because no actual tariffs or historical procurement prices were supplied.

We divide the shortage penalty by the sum of both penalties to choose a demand percentile. One to one gives the fiftieth percentile, P50. Two to one gives P66.7. Four to one gives four divided by five, or eighty percent, so P80. One to two gives P33.3. P means percentile. P80 means eighty percent of outcomes in the estimated demand distribution lie at or below that amount. It does not mean buying eighty percent of the point forecast.

The purchase column adds the ninety-six fifteen-minute purchase amounts for February twenty-eighth, twenty twenty-four. MWh means megawatt-hours, and one megawatt-hour equals one thousand kilowatt-hours. Under four to one, the interval quantities total fourteen point nine nine eight megawatt-hours. The point forecast remains thirteen point zero one five megawatt-hours.

The final column describes a different period: our historical test. For four to one, assumed penalty falls from 1,318,604 to 1,000,926 units. The decrease divided by the original total gives 24.09 percent. It is a simulated penalty reduction, not the amount to buy or a forecast of euro savings. Equal penalties increased test penalty by 0.88 percent, so improvement depends on the assumptions.

In the source file, ML means machine learning and refers to Approach A. Blend is A’s lag benchmark, not Approach B. The file also gives a separate fourteen point zero one one megawatt-hour purchase for net daily settlement. That calculation allows shortages and surpluses at different times to cancel. Our table uses interval settlement, which charges each interval separately. A percentile of a daily sum differs from the sum of interval percentiles, so we keep the two decisions separate.

## Slide 20: 28 February: forecast and purchase recommendation

For February twenty-eighth, twenty twenty-four, Approach A estimates total demand at thirteen point zero one five megawatt-hours across all four hundred and ten households.

If each missing kilowatt-hour costs four times as much as each extra kilowatt-hour, our conditional recommendation is the eightieth percentile, P80, in each fifteen-minute interval. The ninety-six purchase quantities total fourteen point nine nine eight megawatt-hours. The blue curve shows these amounts above the red point forecast. UTC means Coordinated Universal Time, and kWh means kilowatt-hours.

We calibrated demand uncertainty on validation data and kept the rule fixed for the historical test. Under the same four-to-one assumption, it gave twenty-four point zero nine percent less simulated penalty than purchasing the point forecast on the same measured intervals. That evidence supports demonstrating the rule. It does not prove savings for February twenty-eighth, whose actual demand is unavailable.

The four-to-one ratio is an example, not a ratio estimated from actual costs. We recommend using the real shortage and surplus penalties to set the percentile before making a commercial purchase. Optimal here means minimizing expected linear imbalance penalty under our estimated demand distribution and assumed costs. We have not modelled actual electricity prices, fees or purchase constraints. Four homes have no measured demand history, so their uncertainty remains provisional.

## Slide 21: Thank You

Thank you for your attention. We welcome your questions.
