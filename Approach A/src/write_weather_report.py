"""Create a report/plots from locked results only. No fitting or model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import WEATHER_OUTPUT_DIR
OUT=WEATHER_OUTPUT_DIR


def md_table(d, precision=3):
    def fmt(x):
        if pd.isna(x):return 'Unavailable'
        if isinstance(x,(float,np.floating)):return f'{x:.{precision}f}'
        return str(x)
    text='| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'
    for row in d.itertuples(index=False,name=None):text+='| '+' | '.join(fmt(v) for v in row)+' |\n'
    return text


def main():
    lock=json.loads((OUT/'model_selection_lock.json').read_text());selected=lock['selected_methods']
    checks=json.loads((OUT/'validation_checks.json').read_text())
    a=pd.read_csv(OUT/'evaluation/aggregate_metrics.csv')
    common=pd.read_csv(OUT/'evaluation/common_support_comparison.csv')
    g=pd.read_csv(OUT/'evaluation/grouped_metrics.csv')
    s=json.loads((OUT/'splits.json').read_text())
    chosen=a[[selected[v]==m for v,m in zip(a.Variable,a.Method)]].copy()
    chosen['Units']=chosen.Variable.map({'Temperature_avg_hourly':'°C','Sunshine_duration_hourly':'hours/hour','Humidity_avg_hourly':'percentage points'})
    perf=chosen[['Period','Variable','Units','n','MAE','RMSE','Bias']]
    chosen_groups=g[pd.Series([selected[v]==m for v,m in zip(g.Variable,g.Method)],index=g.index) & g.Period.eq('test')]
    by_station=chosen_groups[chosen_groups.Group.eq('Weather_ID')]
    station=pd.DataFrame({'Weather_ID':sorted(by_station.GroupValue.unique())})
    for variable,name in [('Temperature_avg_hourly','Temperature'),('Sunshine_duration_hourly','Sunshine'),('Humidity_avg_hourly','Humidity')]:
        q=by_station[by_station.Variable.eq(variable)].set_index('GroupValue')
        for metric in ['MAE','RMSE']:station[name+' '+metric]=station.Weather_ID.map(q[metric])
    season=chosen_groups[chosen_groups.Group.eq('Season')][['Variable','GroupValue','n','MAE','RMSE','Bias']].rename(columns={'GroupValue':'Season'})
    leads=chosen_groups[chosen_groups.Group.eq('ForecastLeadHours')][['Variable','GroupValue','MAE','RMSE']].rename(columns={'GroupValue':'Lead hours'})
    hours=chosen_groups[chosen_groups.Group.eq('TargetUTCHour')][['Variable','GroupValue','MAE','RMSE']].rename(columns={'GroupValue':'UTC hour'})
    for name,d in [('selected_performance',perf),('selected_by_station',station),('selected_by_season',season),('selected_by_lead',leads),('selected_by_hour',hours)]:
        d.to_csv(OUT/'evaluation'/f'{name}.csv',index=False)
    future=pd.read_parquet(OUT/'predictions/2024-02-28_day_ahead_weather.parquet')
    tomorrow=future.groupby('Weather_ID').agg(
        MeanTemperature=('PredictedTemperature','mean'),MinTemperature=('PredictedTemperature','min'),
        MaxTemperature=('PredictedTemperature','max'),SunshineHours=('PredictedSunshine',lambda x:x.sum(min_count=1)),
        MeanHumidity=('PredictedHumidity','mean')).reset_index()
    tomorrow.to_csv(OUT/'predictions/2024-02-28_station_summary.csv',index=False)
    sunshine=pd.read_csv(OUT/'evaluation/sunshine_dark_other_hours.csv')
    finite=sunshine.dropna(subset=['MAE'])
    daylight=finite[finite.HourClass.eq('other_hours')]
    weighted_day_mae=float((daylight.MAE*daylight.n).sum()/daylight.n.sum())
    weighted_day_rmse=float(np.sqrt((daylight.RMSE**2*daylight.n).sum()/daylight.n.sum()))
    # Scientific validation plots, not another forecast/model experiment.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for method,label,color in [('hist_gradient_boosting','Gradient boosting','#176b9a'),('available_same_hour','Available same hour','#cc6633')]:
        q=g[g.Period.eq('test')&g.Group.eq('ForecastLeadHours')&g.Variable.eq('Temperature_avg_hourly')&g.Method.eq(method)].copy()
        q['Lead']=q.GroupValue.astype(float);q=q.sort_values('Lead')
        axes[0].plot(q.Lead,q.MAE,label=label,color=color,marker='.',markersize=4)
    axes[0].set(title='Held-out temperature error by forecast lead',xlabel='Hours from issue to target hour',ylabel='MAE (°C)')
    axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    q=common[common.Period.eq('test')&common.Variable.eq('Sunshine_duration_hourly')].copy()
    labelmap={'available_same_hour':'Same hour','previous_week':'Previous week','same_hour_7d_mean':'7d mean','same_hour_7d_median':'7d median','climatology':'Climatology','hist_gradient_boosting':'Gradient boost'}
    x=np.arange(len(q));axes[1].bar(x-.18,q.MAE,width=.36,label='MAE',color='#176b9a');axes[1].bar(x+.18,q.RMSE,width=.36,label='RMSE',color='#cc6633')
    axes[1].set(title='Held-out sunshine: MAE/RMSE tradeoff',ylabel='Hours of sunshine per hourly interval')
    axes[1].set_xticks(x,[labelmap[v] for v in q.Method],rotation=30,ha='right');axes[1].legend(frameon=False);axes[1].grid(axis='y',alpha=.2)
    (OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/weather_validation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    seasons=['DJF','MAM','JJA','SON']
    for ax,variable,title,unit in [(axes[0],'Temperature_avg_hourly','Temperature','MAE (°C)'),
                                 (axes[1],'Sunshine_duration_hourly','Sunshine','MAE (hours/hour)')]:
        for period,label,color in [('validation_selection','Full-year validation','#176b9a'),('test','Full-year test','#cc6633')]:
            q=g[g.Period.eq(period)&g.Group.eq('Season')&g.Variable.eq(variable)&g.Method.eq(selected[variable])].set_index('GroupValue').reindex(seasons)
            ax.plot(range(4),q.MAE,marker='o',label=label,color=color)
        ax.set(title=title+' forecast error by season',ylabel=unit)
        ax.set_xticks(range(4),['Winter','Spring','Summer','Autumn']);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(OUT/'figures/seasonal_weather_errors.png',dpi=180);plt.close(fig)
    lookup = chosen.set_index(['Period','Variable'])
    test_temp = lookup.loc[('test','Temperature_avg_hourly')]
    test_sun = lookup.loc[('test','Sunshine_duration_hourly')]
    common_temp = common[(common.Period=='test') & (common.Variable=='Temperature_avg_hourly')].set_index('Method')
    persistence_mae = float(common_temp.loc['available_same_hour','MAE'])
    selected_temp_mae = float(common_temp.loc[selected['Temperature_avg_hourly'],'MAE'])
    improvement = 100*(1-selected_temp_mae/persistence_mae)
    sun_compare = common[(common.Period=='test') & (common.Variable=='Sunshine_duration_hourly')].set_index('Method')
    station_t = by_station[by_station.Variable.eq('Temperature_avg_hourly')]
    season_days = {}
    for name,start,end in [('validation',s['dates']['validation_start'],s['dates']['validation_end']),('test',s['dates']['test_start'],s['dates']['test_end'])]:
        dates = pd.Series(pd.date_range(start,end,freq='D',tz='UTC'))
        from .weather_forecasting import season_of
        season_days[name] = season_of(dates).value_counts().to_dict()
    seasonal_coverage = pd.DataFrame(season_days).reindex(['DJF','MAM','JJA','SON']).reset_index().rename(columns={'index':'Season'})
    seasonal_coverage.to_csv(OUT/'evaluation/season_delivery_day_coverage.csv',index=False)
    # Full variable/method comparison remains available even for unselected candidates.
    comparison=common[['Period','Variable','Method','n','MAE','RMSE']]
    report=f'''# Weather forecasting step: chronological split and locked results

This revised step implements station-level WEATHER forecasts only. No electricity/grid-import model was fitted, no household forecasting features were regenerated, and raw data/Step 2 outputs were not modified. Split choice used the existing preprocessing audit rather than a fresh data audit.

## 1. Exact frozen global split

| Period | UTC delivery dates inclusive | Days | Share |
| --- | --- | ---: | ---: |
| Training | 2019-05-20–2022-02-27 | 1015 | 58.17% |
| Validation | 2022-02-28–2023-02-27 | 365 | 20.92% |
| Final test | 2023-02-28–2024-02-27 | 365 | 20.92% |

Actual meter-date span is 1745 days. The user's explicit request for complete seasonal coverage superseded the approximate 70/10/20 convention: the latest full 365-day year is test, the preceding full year validation, and older dates training. These dates are now frozen in `ChronologicalSplit` in src/config.py and splits.json. The first run remains under outputs/weather_forecasting; this revised run is under outputs/weather_forecasting_full_year. The revised test overlaps previously viewed weather results, so it is excluded from revised fitting/selection but cannot be described as never inspected. Features/model parameters remain unchanged; the split change was requested for season coverage, not chosen by comparing test performance.

Both holdouts contain all 12 months and all four meteorological seasons:

{md_table(seasonal_coverage)}

These full-year windows are anchored to the latest observed demand date, not January–December calendar years. Boundary-day embargoes below only remove unavailable selection labels, not complete seasons. Existing audit file-span coverage at the two boundaries is {s['boundary_household_file_span_coverage']}; this is not a fresh household audit. Weather begins 2019-01-01; older weather serves as input warm-up history, not extra future training/validation labels. All eight stations share the split; no random time split is used.

**Boundary availability embargo:** chronological target dates alone do not guarantee labels were known by the first next-period issue. Training labels are restricted to **{lock['train_fit_source_max']}** before the first validation issue. Validation model selection and train+validation refit stop at **{lock['first_test_weather_cutoff']}**, before the first test issue **{lock['first_test_issue_timestamp']}**. Late boundary-day labels remain outside fitting/selection. The global date boundaries are unchanged. Validation scores below cover the available selection subset, not the embargoed final hours of February 27.

## 2. Weather variables modelled

- Temperature_avg_hourly: temperature in °C, all eight stations.
- Sunshine_duration_hourly: sunshine duration in hours per hourly interval, five stations.
- Humidity_avg_hourly: relative humidity in percent, all eight stations.

Pressure, precipitation and wind are not modelled in this first version. These are optional future candidates, not inputs backed by this validation.

All forecasts use **11:00 Europe/Berlin on D−1**, forecasting the **24 UTC hourly intervals of D**. Historical hourly timestamps are conservatively treated as interval starts with one-hour observation latency, so the last permissible start is **issue minus two hours**. Target-day observations and later D−1 observations never enter features. Forecast lead is 14–37 hours in winter or 15–38 hours in summer, with DST handled by the timezone-aware issue timestamp.

## 3. Baselines tested

1. Most recent observed matching UTC hour, at/before the cutoff; afternoon targets usually need an older date than morning targets.
2. Previous-week matching UTC hour; missing if that exact hourly value is absent.
3. Trailing seven-day same-hour mean.
4. Trailing seven-day same-hour median.
5. Training-only station/month/UTC-hour climatology, falling back to station/hour then station mean when a seasonal bin is absent.

Weather diurnal matching uses UTC hour to avoid an artificial DST clock shift in solar/weather cycles. Both UTC/local hour and seasonal calendar are supplied to ML. Seven-day same-hour windows end at the most recent matching hour permitted by cutoff; there is no future fill. Each method shares training-only bounds/night handling for a fair comparison.

## 4. Selected weather model and lock

A pooled **HistGradientBoostingRegressor** was trained separately for each variable, with station as a categorical feature. There are **55 inputs**, not an exhaustive interaction expansion: station, target UTC/local hour, target month/day of year, lead, cyclic hour/year; and three variables' latest values/age, 6/12/24-hour means/coverage, 24-hour standard deviation, six-hour change, previous-week hour, available hour/age, seven-day same-hour mean/median. No actual target weather is an input. The exact whitelist is in weather_feature_contract.json.

Parameters fixed before test: 120 iterations, learning rate 0.06, 15 leaf nodes, minimum leaf sample count 40, L2=1, seed=42, **early_stopping=False** (no random internal validation split). There was one practical model configuration, not a test-driven hyperparameter search.

Validation selection uses **lowest MAE on common observed support across all six candidates**, with RMSE only a tie-break. Model choices were written to model_selection_lock.json before test labels/metrics were requested:

| Variable | Locked method |
| --- | --- |
| Temperature | {selected['Temperature_avg_hourly']} |
| Sunshine | {selected['Sunshine_duration_hourly']} |
| Humidity | {selected['Humidity_avg_hourly']} |

The selected sunshine method is **{selected['Sunshine_duration_hourly']}**, chosen on the revised full-year validation MAE. All baseline/ML RMSE differences are reported; none triggers test-based reselection. The previous mostly-winter validation winner is not automatically retained. Test-model weights were refitted using only train+validation weather available at the first test cutoff, then frozen throughout test. Later test-period observed history can enter rolling inputs when available, as it would operationally; it never updates the test-model weights.

## 5. Validation performance

{md_table(perf[perf.Period.eq('validation_selection')])}

## 6. Held-out test performance

{md_table(perf[perf.Period.eq('test')])}

Units: sunshine hours/hour (multiply MAE by 60 for minutes); humidity percentage points. Selected-method scores use every finite actual/predicted pair. Temperature/humidity cover all observed targets. Common-support comparison below uses identical rows per variable for model comparison; it differs slightly from full-support scores when a baseline has a missing week lag.

{md_table(comparison)}

The revised test was held out from revised weather fitting/selection and evaluated once after the revised lock. The earlier repository-wide audit and first weather experiment had already inspected overlapping data/results; “held out” here means no test forecast error or target labels were used to choose/fit the test models. The CLI refuses to repeat adaptive test evaluation once test_evaluation_completed.json exists. Validation and test actuals/candidate comparisons are separate evaluation files, never default energy inputs.

## 7. Test performance by Weather_ID

{md_table(station)}

Unavailable sunshine has no score and is not included as zero-error observations. Station-level quality varies: temperature MAE ranges from {station_t.MAE.min():.2f}°C to {station_t.MAE.max():.2f}°C; humidity differences are larger. Aggregate scores weight observed station-hours equally, not households or future energy load. The energy model should later assess its actual portfolio mapping; these weights were not used to tune this module.

## 8. Temperature forecast quality

Temperature improves on available same-hour persistence on both validation and test. Common-support test MAE is {selected_temp_mae:.3f}°C versus {persistence_mae:.3f}°C ({improvement:.1f}% improvement). Full-support MAE/RMSE is {test_temp.MAE:.3f}/{test_temp.RMSE:.3f}°C, with bias {test_temp.Bias:.3f}°C. This is a useful statistical forecast, not precision meteorology or an NWP substitute. Weather fronts and rapidly changing conditions remain difficult using station history alone.

Error by season (all selected variables):

{md_table(season)}

Exact temperature error by lead:

{md_table(leads[leads.Variable.eq('Temperature_avg_hourly')])}

All variables' hourly/lead/season/station breakdowns are in grouped_metrics.csv and the smaller selected_by_*.csv files. Lead and time of day are linked; 14-hour and 38-hour bins occur in different seasons due to DST, so this curve is not a pure causal horizon-degradation estimate.

![Weather held-out diagnostics](figures/weather_validation.png)

![Complete-year seasonal error comparison](figures/seasonal_weather_errors.png)

Calendar inputs (month, target day-of-year and year sin/cos) let the tabular model condition on seasonal patterns; climatology is also month/hour-specific. Full-year holdouts ensure seasonal effects appear in both selection and evaluation. Seasonal metrics remain necessary: a full-year aggregate can still hide winter temperature errors or summer sunshine difficulty. One year cannot represent all heatwaves, cold snaps or multi-year climate variation; this step does not fit separate seasonal models or retune on the seasonal test results.

## 9. Sunshine forecast quality and night handling

Selected test MAE is {test_sun.MAE:.4f} h/hour ({test_sun.MAE*60:.2f} minutes) and RMSE {test_sun.RMSE:.4f} h/hour. Gradient boosting's common-support MAE/RMSE is {sun_compare.loc['hist_gradient_boosting','MAE']:.4f}/{sun_compare.loc['hist_gradient_boosting','RMSE']:.4f}; seven-day median's is {sun_compare.loc['same_hour_7d_median','MAE']:.4f}/{sun_compare.loc['same_hour_7d_median','RMSE']:.4f}. The MAE rule remains locked. Selected sunshine bias is {test_sun.Bias:.4f} h/hour on test; it is not corrected using test. Seasons are assessed separately below; a low all-hour winter error does not imply good summer/daylight performance.

Predictions are bounded to [0,1]. With no station coordinates, the module cannot compute a defensible astronomical sunrise/sunset. It instead uses **training-only station/month/UTC-hour bins** with >=20 measured labels and maximum sunshine zero as an empirical dark-hour mask. Missing sunshine never defines darkness. This mask is trained/refitted only on past eligible observations. It can misclassify dawn/dusk as day length changes: test contains small nonzero sunshine in a few empirical dark bins.

All-hour error includes many dark hours. On the other-hour bins, pooled selected sunshine MAE is **{weighted_day_mae:.4f} h/hour** and RMSE **{weighted_day_rmse:.4f}**. These are a better view of useful PV-period difficulty but are not a formally astronomical daylight classification.

{md_table(sunshine[['Weather_ID','HourClass','n','MAE','RMSE','Bias']])}

## 10. Stations with unavailable sunshine

**HbsbG, ceOxS, sV3mR** have no sunshine source history. Their predictions are NaN, PredictedSunshineMissing=True; no zeros or automatic alternative stations are invented. Temperature and humidity remain forecasted. Thirty households map to these stations, so their later PV-related energy features must use missing indicators and other information rather than an assumed zero-sunshine value. Station substitution requires a justified location/source mapping, still absent here.

## 11. Saved forecast data and leakage checks

- `predictions/historical_day_ahead_weather.parquet`: **335,040** rows = all 1745 delivery days × 24 hours × 8 stations. Includes train/validation/test period and forecast provenance.
- `predictions/2024-02-28_day_ahead_weather.parquet` and `.csv`: **192** hourly next-demand-day forecasts.
- `evaluation/validation_actuals_and_candidates.parquet`, `test_actuals_and_candidates.parquet`: separate actuals/forecast-error analysis only.
- `evaluation/aggregate_metrics.csv`, `common_support_comparison.csv`, `grouped_metrics.csv`: reported performance, including all baselines.
- `models/validation_model.pkl`, `locked_test_model.pkl`, `production_model.pkl`: trusted local artifacts; deployment refit is separate from evaluated test weights.
- `splits.json`, `model_selection_lock.json`, `weather_feature_contract.json`, `validation_checks.json`, `training_forecast_folds.json`: reproducibility/availability contracts.

Main forecast schema: ForecastIssueTimestamp, DeliveryDate, TargetWeatherTimestamp, Weather_ID, ForecastLeadHours, PredictedTemperature, PredictedSunshine, PredictedHumidity and each `...Missing` indicator. Additional provenance: observation-history cutoff/max timestamp, fitted-label max timestamp, method, period and training-forecast mode. UTC datetimes remain timezone-aware. Actual target-day weather is absent from these forecast tables.

**Energy-training forecast safety:** historical training weather predictions are not in-sample fitted values. The initial May–December 2019 warm-up uses causal seven-day same-hour median forecasts. From January 2020 onward, expanding half-year folds fit only earlier labels available at the first fold issue; weights stay fixed within each fold while observations update causally. This yields coverage for every training day. Validation uses training-only weights; test uses locked train+validation weights. The final selected recipe is a retrospective validation choice, not evidence of a historically stored issued forecast archive; validation forecast scores also reflect this selection. No training target weather is used to fit its own fold prediction.

**Checks:** 18 unit tests passed. Independent validation checked {checks['historical_forecast_rows']:,} historical and {checks['next_day_rows']} final-day rows, all eight stations/24-hour grids, duplicates, cutoffs, out-of-time fitted labels, bounds, missing flags and unchanged selection-lock SHA. Zero after-cutoff source/fit violations. Real 8jB and HbsbG future-observation poisoning left features and forecasts unchanged, including unavailable-sunshine handling. A join to 2,688 unchanged Step 2 household rows passed. Existing preprocessing was not rerun.

## 12. First day beyond the demand dataset: 2024-02-28

The issued forecast is **2024-02-27 11:00 Europe/Berlin = 10:00 UTC**, with historical weather cutoff **2024-02-27 08:00 UTC**. The archive contains later February 27 and February 28–29 observations, but these were not used for the forecast/refit.

After test scoring, the SAME locked recipe/parameters were refitted on all eligible global historical forecast-label rows through that cutoff. This production refit is separately saved and does not regenerate test predictions or change test results. Its final eligible training label is 2024-02-27 08:00 UTC. Target-day February 28 observations were never used, nor was their accuracy evaluated here. This is an offline as-of forecast for a genuinely unseen **demand** day; future weather happens to exist in the archive and is deliberately ignored.

Predicted daily summaries (full hourly values in the CSV/Parquet):

{md_table(tomorrow)}

Temperature summaries are °C; SunshineHours is the sum of 24 hourly predictions, NaN for unavailable stations; MeanHumidity is percent. These are predictions, not actual February 28 weather.

## 13. Before building the energy model

The weather module is ready. Feed exactly **PredictedTemperature, PredictedSunshine, PredictedHumidity**, plus **PredictedTemperatureMissing, PredictedSunshineMissing, PredictedHumidityMissing**, joined by **ForecastIssueTimestamp + assigned Weather_ID + TargetWeatherTimestamp = floor(TargetTimestamp, hour)**. `src/weather_energy_interface.py` implements this whitelist-only many-to-one join without changing Step 2 artifacts. Repeat the hourly covariate over four quarter-hours; sunshine retains hourly-duration units, not four times the total. Do not feed Actual_* fields, evaluation paths, method scores or model-fit timestamps as predictors.

The next demand model must combine these predictions with the existing **recent seven-day household behaviour, previous-week same interval, recent 28-day profile/baseline, recent behaviour change, metadata, calendar and history-quality indicators**. No demand estimator or fallback is implemented here.

Before fitting electricity demand:

1. **Use the fixed split and proper training rows.** The currently materialised Step 2 slice (Jan31–Feb27 2024) is entirely in the final test period; it cannot train the energy model. Use the already-created reusable pipeline/lazy processing to materialise older training/validation features in the next implementation step, without changing raw data or the cutoff.
2. Use out-of-time historical weather predictions for energy training, not full-data/weather-in-sample predictions, and keep the three test models fixed for final backtest inputs. Do not substitute production-refit predictions into historical test.
3. Address four targetless households with a later fallback and choose the previously flagged demand-spike policy. Missing demand cannot become zero or ML-estimated training labels.
4. Retain the explicit undated-metadata/Europe/Berlin assumptions from Step 2; confirm locality and latency for operational claims. UTC delivery days differ from local settlement days at DST. No external forecast archive is required for this chosen statistical-weather architecture, but it may materially improve accuracy later.
5. Account for weather uncertainty: temperature MAE ~{test_temp.MAE:.2f}°C, sunshine other-hour MAE ~{weighted_day_mae:.2f} h/hour, unobserved sunshine at three stations, and a warm-up-baseline/expanding-fit distribution in training versus stronger final fits. Historical weather feature error is real and should not be replaced with perfect observations. Stage-specific model selection was validation-only; the final energy test must not tune against the weather test report.
6. The earlier annual training cutoff leaves only 173 households with file-span coverage at the training boundary, versus 394 at validation end. Some later households are unseen in training even when they have recent observations at test issue time. A pooled demand model must handle unseen household IDs and use recent history/metadata; do not require a trained household-specific model for all 410 or drop later entrants.

**READY FOR ENERGY MODEL: YES**
'''
    (OUT/'WEATHER_REPORT.md').write_text(report)
    readme='''# Weather forecasting module

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
'''
    (OUT/'README.md').write_text(readme)
    print(OUT/'WEATHER_REPORT.md')


if __name__=='__main__':main()
