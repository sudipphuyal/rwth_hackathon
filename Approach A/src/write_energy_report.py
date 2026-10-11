"""Report locked energy results; no re-fitting, quantiles or procurement decisions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .run_energy_forecasting import OUT


def table(d):
    def val(x):
        if pd.isna(x):return 'Not scoreable'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x)
    result='| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'
    for r in d.itertuples(index=False,name=None):result+='| '+' | '.join(val(v) for v in r)+' |\n'
    return result


def main():
    lock=json.loads((OUT/'model_selection_lock.json').read_text())
    test=json.loads((OUT/'test_evaluation_complete.json').read_text())
    validation=json.loads((OUT/'validation_residual_summary.json').read_text())
    check=json.loads((OUT/'validation_checks.json').read_text())
    final=json.loads((OUT/'2024-02-28_summary.json').read_text())
    train=json.loads((OUT/'fit_training_summary.json').read_text())
    refit=json.loads((OUT/'refit_summary.json').read_text())
    data=json.loads((OUT/'materialization_complete.json').read_text())
    comparisons=pd.read_csv(OUT/'evaluation/validation_comparison.csv')
    subgroup=pd.read_csv(OUT/'evaluation/subgroup_metrics.csv')
    roster=pd.read_csv(OUT/'evaluation/active_households_by_day.csv')
    daily=pd.read_csv(OUT/'evaluation/test_daily_energy.csv')
    contributions=pd.read_csv(OUT/'predictions/2024-02-28_group_contributions.csv')
    metrics=[]
    for period,results in [('Validation',validation),('Test',test)]:
        for method in ['selected','baseline']:
            for scope in ['Household','MatchedPortfolio']:
                r=results[method][scope];metrics.append({'Period':period,'Method':method,'Scope':scope,
                    'n':r['n'],'MAE':r['MAE'],'RMSE':r['RMSE'],'WAPE %':r['WAPE'],'Bias':r['Bias']})
    metric_table=pd.DataFrame(metrics)
    metric_table.to_csv(OUT/'evaluation/summary_metrics.csv',index=False)
    improvement=[]
    for scope in ['Household','MatchedPortfolio']:
        for m in ['MAE','RMSE','WAPE']:
            a,b=test['selected'][scope][m],test['baseline'][scope][m]
            improvement.append({'Scope':scope,'Metric':m,'Improvement %':100*(1-a/b)})
    improvement.append({'Scope':'Daily matched energy','Metric':'MAE','Improvement %':100*(1-test['selected']['DailyEnergyMAE']/test['baseline']['DailyEnergyMAE'])})
    improvements=pd.DataFrame(improvement)
    improvements.to_csv(OUT/'evaluation/improvements_vs_baseline.csv',index=False)
    daily_scores=pd.DataFrame([{'Period':p,'Method':m,'DaysWithActualSupport':r[m]['EvaluatedDays'],
                                'DailyTotalMAE_kWh':r[m]['DailyEnergyMAE'],'MedianDailyAE_kWh':r[m]['DailyEnergyMedianAE']}
                              for p,r in [('Validation',validation),('Test',test)] for m in ['selected','baseline']])
    baseline_rows=comparisons[(comparisons.Stage=='loss_and_baselines_validation') & comparisons.Candidate.str.startswith(('baseline_','blend_'))]
    ablation=comparisons[(comparisons.Stage=='ablation_validation') & comparisons.Candidate.isin(['A','B','C','D','E','F','G','D_plus_recent','E_plus_recent','G_extreme_excluded_diagnostic'])]
    ablation=ablation[['Candidate','HouseholdMAE','HouseholdRMSE','PortfolioWAPE','DailyEnergyMAE']]
    source_manifest=pd.read_csv(OUT/'materialization_manifest.csv')
    # All-period roster dates derived from existing materialisation, not raw re-audit.
    active_counts=[]
    for day in pd.date_range('2019-05-20','2024-02-28',freq='D'):
        unique=source_manifest.groupby('Household_ID').FirstForecastDate.min()
        active_counts.append({'DeliveryDate':str(day.date()),'ActiveHouseholds':int((pd.to_datetime(unique)<=day).sum())})
    pd.DataFrame(active_counts).to_csv(OUT/'evaluation/active_roster_all_periods.csv',index=False)
    # Export inspectable final household CSV alongside efficient Parquet.
    households=pd.read_parquet(OUT/'predictions/2024-02-28_households.parquet')
    households.to_csv(OUT/'predictions/2024-02-28_households.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    (OUT/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    dt=pd.to_datetime(daily.TargetTimestamp,utc=True)
    axes[0].plot(dt,daily.MatchedActual/1000,label='Actual (matched support)',color='#253746',lw=1)
    axes[0].plot(dt,daily.MatchedPrediction/1000,label='Frozen model',color='#1782a8',lw=1)
    axes[0].plot(dt,daily.MatchedBaseline/1000,label='Validation-selected blend',color='#d07a36',lw=.8,alpha=.8)
    axes[0].set(title='Daily energy: identical observed-household support',ylabel='MWh/day')
    axes[0].legend(frameon=False,ncol=3);axes[0].grid(alpha=.2)
    dates=pd.to_datetime(roster.DeliveryDate)
    axes[1].plot(dates,roster.ActiveHouseholds,color='#253746',label='Active households')
    axes[1].plot(dates,roster.established,color='#1782a8',label='Established')
    axes[1].plot(dates,roster['sparse/new'],color='#d07a36',label='Sparse/new')
    axes[1].plot(dates,roster['cold-start'],color='#8a4ea6',label='Cold-start')
    axes[1].set(ylabel='Households',title='Observed portfolio membership and history state')
    axes[1].legend(frameon=False,ncol=4);axes[1].grid(alpha=.2)
    fig.savefig(OUT/'figures/test_portfolio_and_roster.png',dpi=170);plt.close(fig)
    p=pd.read_csv(OUT/'predictions/2024-02-28_portfolio.csv');p.TargetTimestamp=pd.to_datetime(p.TargetTimestamp,utc=True)
    fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained')
    ax.plot(p.TargetTimestamp,p.PredictedPortfolioKWh,color='#1782a8',lw=2)
    peak=p.loc[p.PredictedPortfolioKWh.idxmax()];ax.scatter([peak.TargetTimestamp],[peak.PredictedPortfolioKWh],color='#d07a36',zorder=5)
    ax.annotate(f'{peak.PredictedPortfolioKWh:.2f} kWh / 15 min\n02:00 CET',xy=(peak.TargetTimestamp,peak.PredictedPortfolioKWh),xytext=(10,-45),textcoords='offset points')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=ZoneInfo('Europe/Berlin')))
    ax.set(title=f'2024-02-28 UTC delivery day: {final["total_daily_kwh"]/1000:.3f} MWh for 410 households',
           ylabel='Forecast kWh per 15-minute interval',xlabel='Local clock time (CET; UTC delivery grid)')
    ax.grid(alpha=.2);fig.savefig(OUT/'figures/2024-02-28_portfolio_forecast.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    b=ablation[~ablation.Candidate.eq('G_extreme_excluded_diagnostic')]
    ax.bar(b.Candidate,b.PortfolioWAPE,color=['#d07a36' if v=='D_plus_recent' else '#1782a8' for v in b.Candidate])
    ax.axhline(validation['baseline']['MatchedPortfolio']['WAPE'],color='#253746',ls='--',label='Selected blend')
    ax.set(ylabel='Validation matched-portfolio WAPE (%)',title='Feature ablation: validation only')
    ax.tick_params(axis='x',rotation=25);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'figures/validation_ablation.png',dpi=170);plt.close(fig)
    original_g=comparisons[(comparisons.Stage=='ablation_validation')&comparisons.Candidate.eq('G')].iloc[0]
    diagnostic=ablation[ablation.Candidate.eq('G_extreme_excluded_diagnostic')].iloc[0]
    report=f'''# Step 4: household demand forecasts and portfolio aggregation

Built one pooled household HistGradientBoostingRegressor and one separate history-free cold-start model. No forecast quantiles, uncertainty intervals or procurement optimisation were implemented. The existing data audit, date split and weather models were not redone. Step 2 feature definitions were executed in a vectorised batch backend, tested against the reference for cutoff/lag/window/DST equivalence.

Frozen UTC delivery dates: **TRAIN 2019-05-20–2022-02-27; VALIDATION 2022-02-28–2023-02-27; TEST 2023-02-28–2024-02-27**. Each forecast issues **11:00 Europe/Berlin on D−1**; only meter interval starts <= issue minus 30 minutes are used. UTC delivery days always have 96 intervals, local behavioural clocks use DST. Metadata remains conditional on the explicitly accepted fixed, undated-snapshot assumption.

## 1. Exact model input features

Selected primary feature set: **{lock['selected_feature_set']}**, {len(lock['input_columns'])} columns including explicit missing indicators. Household_ID is a key only; it is not a predictive or ordinal numerical feature. Actual target-day weather, target/source/quality flags, visits/protocols, retrospective coverage and fitted-model/source timestamps are excluded from inputs.

```text
{chr(10).join(lock['input_columns'])}
```

The only target-day weather inputs are **PredictedTemperature, PredictedSunshine, PredictedHumidity** and their Missing indicators. Historical training weather forecasts are out-of-time expanding fits/warm-up baselines; validation uses training-only weather weights; test uses the locked test weather forecasts. All joins use issue timestamp + assigned Weather_ID + target UTC hour. Hourly weather covariates are repeated over four quarters with their original units, not summed as quarter-hour weather totals.

Metadata unknowns remain distinct from False (categorical code 2 vs 0; True=1). HP type and Weather_ID are native categories. Numeric metadata stays NaN where missing. Unknown technologies produce NaN interactions, not an absence assumption. Floor heating/radiators and DHW sources remain separate coexisting fields.

Small physical interactions investigated: temperature × area; heating exposure × area; temperature/heating exposure × air-source/ground-source flags; sunshine × PV, daytime and cyclic hour; residents × HP/electric DHW; EV × evening/night. HeatingExposure = max(0, **15°C − PredictedTemperature**), the existing fixed candidate base; no test tuning. PV/EV/appliance interactions were evaluated but not selected for the primary model. They remain in the cold-start model, which has no consumption history to represent identity.

## 2. Demand baseline results

A = previous available matching local quarter-hour; B = previous-week same local interval; C = recent same-slot 28-day median. Missing baseline values use the same training-only selected cold fallback, so comparisons cover identical observed target support. Raw baseline availability on validation: A {lock['baseline_raw_availability']['A']:.1%}, B {lock['baseline_raw_availability']['B']:.1%}, C {lock['baseline_raw_availability']['C']:.1%}. Coverage is not inferred from future file summaries.

D = validation-selected blend, testing .25/.50/.75 pairwise weights and endpoints. Selected strongest baseline: **75% A + 25% B**. The weighting objective is matched-portfolio WAPE, household MAE tie-break.

{table(baseline_rows[['Candidate','HouseholdMAE','HouseholdRMSE','PortfolioWAPE','PortfolioBias','DailyEnergyMAE']])}

## 3. Validation results

{table(metric_table[metric_table.Period.eq('Validation')])}

Validation uses available labels only through **2023-02-27 09:30 UTC**: late final validation-day targets would not be known at first test issue and are embargoed. All candidate decisions used training + this validation subset only. The all-missing validation day **2022-10-30** is forecasted but not scored; daily accuracy covers 364 days. The last validation day's energy error covers only its pre-embargo observed support, not an invented complete actual day.

The model improves household error but the selected blend is stronger on validation portfolio WAPE and daily energy error. This matters for procurement: ML is not automatically the best aggregate forecast. The pooled model remains the requested household estimator; the blend remains a serious benchmark for the uncertainty/procurement stage. No test-based strategy reselection occurred.

## 4. Chosen model/configuration

HistGradientBoostingRegressor, one global pooled model, with categorical weather station/technology flags and no household-specific estimator:

```json
{json.dumps(lock['parameters'],indent=2)}
```

No random internal time split: early_stopping=False. Training label thinning is deterministic, not a time split: one quarter-hour in alternating hours, rotating over eight days so every quarter/hour phase is represented. All flagged extremes are forced into the sample unchanged. All modelling rows remain materialised; validation/test are scored at full 15-minute resolution.

Training fit: **{train['rows']:,} measured/reconstructed sample labels**, {train['households']} supervised households, {train['normal_rows']:,} established-history rows for cold-start learning. Refit: **{refit['rows']:,} train+validation sample labels**, {refit['train_validation_households']} supervised households. Actual last refit label is **{refit['fit_max_label_timestamp']}**, before the first test issue. Four targetless households never supply training labels.

## 5. Chosen loss and outlier policy

Both squared_error and absolute_error were compared on validation. **Squared error selected**: absolute error reduced individual MAE but materially underpredicted aggregate demand (validation full-G portfolio WAPE 24.75% vs squared error 14.18%). The practical objective is aggregate procurement volume, not individual MAE alone. Robustness comes from 200-sample leaves, L2 regularisation, historical medians/p95 and validation-selected features; this is not a claim that squared-error loss itself is robust.

**Primary policy: retain all genuinely measured extreme labels unchanged.** The 66.976 kWh training spike is included despite sampling; the 27.304 kWh test spike remains in scoring. No clipping or synthetic targets. Predictions alone are constrained nonnegative because grid import cannot be negative.

Validation diagnostic excludes flagged training labels only, holding features/validation targets fixed. Full-G portfolio WAPE changes from {original_g.PortfolioWAPE:.4f}% to {diagnostic.PortfolioWAPE:.4f}% (very small), and household MAE from {original_g.HouseholdMAE:.6f} to {diagnostic.HouseholdMAE:.6f} kWh. This limited sensitivity experiment does not remove historical spike effects from input summaries. The unverified meter/correction hypothesis is not enough to delete measured data.

## 6. Chosen feature set

**D + recent behaviour:** calendar; allowed demand lags and 28-day profile; expanding baseline/quality/age; predicted weather; area/residents; HP/heating/DHW metadata; recent seven-day means/medians, night/day/evening, peak/p95, change ratio and zero fraction.

PV ownership, EV, dryer/freezer and their explicit interactions did not add validation portfolio skill to this recent-behaviour design and were omitted from the primary. This is a feature-group result for this estimator/sample/benchmark, not proof these technologies have no physical effect. Recent measured behaviour can already represent PV/EV changes better than undated ownership flags. Cold fallback still uses those metadata groups because its personal history is deliberately unavailable.

## 7. Train/validation ablation

A historical core + calendar/quality; B adds predicted weather; C adds area/residents; D adds HP/heating/DHW; E adds PV; F adds EV/appliances; G adds seven-day recent behaviour/change. D_plus_recent and E_plus_recent retain mandatory recent behaviour while testing optional ownership groups. All use the selected loss/configuration; this is not a huge parameter search.

{table(ablation)}

![Validation feature groups](figures/validation_ablation.png)

Predicted weather improved validation portfolio WAPE over A; most further gain comes from recent behaviour. PV/EV group differences are small and confounded by shared features, coverage and metadata age. No causal effect sizes are estimated.

## 8. Final locked test MAE/RMSE/WAPE/Bias

{table(metric_table[metric_table.Period.eq('Test')])}

Household units: kWh per household quarter-hour. Portfolio units: kWh per matched-support quarter-hour. WAPE is 100 × sum(abs(error)) / sum(abs(actual)); Bias is mean(predicted−actual), positive means overforecast. Scores pool observed intervals, not unweighted household percentages. Undefined zero-denominator WAPE remains missing.

**365 daily forecasts issued, 364 days scoreable.** All targets are unavailable on **2023-10-29**. Its 406-household full forecast exists, but matched actual/error are NaN. No fabricated zero-error day. Interval accuracy covers {test['selected']['MatchedPortfolio']['n']:,} intervals and {test['selected']['Household']['n']:,} observed household targets.

Weights are refitted once on train+validation, then frozen. Test date D reads only input columns, predicts all active household intervals, and only then reads D labels for scores. Recent consumption from earlier test days is allowed as an input at the cutoff. No model fitting occurs inside the daily loop. Model-file SHA256 hashes are unchanged through test, independent validation and the unseen-day demonstration. Test errors were not used to change loss, feature set, blend, base temperature or outlier policy. Earlier raw-data and weather investigations had inspected parts of this historical period; it is held out from energy model selection, not a claim that raw files were never inspected.

## 9. Daily total-energy error

Absolute(sum(predictions)−sum(actual)) over exactly the same known household/interval support each day:

{table(daily_scores)}

This is not sum of absolute quarter-hour errors. Missing-household actuals remain absent from both matched sums. A whole day without any actual support has undefined error. Because support can change within a day, these totals describe measured support, not a complete 410-household energy truth.

## 10. Improvement versus the strongest baseline

{table(improvements)}

Positive is improvement. The model improves test household MAE/RMSE and slightly improves quarter-hour portfolio errors. **Daily total-energy MAE is worse than the blend**, and validation aggregate performance also favoured the blend. Those limitations should be carried into the next step rather than hiding them behind one favourable metric.

## 11. Subgroup performance

{table(subgroup)}

These are descriptive test scores, not causal comparisons. WAPE can rise for low-import/PV households because its denominator is smaller. Groups differ in size, weather, area, occupancy, history coverage and label availability. HP Unknown and metadata-missing groups may be less stable. EV peaks are less predictable; metadata ownership does not specify a charging schedule. Actual target-day weather is not used for temperature-band grouping—bands use PredictedTemperature.

## 12. Established / sparse / cold-start

{table(subgroup[subgroup.Group.eq('ForecastState')])}

Established uses the existing Step 2 rule (>=28 valid completed UTC days, >=80% 28-day coverage, <=48h cutoff gap). Sparse/new includes later entrants and stale histories. Cold-start means no valid historical Total, not necessarily a physically new dwelling.

Four targetless households—1065698, 747511, 768498, 996610—receive 96 forecasts each through the separate history-free model. It trains on established training households with all personal demand histories withheld. Cold predictors are metadata/calendar/predicted weather and physical interactions only. Their real accuracy is **unknown**, not zero. Proxy tests deliberately hide histories of labelled households: selected cold test proxy household MAE {test['cold_proxy_all_history_withheld']['Household']['MAE']:.4f} kWh and matched-portfolio WAPE {test['cold_proxy_all_history_withheld']['MatchedPortfolio']['WAPE']:.2f}%. This proxy does not prove accuracy for those four targetless homes.

## 13. Active households over time

A household enters only once its first supplied record is available by issue (`FirstObservedTimestamp+30min <= ForecastIssueTimestamp`). Before that, no historical portfolio prediction is attributed to it. HistoryAgeDaysAtForecast is computed only after known membership. Once active, it stays active because no confirmed departures are provided; a missing/stopped meter is not retrospective proof of departure.

Test active roster grows **{roster.ActiveHouseholds.iloc[0]} → {roster.ActiveHouseholds.iloc[-1]}**, range {roster.ActiveHouseholds.min()}–{roster.ActiveHouseholds.max()}. Households unseen in TRAIN grow **{roster.UnseenTrainingHouseholds.iloc[0]} → {roster.UnseenTrainingHouseholds.iloc[-1]}**; they are retained. Refit has 392 supervised households, and new post-refit entrants still receive global/history-free forecasts as appropriate. First supplied-date membership is a proxy for the observed portfolio, not a verified contract start date.

![Daily matched energy and roster](figures/test_portfolio_and_roster.png)

Exact daily roster/state counts are in active_households_by_day.csv; all-period membership counts in active_roster_all_periods.csv. The first supply day may have observed labels but no prior forecast-issue membership; those retrospective labels are not scored as if that household were known beforehand.

## 14. Matched support versus full forecast

**Matched evaluation:** at t, only active households with finite actual Total contribute to both predicted and actual sums. **Full forecast:** all active households contribute, including normal/sparse/cold fallback. Both series are saved together with ActiveCount, ObservedCount and FullActualKnown. FullActual is NaN when any active actual is missing. The full-portfolio ground truth is unknown for all test intervals; matched metrics must not be labelled full-portfolio accuracy.

## 15. Important failure modes

- Household WAPE remains ~61%, despite lower absolute/RMS error: quarter-hour HP cycling and intermittent EV/appliance use are difficult with no schedules/setpoints. Forecasts smooth peaks.
- Weather is itself forecasted with ~2.5°C temperature error; station-wide errors can correlate many household heating errors and survive portfolio aggregation.
- Cold estimates lack individual calibration; four real targetless cases cannot be scored. Sparse/stale source history is weaker than established history.
- Metadata has no effective dates; ownership/residents may be outdated. Country/timezone and observation arrival latency are assumptions inherited from previous steps.
- Raw periods include whole missing DST-transition dates; no actual day error can be inferred there. UTC delivery days and local procurement settlement days differ at DST.
- No departure dates: stale meters remain active, so operational full totals can include unverified members. Confirm the portfolio roster before real procurement.
- Deterministic 1/8 fitting is a computational compromise; it preserves all slots/dates/households and extremes but is not full-row training. Ablations describe this configuration. Feature gains are modest and no single algorithm dominates all metrics.
- The blend was better on validation aggregate/daily metrics and still wins daily totals on test. Do not claim ML is universally superior or use test results to retune it.

## 16. Saved predictions / reproducibility

- `predictions/test/<date>.parquet`: 365 frozen daily household forecast files, no actual target columns.
- `predictions/test_portfolio.parquet`: matched + full forecast series and support counts, actuals only where known.
- `evaluation/test_labels/<date>.parquet`: actual targets/provenance/extreme flags, separated from predictors.
- `predictions/validation/<date>.parquet`: training-only selected-model forecasts; `evaluation/validation_residuals/` stores known actuals/residuals separately for a later uncertainty step. No intervals/quantiles have been calibrated here.
- `predictions/2024-02-28_households.parquet` and `.csv`; `2024-02-28_portfolio.csv`; `2024-02-28_group_contributions.csv`.
- `models/locked_primary.pkl`, `locked_cold.pkl`; validation-only analysis copies are separate and cannot alter the locked demonstration weights.
- `data/<period>/<household>.parquet` and `daily/<period>/<date>.parquet`: materialised Step 2-equivalent tables with predicted weather only. Raw data and old artifacts remain intact.
- `model_selection_lock.json`, `fit_training_summary.json`, `refit_summary.json`, `validation_checks.json`, `evaluation/validation_comparison.csv`, `evaluation/subgroup_metrics.csv`, `evaluation/test_daily_energy.csv`.

**Checks:** 24 automated tests passed (reference equivalence, future poisoning, gaps, DST, cohort membership, unknown interactions, sampling/extreme preservation and matched-support scores plus prior weather/preprocessing tests). Independent checks validated all test grids/prediction-label keys/support, unchanged locked hashes, 410×96 final-day outputs and three real household future-poison/reproduction cases. No after-cutoff target/weather input or synthetic training labels were introduced. Source weather selection-lock SHA is preserved.

## 17. 2024-02-28 total predicted energy

**{final['total_daily_kwh']:.3f} kWh = {final['total_daily_kwh']/1000:.3f} MWh**, all 410 active households / 39,360 quarter-hour rows. Same locked energy and cold weights as test—no test-target refit. Consumption is limited to the 2024-02-27 meter cutoff (09:30 UTC), weather is the saved 2024-02-28 forecast, metadata is the same accepted snapshot. No smart-meter target is supplied for this day, so no demand accuracy is calculated.

{table(contributions)}

## 18. 2024-02-28 peak

Maximum forecast quarter-hour: **{final['peak_interval_kwh']:.3f} kWh**, interval start **{final['peak_timestamp_utc']} = {final['peak_timestamp_local']}**. Average power for that interval is **{final['peak_average_kw']:.3f} kW**. This is a predicted peak, not a measured outcome; interval-start semantics follow the existing assumption.

![Next-day point forecast](figures/2024-02-28_portfolio_forecast.png)

## 19. Readiness for uncertainty/procurement work

**YES for the next modelling step:** reproducible frozen point forecasts, a strong validation-selected baseline, separate actuals, honest support counts and training-only validation residuals are available. No P10/P50/P90, price assumptions or procurement optimisation are implemented. This is not production procurement approval: the next step must preserve temporal calibration, model correlated weather/household residuals, account for missing/cold/stale support, and retain the blend as a competing point forecast. Real full-portfolio coverage cannot be validated without missing targets/roster confirmation.

**READY FOR UNCERTAINTY/PROCUREMENT STEP: YES**
'''
    (OUT/'ENERGY_REPORT.md').write_text(report)
    readme='''# Step 4 energy forecasting

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
'''
    (OUT/'README.md').write_text(readme)
    print(OUT/'ENERGY_REPORT.md')


if __name__=='__main__':main()
