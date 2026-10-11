"""Freeze splits, train/select weather models, then evaluate the locked test once.

Run: python -m src.run_weather_forecasting
No household/electricity model is fitted. Existing preprocessing is untouched.
"""
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ChronologicalSplit, PipelineConfig, WEATHER_OUTPUT_DIR
from .data_loading import load_weather
from .preprocessing import issue_for_day
from .weather_forecasting import (VARIABLES, MODEL_PARAMETERS, WeatherRecipe, station_features,
    split_masks, weather_cutoff, input_columns, evaluation_labels, metrics, season_of, deployable_table)

OUT = WEATHER_OUTPUT_DIR


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def freeze_splits(split):
    """Use the existing audit, not a rescan or metric-dependent split search."""
    audit = pd.read_csv("outputs/preprocessing/audit/target_coverage.csv")
    first = pd.to_datetime(audit.FirstTimestamp, utc=True).min().normalize()
    last = pd.to_datetime(audit.LastTimestamp, utc=True).max().normalize()
    if str(first.date()) != split.train_start or str(last.date()) != split.test_end:
        raise AssertionError("Frozen split does not match actual audited meter-date bounds")
    counts = {name:(pd.Timestamp(end)-pd.Timestamp(start)).days+1 for name,start,end in [
        ("train",split.train_start,split.train_end),
        ("validation",split.validation_start,split.validation_end),
        ("test",split.test_start,split.test_end)]}
    boundary_coverage = {end:int(((pd.to_datetime(audit.FirstTimestamp,utc=True)<=pd.Timestamp(end,tz="UTC")) &
                            (pd.to_datetime(audit.LastTimestamp,utc=True)>=pd.Timestamp(end,tz="UTC"))).sum())
                         for end in [split.train_end,split.validation_end]}
    payload = {"dates": asdict(split), "delivery_date_timezone": "UTC",
               "selection_basis": "explicit user request: latest full year test, preceding full year validation, older training",
               "days": counts,
               "fractions": {name:n/sum(counts.values()) for name,n in counts.items()},
               "boundary_household_file_span_coverage": boundary_coverage,
               "revision": "full_year_seasonal_coverage",
               "previous_run_preserved": "outputs/weather_forecasting",
               "previously_inspected_overlap": "New test overlaps prior weather results; held out from revised fitting/selection, not never inspected",
               "source": "outputs/preprocessing/audit/target_coverage.csv",
               "weather_pre_history": "2019-01-01..2019-05-19 is input history, not forecast-label split",
               "boundary_embargo": "Fit/selection labels must be available by first following-period issue (hourly start + 2h)"}
    path = OUT / "splits.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise AssertionError("Split artifact already exists and differs; refusing to overwrite")
    save_json(path, payload)
    print(json.dumps(payload, indent=2), flush=True)


def evaluate_candidates(frame, labels, recipe, period, mask=None):
    aggregate, details, predictions, comparison = [], [], {}, []
    eligible = np.ones(len(frame), dtype=bool) if mask is None else np.asarray(mask)
    for c in VARIABLES:
        candidates = recipe.candidates(frame, c)
        predictions[c] = candidates
        actual = labels[c].to_numpy()
        common = eligible & np.isfinite(actual)
        for prediction in candidates.values():
            common &= np.isfinite(prediction)
        for method, prediction in candidates.items():
            overall = metrics(actual[eligible], prediction[eligible])
            overall.update(Period=period, Variable=c, Method=method)
            aggregate.append(overall)
            common_metrics = metrics(actual[common], prediction[common])
            common_metrics.update(Period=period, Variable=c, Method=method)
            comparison.append(common_metrics)
            q = frame[["Weather_ID", "ForecastLeadHours", "TargetUTCHour", "TargetWeatherTimestamp"]].copy()
            q["Season"] = season_of(q.TargetWeatherTimestamp)
            q["Actual"], q["Prediction"] = actual, prediction
            q = q.loc[eligible]
            for group in ["Weather_ID", "ForecastLeadHours", "TargetUTCHour", "Season"]:
                for key, g in q.groupby(group):
                    m = metrics(g.Actual, g.Prediction)
                    m.update(Period=period, Variable=c, Method=method, Group=group, GroupValue=str(key))
                    details.append(m)
    return aggregate, details, predictions, comparison


def choose(comparison):
    d = pd.DataFrame(comparison)
    chosen = {}
    for c in VARIABLES:
        ranked = d[d.Variable.eq(c)].dropna(subset=["MAE"]).sort_values(["MAE", "RMSE", "Method"])
        if ranked.empty:
            raise ValueError(f"No common validation coverage for {c}")
        chosen[c] = ranked.iloc[0].Method
    return chosen


def export_actuals(frame, labels, predictions, period):
    d = frame[["ForecastIssueTimestamp", "TargetWeatherTimestamp", "Weather_ID",
               "ForecastLeadHours", "TargetUTCHour"]].copy()
    d["Season"] = season_of(d.TargetWeatherTimestamp)
    for c in VARIABLES:
        d["Actual_"+c] = labels[c].to_numpy()
        for method, values in predictions[c].items():
            d[c+"__"+method] = values
    d.to_parquet(OUT / "evaluation" / f"{period}_actuals_and_candidates.parquet", index=False)


def training_oof(features, labels, columns, selected, split):
    """Expanding six-month fits; no fold sees its own/future weather labels."""
    train_end_exclusive = pd.Timestamp(split.train_end) + pd.Timedelta(days=1)
    semiannual = pd.date_range("2020-01-01", train_end_exclusive, freq="2QS-JAN")
    boundaries = [split.train_start] + [str(t.date()) for t in semiannual if t < train_end_exclusive]
    boundaries += [str(train_end_exclusive.date())]
    forecasts, records = [], []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        targets = features.loc[features.DeliveryDate.ge(lo) & features.DeliveryDate.lt(hi)].copy()
        cutoff = weather_cutoff(issue_for_day(lo, PipelineConfig()), PipelineConfig())
        fit_mask = features.TargetWeatherTimestamp.le(cutoff) & features.DeliveryDate.lt(lo)
        if i == 0:
            prediction = {c: targets[c+"_same_hour_median_7d"].to_numpy() for c in VARIABLES}
            recipe = None
            methods = {c: "warmup_same_hour_7d_median" for c in VARIABLES}
        else:
            recipe = WeatherRecipe(columns).fit(features.loc[fit_mask], labels.loc[fit_mask])
            prediction = {c: recipe.candidates(targets, c)[selected[c]] for c in VARIABLES}
            methods = selected
        table = deployable_table(targets, prediction, "train", recipe, methods)
        table["TrainingForecastMode"] = "warmup_causal_baseline" if i == 0 else "expanding_half_year_oof"
        forecasts.append(table)
        records.append({"start": start, "end_exclusive": end, "fit_cutoff": str(cutoff),
                        "fit_source_max": str(recipe.fit_source_max) if recipe else None,
                        "rows": len(table), "mode": table.TrainingForecastMode.iloc[0]})
        print(f"OOF weather fold {start}..{end} completed", flush=True)
    save_json(OUT / "training_forecast_folds.json", records)
    return pd.concat(forecasts, ignore_index=True)


def main():
    split, config = ChronologicalSplit(), PipelineConfig()
    for directory in [OUT, OUT / "evaluation", OUT / "models", OUT / "predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    # Protect the test evaluation from being reused for adaptive reruns.
    if (OUT / "test_evaluation_completed.json").exists():
        raise RuntimeError("Final test already evaluated. Refusing adaptive rerun; inspect saved results.")
    freeze_splits(split)
    stations = load_weather(config.data_dir)
    codes = {name:i for i, name in enumerate(sorted(stations))}
    days = pd.date_range(split.train_start, split.unseen_demand_day, freq="D", tz="UTC")
    cached = Path("outputs/weather_forecasting/weather_historical_features.parquet")
    if cached.exists():
        # Features do not depend on split membership: reuse the previous causal table.
        old_contract = json.loads((cached.parent / "weather_feature_contract.json").read_text())
        if old_contract["config"] != config.serializable() or old_contract["station_codes"] != codes:
            raise AssertionError("Cached causal weather feature convention differs")
        features = pd.read_parquet(cached)
        if (features.DeliveryDate.min()!=days.min() or features.DeliveryDate.max()!=days.max()):
            raise AssertionError("Cached weather feature date bounds differ")
        from .weather_forecasting import assert_weather_features
        assert_weather_features(features)
        print("Reusing unchanged causal weather features from the preserved first run",flush=True)
    else:
        features = pd.concat([station_features(station, days, codes[name], name, config)
                              for name, station in stations.items()], ignore_index=True)
    columns = input_columns(features)
    save_json(OUT / "weather_feature_contract.json", {
        "input_columns": columns, "station_codes": codes, "config": config.serializable(),
        "variables": VARIABLES, "model_parameters": MODEL_PARAMETERS,
        "model_selection": "lowest common-support validation MAE; RMSE tie-break; no test tuning",
        "weather_sources": "Only observation timestamps <= issue minus 2h",
        "solar_darkness": "training-only month/UTC-hour bins with >=20 labels and maximum sunshine=0; not astronomical night",
        "next_energy_inputs": ["PredictedTemperature", "PredictedSunshine", "PredictedHumidity",
                               "PredictedTemperatureMissing", "PredictedSunshineMissing", "PredictedHumidityMissing"]})
    features.to_parquet(OUT / "weather_historical_features.parquet", index=False, compression="zstd")
    masks = split_masks(features, split)
    # Training and validation labels only, no test labels loaded into fitting/selection.
    tv_mask = masks["train"] | masks["validation"]
    tv = features.loc[tv_mask]
    labels = evaluation_labels(stations, tv)
    first_val_cutoff = weather_cutoff(issue_for_day(pd.Timestamp(split.validation_start), config), config)
    first_test_cutoff = weather_cutoff(issue_for_day(pd.Timestamp(split.test_start), config), config)
    fit_mask = masks["train"] & features.TargetWeatherTimestamp.le(first_val_cutoff)
    validation = features.loc[masks["validation"]]
    val_labels = labels.loc[validation.index]
    selection_mask = validation.TargetWeatherTimestamp.le(first_test_cutoff)
    print(f"Features: {len(features):,} rows, {len(columns)} model inputs", flush=True)
    with threadpool_limits(limits=2):
        selection_model = WeatherRecipe(columns).fit(features.loc[fit_mask], labels.loc[fit_mask])
        va, vd, vp, vc = evaluate_candidates(validation, val_labels, selection_model,
                                            "validation_selection", selection_mask)
        selected = choose(vc)
        # Lock BEFORE any test labels, predictions or metrics are used.
        lock = {"selected_methods": selected, "model_parameters": MODEL_PARAMETERS,
                "input_columns": columns, "split_dates": asdict(split),
                "train_fit_source_max": str(selection_model.fit_source_max),
                "validation_selection_max_label_timestamp": str(validation.loc[selection_mask].TargetWeatherTimestamp.max()),
                "first_test_issue_timestamp": str(issue_for_day(pd.Timestamp(split.test_start), config)),
                "first_test_weather_cutoff": str(first_test_cutoff)}
        save_json(OUT / "model_selection_lock.json", lock)
        print("Locked methods:", selected, flush=True)
        with open(OUT / "models/validation_model.pkl", "wb") as f:
            pickle.dump(selection_model, f)
        export_actuals(validation, val_labels, vp, "validation")
        val_forecast = deployable_table(validation, {c:vp[c][selected[c]] for c in VARIABLES},
                                       "validation", selection_model, selected)
        val_forecast["TrainingForecastMode"] = "training_only_fit_validation_selection"
        # Refit locked recipe on train+validation observations available by first test issue.
        final_fit_mask = tv_mask & features.TargetWeatherTimestamp.le(first_test_cutoff)
        test_model = WeatherRecipe(columns).fit(features.loc[final_fit_mask], labels.loc[final_fit_mask])
        with open(OUT / "models/locked_test_model.pkl", "wb") as f:
            pickle.dump(test_model, f)
        test = features.loc[masks["test"]]
        # Labels are first requested for test only AFTER selection and test-model lock.
        test_labels = evaluation_labels(stations, test)
        ta, td, tp, tc = evaluate_candidates(test, test_labels, test_model, "test")
        test_forecast = deployable_table(test, {c:tp[c][selected[c]] for c in VARIABLES},
                                        "test", test_model, selected)
        test_forecast["TrainingForecastMode"] = "locked_train_validation_fit"
        export_actuals(test, test_labels, tp, "test")
        pd.DataFrame(va+ta).to_csv(OUT / "evaluation/aggregate_metrics.csv", index=False)
        pd.DataFrame(vc+tc).to_csv(OUT / "evaluation/common_support_comparison.csv", index=False)
        pd.DataFrame(vd+td).to_csv(OUT / "evaluation/grouped_metrics.csv", index=False)
        lock_sha = hashlib.sha256((OUT / "model_selection_lock.json").read_bytes()).hexdigest()
        save_json(OUT / "test_evaluation_completed.json", {"selected_methods": selected,
            "lock_sha256": lock_sha, "test_model_fit_source_max": str(test_model.fit_source_max),
            "no_test_labels_used_in_fit": True, "test_rows": len(test_forecast),
            "test_metrics": [m for m in ta if selected[m["Variable"]] == m["Method"]]})
        print("Locked final test evaluation completed", flush=True)
        # All training weather predictions are causal cross-fits, not in-sample predictions.
        train = features.loc[masks["train"]]
        train_labels = labels.loc[train.index]
        train_forecast = training_oof(train, train_labels, columns, selected, split)
        history_forecast = pd.concat([train_forecast, val_forecast, test_forecast], ignore_index=True)
        history_forecast["WeatherModelFitSourceMaxTimestamp"] = pd.to_datetime(
            history_forecast.WeatherModelFitSourceMaxTimestamp, utc=True)
        history_forecast.to_parquet(OUT / "predictions/historical_day_ahead_weather.parquet", index=False, compression="zstd")
        # Post-evaluation production refit: SAME locked recipe, only labels available by D-1 cutoff.
        unseen_day = pd.Timestamp(split.unseen_demand_day, tz="UTC")
        last_issue = issue_for_day(unseen_day, config)
        last_cutoff = weather_cutoff(last_issue, config)
        production_mask = features.TargetWeatherTimestamp.le(last_cutoff)
        production_frame = features.loc[production_mask]
        production_labels = evaluation_labels(stations, production_frame)
        production = WeatherRecipe(columns).fit(production_frame, production_labels)
        with open(OUT / "models/production_model.pkl", "wb") as f:
            pickle.dump(production, f)
        tomorrow = features.loc[features.DeliveryDate.eq(unseen_day)]
        predicted = {c:production.candidates(tomorrow, c)[selected[c]] for c in VARIABLES}
        last_forecast = deployable_table(tomorrow, predicted, "unseen_demand_day", production, selected)
        last_forecast["TrainingForecastMode"] = "post_evaluation_locked_recipe_refit"
        last_forecast.to_parquet(OUT / "predictions/2024-02-28_day_ahead_weather.parquet", index=False)
        last_forecast.to_csv(OUT / "predictions/2024-02-28_day_ahead_weather.csv", index=False)
        save_json(OUT / "production_refit.json", {"same_locked_methods": selected,
            "issue": str(last_issue), "weather_cutoff": str(last_cutoff),
            "max_fit_label_timestamp": str(production.fit_source_max),
            "actual_2024_02_28_weather_used": False,
            "test_predictions_regenerated_with_production_model": False})
    print("Saved weather predictions; no electricity model fitted.", flush=True)


if __name__ == "__main__":
    main()
