# ============================================================

# SUMMER SCHOOL - 410-HOUSEHOLD DAY-AHEAD FORECAST

#

# Primary model:

#   Global Ridge Regression

#   + Split Conformal Prediction

#

# Fallback hierarchy:

#   1. Ridge model

#   2. Same time previous week

#   3. Same time previous day

#   4. Portfolio-based imputation

#

# Historical validation:

#   2023-03-15

#   2023-09-05

#   2024-02-27

#

# Final forecast:

#   2024-02-28

# ============================================================





# ============================================================

# 1. IMPORTS

# ============================================================



from pathlib import Path
import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline





# ============================================================

# 2. SETTINGS

# ============================================================



DATA_DIR = Path(os.environ.get("METER_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data" / "15min")))



VALIDATION_DATES = [

    pd.Timestamp("2023-03-15", tz="UTC"),

    pd.Timestamp("2023-09-05", tz="UTC"),

    pd.Timestamp("2024-02-27", tz="UTC"),

]



FORECAST_DATE = pd.Timestamp(

    "2024-02-28",

    tz="UTC"

)



CALIBRATION_DAYS = 14

TARGET_COVERAGE = 0.80

RIDGE_ALPHA = 1.0



FEATURES = [

    "lag_1d",

    "lag_7d",

    "sin_time",

    "cos_time",

    "sin_dow",

    "cos_dow",

]





# ============================================================

# 3. LOAD ALL 410 HOUSEHOLD FILES

# ============================================================



files = sorted(

    DATA_DIR.glob("*.csv")

)



print(

    "Household files found:",

    len(files)

)





# Keep every household file, including files with no usable

# consumption observations.



households = {}





for file in files:



    try:



        df = pd.read_csv(

            file,

            sep=";",

            usecols=[

                "Timestamp",

                "kWh_received_Total"

            ]

        )



        df["Timestamp"] = pd.to_datetime(

            df["Timestamp"],

            utc=True,

            errors="coerce"

        )



        df["load"] = pd.to_numeric(

            df["kWh_received_Total"],

            errors="coerce"

        )



        df = df.dropna(

            subset=[

                "Timestamp",

                "load"

            ]

        )



        df = df.sort_values(

            "Timestamp"

        )



        df = df.drop_duplicates(

            subset="Timestamp"

        )



        households[file.stem] = df[

            [

                "Timestamp",

                "load"

            ]

        ].copy()





    except Exception as e:



        print(

            "Could not read",

            file.name,

            ":",

            e

        )



        # Keep household in portfolio even if file

        # cannot provide usable observations.



        households[file.stem] = pd.DataFrame(

            columns=[

                "Timestamp",

                "load"

            ]

        )





print(

    "Total households retained:",

    len(households)

)





usable_count = sum(

    len(df) > 0

    for df in households.values()

)





print(

    "Households with usable consumption:",

    usable_count

)



print(

    "Households requiring complete imputation:",

    len(households) - usable_count

)





# ============================================================

# 4. HELPER: GET EXACT HISTORICAL PROFILE

# ============================================================



def get_profile(df, target_index):



    """

    Return consumption corresponding exactly to target_index.



    Returns None if any of the 96 values are unavailable.

    """



    if len(df) == 0:

        return None



    s = (

        df.set_index("Timestamp")["load"]

        .sort_index()

    )



    values = (

        s.reindex(target_index)

        .to_numpy(dtype=float)

    )



    if len(values) != 96:

        return None



    if np.isnan(values).any():

        return None



    return values





# ============================================================

# 5. CREATE RIDGE DATASET

# ============================================================



def create_ridge_dataset(

    df,

    target_date,

    require_actual=False

):



    """

    Reproduce the original Ridge forecasting procedure.



    Training:

        Before the 14-day calibration period.



    Calibration:

        14 complete days immediately before target date.



    Features:

        lag 1 day

        lag 7 days

        cyclic time

        cyclic weekday

    """



    if len(df) == 0:

        return None





    calibration_start = (

        target_date

        - pd.Timedelta(

            days=CALIBRATION_DAYS

        )

    )





    x = df.copy()



    x = x.sort_values(

        "Timestamp"

    )





    # --------------------------------------------------------

    # Historical timeline only

    # --------------------------------------------------------



    start = x["Timestamp"].min()



    end = (

        target_date

        - pd.Timedelta(minutes=15)

    )





    index = pd.date_range(

        start=start,

        end=end,

        freq="15min",

        tz="UTC"

    )





    load = (

        x.set_index("Timestamp")["load"]

        .reindex(index)

    )





    data = pd.DataFrame(

        index=index

    )



    data["y"] = load





    # --------------------------------------------------------

    # Lagged consumption

    # --------------------------------------------------------



    data["lag_1d"] = (

        data["y"].shift(96)

    )



    data["lag_7d"] = (

        data["y"].shift(

            96 * 7

        )

    )





    # --------------------------------------------------------

    # Calendar features

    # --------------------------------------------------------



    quarter = (

        data.index.hour * 4

        + data.index.minute // 15

    )



    dow = (

        data.index.dayofweek

    )





    data["sin_time"] = np.sin(

        2 * np.pi

        * quarter / 96

    )



    data["cos_time"] = np.cos(

        2 * np.pi

        * quarter / 96

    )



    data["sin_dow"] = np.sin(

        2 * np.pi

        * dow / 7

    )



    data["cos_dow"] = np.cos(

        2 * np.pi

        * dow / 7

    )





    # --------------------------------------------------------

    # Training

    # --------------------------------------------------------



    train = data[

        data.index < calibration_start

    ].dropna(

        subset=FEATURES + ["y"]

    )





    # --------------------------------------------------------

    # Calibration

    # --------------------------------------------------------



    calibration = data[

        (data.index >= calibration_start)

        &

        (data.index < target_date)

    ].dropna(

        subset=FEATURES + ["y"]

    )





    # Original eligibility requirements



    if len(train) < 96 * 14:

        return None





    if len(calibration) != (

        96 * CALIBRATION_DAYS

    ):

        return None





    # --------------------------------------------------------

    # Target timestamps

    # --------------------------------------------------------



    future_index = pd.date_range(

        start=target_date,

        periods=96,

        freq="15min",

        tz="UTC"

    )





    lag1_index = (

        future_index

        - pd.Timedelta(days=1)

    )





    lag7_index = (

        future_index

        - pd.Timedelta(days=7)

    )





    lag_1d = (

        data["y"]

        .reindex(lag1_index)

        .to_numpy()

    )





    lag_7d = (

        data["y"]

        .reindex(lag7_index)

        .to_numpy()

    )





    if (

        np.isnan(lag_1d).any()

        or

        np.isnan(lag_7d).any()

    ):

        return None





    # --------------------------------------------------------

    # Future calendar features

    # --------------------------------------------------------



    future_quarter = (

        future_index.hour * 4

        + future_index.minute // 15

    )



    future_dow = (

        future_index.dayofweek

    )





    future = pd.DataFrame({



        "lag_1d":

            lag_1d,



        "lag_7d":

            lag_7d,



        "sin_time":

            np.sin(

                2 * np.pi

                * future_quarter / 96

            ),



        "cos_time":

            np.cos(

                2 * np.pi

                * future_quarter / 96

            ),



        "sin_dow":

            np.sin(

                2 * np.pi

                * future_dow / 7

            ),



        "cos_dow":

            np.cos(

                2 * np.pi

                * future_dow / 7

            ),

    })





    result = {



        "X_train":

            train[FEATURES].to_numpy(),



        "y_train":

            train["y"].to_numpy(),



        "X_cal":

            calibration[FEATURES].to_numpy(),



        "y_cal":

            calibration["y"].to_numpy(),



        "X_future":

            future[FEATURES].to_numpy(),



        "timestamps":

            future_index,

    }





    # Actual target date is used ONLY for validation.



    if require_actual:



        actual = get_profile(

            df,

            future_index

        )



        result["y_actual"] = actual





    return result





# ============================================================

# 6. TRAIN GLOBAL RIDGE MODEL FOR TARGET DATE

# ============================================================



def train_ridge_for_date(

    target_date,

    require_actual=False

):



    ridge_datasets = {}





    for household_id, df in households.items():



        d = create_ridge_dataset(

            df,

            target_date,

            require_actual=require_actual

        )



        if d is not None:



            ridge_datasets[

                household_id

            ] = d





    ridge_ids = list(

        ridge_datasets.keys()

    )





    if len(ridge_ids) == 0:



        raise ValueError(

            "No households available for Ridge training."

        )





    # --------------------------------------------------------

    # Global training data

    # --------------------------------------------------------



    X_train = np.vstack([



        ridge_datasets[h][

            "X_train"

        ]



        for h in ridge_ids

    ])





    y_train = np.concatenate([



        ridge_datasets[h][

            "y_train"

        ]



        for h in ridge_ids

    ])





    # --------------------------------------------------------

    # Train Ridge

    # --------------------------------------------------------



    model = make_pipeline(



        StandardScaler(),



        Ridge(

            alpha=RIDGE_ALPHA

        )

    )





    model.fit(

        X_train,

        y_train

    )





    # --------------------------------------------------------

    # Conformal calibration

    # --------------------------------------------------------



    calibration_errors = []





    for h in ridge_ids:



        d = ridge_datasets[h]





        pred = model.predict(

            d["X_cal"]

        )





        pred = np.maximum(

            pred,

            0

        )





        errors = np.abs(

            d["y_cal"]

            - pred

        )





        calibration_errors.append(

            errors

        )





    calibration_errors = (

        np.concatenate(

            calibration_errors

        )

    )





    n_calibration = len(

        calibration_errors

    )





    quantile_level = (



        np.ceil(



            (n_calibration + 1)

            * TARGET_COVERAGE



        )



        / n_calibration

    )





    quantile_level = min(

        quantile_level,

        1.0

    )





    conformal_q = np.quantile(



        calibration_errors,



        quantile_level,



        method="higher"

    )





    return (

        ridge_datasets,

        ridge_ids,

        model,

        conformal_q,

        len(y_train),

        n_calibration

    )





# ============================================================

# 7. FALLBACK FORECAST

# ============================================================



def fallback_forecast(

    df,

    target_date,

    portfolio_reference=None

):



    """

    Forecast hierarchy for households that cannot use Ridge.



    1. Same 15-minute profile one week earlier

    2. Same profile one day earlier

    3. Household historical quarter-hour mean

    4. Portfolio reference profile

    """



    target_index = pd.date_range(

        start=target_date,

        periods=96,

        freq="15min",

        tz="UTC"

    )





    # --------------------------------------------------------

    # Fallback 1: previous week

    # --------------------------------------------------------



    week_index = (

        target_index

        - pd.Timedelta(days=7)

    )





    week_profile = get_profile(

        df,

        week_index

    )





    if week_profile is not None:



        return (

            np.maximum(

                week_profile,

                0

            ),

            "Previous-week"

        )





    # --------------------------------------------------------

    # Fallback 2: previous day

    # --------------------------------------------------------



    day_index = (

        target_index

        - pd.Timedelta(days=1)

    )





    day_profile = get_profile(

        df,

        day_index

    )





    if day_profile is not None:



        return (

            np.maximum(

                day_profile,

                0

            ),

            "Previous-day"

        )





    # --------------------------------------------------------

    # Fallback 3:

    # Household historical quarter-hour profile

    # --------------------------------------------------------



    if len(df) > 0:



        hist = df[

            df["Timestamp"] < target_date

        ].copy()





        if len(hist) > 0:



            hist["quarter"] = (



                hist["Timestamp"].dt.hour

                * 4



                +



                hist["Timestamp"].dt.minute

                // 15

            )





            quarter_mean = (



                hist.groupby(

                    "quarter"

                )["load"]

                .mean()



                .reindex(

                    range(96)

                )

            )





            if not quarter_mean.isna().any():



                return (



                    np.maximum(

                        quarter_mean.to_numpy(),

                        0

                    ),



                    "Historical-profile"

                )





    # --------------------------------------------------------

    # Fallback 4:

    # Portfolio-based imputation

    # --------------------------------------------------------



    if portfolio_reference is not None:



        return (

            np.maximum(

                portfolio_reference.copy(),

                0

            ),

            "Portfolio-imputation"

        )





    # Should normally never be reached.



    return (

        np.zeros(96),

        "Zero-imputation"

    )





# ============================================================

# 8. FORECAST ALL 410 HOUSEHOLDS

# ============================================================



def forecast_all_households(

    target_date,

    require_actual=False

):



    print("\n")

    print("=" * 70)



    print(

        "Target date:",

        target_date.date()

    )



    print("=" * 70)





    (

        ridge_datasets,

        ridge_ids,

        model,

        conformal_q,

        n_training,

        n_calibration

    ) = train_ridge_for_date(

        target_date,

        require_actual=require_actual

    )





    # --------------------------------------------------------

    # First forecast Ridge households

    # --------------------------------------------------------



    predictions = {}



    methods = {}





    ridge_prediction_matrix = []





    for h in ridge_ids:



        d = ridge_datasets[h]





        pred = model.predict(

            d["X_future"]

        )





        pred = np.maximum(

            pred,

            0

        )





        predictions[h] = pred



        methods[h] = "Ridge"



        ridge_prediction_matrix.append(

            pred

        )





    # --------------------------------------------------------

    # Mean Ridge-household profile

    #

    # Used only as final fallback for households with no

    # usable individual history.

    # --------------------------------------------------------



    ridge_prediction_matrix = np.vstack(

        ridge_prediction_matrix

    )





    portfolio_reference = np.mean(

        ridge_prediction_matrix,

        axis=0

    )





    # --------------------------------------------------------

    # Forecast all households excluded from Ridge

    # --------------------------------------------------------



    all_ids = list(

        households.keys()

    )





    for h in all_ids:



        if h in predictions:

            continue





        pred, method = fallback_forecast(



            households[h],



            target_date,



            portfolio_reference=portfolio_reference

        )





        predictions[h] = pred



        methods[h] = method





    # --------------------------------------------------------

    # Safety check

    # --------------------------------------------------------



    if len(predictions) != len(households):



        raise ValueError(

            "Not all households received a forecast."

        )





    # --------------------------------------------------------

    # Aggregate all household forecasts

    # --------------------------------------------------------



    portfolio_forecast = np.sum(



        np.vstack([



            predictions[h]



            for h in all_ids



        ]),



        axis=0

    )





    # --------------------------------------------------------

    # Count forecasting methods

    # --------------------------------------------------------



    method_counts = (



        pd.Series(

            methods

        )



        .value_counts()



        .sort_index()

    )





    print(

        "Total household files:",

        len(files)

    )





    print(

        "Total households forecast:",

        len(predictions)

    )





    print(

        "\nForecasting methods:"

    )





    for method, count in method_counts.items():



        print(

            f"  {method}: {count}"

        )





    print(

        "\nTraining observations:",

        n_training

    )





    print(

        "Calibration observations:",

        n_calibration

    )





    print(

        "Conformal q:",

        round(

            conformal_q,

            4

        ),

        "kWh / 15 min"

    )





    # --------------------------------------------------------

    # Validation actual portfolio

    # --------------------------------------------------------



    actual_portfolio = None

    actual_households = 0





    if require_actual:



        actual_profiles = []





        for h in all_ids:



            target_index = pd.date_range(

                start=target_date,

                periods=96,

                freq="15min",

                tz="UTC"

            )





            actual = get_profile(

                households[h],

                target_index

            )





            if actual is not None:



                actual_profiles.append(

                    actual

                )



                actual_households += 1





        # IMPORTANT:

        # Validation metrics are calculated only if actual

        # observations are available for all 410 households.



        if actual_households == len(all_ids):



            actual_portfolio = np.sum(

                np.vstack(

                    actual_profiles

                ),

                axis=0

            )





    return {



        "predictions":

            predictions,



        "methods":

            methods,



        "method_counts":

            method_counts,



        "portfolio_forecast":

            portfolio_forecast,



        "actual_portfolio":

            actual_portfolio,



        "actual_households":

            actual_households,



        "conformal_q":

            conformal_q,

    }





# ============================================================

# 9. HISTORICAL VALIDATION

# ============================================================



print("\n")

print("=" * 70)

print("OUT-OF-SAMPLE HISTORICAL VALIDATION")

print("=" * 70)





validation_summary = []





for validation_date in VALIDATION_DATES:



    result = forecast_all_households(



        validation_date,



        require_actual=True

    )





    forecast = result[

        "portfolio_forecast"

    ]





    actual = result[

        "actual_portfolio"

    ]





    print(

        "\nActual household profiles available:",

        result["actual_households"],

        "/",

        len(households)

    )





    # --------------------------------------------------------

    # Full 410-household validation possible

    # --------------------------------------------------------



    if actual is not None:



        error = (

            actual

            - forecast

        )





        mae = np.mean(

            np.abs(error)

        )





        rmse = np.sqrt(

            np.mean(

                error ** 2

            )

        )





        valid = (

            actual > 0

        )





        mape = (



            np.mean(



                np.abs(



                    error[valid]

                    /

                    actual[valid]



                )



            )



            * 100

        )





        actual_daily = (

            actual.sum()

        )





        forecast_daily = (

            forecast.sum()

        )





        daily_error = (



            100



            * (

                forecast_daily

                - actual_daily

            )



            / actual_daily

        )





        print("\nValidation results")





        print(

            "Actual daily energy:",

            round(

                actual_daily,

                2

            ),

            "kWh"

        )





        print(

            "Forecast daily energy:",

            round(

                forecast_daily,

                2

            ),

            "kWh"

        )





        print(

            "Daily energy error:",

            round(

                daily_error,

                2

            ),

            "%"

        )





        print(

            "MAE:",

            round(

                mae,

                4

            ),

            "kWh / 15 min"

        )





        print(

            "RMSE:",

            round(

                rmse,

                4

            ),

            "kWh / 15 min"

        )





        print(

            "MAPE:",

            round(

                mape,

                2

            ),

            "%"

        )





        validation_summary.append({



            "Date":

                validation_date.date(),



            "Households":

                len(households),



            "Actual_daily_kWh":

                actual_daily,



            "Forecast_daily_kWh":

                forecast_daily,



            "Daily_error_percent":

                daily_error,



            "MAE_kWh_15min":

                mae,



            "RMSE_kWh_15min":

                rmse,



            "MAPE_percent":

                mape,

        })





        # ----------------------------------------------------

        # Plot

        # ----------------------------------------------------



        hours = np.arange(

            96

        ) / 4





        plt.figure(

            figsize=(11, 5)

        )





        plt.plot(

            hours,

            actual,

            linewidth=2,

            label="Actual"

        )





        plt.plot(

            hours,

            forecast,

            linewidth=2,

            label="Forecast"

        )





        plt.xlabel(

            "Hour of day"

        )





        plt.ylabel(

            "Portfolio energy (kWh / 15 min)"

        )





        plt.title(

            "410-Household Validation - "

            + validation_date.strftime(

                "%Y-%m-%d"

            )

        )





        plt.xticks(

            np.arange(

                0,

                25,

                2

            )

        )





        plt.xlim(

            0,

            24

        )





        plt.grid(

            alpha=0.3

        )





        plt.legend()





        plt.tight_layout()





        plt.show()





    else:



        print(

            "\nFull 410-household validation cannot be "

            "calculated because actual observations are "

            "not available for every household on this date."

        )





# ============================================================

# 10. VALIDATION SUMMARY

# ============================================================



if len(validation_summary) > 0:



    validation_summary_df = pd.DataFrame(

        validation_summary

    )





    print("\n")

    print("=" * 70)

    print("VALIDATION SUMMARY")

    print("=" * 70)





    print(



        validation_summary_df

        .round(3)

        .to_string(

            index=False

        )



    )





    validation_summary_df.to_csv(

        "VALIDATION_410_households.csv",

        index=False

    )





# ============================================================

# 11. FINAL FEBRUARY 28 FORECAST

# ============================================================



print("\n")

print("=" * 70)

print("FINAL 410-HOUSEHOLD FORECAST")

print("=" * 70)





final_result = forecast_all_households(



    FORECAST_DATE,



    require_actual=False

)





portfolio_forecast = final_result[

    "portfolio_forecast"

]





daily_energy_kwh = (

    portfolio_forecast.sum()

)





daily_energy_mwh = (

    daily_energy_kwh

    / 1000

)





print("\n")

print("=" * 70)

print("FINAL DAY-AHEAD RESULT")

print("=" * 70)





print(

    "Forecast date:",

    FORECAST_DATE.date()

)





print(

    "Total households forecast:",

    len(

        final_result[

            "predictions"

        ]

    )

)





print(

    "Daily portfolio forecast:",

    round(

        daily_energy_kwh,

        2

    ),

    "kWh"

)





print(

    "Daily portfolio forecast:",

    round(

        daily_energy_mwh,

        3

    ),

    "MWh"

)





# ============================================================

# 12. SAVE FINAL PORTFOLIO FORECAST

# ============================================================



timestamps = pd.date_range(

    start=FORECAST_DATE,

    periods=96,

    freq="15min",

    tz="UTC"

)





portfolio_df = pd.DataFrame({



    "Timestamp":

        timestamps,



    "Point_forecast_kWh":

        portfolio_forecast,

})





portfolio_df.to_csv(

    "FINAL_portfolio_forecast_410_households_2024-02-28.csv",

    index=False

)





# ============================================================

# 13. SAVE HOUSEHOLD-LEVEL FORECASTS

# ============================================================



household_rows = []





for h, prediction in final_result[

    "predictions"

].items():



    method = final_result[

        "methods"

    ][h]





    for i in range(96):



        household_rows.append({



            "Household_ID":

                h,



            "Timestamp":

                timestamps[i],



            "Point_forecast_kWh":

                prediction[i],



            "Forecast_method":

                method,

        })





household_df = pd.DataFrame(

    household_rows

)





household_df.to_csv(

    "FINAL_household_forecast_410_households_2024-02-28.csv",

    index=False

)





# ============================================================

# 14. PLOT FINAL 410-HOUSEHOLD FORECAST

# ============================================================



hours = np.arange(

    96

) / 4





plt.figure(

    figsize=(11, 5)

)





plt.plot(

    hours,

    portfolio_forecast,

    linewidth=2,

    label="Forecast"

)





plt.xlabel(

    "Hour of day"

)





plt.ylabel(

    "Portfolio energy (kWh / 15 min)"

)





plt.title(

    "Day-Ahead Portfolio Forecast - "

    "410 Households - February 28, 2024"

)





plt.xticks(

    np.arange(

        0,

        25,

        2

    )

)





plt.xlim(

    0,

    24

)





plt.grid(

    alpha=0.3

)





plt.legend()





plt.tight_layout()





plt.show()