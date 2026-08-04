import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


CSV_PATH = "single_ecg_signal.csv"
OUTPUT_DIR = "outputs_q4"

TRAIN_SIZE = 1800
SAMPLING_RATE = 360
MAX_MODEL_ORDER = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_signal(csv_path):
    df = pd.read_csv(csv_path)

    signal = df.iloc[:, 0].astype(float).values

    return signal


def run_adf_test(series, name):
    clean_series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()

    result = adfuller(clean_series)

    adf_statistic = result[0]
    p_value = result[1]
    used_lags = result[2]
    n_observations = result[3]
    critical_values = result[4]

    return {
        "Signal version": name,
        "ADF statistic": adf_statistic,
        "p-value": p_value,
        "Used lags": used_lags,
        "No. observations": n_observations,
        "Critical value 1%": critical_values["1%"],
        "Critical value 5%": critical_values["5%"],
        "Critical value 10%": critical_values["10%"],
        "Stationary at 5%": p_value < 0.05
    }


def create_transformations(signal):
    original = pd.Series(signal, name="Original")

    first_difference = original.diff().dropna()
    first_difference.name = "First difference"

    min_value = original.min()
    shifted = original - min_value + 1e-6

    square_root = np.sqrt(shifted)
    square_root = pd.Series(square_root, name="Square root")

    log_transform = np.log(shifted)
    log_transform = pd.Series(log_transform, name="Log")

    transformations = {
        "Original": original,
        "First difference": first_difference,
        "Square root": square_root,
        "Log": log_transform
    }

    return transformations


def stationarity_analysis(transformations):
    adf_rows = []

    for name, series in transformations.items():
        adf_rows.append(run_adf_test(series, name))

    adf_df = pd.DataFrame(adf_rows)

    adf_df_sorted = adf_df.sort_values(
        by=["p-value", "ADF statistic"],
        ascending=[True, True]
    )

    selected_name = adf_df_sorted.iloc[0]["Signal version"]
    selected_signal = transformations[selected_name].dropna().reset_index(drop=True)

    adf_path = os.path.join(OUTPUT_DIR, "q4_adf_results.csv")
    adf_df.to_csv(adf_path, index=False)

    print("\nADF test results:")
    print(adf_df)

    return adf_df, selected_name, selected_signal


def plot_original_signal(signal):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title("Original ECG signal")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "q4_original_signal.png"), dpi=300)
    plt.close()


def plot_transformed_signals(transformations):
    for name, series in transformations.items():
        safe_name = name.lower().replace(" ", "_")

        plt.figure(figsize=(12, 4))
        plt.plot(series.values)
        plt.title(f"ECG signal - {name}")
        plt.xlabel("Sample index")
        plt.ylabel("Transformed amplitude")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"q4_signal_{safe_name}.png"),
            dpi=300
        )
        plt.close()


def plot_selected_signal(selected_signal, selected_name):
    plt.figure(figsize=(12, 4))
    plt.plot(selected_signal.values)
    plt.title(f"Selected signal for forecasting: {selected_name}")
    plt.xlabel("Sample index")
    plt.ylabel("Signal value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "q4_selected_signal.png"), dpi=300)
    plt.close()


def estimate_seasonal_period(signal, sampling_rate=360):
    return sampling_rate


def perform_seasonal_decomposition(selected_signal, selected_name):
    period = estimate_seasonal_period(selected_signal)

    try:
        decomposition = seasonal_decompose(
            selected_signal,
            model="additive",
            period=period
        )

        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(
            f"Seasonal decomposition of selected signal: {selected_name}",
            y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "q4_seasonal_decomposition.png"),
            dpi=300
        )
        plt.close()

        return period

    except Exception as e:
        print(f"Seasonal decomposition failed: {e}")
        return period


def suggest_orders_from_acf_pacf(train, max_lag=40, max_order=5):
    train_values = pd.Series(train).dropna().values
    n = len(train_values)

    significance_threshold = 1.96 / np.sqrt(n)

    acf_values = acf(train_values, nlags=max_lag, fft=True)
    pacf_values = pacf(train_values, nlags=max_lag, method="ols")

    # Exclude lag 0.
    significant_acf_lags = np.where(np.abs(acf_values[1:]) > significance_threshold)[0] + 1
    significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > significance_threshold)[0] + 1

    if len(significant_pacf_lags) == 0:
        p = 1
    else:
        p = int(min(significant_pacf_lags[-1], max_order))

    if len(significant_acf_lags) == 0:
        q = 1
    else:
        q = int(min(significant_acf_lags[-1], max_order))

    p = max(1, p)
    q = max(1, q)

    return p, q, significance_threshold


def plot_acf_pacf_figures(train):
    y = pd.Series(train).dropna().values
    lag_acf = acf(y, nlags=50)
    lag_pacf = pacf(y, nlags=50, method="ols")

    fig = plt.figure(figsize=(12, 7))

    plt.subplot(121)
    plt.plot(lag_acf, color="teal")
    plt.axhline(y=0, linestyle="--", color="gray")
    plt.axhline(y=-1.96 / np.sqrt(len(y)), linestyle="--", color="gray")
    plt.axhline(y=1.96 / np.sqrt(len(y)), linestyle="--", color="gray")
    plt.title("Autocorrelation Function")

    plt.subplot(122)
    plt.plot(lag_pacf, color="plum")
    plt.axhline(y=0, linestyle="--", color="gray")
    plt.axhline(y=-1.96 / np.sqrt(len(y)), linestyle="--", color="gray")
    plt.axhline(y=1.96 / np.sqrt(len(y)), linestyle="--", color="gray")
    plt.title("Partial Autocorrelation Function")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "q4_acf_pacf.png"), dpi=300)
    plt.close()


def calculate_rss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sum((y_true - y_pred) ** 2)


def fit_forecast_model(train, test, order, model_name):
    model = ARIMA(train, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test))
    model_rss = calculate_rss(test, forecast)

    return {
        "Model": model_name,
        "Order": order,
        "RSS": model_rss,
        "Forecast": forecast,
        "AIC": fitted_model.aic,
        "BIC": fitted_model.bic
    }


def run_forecasting_models(train, test, p, q):
    model_specs = [
        ("AR", (p, 0, 0)),
        ("MA", (0, 0, q)),
        ("ARMA", (p, 0, q)),
        ("ARIMA", (p, 1, q))
    ]

    results = []

    for model_name, order in model_specs:
        try:
            result = fit_forecast_model(train, test, order, model_name)
            results.append(result)
        except Exception as e:
            print(f"{model_name} with order={order} failed: {e}")

            # Use a simpler order if fitting fails.
            fallback_order = {
                "AR": (1, 0, 0),
                "MA": (0, 0, 1),
                "ARMA": (1, 0, 1),
                "ARIMA": (1, 1, 1)
            }[model_name]

            try:
                result = fit_forecast_model(train, test, fallback_order, model_name)
                results.append(result)
            except Exception as e2:
                print(f"Fallback {model_name} also failed: {e2}")

    return results


def save_model_comparison(results):
    rows = []

    for r in results:
        rows.append({
            "Model": r["Model"],
            "Order": r["Order"],
            "RSS": r["RSS"],
            "AIC": r["AIC"],
            "BIC": r["BIC"]
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by="RSS", ascending=True)

    output_path = os.path.join(OUTPUT_DIR, "q4_model_comparison.csv")
    comparison_df.to_csv(output_path, index=False)

    print("\nModel comparison table:")
    print(comparison_df)

    return comparison_df


def plot_forecast_comparison(test, results):
    plt.figure(figsize=(12, 5))

    plt.plot(
        np.arange(len(test)),
        test.values,
        label="Test signal",
        linewidth=2
    )

    for r in results:
        forecast = np.asarray(r["Forecast"])
        plt.plot(
            np.arange(len(forecast)),
            forecast,
            label=f"{r['Model']} forecast"
        )

    plt.title("Forecast comparison on test set")
    plt.xlabel("Test sample index")
    plt.ylabel("Signal value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "q4_forecast_comparison.png"), dpi=300)
    plt.close()


def main():
    # Load the signal.
    signal = load_signal(CSV_PATH)

    plot_original_signal(signal)

    # Build transformed signals.
    transformations = create_transformations(signal)

    plot_transformed_signals(transformations)

    # Run ADF stationarity tests.
    adf_df, selected_name, selected_signal = stationarity_analysis(transformations)

    plot_selected_signal(selected_signal, selected_name)

    # Split the train and test sets.
    if len(selected_signal) <= TRAIN_SIZE:
        raise ValueError(
            f"Selected signal length is {len(selected_signal)}, "
            f"which is not greater than TRAIN_SIZE={TRAIN_SIZE}."
        )

    train = selected_signal.iloc[:TRAIN_SIZE]
    test = selected_signal.iloc[TRAIN_SIZE:]

    # Seasonal decomposition.
    seasonal_period = perform_seasonal_decomposition(selected_signal, selected_name)

    # ACF and PACF.
    plot_acf_pacf_figures(train)

    p, q, threshold = suggest_orders_from_acf_pacf(
        train,
        max_lag=40,
        max_order=MAX_MODEL_ORDER
    )

    # Save model orders.
    order_df = pd.DataFrame([
        {
            "Selected signal": selected_name,
            "Suggested p from PACF": p,
            "Suggested q from ACF": q,
            "ACF/PACF significance threshold": threshold,
            "Seasonal period used": seasonal_period
        }
    ])

    order_path = os.path.join(OUTPUT_DIR, "q4_selected_orders.csv")
    order_df.to_csv(order_path, index=False)

    # Fit AR, MA, ARMA, and ARIMA models.
    results = run_forecasting_models(train, test, p, q)

    if len(results) == 0:
        raise RuntimeError("No forecasting model was fitted successfully.")

    # Compare models with RSS.
    comparison_df = save_model_comparison(results)

    best_model = comparison_df.iloc[0]
    print(
        f"\nBest model by RSS: {best_model['Model']} "
        f"with order={best_model['Order']} and RSS={best_model['RSS']:.4f}"
    )

    plot_forecast_comparison(test, results)


if __name__ == "__main__":
    main()
