import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import settings
from .meteostat import load_meteostat_data


def _get_date_weather_features(dates_array):
    meteo_df = load_meteostat_data()
    meteo_df["date"] = pd.to_datetime(meteo_df["date"])
    weather_map = meteo_df.set_index("date").to_dict("index")

    features = []
    for d in dates_array:
        wd = weather_map.get(pd.Timestamp(d).normalize(), {"tavg": 0, "tmin": 0, "tmax": 0, "prcp": 0, "snow": 0})
        row = [
            np.sin(2 * np.pi * d.timetuple().tm_yday / 365),
            np.cos(2 * np.pi * d.timetuple().tm_yday / 365),
            np.sin(2 * np.pi * d.month / 12),
            np.cos(2 * np.pi * d.month / 12),
            d.year,
            wd["tavg"],
            wd["tmin"],
            wd["tmax"],
            wd["prcp"],
            wd["snow"],
        ]
        features.append(row)
    return np.array(features)


def train_water_level_model(dates, water_levels):
    x = _get_date_weather_features(dates)
    x_train, _, y_train, _ = train_test_split(x, water_levels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model, scaler


def predict_future_water_levels(model, scaler, last_date, periods=None):
    if periods is None:
        periods = settings.PREDICTION_MONTHS

    future_dates = pd.date_range(start=last_date, periods=periods, freq='ME')
    x_future = _get_date_weather_features(future_dates)
    x_future_scaled = scaler.transform(x_future)
    predictions = np.clip(model.predict(x_future_scaled), 0, 1)

    return future_dates, predictions


def inspect_weather_correlations(dates, water_levels):
    x = _get_date_weather_features(dates)
    df = pd.DataFrame(x, columns=[
        "sin_doy", "cos_doy", "sin_month", "cos_month", "year",
        "tavg", "tmin", "tmax", "prcp", "snow"
    ])
    df["water"] = water_levels
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_corr = corr["water"].drop("water", errors='ignore').dropna()
    plot_corr.sort_values().plot(
        kind="barh", ax=ax, color="steelblue", edgecolor="black"
    )
    ax.set_title("Korelacja cech z poziomem wody", fontsize=14)
    ax.set_xlabel("Współczynnik korelacji", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    return fig


def compare_models_with_and_without_weather(dates, water_levels):
    x1 = _get_date_weather_features(dates)
    x2 = np.array([
        [np.sin(2 * np.pi * d.timetuple().tm_yday / 365), np.cos(2 * np.pi * d.timetuple().tm_yday / 365)]
        for d in dates
    ])

    x1_train, x1_test, y_train, y_test = train_test_split(x1, water_levels, test_size=0.3, random_state=42)
    x2_train, x2_test, _, _ = train_test_split(x2, water_levels, test_size=0.3, random_state=42)

    s1 = StandardScaler()
    s2 = StandardScaler()
    x1_train_scaled = s1.fit_transform(x1_train)
    x1_test_scaled = s1.transform(x1_test)

    x2_train_scaled = s2.fit_transform(x2_train)
    x2_test_scaled = s2.transform(x2_test)

    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = RandomForestRegressor(n_estimators=100, random_state=42)
    model1.fit(x1_train_scaled, y_train)
    model2.fit(x2_train_scaled, y_train)

    pred1 = model1.predict(x1_test_scaled)
    pred2 = model2.predict(x2_test_scaled)

    mse1 = mean_squared_error(y_test, pred1)
    mae1 = mean_absolute_error(y_test, pred1)
    mse2 = mean_squared_error(y_test, pred2)
    mae2 = mean_absolute_error(y_test, pred2)

    print("\nModel z danymi pogodowymi:")
    print(f"  MSE: {mse1:.4f}")
    print(f"  MAE: {mae1:.4f}")

    print("Model bez danych pogodowych:")
    print(f"  MSE: {mse2:.4f}")
    print(f"  MAE: {mae2:.4f}")

    if mse1 < mse2 and mae1 < mae2:
        print("\nModel z danymi pogodowymi wykazuje lepszą dokładność.")
    elif mse1 > mse2 and mae1 > mae2:
        print("\nModel bez danych pogodowych wykazuje lepszą dokładność.")
    else:
        print("\nRóżnice MSE i MAE nie wskazują na przewagę jednego z modeli.")