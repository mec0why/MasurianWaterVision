import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from config import settings


def _get_date_features(dates_array):
    return np.array([
        [np.sin(2 * np.pi * d.timetuple().tm_yday / 365), np.cos(2 * np.pi * d.timetuple().tm_yday / 365),
         np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12), d.year]
        for d in dates_array
    ])


def train_water_level_model(dates, water_levels):
    x = _get_date_features(dates)
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
    x_future = _get_date_features(future_dates)
    x_future_scaled = scaler.transform(x_future)
    predictions = np.clip(model.predict(x_future_scaled), 0, 1)

    return future_dates, predictions