from datetime import datetime, timedelta
from meteostat import Daily
from config import settings


def load_meteostat_data():
    station_id = settings.METEOSTAT_STATION_ID
    end_date = datetime.now() - timedelta(weeks=6)
    start_date = end_date - timedelta(days=365 * settings.TIME_RANGE_YEARS)

    start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    data = Daily(station_id, start, end)
    df = data.fetch()
    df = df.reset_index()
    df = df.rename(columns={"time": "date"})
    df = df[["date", "tavg", "tmin", "tmax", "prcp", "snow"]]
    df = df.ffill().bfill()
    return df