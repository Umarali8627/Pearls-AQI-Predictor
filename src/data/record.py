from src.data.coordinates import get_coordinates
from src.data.pollutants import get_pollutants,get_historical_pollution
from src.data.weather import get_weather,get_historical_weather
import pandas as pd
from datetime import datetime,timedelta,timezone
from dotenv import load_dotenv
import os 

load_dotenv()


def create_new_record() -> dict:
    """Fetch current weather + pollution and build one feature row."""
    city = os.getenv('CITY', "Akora Khattak")
    lat, long = get_coordinates(city)

    weather = get_weather(lat, long)
    # pollution = get_pollutants(lat, long)

    current = weather['current']
    # components = pollution['list'][0]['components']
    components = get_pollutants(lat,long)

    dt = pd.to_datetime(current['time'])

    record = {
        "datetime": dt,
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),

        "temperature": current['temperature_2m'],
        "humidity": current['relative_humidity_2m'],
        "pressure": current['pressure_msl'],

        "co": components['co'],
        "no": components['no'],
        "no2": components['no2'],
        "o3": components['o3'],
        "so2": components['so2'],
        "pm2_5": components['pm2_5'],
        "pm10": components['pm10'],
        # "nh3": components['nh3'],

        "aqi": components['aqi'],

        "hour": int(dt.hour),
        "day": int(dt.day),
        "month": int(dt.month),
        "day_of_week": int(dt.dayofweek),
        "week_of_year": int(dt.isocalendar().week),
    }
    return record


def to_unix(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def backfill(start_date: str, end_date: str) -> pd.DataFrame:
    """Build historical features + targets as a DataFrame (no CSV, no Hopsworks push here)."""
    city = os.getenv('CITY', "Akora Khattak")
    lat, lon = get_coordinates(city)

    weather_data = get_historical_weather(lat, lon, start_date, end_date)
    pollution_data = get_historical_pollution(lat, lon, start_date, end_date)

    weather_df = pd.DataFrame(weather_data["hourly"])
    weather_df["datetime"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "pressure_msl": "pressure",
    }).drop(columns=["time"])

    pollution_df = pd.DataFrame(pollution_data["hourly"])
    pollution_df["datetime"] = pd.to_datetime(pollution_df["time"])
    pollution_df = pollution_df.rename(columns={
        "carbon_monoxide": "co",
        "nitrogen_monoxide": "no",
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
        "sulphur_dioxide": "so2",
        "ammonia": "nh3",
        "us_aqi": "aqi",
    }).drop(columns=["time"])

    df = pd.merge(weather_df, pollution_df, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)

    # df["city"] = city
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["aqi_change_rate"] = df["aqi"].diff()
    df.drop('nh3',axis=1,inplace=True)
    # Hopsworks doesn't like NaN in numeric feature columns on first insert
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0.0)

    return df

if __name__ == "__main__":
    # backfill("2026-04-17", "2026-07-17", "src/data/aqi_history.csv")
    record= create_new_record()
    print(record)