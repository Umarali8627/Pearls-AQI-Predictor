import requests
from dotenv import load_dotenv
import os 

load_dotenv()

API_KEY=os.getenv('WEATHER_API_KEY')

import requests
from datetime import datetime


def get_pollutants(latitude: float, longitude: float):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={latitude}&longitude={longitude}"
        "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_monoxide,"
        "nitrogen_dioxide,ozone,sulphur_dioxide,ammonia,us_aqi"
        "&forecast_hours=1"
        "&timezone=auto"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()

        pollution = response.json()

        hourly_data = pollution["hourly"]

        # Get the first/current hour's data
        current_data = {
            "time": hourly_data["time"][0],
            "pm10": hourly_data["pm10"][0],
            "pm2_5": hourly_data["pm2_5"][0],
            "co": hourly_data["carbon_monoxide"][0],
            "no": hourly_data["nitrogen_monoxide"][0],
            "no2": hourly_data["nitrogen_dioxide"][0],
            "o3": hourly_data["ozone"][0],
            "so2": hourly_data["sulphur_dioxide"][0],
            "nh3": hourly_data["ammonia"][0],
            "aqi": hourly_data["us_aqi"][0]
        }

        return current_data

    except Exception as ex:
        raise ValueError(f"Error occurred: {ex}")
   
    
def get_historical_pollution(latitude: float, longitude: float, start_date: str, end_date: str):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_monoxide,nitrogen_dioxide,"
        "ozone,sulphur_dioxide,ammonia,us_aqi"
        "&timezone=auto"
    )
    try:
        return requests.get(url).json()
    except Exception as ex:
        raise ValueError(f'Value Error {ex}')
# pollution = get_pollutants(34.00337 ,72.12561)
# print(pollution)
if __name__ == "__main__":
    pollution = get_pollutants(34.00337 ,72.12561)
    print(pollution)