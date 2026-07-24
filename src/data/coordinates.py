import requests
from dotenv import load_dotenv
import os 

load_dotenv()
city = os.getenv('CITY')

def get_coordinates(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "results" in data and len(data["results"]) > 0:
        latitude = data["results"][0]["latitude"]
        longitude = data["results"][0]["longitude"]
        return latitude, longitude
    else:
        return None, None

# lat ,long = get_coordinates(city)
# print(lat,long)