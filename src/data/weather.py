import requests 




def get_weather(latitude: float,longitude:float):
    """Weather function that takes the langitude and latitude of the city 
      and return the weather data of the current data """
    weather_url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}&longitude={longitude}"
    "&current=temperature_2m,relative_humidity_2m,pressure_msl"
    "&timezone=auto"
)
    try:
        weather = requests.get(weather_url).json()
       
       
        return weather
    except Exception as ex:
        raise ValueError(f'Value Error {ex}')
  
def get_historical_weather(latitude: float, longitude: float, start_date: str, end_date: str):
    """start_date/end_date format: 'YYYY-MM-DD'"""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl"
        "&timezone=auto"
    )
    try:
        return requests.get(url).json()
    except Exception as ex:
        raise ValueError(f'Value Error {ex}')
    

if __name__ == "__main__":    
    weather = get_weather(34.00337,72.12561)
    print(f"Weather :: {weather}")
# current = weather['current']
# datetime = current['time']
# date_time = datetime.split('T')
# date = date_time[0]
# time = date_time[1]
# print(f'DATE :: {date} Time :: {time}')
