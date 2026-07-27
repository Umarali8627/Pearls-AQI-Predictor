from fastapi import FastAPI,Response
from src.pipelines.inference_pipeline import (run_inference,
                                              current_data,HORIZONS,connect
                                              )
# from src.pipelines import feature_pipeline 
from fastapi.middleware.cors import CORSMiddleware
import threading



project = connect()
fs = project.get_feature_store()
hopsworks_lock = threading.Lock()




# set the AQI Category 
AQI_CATEGORIES= [
    (0, 50, "good", "Good"),
    (51, 100, "moderate", "Moderate"),
    (101, 150, "usg", "Unhealthy for Sensitive Groups"),
    (151, 200, "unhealthy", "Unhealthy"),
    (201, 300, "very_unhealthy", "Very Unhealthy"),
    (301, 10_000, "hazardous", "Hazardous"),
]  

ALERT_THRESHOLD =151
project = connect()
fs = project.get_feature_store()
def classify_aqi(value: float) -> dict:
    for low, high, key, category_label in AQI_CATEGORIES:
        if low <= value <= high:
            return {
                "category": key,
                "category_label": category_label,
                "alert": value >= ALERT_THRESHOLD,
            }
    return {"category": "unknown", "category_label": "unknown", "alert": False}



app = FastAPI(
    title = "Pearls AQI Predictor",
    description="City Based AQI Predictor that predict the aqi of the city of next 3 days",
    version="1.0.0"
)
# Set the middleware 
app.add_middleware(
    CORSMiddleware,
     allow_origins=[
        "http://localhost:5173",  
        "https://pearlsair.vercel.app", 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def home():
    return {"Hello ! From Pearls AQI Predictor Everything Work fine sever started"}
from fastapi import Response

@app.get('/data')
def get_current_24hrs_data():
    with hopsworks_lock:
        df = current_data(project)
    return Response(content=df.to_json(orient="records", date_format="iso"),
                     media_type="application/json")
@app.get('/prediction')
def get_3_days_prediction():
    with hopsworks_lock:
        forecasts = run_inference(project)
    for key, data in forecasts.items():
        if key == "Today":
            current_aqi = data.get("aqi")
            if current_aqi is not None:
                data["classification"] = classify_aqi(current_aqi)
        else:
            predicted_aqi = data.get("predicted_aqi")
            if predicted_aqi is not None:
                data["classification"] = classify_aqi(predicted_aqi)
    return forecasts
