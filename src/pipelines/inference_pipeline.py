"""
Inference pipeline for AQI forecasting.

Loads the best-performing registered model for each horizon (24h/48h/72h)
from the Hopsworks Model Registry, pulls the latest feature row from the
feature store, and returns AQI forecasts.

Model type (Ridge / RandomForest / NeuralNetwork) is NOT assumed from the
horizon -- it's detected from the downloaded artifact contents, because
different training runs can register different winning model types under
the same registry name (e.g. aqi_forecast_24h v1 = Ridge, v2 = RandomForest).
"""

import os
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import hopsworks
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

FG_NAME = "aqi_features"
FG_VERSION = 1

HORIZONS = {24: "24h (Day 1)", 48: "48h (Day 2)", 72: "72h (Day 3)"}

FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure",
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "day_of_week", "week_of_year",
    "aqi", "aqi_change_rate",
]

DOWNLOAD_DIR = "downloaded_models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging (mirrors the training pipeline's setup)
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"inference_{datetime.now():%Y%m%d_%H%M%S}.log")

logger = logging.getLogger("aqi_inference")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def connect():
    project_name = os.getenv("HOPSWORKS_PROJECT", "abd")
    logger.info("Connecting to Hopsworks project '%s'...", project_name)
    project = hopsworks.login(
        project=project_name,
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    logger.info("Connected to Hopsworks project '%s'.", project_name)
    return project


# ---------------------------------------------------------------------------
# Feature retrieval
# ---------------------------------------------------------------------------
def get_latest_feature_row(project) -> pd.DataFrame:
    """
    Pull the most recent row from the feature group. This is the row of
    current conditions (temp, pollutants, etc.) the models use to forecast
    AQI 24h / 48h / 72h into the future -- same input shape used at
    training time, just for the single most recent timestamp.
    """
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()

    if df.empty:
        raise RuntimeError(f"Feature group '{FG_NAME}' v{FG_VERSION} is empty.")

    df = df.sort_values("datetime").reset_index(drop=True)
    latest = df.iloc[[-1]].reset_index(drop=True)  # keep as DataFrame
    logger.info("Latest feature row timestamp: %s", latest["datetime"].iloc[0])

    missing = [c for c in FEATURE_COLUMNS if c not in latest.columns]
    if missing:
        raise RuntimeError(f"Feature group is missing expected columns: {missing}")

    return latest[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_for_horizon(project, horizon: int, use: str = "best"):
    """
    Download and load the registered model for a given horizon.

    Parameters
    ----------
    use : "best" to fetch the lowest-RMSE version ever registered under this
          name (recommended -- matches your training pipeline's own
          selection rule), or "latest" to always take the most recently
          registered version regardless of RMSE.
    """
    mr = project.get_model_registry()
    registry_name = f"aqi_forecast_{horizon}h"

    if use == "best":
        hw_model = mr.get_best_model(registry_name, "rmse", "min")
    elif use == "latest":
        versions = mr.get_models(registry_name)
        if not versions:
            hw_model = None
        else:
            hw_model = max(versions, key=lambda m: m.version)
    else:
        raise ValueError("use must be 'best' or 'latest'")

    if hw_model is None:
        raise RuntimeError(f"No registered model found for '{registry_name}'.")

    rmse = hw_model.training_metrics.get("rmse")
    logger.info(
        "Selected '%s' v%s (rmse=%s) [%s]",
        registry_name, hw_model.version, rmse, use,
    )

    local_dir = hw_model.download()

    scaler_path = os.path.join(local_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    sklearn_path = os.path.join(local_dir, "model.pkl")
    tf_path = os.path.join(local_dir, "model")

    if os.path.exists(sklearn_path):
        model = joblib.load(sklearn_path)
        kind = "sklearn"
    elif os.path.isdir(tf_path):
        model = tf.saved_model.load(tf_path)
        kind = "tensorflow"
    else:
        raise FileNotFoundError(
            f"No recognizable model artifact (model.pkl or model/) in {local_dir}"
        )

    logger.debug("Loaded '%s' as a %s model (scaler=%s).", registry_name, kind, scaler is not None)

    return {
        "model": model,
        "scaler": scaler,
        "kind": kind,
        "version": hw_model.version,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_for_horizon(model_bundle: dict, X: pd.DataFrame) -> float:
    X_input = X[FEATURE_COLUMNS]

    if model_bundle["scaler"] is not None:
        X_input = model_bundle["scaler"].transform(X_input)
    else:
        X_input = X_input.to_numpy(dtype="float32")

    if model_bundle["kind"] == "tensorflow":
        infer = model_bundle["model"].signatures["serving_default"]
        output = infer(tf.constant(X_input, dtype=tf.float32))
        pred = list(output.values())[0].numpy().ravel()[0]
    else:
        pred = model_bundle["model"].predict(X_input).ravel()[0]

    return float(pred)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_inference(project,use: str = "best") -> dict:
    # project = connect()
    X_latest = get_latest_feature_row(project)
  
    forecasts = {}
    latest_data = X_latest.iloc[0].to_dict()
    for h, label in HORIZONS.items():
        bundle = load_model_for_horizon(project, h, use=use)
        pred = predict_for_horizon(bundle, X_latest)
        forecasts[h] = {
            "label": label,
            "predicted_aqi": round(pred, 2),
            "model_version": bundle["version"],
            "model_rmse": bundle["rmse"],
        }
        logger.info("%s -> predicted AQI = %.2f (v%s)", label, pred, bundle["version"])
    forecasts["Today"]=latest_data
    return forecasts
# def current_data(project):
#        fs = project.get_feature_store()
#        fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
#        df = fg.read()
       
#        if df.empty:
#               raise RuntimeError(f"Feature group '{FG_NAME}' v{FG_VERSION} is empty.")
      
#        df = df.sort_values("datetime").reset_index(drop=True)

#        df.to_csv("current_data",index=False)
#        print("Data saved successfully..")
import datetime

def current_data(project):
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # Cutoff: last 24 hours (adjust window as needed)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=1)

    df = fg.filter(fg.datetime >= cutoff).read()

    if df.empty:
        raise RuntimeError(f"Feature group '{FG_NAME}' v{FG_VERSION} has no recent data.")

    df = df.sort_values("datetime").reset_index(drop=True)

    # df.to_csv("current_data", index=False)
    # print("Data saved successfully..")
    return df 




if __name__ == "__main__":
    results = run_inference(use="best")
    print(json.dumps(results, indent=2))
    # project = connect()
    # X_latest = get_latest_feature_row(project)
    # print(X_latest)
    # current_data(project)