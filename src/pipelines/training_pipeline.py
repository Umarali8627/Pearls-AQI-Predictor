# now building a training Pipeline
import hopsworks
import pandas as pd
from dotenv import load_dotenv
import os
import shutil
import time
import logging
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np
import joblib

load_dotenv()
FG_NAME = "aqi_features"
FG_VERSION = 1
# model directory
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# define the features columns
FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure",
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10",
    "hour", "day", "month", "day_of_week", "week_of_year",
    "aqi", "aqi_change_rate",
]
HORIZONS = {24: "24h (Day 1)", 48: "48h (Day 2)", 72: "72h (Day 3)"}

# Which model types need scaled input at inference time.
SCALED_MODEL_TYPES = {"Ridge", "NeuralNetwork"}

# Registration reliability settings -- Hopsworks' model-upload staging
# component has a known intermittent 500 ("Singleton StagingManager is
# unavailable") that is transient and typically clears within seconds
# to minutes. Retry rather than treating it as a hard failure.
SAVE_MAX_ATTEMPTS = 3
SAVE_RETRY_BACKOFF_SECONDS = 15  # multiplied by attempt number


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")

logger = logging.getLogger("aqi_pipeline")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # keep console concise

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # full detail goes to the log file

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.info("Logging initialized. Full log at: %s", LOG_FILE)


# funtion to connect
def connect():
    project_name = os.getenv("HOPSWORKS_PROJECT", "abd")
    logger.info("Connecting to Hopsworks project '%s'...", project_name)
    project = hopsworks.login(
        project=project_name,
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    logger.info("Connected to Hopsworks project '%s'.", project_name)
    return project


# function to get feature store
def load_data(project) -> pd.DataFrame:
    logger.info("Fetching feature group '%s' (v%s)...", FG_NAME, FG_VERSION)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()
    logger.info("Loaded %d rows, %d columns from the feature store.", len(df), df.shape[1])
    return df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:

    for h in HORIZONS:
        df[f"aqi_target_{h}h"] = df["aqi"].shift(-h)

    target_cols = [f"aqi_target_{h}h" for h in HORIZONS]
    rows_before = len(df)
    df_model = df.dropna(subset=target_cols).reset_index(drop=True)
    logger.debug(
        "add_future_targets: dropped %d rows without full targets (%d -> %d).",
        rows_before - len(df_model), rows_before, len(df_model),
    )
    return df_model


# function to preprocess the data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data (%d rows)...", len(df))
    # sort data according to dates
    df = df.sort_values("datetime").reset_index(drop=True)
    # now add future targets 24/48/72 hrs
    df = add_future_targets(df)
    logger.info("Preprocessing complete. %d rows remain after target creation.", len(df))
    return df


# now function to split the data for training with time period
def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y_by_horizon = {h: df[f"aqi_target_{h}h"] for h in HORIZONS}

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    y_train_by_horizon = {h: y.iloc[:split_idx] for h, y in y_by_horizon.items()}
    y_test_by_horizon = {h: y.iloc[split_idx:] for h, y in y_by_horizon.items()}
    logger.info(
        "Split data (time-based, 80/20): %d train rows, %d test rows.",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_test_by_horizon, y_train_by_horizon


# scaling function for some models
def scale_data(X_train, X_test):
    logger.debug("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.debug("Scaling complete.")

    return scaler, X_train_scaled, X_test_scaled


# now train the models
def train_ridge_models(X_train_scaled, y_train_by_horizon):
    logger.info("Training Ridge models for horizons: %s", list(HORIZONS))
    models = {}

    for h in HORIZONS:
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train_by_horizon[h])
        logger.debug("Ridge model trained for %sh horizon.", h)

        models[h] = model

    logger.info("Ridge training complete.")
    return models


def train_rf_models(X_train, y_train_by_horizon):
    logger.info("Training RandomForest models for horizons: %s", list(HORIZONS))
    models = {}

    for h in HORIZONS:

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train_by_horizon[h])
        logger.debug("RandomForest model trained for %sh horizon.", h)

        models[h] = model

    logger.info("RandomForest training complete.")
    return models


def train_nn_models(X_train_scaled, y_train_by_horizon):
    logger.info("Training Neural Network models for horizons: %s", list(HORIZONS))
    nn_models = {}
    nn_histories = {}
    keras.utils.set_random_seed(42)
    for h, label in HORIZONS.items():
        logger.debug("Building NN for %sh horizon (%s)...", h, label)
        nn_model = keras.Sequential([
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])

        nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        early_stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

        history = nn_model.fit(
            X_train_scaled, y_train_by_horizon[h],
            validation_split=0.15,
            epochs=200,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0,
        )

        epochs_ran = len(history.history["loss"])
        final_loss = history.history["loss"][-1]
        final_val_loss = history.history.get("val_loss", [None])[-1]
        logger.debug(
            "NN for %sh horizon stopped after %d epochs (loss=%.4f, val_loss=%s).",
            h, epochs_ran, final_loss,
            f"{final_val_loss:.4f}" if final_val_loss is not None else "n/a",
        )

        nn_models[h] = nn_model
        nn_histories[h] = history

    logger.info("Neural Network training complete.")
    return nn_models, nn_histories


# now evaluate the models
def evaluate_model(models, X_test, y_test_by_horizon, model_name):
    """
    Evaluate models for each forecasting horizon.

    Parameters
    ----------
    models : dict
        Dictionary of trained models.
    X_test : pd.DataFrame or np.ndarray
        Test features (already scaled if the model type requires it).
    y_test_by_horizon : dict
        Dictionary of test targets.
    model_name : str
        Name of the model (e.g., Ridge, RandomForest, NeuralNetwork).

    Returns
    -------
    results : list
        List of dictionaries containing evaluation metrics.
    """

    results = []

    logger.info("%s", "=" * 60)
    logger.info("Evaluating %s", model_name)
    logger.info("%s", "=" * 60)

    for h in HORIZONS:

        model = models[h]

        y_true = y_test_by_horizon[h]

        y_pred = model.predict(X_test)

        # Keras predict returns (n,1)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results.append({
            "horizon": h,
            "model": model_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

        logger.info(
            "%sh -> RMSE: %.2f | MAE: %.2f | R2: %.3f",
            h, rmse, mae, r2,
        )

    return results


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def select_best_models(all_results, models_by_type):
    """
    Pick the best-performing model type (lowest RMSE) for each horizon.

    Parameters
    ----------
    all_results : list[dict]
        Combined evaluation results from every model type/horizon,
        e.g. the concatenation of evaluate_model() outputs.
    models_by_type : dict
        {model_name: {horizon: trained_model}}

    Returns
    -------
    best : dict
        {horizon: {"model_name": str, "model": obj, "metrics": dict}}
    """
    logger.info("Selecting best model per horizon by lowest RMSE...")
    results_df = pd.DataFrame(all_results)
    best = {}

    for h in HORIZONS:
        h_results = results_df[results_df["horizon"] == h]
        best_row = h_results.loc[h_results["rmse"].idxmin()]
        model_name = best_row["model"]

        best[h] = {
            "model_name": model_name,
            "model": models_by_type[model_name][h],
            "metrics": {
                "rmse": float(best_row["rmse"]),
                "mae": float(best_row["mae"]),
                "r2": float(best_row["r2"]),
            },
        }
        logger.info(
            "Horizon %sh -> best model: %s (RMSE=%.2f)",
            h, model_name, best_row["rmse"],
        )

    return best


# ---------------------------------------------------------------------------
# Local artifact saving (models must live on disk before they can be
# uploaded to the Hopsworks Model Registry)
# ---------------------------------------------------------------------------
def save_model_artifacts(model_name, model, horizon, scaler=None):
    """
    Persist a trained model (and its scaler, if the model type needs one
    at inference time) into a clean local directory.

    Returns
    -------
    model_dir : str
        Local folder path containing everything the registry upload needs.
    """
    model_dir = os.path.join(MODEL_DIR, f"aqi_{model_name.lower()}_{horizon}h")

    # start clean so stale artifacts from a previous run don't get uploaded
    if os.path.exists(model_dir):
        logger.debug("Removing stale artifact directory: %s", model_dir)
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    if model_name == "NeuralNetwork":
        # SavedModel format, required for the Hopsworks `tensorflow` model API
        model.export(os.path.join(model_dir, "model"))
    else:
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        logger.debug("Saved scaler.pkl alongside %s (%sh).", model_name, horizon)

    # keep the feature order alongside the model so inference code doesn't
    # need to guess column ordering
    with open(os.path.join(model_dir, "feature_columns.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLUMNS))

    logger.info("Saved %s (%sh) artifacts to %s", model_name, horizon, model_dir)
    return model_dir


# ---------------------------------------------------------------------------
# Reliable save: retry on transient backend errors, then verify the
# upload actually landed by downloading it back before trusting it.
# ---------------------------------------------------------------------------
def save_and_verify(hw_model, local_dir, model_name, horizon):
    """
    Wraps hw_model.save(local_dir) with:
      1. Retries on transient Hopsworks backend errors (e.g. the known
         intermittent "Singleton StagingManager is unavailable" 500).
      2. A post-save verification download, so a partially-succeeded
         upload (metadata registered, files missing) is caught here and
         now -- not later during inference.

    Raises the last exception if every attempt fails.
    """
    last_exc = None

    for attempt in range(1, SAVE_MAX_ATTEMPTS + 1):
        try:
            hw_model.save(local_dir)

            # Verification: download what we just uploaded to a throwaway
            # path and confirm the expected files are actually present.
            verify_dir = hw_model.download()
            expected = ["model.pkl"] if model_name != "NeuralNetwork" else ["model"]
            missing = [f for f in expected if not os.path.exists(os.path.join(verify_dir, f))]
            if missing:
                raise RuntimeError(
                    f"Upload for '{hw_model.name}' v{hw_model.version} reported success "
                    f"but expected artifact(s) {missing} are missing after download -- "
                    f"treating as a failed upload."
                )

            logger.debug(
                "Verified '%s' (%sh) upload: found %s at %s.",
                model_name, horizon, expected, verify_dir,
            )
            return  # success

        except Exception as e:
            last_exc = e
            if attempt < SAVE_MAX_ATTEMPTS:
                wait = SAVE_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Save/verify attempt %d/%d failed for '%s' (%sh horizon): %s "
                    "-- retrying in %ds.",
                    attempt, SAVE_MAX_ATTEMPTS, model_name, horizon, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "All %d save/verify attempts failed for '%s' (%sh horizon).",
                    SAVE_MAX_ATTEMPTS, model_name, horizon,
                )

    raise last_exc


# ---------------------------------------------------------------------------
# Push the winning model per horizon to the Hopsworks Model Registry
# ---------------------------------------------------------------------------
def push_to_model_registry(project, best_models, scaler, X_train):
    """
    Register the best model for each forecasting horizon in the
    Hopsworks Model Registry.

    Parameters
    ----------
    project : hopsworks Project handle (from connect())
    best_models : dict
        Output of select_best_models().
    scaler : fitted StandardScaler
        Needed at inference time for Ridge / NeuralNetwork models.
    X_train : pd.DataFrame
        Used to build a representative input_example for the registry.

    Returns
    -------
    registered : dict
        {horizon: hsml Model object}
    """
    logger.info("Connecting to Hopsworks Model Registry...")
    mr = project.get_model_registry()
    registered = {}
    failed = []

    for h, info in best_models.items():
        model_name = info["model_name"]
        model = info["model"]
        metrics = info["metrics"]

        try:
            needs_scaler = model_name in SCALED_MODEL_TYPES
            local_dir = save_model_artifacts(
                model_name, model, h, scaler=scaler if needs_scaler else None
            )

            registry_name = f"aqi_forecast_{h}h"
            description = (
                f"Best AQI model for the {HORIZONS[h]} horizon, selected as "
                f"'{model_name}' by lowest test RMSE. "
                f"Requires {'scaled (see scaler.pkl)' if needs_scaler else 'raw'} "
                f"input in the order given by feature_columns.txt."
            )
            input_example = X_train.iloc[:1]

            logger.info("Registering '%s' as a %s model...", registry_name, model_name)
            if model_name == "NeuralNetwork":
                hw_model = mr.tensorflow.create_model(
                    name=registry_name,
                    metrics=metrics,
                    description=description,
                    input_example=input_example,
                )
            else:
                hw_model = mr.python.create_model(
                    name=registry_name,
                    metrics=metrics,
                    description=description,
                    input_example=input_example,
                )

            save_and_verify(hw_model, local_dir, model_name, h)
            registered[h] = hw_model

            logger.info(
                "Registered '%s' (v%s) -> %s  RMSE=%.2f  R2=%.3f",
                registry_name, hw_model.version, model_name,
                metrics["rmse"], metrics["r2"],
            )
        except Exception:
            logger.exception("Failed to register model for the %sh horizon.", h)
            failed.append(h)

    if failed:
        logger.warning("Model registry push finished with failures for horizons: %s", failed)
        logger.warning(
            "Failed horizons may have left an empty/broken version registered in "
            "Hopsworks (metadata created, files missing) -- delete those versions "
            "from the Model Registry UI before the next inference run, or leave "
            "them: the inference pipeline's fallback will skip broken versions "
            "and fall through to the next-best one."
        )
    else:
        logger.info("All models registered successfully.")

    return registered


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    pipeline_start = time.perf_counter()
    logger.info("Starting AQI training pipeline.")

    try:
        project = connect()

        df = load_data(project)
        df = preprocess_data(df)

        X_train, X_test, y_test_by_horizon, y_train_by_horizon = split_data(df)
        scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

        t0 = time.perf_counter()
        ridge_models = train_ridge_models(X_train_scaled, y_train_by_horizon)
        logger.info("Ridge training took %.1fs.", time.perf_counter() - t0)

        t0 = time.perf_counter()
        rf_models = train_rf_models(X_train, y_train_by_horizon)
        logger.info("RandomForest training took %.1fs.", time.perf_counter() - t0)

        t0 = time.perf_counter()
        nn_models, nn_histories = train_nn_models(X_train_scaled, y_train_by_horizon)
        logger.info("Neural Network training took %.1fs.", time.perf_counter() - t0)

        all_results = []
        all_results += evaluate_model(ridge_models, X_test_scaled, y_test_by_horizon, "Ridge")
        all_results += evaluate_model(rf_models, X_test, y_test_by_horizon, "RandomForest")
        all_results += evaluate_model(nn_models, X_test_scaled, y_test_by_horizon, "NeuralNetwork")

        models_by_type = {
            "Ridge": ridge_models,
            "RandomForest": rf_models,
            "NeuralNetwork": nn_models,
        }

        best_models = select_best_models(all_results, models_by_type)

        push_to_model_registry(project, best_models, scaler, X_train)

        logger.info("Pipeline completed in %.1fs.", time.perf_counter() - pipeline_start)

    except Exception:
        logger.exception("Pipeline failed after %.1fs.", time.perf_counter() - pipeline_start)
        raise


if __name__ == "__main__":
    main()