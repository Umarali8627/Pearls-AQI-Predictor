from src.data.coordinates import get_coordinates
from src.data.pollutants import get_pollutants, get_historical_pollution
from src.data.weather import get_weather, get_historical_weather
from src.data.record import create_new_record,backfill
import pandas as pd
from dotenv import load_dotenv
import os
import hopsworks


load_dotenv()

FG_NAME = "aqi_features"
FG_VERSION = 1


def get_feature_store():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
    )
    return project.get_feature_store()


def get_or_create_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="Hourly weather + pollution features for AQI prediction",
        primary_key=["datetime"],
        event_time="datetime",
        online_enabled=False,
        time_travel_format = "HUDI"
    )

def run_backfill_to_hopsworks(start_date: str, end_date: str):
    """One-time job: build historical data and insert into the Hopsworks feature group."""
    df = backfill(start_date, end_date)

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    fg.insert(df, write_options={"wait_for_job": True,"internal_kafka": False,
        "kafka_timeout": 30})
    print(f"Backfilled {len(df)} rows into Hopsworks feature group '{FG_NAME}' v{FG_VERSION}")
    return df


def run_hourly_to_hopsworks():
    """Hourly job: fetch the current record, compute AQI change rate
    against the last stored row, and insert it into Hopsworks.
    """

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    record = create_new_record()

    try:
        existing_df = fg.read()

        if not existing_df.empty:
            existing_df = existing_df.sort_values("datetime")

            last_aqi = float(existing_df.iloc[-1]["aqi"])

            record["aqi_change_rate"] = float(
                record["aqi"] - last_aqi
            )

        else:
            record["aqi_change_rate"] = 0.0

    except Exception as e:
        print(
            f"Could not read existing feature group "
            f"(may be empty): {e}"
        )

        record["aqi_change_rate"] = 0.0

    # Create DataFrame
    new_df = pd.DataFrame([record])

    # Explicitly match Hopsworks Feature Group schema
    new_df["hour"] = new_df["hour"].astype("int32")
    new_df["day"] = new_df["day"].astype("int32")
    new_df["month"] = new_df["month"].astype("int32")
    new_df["day_of_week"] = new_df["day_of_week"].astype("int32")

    # Ensure AQI change rate is double
    new_df["aqi_change_rate"] = (
        new_df["aqi_change_rate"].astype("float64")
    )

    print(new_df.dtypes)

    fg.insert(
        new_df,
        write_options={"wait_for_job": True}
    )

    print(
        f"Inserted new record for "
        f"{record['datetime']} into Hopsworks"
    )

    return new_df
def run_feature_Pipeline(back_fill :bool = False,hourly_run:bool = False):
    if back_fill:
        run_backfill_to_hopsworks("2026-04-17", "2026-07-17")
    if hourly_run:
        run_hourly_to_hopsworks()

if __name__ == "__main__":
    #  one-time historical backfill (run once)
# run_backfill_to_hopsworks("2026-04-17", "2026-07-23")

#     # hourly job (this is what your GitHub Action should call every hour)
     run_hourly_to_hopsworks()