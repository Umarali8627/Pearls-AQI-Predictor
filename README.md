# Pearls AQI Predictor

A FastAPI backend that serves real-time air quality data and 3-day AQI forecasts for Akora Khattak, Nowshera — powered by a Hopsworks feature store and scikit-learn/TensorFlow forecasting models.

## Features

- **Live data** — last 24 hours of pollutant and weather readings (`/data`)
- **AQI forecasting** — 24h / 48h / 72h ahead predictions using the best-performing registered model per horizon (`/prediction`)
- **Automated MLOps pipeline** — hourly feature ingestion and daily model retraining via GitHub Actions
- **AQI classification** — categorizes predictions into EPA-style bands (Good → Hazardous) with alert thresholds

## Tech Stack

- **API:** FastAPI, Uvicorn
- **Feature Store / Model Registry:** [Hopsworks](https://www.hopsworks.ai/)
- **ML:** scikit-learn (Ridge, RandomForest), TensorFlow
- **Frontend:** React + TanStack Start ([repo](https://github.com/Umarali8627/pearls-air-insight))
- **CI/CD:** GitHub Actions (scheduled feature & training pipelines)

## Architecture
Feature Pipeline (hourly) → Hopsworks Feature Store
↓
Train Pipeline (daily) → Hopsworks Model Registry
↓
FastAPI Backend (this repo)
↓
React Frontend


## API Endpoints

| Method | Endpoint      | Description                                  |
|--------|---------------|-----------------------------------------------|
| GET    | `/`           | Health check                                  |
| GET    | `/data`       | Last 24h of feature/pollutant data            |
| GET    | `/prediction` | 24h/48h/72h AQI forecasts with classification |

## Getting Started

### Prerequisites
- Python 3.11
- A [Hopsworks](https://www.hopsworks.ai/) account and API key

### Installation

```bash
git clone https://github.com/Umarali8627/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

HOPSWORKS_PROJECT=your_project_name
HOPSWORKS_API_KEY=your_api_key


### Run the API locally

```bash
fastapi dev main.py --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs.

## Automated Pipelines (GitHub Actions)

This repo runs two scheduled workflows:

| Workflow           | Schedule                  | Purpose                                  |
|---------------------|---------------------------|-------------------------------------------|
| Feature Pipeline    | Hourly (`0 * * * *`)      | Ingests new AQI/weather data into Hopsworks |
| Train Pipeline      | Daily (`15 2 * * *`)      | Retrains and registers forecasting models |

Both can also be triggered manually from the **Actions** tab.

## Project Structure

├── main.py # FastAPI app
├── src/
│ ├── pipelines/
│ │ ├── feature_pipeline.py
│ │ ├── trainiing.py
│ │ └── inference_pipeline.py
│ ├── data/
│ └── notebook/
├── .github/workflows/ # CI/CD pipeline definitions
└── requirements.txt


## License

MIT (or update as you prefer)
