# 🌍 Pearls AQI Predictor

An end-to-end **MLOps-powered Air Quality Index (AQI) forecasting platform** that provides real-time air-quality data and predicts AQI for the next **24, 48, and 72 hours** for **Akora Khattak, Nowshera, Pakistan**.

The project demonstrates a complete machine learning production pipeline:

> **Data Collection → Feature Engineering → Feature Store → Model Training → Model Registry → FastAPI Inference → React Dashboard**

The system uses historical and real-time environmental data from **Open-Meteo**, stores engineered features in **Hopsworks**, trains multiple forecasting models, registers the best-performing models, and serves predictions through a production-ready FastAPI backend.

---

## 🚀 Live Project Overview

Pearls AQI Predictor is designed to answer:

> **"What will the air quality be in the next 24, 48, and 72 hours?"**

The platform:

* Collects historical air-quality and weather data
* Builds a three-month dataset for model training
* Continuously ingests new hourly environmental data
* Performs feature engineering and target generation
* Stores features in a Hopsworks Feature Store
* Trains multiple machine learning models
* Evaluates models using forecasting metrics
* Registers the best model for each prediction horizon
* Serves real-time data and AQI forecasts through FastAPI
* Displays results through a React frontend
* Automates the entire workflow using GitHub Actions

---

# ✨ Features

## 📡 Real-Time Air Quality Data

The API provides the latest environmental measurements, including:

* PM2.5
* PM10
* Carbon Monoxide (CO)
* Nitrogen Monoxide (NO)
* Nitrogen Dioxide (NO₂)
* Ozone (O₃)
* Sulfur Dioxide (SO₂)
* Temperature
* Humidity
* Atmospheric Pressure

The latest data is continuously ingested into the feature store through an automated hourly pipeline.

---

## 🔮 Multi-Horizon AQI Forecasting

The system generates forecasts for:

| Horizon  | Prediction                        |
| -------- | --------------------------------- |
| 24 Hours | AQI forecast for the next day     |
| 48 Hours | AQI forecast for two days ahead   |
| 72 Hours | AQI forecast for three days ahead |

Each forecasting horizon can use a different machine learning model based on its validation performance.

For example:

```text
24h → Ridge Regression
48h → Random Forest
72h → TensorFlow Neural Network
```

The best-performing model for each horizon is selected using validation metrics.

---

## 🏷️ AQI Classification

The predicted AQI values are converted into understandable air-quality categories:

| AQI Range | Category                       |
| --------- | ------------------------------ |
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| 301+      | Hazardous                      |

The API also generates alerts based on AQI severity.

---

# 🏗️ System Architecture

```text
                 ┌────────────────────────┐
                 │     Open-Meteo APIs     │
                 │                          │
                 │ Historical + Live Data  │
                 └──────────────┬─────────┘
                                │
                                ▼
                 ┌────────────────────────┐
                 │   Feature Pipeline     │
                 │                          │
                 │ Data Collection         │
                 │ Feature Engineering     │
                 │ Target Generation       │
                 └──────────────┬─────────┘
                                │
                                ▼
                 ┌────────────────────────┐
                 │   Hopsworks Feature    │
                 │        Store           │
                 │                          │
                 │ Historical + Live Data │
                 └──────────────┬─────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
 ┌────────────────────────┐        ┌────────────────────────┐
 │    Training Pipeline   │        │   Inference Pipeline   │
 │                        │        │                        │
 │ Load Features          │        │ Load Latest Features   │
 │ Create Time Features   │        │ Load Registered Models │
 │ Train Multiple Models  │        │ Generate Predictions   │
 │ Evaluate Models        │        │ Classify AQI           │
 │ Select Best Model      │        └──────────────┬─────────┘
 └──────────────┬─────────┘                       │
                ▼                                 ▼
 ┌────────────────────────┐        ┌────────────────────────┐
 │ Hopsworks Model        │        │      FastAPI Backend    │
 │ Registry               │        │                        │
 │                        │        │ /data                  │
 │ Best Model per Horizon │        │ /prediction            │
 └────────────────────────┘        └──────────────┬─────────┘
                                                   │
                                                   ▼
                                    ┌────────────────────────┐
                                    │   React Frontend        │
                                    │   TanStack Start        │
                                    └────────────────────────┘
```

---

# 📊 Data Source

The project uses the **Open-Meteo API** as the primary data source.

Open-Meteo provides historical and current:

### Air Quality Data

* PM2.5
* PM10
* CO
* NO
* NO₂
* O₃
* SO₂

### Weather Data

* Temperature
* Relative Humidity
* Surface Pressure

The system first collects approximately **three months of historical hourly data** to create the initial training dataset.

The historical data provides enough temporal information for the models to learn:

* Daily pollution patterns
* Hourly pollution cycles
* Weather-pollution relationships
* Short-term AQI trends
* Seasonal changes within the available training period

After the initial historical dataset is created, new data is continuously collected hourly through the feature pipeline.

---

# 🗃️ Historical Data Collection

The initial data preparation process follows this workflow:

```text
Open-Meteo Historical API
          │
          ▼
Fetch 3 Months of Hourly Data
          │
          ▼
Combine Weather + Pollutant Data
          │
          ▼
Clean Missing Values
          │
          ▼
Create Features and Targets
          │
          ▼
Store in Hopsworks Feature Store
```

The historical dataset acts as the foundation for model training.

After the initial dataset is available, the system continuously adds new observations:

```text
Historical Data
      +
Hourly New Data
      │
      ▼
Feature Store
```

This allows the training dataset to grow over time.

---

# 🧠 Feature Engineering

Feature engineering is one of the most important parts of the project.

The raw data contains pollutant and weather measurements. These raw values are transformed into meaningful features that help the models understand air-quality behavior.

## Raw Environmental Features

```text
temperature
humidity
pressure

co
no
no2
o3
so2
pm2_5
pm10
```

---

## ⏰ Time-Based Features

Time features are extracted from the timestamp:

```text
hour
day
month
day_of_week
```

These features help the model learn temporal patterns.

For example:

```text
hour = 8
```

may represent morning rush-hour pollution.

```text
hour = 23
```

may represent nighttime pollution patterns.

Similarly:

```text
day_of_week
```

can help the model understand differences between weekdays and weekends.

---

# 🎯 Target Engineering

The model does not simply predict the current AQI.

Instead, future AQI values are created as forecasting targets.

For example:

```text
aqi24h = AQI value 24 hours in the future

aqi48h = AQI value 48 hours in the future

aqi72h = AQI value 72 hours in the future
```

Conceptually:

```text
Current Features
      │
      ├──────────────► aqi24h
      │
      ├──────────────► aqi48h
      │
      └──────────────► aqi72h
```

The training dataset therefore contains:

```text
Features at Time T
        │
        ▼
Predict Future AQI
        │
        ├── T + 24 hours
        ├── T + 48 hours
        └── T + 72 hours
```

This transforms the project from a simple classification problem into a **multi-horizon time-series forecasting problem**.

---

# 🧪 Feature-Target Example

Example:

```text
Timestamp: 2026-07-20 10:00
```

The model receives features from:

```text
2026-07-20 10:00
```

And learns to predict:

```text
aqi24h → AQI at 2026-07-21 10:00

aqi48h → AQI at 2026-07-22 10:00

aqi72h → AQI at 2026-07-23 10:00
```

This allows the system to provide independent forecasts for each prediction horizon.

---

# 🏭 FTI Pipeline

The project follows an **FTI pipeline architecture**:

```text
Feature Pipeline
        │
        ▼
Training Pipeline
        │
        ▼
Inference Pipeline
```

---

## 1️⃣ Feature Pipeline

The Feature Pipeline runs hourly.

### Workflow

```text
Open-Meteo API
      │
      ▼
Fetch Latest Hourly Data
      │
      ▼
Clean Raw Data
      │
      ▼
Calculate Derived Features
      │
      ▼
Create Time Features
      │
      ▼
Calculate AQI / Targets
      │
      ▼
Insert Record into Hopsworks
```

The feature pipeline is responsible for maintaining fresh data in the feature store.

It acts as the connection between:

```text
External Data Source
        ↓
Feature Store
```

---

## 2️⃣ Training Pipeline

The Training Pipeline runs daily.

### Workflow

```text
Hopsworks Feature Store
          │
          ▼
Read Historical Features
          │
          ▼
Prepare Training Dataset
          │
          ▼
Create Train/Validation/Test Sets
          │
          ▼
Train Multiple Models
          │
          ▼
Evaluate Models
          │
          ▼
Select Best Model
          │
          ▼
Register Model
```

The pipeline can train different models such as:

### Ridge Regression

A fast linear baseline model.

```text
Features → Linear Relationships → AQI Prediction
```

### Random Forest

A tree-based ensemble model capable of learning nonlinear relationships between:

```text
Pollutants
Weather
Time
Historical Values
```

### TensorFlow Neural Network

A neural network can learn more complex relationships within the environmental data.

---

# 📏 Model Evaluation

Models are evaluated using regression metrics.

## RMSE

Root Mean Squared Error measures the average prediction error while giving more weight to larger errors.

```text
RMSE = √(average squared error)
```

Lower RMSE is better.

---

## MAE

Mean Absolute Error represents the average absolute difference between:

```text
Actual AQI
      vs
Predicted AQI
```

Lower MAE is better.

---

## R² Score

R² measures how much variance in the target variable is explained by the model.

Higher values generally indicate better performance.

---

# 🏆 Best Model Selection

The system does not assume that one model is best for every forecasting horizon.

Instead, each horizon is evaluated independently.

Example:

```text
24-Hour Forecast
├── Ridge RMSE: 12.4
├── Random Forest RMSE: 9.8
└── TensorFlow RMSE: 11.2

Best Model → Random Forest
```

For another horizon:

```text
72-Hour Forecast
├── Ridge RMSE: 25.1
├── Random Forest RMSE: 21.7
└── TensorFlow RMSE: 18.9

Best Model → TensorFlow
```

The best-performing model is registered in the Hopsworks Model Registry.

This creates a model selection strategy based on actual validation performance.

---

# 🔭 Forecasting Horizons

The system supports three independent forecasting horizons.

## 24-Hour Forecast

Short-term prediction.

The model focuses on:

* Recent pollutant levels
* Latest weather conditions
* Hourly patterns
* Recent AQI trends

---

## 48-Hour Forecast

Medium-term prediction.

The model must handle increased uncertainty and changing environmental conditions.

---

## 72-Hour Forecast

Longer-term prediction.

This horizon is more challenging because prediction uncertainty increases as the forecast moves further into the future.

Therefore, each horizon can have a separate optimized model.

```text
Current Data
     │
     ├──────────────► 24-Hour AQI Forecast
     │
     ├──────────────► 48-Hour AQI Forecast
     │
     └──────────────► 72-Hour AQI Forecast
```

---

# 🔁 MLOps Automation

The entire workflow is automated using **GitHub Actions**.

## Hourly Feature Pipeline

```text
Every Hour
    │
    ▼
GitHub Actions Trigger
    │
    ▼
Run Feature Pipeline
    │
    ▼
Fetch Latest Open-Meteo Data
    │
    ▼
Engineer Features
    │
    ▼
Store Data in Hopsworks
```

Schedule:

```text
0 * * * *
```

This means the pipeline runs at the beginning of every hour.

---

## Daily Training Pipeline

```text
Every Day
    │
    ▼
GitHub Actions Trigger
    │
    ▼
Read Latest Feature Store Data
    │
    ▼
Train Multiple Models
    │
    ▼
Evaluate Models
    │
    ▼
Select Best Model Per Horizon
    │
    ▼
Register Models in Hopsworks
```

Schedule:

```text
15 2 * * *
```

The training pipeline can also be triggered manually from the GitHub Actions interface.

---

# 🔄 Complete Production Workflow

```text
┌─────────────────────┐
│   Open-Meteo API    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Feature Pipeline   │
│      Hourly         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hopsworks Feature   │
│       Store         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Training Pipeline   │
│       Daily         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hopsworks Model     │
│      Registry       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   FastAPI Backend   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   React Frontend    │
└─────────────────────┘
```

---

# 🛠️ Tech Stack

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Data Source

* Open-Meteo Historical Weather API
* Open-Meteo Air Quality API

### Feature Store & Model Registry

* Hopsworks

### Machine Learning

* scikit-learn
* Ridge Regression
* Random Forest
* TensorFlow

### Frontend

* React
* TanStack Start

### MLOps

* GitHub Actions
* Automated scheduled pipelines
* Hopsworks Model Registry
* Hopsworks Feature Store

---

# 📡 API Endpoints

| Method | Endpoint      | Description                                       |
| ------ | ------------- | ------------------------------------------------- |
| GET    | `/`           | API health check                                  |
| GET    | `/data`       | Returns the latest 24 hours of environmental data |
| GET    | `/prediction` | Returns 24h, 48h, and 72h AQI forecasts           |

---

## Example Prediction Response

```json
{
  "predictions": {
    "24h": {
      "aqi": 82,
      "category": "Moderate"
    },
    "48h": {
      "aqi": 118,
      "category": "Unhealthy for Sensitive Groups"
    },
    "72h": {
      "aqi": 156,
      "category": "Unhealthy"
    }
  }
}
```

---

# 📁 Project Structure

```text
Pearls-AQI-Predictor/
│
├── main.py
│
├── src/
│   ├── pipelines/
│   │   ├── feature_pipeline.py
│   │   ├── trainiing.py
│   │   └── inference_pipeline.py
│   │
│   ├── data/
│   │
│   └── notebook/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── train_pipeline.yml
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Getting Started

## Prerequisites

* Python 3.11
* Hopsworks account
* Hopsworks API key
* Git
* GitHub account

---

## Installation

```bash
git clone https://github.com/Umarali8627/Pearls-AQI-Predictor.git

cd Pearls-AQI-Predictor

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
HOPSWORKS_PROJECT=your_project_name
HOPSWORKS_API_KEY=your_api_key
```

Never commit your API keys to GitHub.

---

# ▶️ Run the API Locally

```bash
fastapi dev main.py --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

This opens the interactive Swagger API documentation.

---

# 🤖 GitHub Actions Automation

The project uses GitHub Actions to automate the MLOps workflow.

Required GitHub Secrets:

```text
HOPSWORKS_PROJECT
HOPSWORKS_API_KEY
```

The automated workflows are:

```text
feature_pipeline.yml
```

Runs hourly and adds new environmental data.

```text
train_pipeline.yml
```

Runs daily and retrains the forecasting models.

This creates a continuously updating ML system:

```text
New Data
   │
   ▼
Feature Store Updated
   │
   ▼
Daily Retraining
   │
   ▼
Best Model Registered
   │
   ▼
New Predictions Served
```

---

# 🎯 Project Objective

The objective of Pearls AQI Predictor is to demonstrate how a machine learning model can be transformed into a continuously operating production system.

This project combines:

* Data Engineering
* Feature Engineering
* Time-Series Forecasting
* Machine Learning
* Model Evaluation
* Feature Stores
* Model Registries
* FastAPI
* React
* CI/CD
* MLOps Automation

Instead of training a model once inside a notebook, the system continuously:

```text
Collects Data
     ↓
Processes Data
     ↓
Stores Features
     ↓
Retrains Models
     ↓
Registers Best Models
     ↓
Serves Predictions
```

This makes Pearls AQI Predictor a complete end-to-end **production-oriented MLOps project**.

---

# 👨‍💻 Author

**Umar Ali**

BSCS Student | AI Engineer | Data Science & MLOps Enthusiast

Interested in:

* Machine Learning
* Deep Learning
* MLOps
* AI Engineering
* Data Engineering
* Full-Stack AI Applications

---

# 🔗 Related Links

* **Backend Repository:** https://github.com/Umarali8627/Pearls-AQI-Predictor
* **Frontend Repository:** https://github.com/Umarali8627/pearls-air-insight

---

# 📄 License

This project is licensed under the MIT License.
