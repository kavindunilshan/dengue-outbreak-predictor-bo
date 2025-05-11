from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

# Load model and scalers
model = joblib.load("models/model.pkl")
feature_scaler = joblib.load("models/feature_scaler.pkl")
target_scaler = joblib.load("models/target_scaler.pkl")

print(feature_scaler.feature_names_in_)

scaler_vars = [
    'temp_avg',
    'humidity_avg',
    'cases_per_100k',
    'vim',
    'cases_lag0',
    'cases_lag1',
    'precipitation_avg_ordinary_kriging_lag3',
    'precipitation_avg_ordinary_kriging_lag4'
]

# Final feature order (important)
feature_order = [
    "week",
    "cases_per_100k",
    "temp_avg",
    "humidity_avg",
    "vim",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "cases_lag0",
    "cases_lag1",
    "precipitation_avg_ordinary_kriging_lag3",
    "precipitation_avg_ordinary_kriging_lag4"
]

# Input schema
class DengueInput(BaseModel):
    week: int
    temp_avg: float
    humidity_avg: float
    vim: float
    cases_lag0: float
    cases_lag1: float
    precipitation_avg_ordinary_kriging_lag3: float
    precipitation_avg_ordinary_kriging_lag4: float
    population: int  # Used only for derived feature


@app.get("/test")
def test():
    return {"message": "API is working!"}


@app.post("/predict-week2")
def predict_week2(input: DengueInput):
    data = input.dict()

    # Derived feature
    cases_per_100k = (data["cases_lag0"] / data["population"]) * 100000

    # Extract week and month
    week = data["week"]
    year = int(str(week)[:4])
    week_of_year = int(str(week)[-2:])
    month = ((week_of_year - 1) // 4 + 1)

    # Temporal features
    month_sin = np.sin((2 * np.pi * month) / 12)
    month_cos = np.cos((2 * np.pi * month) / 12)
    week_angle = 2 * np.pi * week_of_year / 52
    week_sin = np.sin(week_angle)
    week_cos = np.cos(week_angle)

    # Assemble input
    input_df = pd.DataFrame([{
        "week": data["week"],
        "cases_per_100k": cases_per_100k,
        "temp_avg": data["temp_avg"],
        "humidity_avg": data["humidity_avg"],
        "vim": data["vim"],
        "month_sin": month_sin,
        "month_cos": month_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "cases_lag0": data["cases_lag0"],
        "cases_lag1": data["cases_lag1"],
        "precipitation_avg_ordinary_kriging_lag3": data["precipitation_avg_ordinary_kriging_lag3"],
        "precipitation_avg_ordinary_kriging_lag4": data["precipitation_avg_ordinary_kriging_lag4"]
    }])

    # Scale only the required features
    input_df[scaler_vars] = feature_scaler.transform(input_df[scaler_vars])

    # Arrange columns in the right order
    input_df = input_df[feature_order]

    # Predict
    prediction_scaled = model.predict(input_df)
    prediction = target_scaler.inverse_transform(np.array(prediction_scaled).reshape(-1, 1)).flatten()

    return {
        "weekAfter": round(prediction[0])  # 2-week-ahead prediction
    }