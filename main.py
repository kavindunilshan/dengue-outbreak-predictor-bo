from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

# Allow frontend to communicate (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace "*" with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DengueInput(BaseModel):
    week: int
    temp_avg: float
    humidity_avg: float
    vim: float
    cases_lag0: float
    cases_lag1: float
    precipitation_avg_ordinary_kriging_lag3: float
    precipitation_avg_ordinary_kriging_lag4: float
    population: int


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

feature_order = [
    "week", "cases_per_100k", "temp_avg", "humidity_avg", "vim",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "cases_lag0", "cases_lag1",
    "precipitation_avg_ordinary_kriging_lag3", "precipitation_avg_ordinary_kriging_lag4"
]

models = {
    "week1": {
        "model": joblib.load("models/model1/model.pkl"),
        "feature_scaler": joblib.load("models/model1/feature_scaler.pkl"),
        "target_scaler": joblib.load("models/model1/target_scaler.pkl"),
    },
    "week2": {
        "model": joblib.load("models/model2/model.pkl"),
        "feature_scaler": joblib.load("models/model2/feature_scaler.pkl"),
        "target_scaler": joblib.load("models/model2/target_scaler.pkl"),
    }
}


def prepare_input(data: dict) -> pd.DataFrame:
    cases_per_100k = (data["cases_lag0"] / data["population"]) * 100000
    week = data["week"]
    week_of_year = int(str(week)[-2:])
    month = ((week_of_year - 1) // 4 + 1)

    month_sin = np.sin((2 * np.pi * month) / 12)
    month_cos = np.cos((2 * np.pi * month) / 12)
    week_angle = 2 * np.pi * week_of_year / 52
    week_sin = np.sin(week_angle)
    week_cos = np.cos(week_angle)

    df = pd.DataFrame([{
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
    return df


def predict(model_key: str, input_data: dict):
    config = models[model_key]
    df = prepare_input(input_data)
    df[scaler_vars] = config["feature_scaler"].transform(df[scaler_vars])
    df = df[feature_order]
    scaled_pred = config["model"].predict(df)
    pred = config["target_scaler"].inverse_transform(np.array(scaled_pred).reshape(-1, 1)).flatten()
    return round(pred[0])


@app.post("/bulk-predict")
async def bulk_predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    required_columns = {
        "week", "temp_avg", "humidity_avg", "vim", "cases_lag0",
        "cases_lag1", "precipitation_avg_ordinary_kriging_lag3",
        "precipitation_avg_ordinary_kriging_lag4", "population"
    }

    if not required_columns.issubset(df.columns):
        return {"error": "Missing one or more required columns."}

    results = []
    for _, row in df.iterrows():
        row_data = row.to_dict()
        try:
            week1_pred = predict("week1", row_data)
            week2_pred = predict("week2", row_data)
            results.append({
                **row_data,
                "prediction_week1": week1_pred,
                "prediction_week2": week2_pred
            })
        except Exception as e:
            results.append({**row_data, "error": str(e)})

    return results


@app.get("/test")
def test():
    return {"message": "API is working!"}


@app.post("/predict-week1")
def predict_week1(data: DengueInput):
    return {"prediction": predict("week1", data)}


@app.post("/predict-week2")
def predict_week2(data: DengueInput):
    return {"prediction": predict("week2", data)}
