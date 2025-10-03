from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import uvicorn


# Input schema
class WildfireInput(BaseModel):
    temperature_c: float
    relative_humidity: float
    wind_speed: float
    fuel_moisture: float
    drought_code: float
    fine_fuel_moisture: float
    fwi: float


app = FastAPI(title="Wildfire Prediction API")
xgb_model = None


# ===== Synthetic dataset generator (replace later with real dataset) =====
def generate_data(n_samples=10000):
    np.random.seed(42)
    df = pd.DataFrame({
        "temperature_c": np.random.uniform(10, 50, n_samples),
        "relative_humidity": np.random.uniform(5, 80, n_samples),
        "wind_speed": np.random.uniform(0, 20, n_samples),
        "fuel_moisture": np.random.uniform(2, 40, n_samples),
        "drought_code": np.random.uniform(100, 800, n_samples),
        "fine_fuel_moisture": np.random.uniform(2, 30, n_samples),
        "fwi": np.random.uniform(0, 40, n_samples)
    })

    fire_prob = (
        0.03 * (df["temperature_c"] - 20) +
        0.04 * (40 - df["relative_humidity"]) +
        0.05 * df["wind_speed"] +
        0.06 * (40 - df["fuel_moisture"]) +
        0.07 * (df["drought_code"] / 800) +
        0.05 * (30 - df["fine_fuel_moisture"]) +
        0.08 * df["fwi"] / 40
    )
    fire_prob = 1 / (1 + np.exp(-fire_prob))
    labels = (fire_prob > 0.5).astype(int)
    return df, labels


# ===== Train XGBoost model =====
def train_model():
    global xgb_model
    X, y = generate_data()
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=200)


# ===== Prediction endpoint =====
@app.post("/predict")
def predict_fire(data: WildfireInput):
    global xgb_model
    if xgb_model is None:
        train_model()
    X_input = pd.DataFrame([data.dict()])
    dmatrix = xgb.DMatrix(X_input)
    prob = float(xgb_model.predict(dmatrix)[0])
    risk = "HIGH" if prob > 0.6 else "MODERATE" if prob > 0.3 else "LOW"
    return {"wildfire_probability": round(prob, 4), "risk_level": risk}


if __name__ == "__main__":
    train_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
