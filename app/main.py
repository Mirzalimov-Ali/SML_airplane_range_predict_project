import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path("pipeline/full_pipeline.joblib")

app = FastAPI(
    title="Airplane Range in KM Prediction API",
    version="1.0"
)

pipeline = None


# --------------------------------------------------
# Load model ON STARTUP
# --------------------------------------------------

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")


# --------------------------------------------------
# Schemas
# --------------------------------------------------

class AirplaneInput(BaseModel):
    Model: str
    Year_of_Manufacture: int
    Number_of_Engines: int
    Engine_Type: str
    Capacity: int
    Fuel_Consumption_L_per_hour: float
    Hourly_Maintenance_Cost_USD: float
    Age: int
    Sales_Region: str
    Price_USD: float


class PredictionOutput(BaseModel):
    prediction: float
    range_min: float
    range_max: float


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Predict
# --------------------------------------------------

@app.post("/predict", response_model=PredictionOutput)
def predict(data: AirplaneInput):
    if pipeline is None:
        raise RuntimeError("Model not loaded")

    df = pd.DataFrame([data.model_dump()])

    df = df.rename(columns={
        "Fuel_Consumption_L_per_hour": "Fuel_Consumption_(L/hour)",
        "Hourly_Maintenance_Cost_USD": "Hourly_Maintenance_Cost_($)",
        "Price_USD": "Price_($)"
    })

    # ❌ УБРАТЬ ЭТО
    # expected_cols = pipeline.feature_names_in_
    # df = df.reindex(columns=expected_cols)

    pred = float(pipeline.predict(df)[0])

    delta = abs(pred) * 0.1

    return PredictionOutput(
        prediction=round(pred, 2),
        range_min=round(pred - delta, 2),
        range_max=round(pred + delta, 2)
    )
