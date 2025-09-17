import logging
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
import joblib
from fastapi import FastAPI, Query, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.data_utils import get_engine, load_participants
from src.stats_models import run_logistic_regression
from src.stats_inf import run_hypothesis_tests

logger = logging.getLogger(__name__)
app = FastAPI(title="Graduation Insights API")


# Paths for saved models

MODEL_RF = Path("models") / "rf.pkl"
MODEL_LR = Path("models") / "logreg.pkl"


# Pydantic Models

class PredictRequest(BaseModel):
    total_score: Optional[float] = Field(None, example=70.0)
    hours_per_week: Optional[float] = Field(None, example=10.0)
    years_experience: Optional[float] = Field(None, example=1.0)
    skill_level: Optional[float] = Field(None, example=5.0)
    track_name: Optional[str] = Field(None, example="Data analysis")
    country_name: Optional[str] = Field(None, example="Kenya")
    gender: Optional[str] = Field(None, example="Male")
    age_range: Optional[str] = Field(None, example="25-34 years")
    heard_about: Optional[str] = Field(None, example="WhatsApp")

class PredictResponse(BaseModel):
    model: str
    probability: float
    details: Optional[Dict[str, Any]] = None

# Helper Methods

def _load_best_model():
    if MODEL_RF.exists():
        return "rf", joblib.load(MODEL_RF)
    if MODEL_LR.exists():
        return "logreg", joblib.load(MODEL_LR)
    return None, None

def _df_to_jsonable(df: pd.DataFrame):
    return jsonable_encoder(df.to_dict(orient="records"))

# Defining Endpoints

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/participants")
def participants(limit: int = Query(100, ge=1, le=1000),
                 offset: int = Query(0, ge=0)):
    try:
        df = load_participants()
    except Exception as e:
        logger.error("Failed loading participants: %s", e)
        raise HTTPException(status_code=500, detail="DB query failed")

    df = df.iloc[offset: offset+limit]
    return {"rows": _df_to_jsonable(df), "count": len(df)}

@app.get("/stats/graduation-by-track")
def graduation_by_track():
    df = load_participants()
    grouped = (df.groupby("track_name")
                 .agg(total=("id", "count"),
                      graduates=("graduation_status", "sum"))
                 .reset_index())
    grouped["graduation_rate"] = (grouped["graduates"] / grouped["total"]).round(4)
    return {"data": _df_to_jsonable(grouped)}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model_name, model = _load_best_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model found.")

    row = {k: v for k, v in payload.dict().items()}
    df = pd.DataFrame([row])

    try:
        probs = model.predict_proba(df)[:, 1]
        prob = float(probs[0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")
@app.get("/")
def root():
    return {"message": "Graduation Insights API is running. All systems GO"}


    return PredictResponse(model=model_name, probability=prob, details={"input": row})
