"""
FastAPI application. All responses are JSON-serializable via utils.to_json_serializable
Run: uvicorn src.api:app --reload
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd

from src.config import settings
from src.utils import to_json_serializable
import src.stats_viz as stats_viz
import src.stats_inf as stats_inf
from models.registry import get_registry, load_model
from src.data_utils import load_clean_data

app = FastAPI(title="Graduation Prediction API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Participant(BaseModel):
    total_score: float
    hours_per_week: float
    years_experience: float
    skill_level: float
    track_name: str
    country_name: str
    gender: str
    age_range: str
    heard_about: str

@app.get("/")
def root():
    return {"message": "Graduation Prediction API is running. See /docs"}

@app.get("/models")
def list_models():
    return {"available_models": list(get_registry().keys())}

@app.post("/models/train")
def train_models(retrain: bool = Query(True)):
    df = load_clean_data()
    if df.empty:
        raise HTTPException(status_code=400, detail="No data to train on - run ETL first.")
    # import models module here to avoid top-level heavy imports
    from models.stats_models import train_all
    registry, metrics = train_all(df, overwrite=retrain)
    return to_json_serializable({"registry": registry, "metrics": metrics})

@app.post("/predict/{model_name}")
def predict_single(model_name: str, record: Participant):
    reg = get_registry()
    if model_name not in reg:
        raise HTTPException(status_code=404, detail="Model not found")
    model = load_model(model_name)
    X = pd.DataFrame([record.dict()])
    try:
        pred = bool(int(model.predict(X)[0]))
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception:
                prob = None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return to_json_serializable({"model": model_name, "prediction": pred, "probability": prob})

@app.post("/predict-batch/{model_name}")
def predict_batch(model_name: str, records: List[Participant]):
    reg = get_registry()
    if model_name not in reg:
        raise HTTPException(status_code=404, detail="Model not found")
    model = load_model(model_name)
    X = pd.DataFrame([r.dict() for r in records])
    try:
        preds = [bool(int(x)) for x in model.predict(X).tolist()]
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)[:, 1].tolist()
            except Exception:
                probs = [None] * len(preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return to_json_serializable({"model": model_name, "predictions": preds, "probabilities": probs})

@app.get("/sample-data")
def sample_data(limit: int = 20):
    df = load_clean_data(limit=limit)
    if df.empty:
        return []
    return to_json_serializable(df.head(limit))

@app.get("/eda/summary")
def eda_summary():
    df = load_clean_data()
    if df.empty:
        return {"error": "no data"}
    return stats_inf.get_summary(df)

@app.get("/eda/null_counts")
def eda_null_counts():
    df = load_clean_data()
    if df.empty:
        return {"error": "no data"}
    return stats_inf.get_null_counts(df)

@app.get("/viz/{viz_type}")
def viz_data(viz_type: str, bins: int = 30, sample: int = 1000, model: str | None = None):
    df = load_clean_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="no data")
    if viz_type == "distribution":
        return stats_viz.get_score_distribution_data(df, bins=bins)
    if viz_type == "graduation_by_track":
        return stats_viz.get_graduation_by_track_data(df)
    if viz_type == "graduation_by_country":
        return stats_viz.get_graduation_by_country_data(df)
    if viz_type == "graduation_by_gender":
        return stats_viz.get_graduation_by_gender_data(df)
    if viz_type == "score_by_track":
        return stats_viz.get_score_by_track_data(df)
    if viz_type == "experience_vs_score":
        return stats_viz.get_experience_vs_score_data(df, sample=sample)
    if viz_type == "correlations":
        return stats_viz.get_correlation_data(df)
    if viz_type == "feature_importance":
        if not model:
            raise HTTPException(status_code=400, detail="model parameter required")
        mdl = load_model(model)
        feat_names = []
        try:
            # attempt to infer feature names
            from models.stats_models import _extract_feature_names
            feat_names = _extract_feature_names(mdl.named_steps["pre"], df.head(20))
        except Exception:
            feat_names = []
        return stats_viz.get_feature_importance_data(mdl, feat_names)
    raise HTTPException(status_code=400, detail="Unknown visualization type")
