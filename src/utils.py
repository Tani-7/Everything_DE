"""
src/utils.py
Shared helpers for JSON-safe conversions and env loading.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Any
from dotenv import load_dotenv
import os

load_dotenv()

def clean_scalar(v: Any):
    if pd.isna(v) or v is None or v in [np.inf, -np.inf]:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return v.isoformat()
    return v

def to_json_serializable(obj: Any):
    if isinstance(obj, pd.DataFrame):
        return obj.where(pd.notnull(obj), None).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.where(pd.notnull(obj), None).to_dict()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return [to_json_serializable(x) for x in list(obj)]
    return clean_scalar(obj)
