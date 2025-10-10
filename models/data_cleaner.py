"""
models/data_cleaner.py
Centralized dataframe cleaning utilities for the pipeline.
"""
import pandas as pd
import numpy as np
from typing import Any

def to_bool_safe(v: Any) -> bool | None:
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not np.isnan(v):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "t", "1", "completed", "complete"}:
        return True
    if s in {"no", "n", "false", "f", "0", "not completed", "incomplete"}:
        return False
    return None

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert to numeric, coerce errors to NaN."""
    return pd.to_numeric(series, errors="coerce")

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """General sanitization: strip strings, replace infinite with NaN."""
    df = df.copy()
    # strip string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": None})
    # replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
