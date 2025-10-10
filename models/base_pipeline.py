"""
base_pipeline.py
Defines a reusable preprocessing transformer for all models.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    return preprocessor
