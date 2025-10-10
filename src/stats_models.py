# src/stats_models.py
import logging
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix
)

# safer import if run as a script
try:
    from src.config import settings
except ModuleNotFoundError:
    from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = Path(settings.model_dir)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------
# Preprocessing setup
# ----------------------------
def _build_preprocessor(num_features, cat_features):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    transformers = []
    if num_features:
        transformers.append(("num", num_transformer, num_features))
    if cat_features:
        transformers.append(("cat", cat_transformer, cat_features))
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(model, num_features, cat_features):
    preprocessor = _build_preprocessor(num_features, cat_features)
    return Pipeline([("pre", preprocessor), ("clf", model)])


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "auc": float(roc_auc_score(y_test, probs)) if probs is not None else None,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_test, probs)) if probs is not None else None,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }


def _extract_feature_names_from_pipeline(pipeline, X_sample: pd.DataFrame):
    pre = pipeline.named_steps["pre"]
    feature_names = []
    for name, transformer, cols in pre.transformers_:
        if name == "remainder":
            continue
        if transformer is None:
            feature_names.extend(cols)
            continue
        last = transformer
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
        try:
            names = last.get_feature_names_out(cols)
            feature_names.extend(list(names))
        except Exception:
            feature_names.extend(list(cols))
    return feature_names


# ----------------------------
# Training
# ----------------------------
def _train_single(df, model, model_name, num_features, cat_features, overwrite=False):
    available_num = [c for c in num_features if c in df.columns]
    available_cat = [c for c in cat_features if c in df.columns]
    features = available_num + available_cat

    if not features:
        raise ValueError("No valid features found in DataFrame!")

    X = df[features].copy()
    y = df["graduation_status"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    out_path = MODEL_DIR / f"{model_name}.pkl"
    if out_path.exists() and not overwrite:
        logging.info("Loading existing model %s (overwrite=False)", out_path)
        pipe = joblib.load(out_path)
        return pipe, None, None, None, None

    pipe = build_pipeline(model, num_features=available_num, cat_features=available_cat)
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, out_path)
    logging.info("Saved model %s to %s", model_name, out_path)

    feat_names = _extract_feature_names_from_pipeline(pipe, X_train)
    metrics = evaluate(pipe, X_test, y_test)
    return pipe, feat_names, X_test, y_test, metrics


def train_all_models(df: pd.DataFrame, overwrite=False):
    num_features = ["total_score", "hours_per_week", "years_experience", "skill_level"]
    cat_features = ["track_name", "country_name", "gender", "age_range", "heard_about"]

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42)
    }

    registry, metrics = {}, {}

    for name, clf in models.items():
        pipe, feat_names, X_test, y_test, m = _train_single(
            df=df,
            model=clf,
            model_name=name,
            num_features=num_features,
            cat_features=cat_features,
            overwrite=overwrite
        )
        registry[name] = f"{name}.pkl"
        metrics[name] = {"metrics": m, "feature_names": feat_names}

    reg_path = MODEL_DIR / "registry.json"
    with open(reg_path, "w") as fh:
        json.dump(registry, fh, indent=2)

    logging.info("Registry written to %s", reg_path)
    return registry, metrics
