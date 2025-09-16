import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def _build_preprocessor(num_features, cat_features):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])


def build_pipeline(model, num_features, cat_features):
    preprocessor = _build_preprocessor(num_features, cat_features)
    return Pipeline([("pre", preprocessor), ("clf", model)])


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }


def run_logistic_regression(df, num_features=None, cat_features=None, test_size=0.2,
                            random_state=42, load_saved=True, saved_model_path=MODEL_DIR / "logreg.pkl"):
    if num_features is None:
        num_features = ["total_score", "hours_per_week", "years_experience", "skill_level"]
    if cat_features is None:
        cat_features = ["track_name", "country_name", "gender", "age_range", "heard_about"]

    features = [f for f in (num_features + cat_features) if f in df.columns]
    X = df[features].copy()
    y = df["graduation_status"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    if load_saved and saved_model_path.exists():
        try:
            model = joblib.load(saved_model_path)
            logging.info("Loaded saved logistic model from %s", saved_model_path)
            return X, y, model, X_train, X_test, y_train, y_test
        except Exception as e:
            logging.warning("Failed to load saved model (%s): %s — training a fresh one", saved_model_path, e)

    pipe = build_pipeline(LogisticRegression(max_iter=500, solver="liblinear"),
                          num_features=[c for c in num_features if c in df.columns],
                          cat_features=[c for c in cat_features if c in df.columns])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, saved_model_path)
    logging.info("Saved logistic model to %s", saved_model_path)
    return X, y, pipe, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    return evaluate(model, X_test, y_test)
