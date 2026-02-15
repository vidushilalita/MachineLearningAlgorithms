"""
predict.py – Load a trained model and make predictions.
"""

import pandas as pd
import joblib
from pathlib import Path
from .preprocessing import preprocess_dataframe

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def load_model(name: str):
    return joblib.load(MODELS_DIR / MODEL_FILES[name])


def predict_single(input_dict: dict, model_name: str = "Random Forest"):
    """Predict for a single patient (dict of features). Returns (label, probability)."""
    df = pd.DataFrame([input_dict])
    X, _, _ = preprocess_dataframe(df)
    model = load_model(model_name)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]
    return ("YES" if pred == 1 else "NO"), float(prob)


def predict_batch(df: pd.DataFrame, model_name: str = "Random Forest"):
    """Predict for a batch. Returns df with Prediction & Probability columns."""
    X, _, _ = preprocess_dataframe(df)
    model = load_model(model_name)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    result = df.copy()
    result["Prediction"] = ["YES" if p == 1 else "NO" for p in preds]
    result["Probability"] = probs.round(4)
    return result
