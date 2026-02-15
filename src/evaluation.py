"""
evaluation.py – Compute metrics for trained models.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def load_all_models():
    """Load models whose .pkl files exist."""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)
    return models


def compute_all_metrics(models: dict, X_test, y_test) -> pd.DataFrame:
    """Return a comparison DataFrame with all metrics."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "MCC": matthews_corrcoef(y_test, y_pred),
        })
    return pd.DataFrame(rows).set_index("Model").round(4)
