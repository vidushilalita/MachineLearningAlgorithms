"""
model_training.py – Train all models with GridSearchCV and save as .pkl files.
Run as script: python -m src.model_training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import joblib
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def get_model_configs(spw: float = 1.0):
    """Return {name: (estimator, param_grid)} for all models."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"],
             "solver": ["liblinear"], "class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10],
             "criterion": ["gini", "entropy"], "class_weight": [None, "balanced"]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        ),
        "naive_bayes": (
            GaussianNB(),
            {"var_smoothing": np.logspace(-12, -1, 30)},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None],
             "class_weight": [None, "balanced"]},
        ),
        "xgboost": (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.1], "scale_pos_weight": [1, spw]},
        ),
    }


def train_and_save_all(X_train, y_train, X_test, y_test):
    """Train all models, print results, save .pkl files."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    configs = get_model_configs(spw)

    for name, (estimator, param_grid) in configs.items():
        print(f"Training {name} ...")
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}  "
              f"F1: {f1_score(y_test, y_pred):.4f}")
        joblib.dump(best, MODELS_DIR / f"{name}.pkl")

    print("Done — models saved to", MODELS_DIR)


if __name__ == "__main__":
    from data_loader import load_processed_splits
    X_train, X_test, y_train, y_test = load_processed_splits()
    train_and_save_all(X_train, y_train, X_test, y_test)
