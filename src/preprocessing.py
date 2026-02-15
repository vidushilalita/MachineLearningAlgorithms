"""
preprocessing.py – Preprocess raw data for model training / prediction.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def preprocess_dataframe(df: pd.DataFrame, scaler=None, fit_scaler=False):
    """
    Preprocess raw survey dataframe.
    Returns (X, y_or_None, scaler).
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Encode GENDER
    if df["GENDER"].dtype == object:
        df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})

    # Extract target if present
    y = None
    if "LUNG_CANCER" in df.columns:
        if df["LUNG_CANCER"].dtype == object:
            df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
        y = df.pop("LUNG_CANCER")

    # Binary (1,2) → (0,1)
    for col in [c for c in df.columns if c not in ("GENDER", "AGE")]:
        if set(df[col].unique()).issubset({1, 2}):
            df[col] = df[col].map({1: 0, 2: 1})

    # Scale AGE
    if scaler is None and not fit_scaler:
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    if fit_scaler:
        scaler = StandardScaler()
        df["AGE"] = scaler.fit_transform(df[["AGE"]])
    else:
        df["AGE"] = scaler.transform(df[["AGE"]])

    return df.astype(float), y, scaler
    df = pd.DataFrame([input_dict])
    df_processed, _, scaler = preprocess_dataframe(df, scaler=scaler, fit_scaler=False)
    return df_processed
