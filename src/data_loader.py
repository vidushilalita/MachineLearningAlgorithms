"""
data_loader.py – Load raw and processed data.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "survey lung cancer.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_processed_splits():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").astype(float)
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv").astype(float)
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel().astype(int)
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel().astype(int)
    return X_train, X_test, y_train, y_test
