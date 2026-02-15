"""
Streamlit App – Lung Cancer Prediction
=======================================
Run with:  streamlit run src/streamlit_app.py

Features
--------
a. CSV upload (test data with LUNG_CANCER column)
b. Model selection dropdown
c. Evaluation metrics display
d. Confusion matrix & classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLE_DATA_PATH = ROOT / "data" / "raw" / "test_streamlit_app.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# ── Helpers ──────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return joblib.load(MODELS_DIR / MODEL_FILES[name])


@st.cache_resource
def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")


def get_available_models():
    return [n for n, f in MODEL_FILES.items() if (MODELS_DIR / f).exists()]


def preprocess(df: pd.DataFrame):
    """Preprocess raw CSV data. Returns (X, y) where y is None if
    LUNG_CANCER column is absent."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Encode GENDER
    if "GENDER" in df.columns and df["GENDER"].dtype == object:
        df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})

    # Extract target if present
    y = None
    if "LUNG_CANCER" in df.columns:
        if df["LUNG_CANCER"].dtype == object:
            df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
        y = df.pop("LUNG_CANCER").values.astype(int)

    # Map binary features (1,2) → (0,1)
    binary_cols = [c for c in df.columns if c not in ("GENDER", "AGE")]
    for col in binary_cols:
        if set(df[col].unique()).issubset({1, 2}):
            df[col] = df[col].map({1: 0, 2: 1})

    # Scale AGE
    scaler = load_scaler()
    df["AGE"] = scaler.transform(df[["AGE"]])
    return df.astype(float), y


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁")

st.markdown(
    """
    <style>
    .stApp { background-color: #f0f8f0; }
    header[data-testid="stHeader"] { background-color: #1a6b3c; }
    [data-testid="stSidebar"] { background-color: #e8f4f8; }
    h1, h2, h3 { color: #1a6b3c; }
    .stMetric label { color: #1a6b3c !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0d5e8a !important; }
    div.stButton > button[kind="primary"] {
        background-color: #1a6b3c; color: white; border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0d5e8a; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Lung Cancer Prediction")

available = get_available_models()
if not available:
    st.error("No trained models found in models/. Run the training notebooks first.")
    st.stop()

# ── (b) Model selection dropdown ─────────────────────────────────
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Select Model", available)
st.sidebar.markdown("---")
st.sidebar.caption("ML Assignment – Lung Cancer Survey")

# ── (a) Dataset upload (CSV – test data only) ────────────────────
st.subheader(" Upload Test Data (CSV)")
st.caption("Upload a CSV file containing test samples. "
           "Include the **LUNG_CANCER** column (YES/NO) to see evaluation metrics.")

# Sample data section
if SAMPLE_DATA_PATH.exists():
    sample_df = pd.read_csv(SAMPLE_DATA_PATH)
    sample_df.columns = [c.strip() for c in sample_df.columns]
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")

    st.markdown("##### 📎 Sample Test Data")
    col_a, col_b = st.columns(2)
    use_sample = col_a.button("▶ Use Sample Data", type="primary")
    col_b.download_button("⬇ Download Sample CSV", sample_csv,
                          "test_streamlit_app.csv", "text/csv")
else:
    use_sample = False

uploaded = st.file_uploader("Or upload your own CSV", type=["csv"])

# Decide which data to use
if use_sample and SAMPLE_DATA_PATH.exists():
    raw_df = sample_df.copy()
    st.success("Using sample test data (500 rows)")
elif uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    raw_df.columns = [c.strip() for c in raw_df.columns]
else:
    raw_df = None

if raw_df is not None:

    st.markdown("##### Uploaded Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    has_target = "LUNG_CANCER" in raw_df.columns

    if st.button("Evaluate", type="primary"):
        X, y_true = preprocess(raw_df)
        model = load_model(model_name)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # ── (c) Display evaluation metrics ───────────────────────
        if has_target and y_true is not None:
            st.markdown("---")
            st.subheader("Evaluation Metrics")

            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            mcc  = matthews_corrcoef(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float("nan")

            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("F1 Score", f"{f1:.4f}")
            m5.metric("AUC-ROC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
            m6.metric("MCC", f"{mcc:.4f}")

            # ── (d) Confusion matrix ─────────────────────────────
            st.markdown("---")
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu",
                        xticklabels=["NO", "YES"],
                        yticklabels=["NO", "YES"], ax=ax,
                        linewidths=1, linecolor="white")
            ax.set_xlabel("Predicted", fontsize=12, color="#0d5e8a")
            ax.set_ylabel("Actual", fontsize=12, color="#0d5e8a")
            ax.set_title(f"Confusion Matrix – {model_name}",
                         fontsize=13, color="#1a6b3c", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

            # ── (d) Classification report ────────────────────────
            st.subheader("Classification Report")
            report = classification_report(
                y_true, y_pred,
                target_names=["NO (0)", "YES (1)"],
                zero_division=0,
            )
            st.code(report, language="text")

        else:
            # No target column → just show predictions
            st.markdown("---")
            st.info("No **LUNG_CANCER** column found — showing predictions only.")

        # Always show prediction results
        st.subheader("Predictions")
        result_df = raw_df.copy()
        result_df["Prediction"] = ["YES" if p == 1 else "NO" for p in y_pred]
        result_df["Probability"] = y_prob.round(4)
        st.dataframe(result_df, use_container_width=True)

        pred_counts = pd.Series(y_pred).value_counts()
        c1, c2 = st.columns(2)
        c1.metric("Predicted YES", int(pred_counts.get(1, 0)))
        c2.metric("Predicted NO", int(pred_counts.get(0, 0)))

else:
    st.info("Upload a CSV file or click **Use Sample Data** to get started.")
