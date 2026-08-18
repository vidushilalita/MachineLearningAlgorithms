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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

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

FEATURE_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]

COLUMN_ALIASES = {
    "CHRONIC_DISEASE": "CHRONIC DISEASE",
    "ALCOHOL_CONSUMING": "ALCOHOL CONSUMING",
    "SHORTNESS_OF_BREATH": "SHORTNESS OF BREATH",
    "SWALLOWING_DIFFICULTY": "SWALLOWING DIFFICULTY",
    "CHEST_PAIN": "CHEST PAIN",
}

BINARY_FEATURES = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]


def normalize_column_names(columns):
    renamed = {}
    for col in columns:
        original = str(col).strip()
        key = original.upper().replace(" ", "_")
        renamed[original] = COLUMN_ALIASES.get(key, original)
    return renamed


@st.cache_resource
def load_model(name):
    return joblib.load(MODELS_DIR / MODEL_FILES[name])


@st.cache_resource
def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")


def get_available_models():
    return [name for name, file_name in MODEL_FILES.items() if (MODELS_DIR / file_name).exists()]


def normalize_gender(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned in {"M", "MALE"}:
            return 1
        if cleaned in {"F", "FEMALE"}:
            return 0
    return int(value)


def normalize_binary_feature(series):
    if series.dtype == object:
        series = series.astype(str).str.strip().str.upper()
        mapping = {"YES": 1, "Y": 1, "1": 1, "TRUE": 1, "NO": 0, "N": 0, "0": 0, "FALSE": 0, "2": 1}
        return series.map(mapping).astype(float)

    series = pd.to_numeric(series, errors="coerce")
    if set(series.dropna().unique()).issubset({1, 2}):
        return series.map({1: 0, 2: 1}).astype(float)
    return series.astype(float)


def preprocess(df: pd.DataFrame):
    """Return (X, y) for model input."""
    df = df.copy()
    df = df.rename(columns=normalize_column_names(df.columns))
    df.columns = [str(col).strip() for col in df.columns]

    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            raise ValueError(f"Missing required column: {feature}")

    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].apply(normalize_gender).astype(float)

    y = None
    if "LUNG_CANCER" in df.columns:
        if df["LUNG_CANCER"].dtype == object:
            df["LUNG_CANCER"] = (
                df["LUNG_CANCER"].astype(str).str.strip().str.upper().map({"YES": 1, "Y": 1, "1": 1, "NO": 0, "N": 0, "0": 0})
            )
        df["LUNG_CANCER"] = pd.to_numeric(df["LUNG_CANCER"], errors="coerce").fillna(1)
        y = df["LUNG_CANCER"].astype(int).to_numpy()
        df = df.drop(columns=["LUNG_CANCER"])

    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = normalize_binary_feature(df[col])

    if "AGE" in df.columns:
        scaler = load_scaler()
        df["AGE"] = scaler.transform(df[["AGE"]]).ravel()

    X = df[FEATURE_COLUMNS].copy()
    X = X.astype(float)
    return X, y


def format_prediction_label(value):
    return "YES" if int(value) == 1 else "NO"


def render_styles():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f5fbf7 0%, #eef7ff 100%); }
        header[data-testid="stHeader"] { background-color: rgba(20, 97, 67, 0.9); }
        [data-testid="stSidebar"] { background-color: rgba(227, 242, 233, 0.95); }
        h1, h2, h3, h4 { color: #155f46; }
        .stMetric label { color: #155f46 !important; }
        .stMetric [data-testid="stMetricValue"] { color: #0d5e8a !important; }
        div.stButton > button[kind="primary"] {
            background-color: #197a56; color: white; border: none; border-radius: 0.5rem;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #0f5b42; color: white;
        }
        .prediction-card {
            background: linear-gradient(135deg, #e8f7ee 0%, #f4fbff 100%);
            padding: 1.2rem; border-radius: 1rem; border: 1px solid #dfeee6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_predict_proba(model, X):
    """Return probability estimates, with a fallback for compatibility issues."""
    try:
        return model.predict_proba(X)
    except AttributeError:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            if scores.ndim == 1:
                probs = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - probs, probs])
            if scores.shape[1] == 2:
                exp_scores = np.exp(scores)
                return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        raise


def safe_predict(model, X):
    try:
        return model.predict(X)
    except Exception:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return (scores >= 0).astype(int)
        raise


def render_single_prediction(model_name):
    st.subheader("🩺 Single Patient Prediction")
    st.caption("Enter patient details to get an instant lung cancer risk estimate.")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            age = st.slider("Age", 18, 90, 45)
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
            anxiety = st.selectbox("Anxiety", ["No", "Yes"])
            peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
            chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        with col2:
            fatigue = st.selectbox("Fatigue", ["No", "Yes"])
            allergy = st.selectbox("Allergy", ["No", "Yes"])
            wheezing = st.selectbox("Wheezing", ["No", "Yes"])
            alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
            coughing = st.selectbox("Coughing", ["No", "Yes"])
            shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
            swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
            chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

        submit = st.form_submit_button("Predict Risk", type="primary")

    if submit:
        data = {
            "GENDER": 1 if gender == "Male" else 0,
            "AGE": age,
            "SMOKING": 1 if smoking == "Yes" else 0,
            "YELLOW_FINGERS": 1 if yellow_fingers == "Yes" else 0,
            "ANXIETY": 1 if anxiety == "Yes" else 0,
            "PEER_PRESSURE": 1 if peer_pressure == "Yes" else 0,
            "CHRONIC_DISEASE": 1 if chronic_disease == "Yes" else 0,
            "FATIGUE": 1 if fatigue == "Yes" else 0,
            "ALLERGY": 1 if allergy == "Yes" else 0,
            "WHEEZING": 1 if wheezing == "Yes" else 0,
            "ALCOHOL_CONSUMING": 1 if alcohol == "Yes" else 0,
            "COUGHING": 1 if coughing == "Yes" else 0,
            "SHORTNESS_OF_BREATH": 1 if shortness_of_breath == "Yes" else 0,
            "SWALLOWING_DIFFICULTY": 1 if swallowing_difficulty == "Yes" else 0,
            "CHEST_PAIN": 1 if chest_pain == "Yes" else 0,
        }

        df = pd.DataFrame([data])
        X, _ = preprocess(df)
        model = load_model(model_name)
        prediction = safe_predict(model, X)[0]
        probability = float(safe_predict_proba(model, X)[0, 1])

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        risk_color = "#d32f2f" if prediction == 1 else "#2e7d32"
        st.markdown(
            f"<h3 style='color:{risk_color}; margin:0;'>Risk: {risk_level}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p><strong>Prediction:</strong> {format_prediction_label(prediction)}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Probability of Lung Cancer:</strong> {probability:.2%}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if prediction == 1:
            st.warning("This patient is predicted to be at risk of lung cancer. Please consult a medical professional for follow-up.")
        else:
            st.success("This patient is predicted to have a lower risk of lung cancer based on the selected model.")


def render_dataset_evaluation(model_name):
    st.subheader("📊 Dataset Evaluation")
    st.caption("Upload a CSV file or use the sample dataset to evaluate model performance.")

    if SAMPLE_DATA_PATH.exists():
        sample_df = pd.read_csv(SAMPLE_DATA_PATH)
        sample_df.columns = [str(col).strip() for col in sample_df.columns]
        sample_csv = sample_df.to_csv(index=False).encode("utf-8")

        col_a, col_b = st.columns([1, 1])
        if col_a.button("Use Sample Dataset", type="primary"):
            st.session_state["use_sample_data"] = True
        col_b.download_button(
            "Download Sample CSV",
            sample_csv,
            "sample_lung_cancer_dataset.csv",
            "text/csv",
        )

    uploaded = st.file_uploader("Upload your own CSV", type=["csv"])

    if uploaded is not None:
        st.session_state["use_sample_data"] = False
        raw_df = pd.read_csv(uploaded)
        raw_df.columns = [str(col).strip() for col in raw_df.columns]
    elif st.session_state.get("use_sample_data") and SAMPLE_DATA_PATH.exists():
        raw_df = pd.read_csv(SAMPLE_DATA_PATH)
        raw_df.columns = [str(col).strip() for col in raw_df.columns]
        st.success("Using the sample dataset.")
    else:
        raw_df = None

    if raw_df is not None:
        st.markdown("##### Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        if st.button("Evaluate Model", type="primary"):
            try:
                X, y_true = preprocess(raw_df)
            except ValueError as exc:
                st.error(f"Unable to process the CSV file: {exc}")
                return

            if y_true is None:
                st.info("No target column detected. Showing prediction-only output.")
                model = load_model(model_name)
                y_pred = safe_predict(model, X)
                y_prob = safe_predict_proba(model, X)[:, 1]

                result_df = raw_df.copy()
                result_df["Prediction"] = [format_prediction_label(v) for v in y_pred]
                result_df["Probability"] = np.round(y_prob, 4)
                st.dataframe(result_df, use_container_width=True)
                return

            model = load_model(model_name)
            y_pred = safe_predict(model, X)
            y_prob = safe_predict_proba(model, X)[:, 1]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float("nan")

            st.markdown("---")
            st.subheader("Model Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Precision", f"{prec:.4f}")
            m3.metric("Recall", f"{rec:.4f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("F1 Score", f"{f1:.4f}")
            m5.metric("AUC-ROC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
            m6.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="GnBu",
                xticklabels=["NO", "YES"],
                yticklabels=["NO", "YES"],
                ax=ax,
                linewidths=1,
                linecolor="white",
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {model_name}")
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Classification Report")
            report = classification_report(
                y_true,
                y_pred,
                target_names=["NO (0)", "YES (1)"],
                zero_division=0,
            )
            st.code(report, language="text")

            st.subheader("Predictions")
            result_df = raw_df.copy()
            result_df["Prediction"] = [format_prediction_label(v) for v in y_pred]
            result_df["Probability"] = np.round(y_prob, 4)
            st.dataframe(result_df, use_container_width=True)

    else:
        st.info("Upload a CSV file or use the sample dataset to begin evaluation.")


def main():
    render_styles()
    st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="wide")

    st.title("🫁 Lung Cancer Prediction Dashboard")
    st.caption("Explore model predictions and evaluate performance on custom datasets.")

    available_models = get_available_models()
    if not available_models:
        st.error("No trained model files were found in the `models/` folder. Please train the models before running the app.")
        st.stop()

    st.sidebar.title("App Controls")
    model_name = st.sidebar.selectbox("Choose a model", available_models)
    st.sidebar.markdown("---")
    st.sidebar.caption("This project compares multiple ML models for binary lung cancer prediction.")

    tab1, tab2 = st.tabs(["Single Patient", "Dataset Evaluation"])
    with tab1:
        render_single_prediction(model_name)
    with tab2:
        render_dataset_evaluation(model_name)


if __name__ == "__main__":
    main()
