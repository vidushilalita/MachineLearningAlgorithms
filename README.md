# Lung Cancer Prediction Using Machine Learning

This project predicts whether a patient is likely to have lung cancer using a survey-based dataset and several machine learning models.

## Overview

- Binary classification problem: YES / NO
- Models included: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost
- Built with Python, scikit-learn, Streamlit, Pandas, NumPy, Matplotlib, and Seaborn
- Includes a user-friendly dashboard with:
  - single-patient prediction form
  - CSV upload evaluation
  - model metrics and confusion matrix

## Project structure

- `app.py` — main Streamlit entry point for deployment
- `src/streamlit_app.py` — dashboard logic and model interactions
- `models/` — trained model files and scaler
- `data/` — raw and processed datasets
- `results/` — saved metrics
- `notebooks/` — training and analysis notebooks

## Model performance summary

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8839 | 0.8929 | 0.9126 | 0.9592 | 0.9353 | 0.3848 |
| Decision Tree | 0.9286 | 0.9767 | 0.9412 | 0.9796 | 0.9600 | 0.6391 |
| kNN | 0.9821 | 0.9271 | 0.9800 | 1.0000 | 0.9899 | 0.9165 |
| Naive Bayes | 0.9196 | 0.9111 | 0.9406 | 0.9694 | 0.9548 | 0.6010 |
| Random Forest | 0.9732 | 0.9537 | 0.9703 | 1.0000 | 0.9849 | 0.8731 |
| XGBoost | 0.9732 | 0.9789 | 0.9798 | 0.9898 | 0.9848 | 0.8745 |

## Best model

The kNN model achieved the strongest overall accuracy and F1 score in this project, while XGBoost had the highest AUC score.

## Notes

- The app can load a sample CSV or accept a custom dataset for evaluation.
- The single-patient form is designed to make the app easier for non-technical users.
- For healthcare use, always validate predictions with domain experts and appropriate clinical guidance.
