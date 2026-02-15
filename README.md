# Lung Cancer Prediction Using Machine Learning

## a. Problem Statement

Predict whether a patient is at risk of lung cancer based on survey responses covering lifestyle habits and symptoms. The task is a binary classification problem (YES / NO) using 6 different machine learning models.

## b. Dataset Description

- **Name:** Survey Lung Cancer
- **Samples:** 559
- **Features:** 15 (GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN)
- **Target:** LUNG_CANCER (YES / NO)
- **Class Distribution:** YES — 488 (87.3%), NO — 71 (12.7%)
- **Missing Values:** None

## c. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8839 | 0.8929 | 0.9126 | 0.9592 | 0.9353 | 0.3848 |
| Decision Tree | 0.9286 | 0.9767 | 0.9412 | 0.9796 | 0.9600 | 0.6391 |
| kNN | 0.9821 | 0.9271 | 0.9800 | 1.0000 | 0.9899 | 0.9165 |
| Naive Bayes | 0.9196 | 0.9111 | 0.9406 | 0.9694 | 0.9548 | 0.6010 |
| Random Forest (Ensemble) | 0.9732 | 0.9537 | 0.9703 | 1.0000 | 0.9849 | 0.8731 |
| XGBoost (Ensemble) | 0.9732 | 0.9789 | 0.9798 | 0.9898 | 0.9848 | 0.8745 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Lowest accuracy (0.8839) and MCC (0.3848) among all models. The linear decision boundary is too simple to capture the patterns in this dataset. High recall (0.9592) but comparatively low precision indicates it over-predicts the positive class. |
| Decision Tree | Moderate performance with good AUC (0.9767). Prone to overfitting on small datasets. Performs better than Logistic Regression and Naive Bayes but falls behind ensemble methods and kNN. |
| kNN | Best accuracy (0.9821), best F1 (0.9899), and perfect recall (1.0000). Achieves the highest MCC (0.9165), indicating strong performance on both classes despite the imbalance. AUC (0.9271) is slightly lower since kNN probability estimates are less calibrated. |
| Naive Bayes | Second-lowest accuracy (0.9196) and MCC (0.6010). The conditional independence assumption limits its ability to model feature interactions. Still achieves reasonable recall (0.9694). |
| Random Forest (Ensemble) | Strong overall performance with perfect recall (1.0000) and high accuracy (0.9732). As a bagged ensemble it reduces variance compared to a single Decision Tree. MCC (0.8731) confirms balanced classification. |
| XGBoost (Ensemble) | Tied with Random Forest for accuracy (0.9732) and achieves the highest AUC (0.9789), indicating the best-calibrated probability estimates. Highest precision (0.9798) among top models and strong MCC (0.8745). |
