import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


# ==========================================
# STEP 10 - FINAL ML MODEL PIPELINE
# ==========================================

DATA_PATH = "../data/machine_data_v2.csv"
MODEL_PATH = "../models/gradient_boosting_model.pkl"
RESULT_PATH = "../experiments/final_model_results.csv"


# ==========================================
# 1. LOAD DATA
# ==========================================

print("==========================================")
print(" FINAL ML MODEL PIPELINE")
print("==========================================")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================
# 2. SORT BY TIME
# ==========================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ==========================================
# 3. DEFINE LEGITIMATE FEATURES
# ==========================================

feature_columns = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]

target_column = "failure"


X = df[feature_columns]
y = df[target_column]


# ==========================================
# 4. CHECK DATA
# ==========================================

print("\n========== DATA CHECK ==========")

print("Missing feature values:")
print(X.isnull().sum())

print("\nMissing target values:")
print(y.isnull().sum())

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 5. RANDOM REFERENCE SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. CREATE FINAL GRADIENT BOOSTING MODEL
# ==========================================

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining final Gradient Boosting model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==========================================
# 8. PREDICT
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 9. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 10. DISPLAY PERFORMANCE
# ==========================================

print("\n========== FINAL MODEL PERFORMANCE ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(
    importance.to_string(
        index=False
    )
)


# ==========================================
# 12. SAVE MODEL
# ==========================================

Path("../models").mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)


# ==========================================
# 13. SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

Path("../experiments").mkdir(
    exist_ok=True
)

results.to_csv(
    RESULT_PATH,
    index=False
)

print("\nResults saved to:")
print(RESULT_PATH)


# ==========================================
# 14. SAMPLE PREDICTION
# ==========================================

sample = X_test.iloc[[0]]

sample_prediction = model.predict(
    sample
)[0]

sample_probability = model.predict_proba(
    sample
)[0][1]

print("\n========== SAMPLE PREDICTION ==========")

print(
    "Predicted failure:",
    int(sample_prediction)
)

print(
    "Failure probability:",
    round(sample_probability, 4)
)


print("\n==========================================")
print("FINAL ML MODEL PIPELINE COMPLETED")
print("==========================================")