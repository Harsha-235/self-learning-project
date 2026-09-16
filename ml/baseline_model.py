import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================
# STEP 3 - REFERENCE GRADIENT BOOSTING MODEL
# ==========================================

# 1. Load dataset
df = pd.read_csv("../data/machine_data_leaky.csv")

print("========== DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# 2. Define input features
feature_columns = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent",
    "future_failure_signal"
]

X = df[feature_columns]

# 3. Define target
y = df["failure"]


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Create Gradient Boosting model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# 6. Train the model
print("\nTraining Gradient Boosting model...")

model.fit(X_train, y_train)

print("Training completed.")


# 7. Make predictions
y_pred = model.predict(X_test)

# Probability of failure = 1
y_probability = model.predict_proba(X_test)[:, 1]


# 8. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

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


# 9. Display results
print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# 11. Feature importance
importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.to_string(index=False))


print("\n========================================")
print("REFERENCE MODEL COMPLETED")
print("========================================")