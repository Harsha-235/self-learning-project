import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# STEP 6 - TEMPORAL VALIDATION
# ==========================================

# 1. Load clean dataset
df = pd.read_csv("../data/machine_data_v2.csv")

# 2. Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 3. Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

print("========== TEMPORAL VALIDATION ==========")
print("Total rows:", len(df))

# 4. Define legitimate features
feature_columns = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]

# 5. Calculate temporal split
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("\n========== TEMPORAL SPLIT ==========")

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

print("\nTraining time:")
print(train_df["timestamp"].min(), "to", train_df["timestamp"].max())

print("\nTesting time:")
print(test_df["timestamp"].min(), "to", test_df["timestamp"].max())


# 6. Create X and y
X_train = train_df[feature_columns]
y_train = train_df["failure"]

X_test = test_df[feature_columns]
y_test = test_df["failure"]


# 7. Train Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

print("\nTraining temporal model...")

model.fit(X_train, y_train)

print("Training completed.")


# 8. Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# 9. Evaluation
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


# 10. Results
print("\n========== TEMPORAL PERFORMANCE ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


print("\n========================================")
print("TEMPORAL VALIDATION COMPLETED")
print("========================================")