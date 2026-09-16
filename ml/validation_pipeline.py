import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# STEP 8 - INTEGRATED VALIDATION PIPELINE
# ==========================================

# ------------------------------------------
# 1. LOAD CLEAN DATASET
# ------------------------------------------

df = pd.read_csv("../data/machine_data_v2.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("==========================================")
print(" INTEGRATED VALIDATION PIPELINE")
print("==========================================")

print("\nDataset rows:", len(df))
print("Dataset columns:", len(df.columns))


# ------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# ------------------------------------------

feature_columns = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]

X = df[feature_columns]
y = df["failure"]


# ------------------------------------------
# 3. FUNCTION TO TRAIN AND EVALUATE MODEL
# ------------------------------------------

def evaluate_model(X_train, X_test, y_train, y_test):

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

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

    return accuracy, precision, recall, f1, roc_auc


# ==========================================
# VALIDATION 1 - RANDOM SPLIT
# ==========================================

print("\n")
print("========== 1. RANDOM VALIDATION ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

random_results = evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test
)

print("Accuracy :", round(random_results[0], 4))
print("Precision:", round(random_results[1], 4))
print("Recall   :", round(random_results[2], 4))
print("F1 Score :", round(random_results[3], 4))
print("ROC-AUC  :", round(random_results[4], 4))


# ==========================================
# VALIDATION 2 - TEMPORAL SPLIT
# ==========================================

print("\n")
print("========== 2. TEMPORAL VALIDATION ==========")

df_time = df.sort_values("timestamp").reset_index(drop=True)

split_index = int(len(df_time) * 0.80)

train_df = df_time.iloc[:split_index]
test_df = df_time.iloc[split_index:]

X_train = train_df[feature_columns]
y_train = train_df["failure"]

X_test = test_df[feature_columns]
y_test = test_df["failure"]

temporal_results = evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test
)

print("Training period:")
print(
    train_df["timestamp"].min(),
    "to",
    train_df["timestamp"].max()
)

print("Testing period:")
print(
    test_df["timestamp"].min(),
    "to",
    test_df["timestamp"].max()
)

print("\nAccuracy :", round(temporal_results[0], 4))
print("Precision:", round(temporal_results[1], 4))
print("Recall   :", round(temporal_results[2], 4))
print("F1 Score :", round(temporal_results[3], 4))
print("ROC-AUC  :", round(temporal_results[4], 4))


# ==========================================
# VALIDATION 3 - GROUP SPLIT
# ==========================================

print("\n")
print("========== 3. GROUP VALIDATION ==========")

machines = sorted(df["machine_id"].unique())

train_machines = machines[:8]
test_machines = machines[8:]

train_df = df[
    df["machine_id"].isin(train_machines)
]

test_df = df[
    df["machine_id"].isin(test_machines)
]

X_train = train_df[feature_columns]
y_train = train_df["failure"]

X_test = test_df[feature_columns]
y_test = test_df["failure"]

group_results = evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test
)

print("Training machines:", train_machines)
print("Testing machines :", test_machines)

print("\nAccuracy :", round(group_results[0], 4))
print("Precision:", round(group_results[1], 4))
print("Recall   :", round(group_results[2], 4))
print("F1 Score :", round(group_results[3], 4))
print("ROC-AUC  :", round(group_results[4], 4))


# ==========================================
# FINAL COMPARISON
# ==========================================

print("\n")
print("==========================================")
print(" VALIDATION COMPARISON")
print("==========================================")

comparison = pd.DataFrame({
    "Validation": [
        "Random",
        "Temporal",
        "Group"
    ],

    "Accuracy": [
        random_results[0],
        temporal_results[0],
        group_results[0]
    ],

    "Precision": [
        random_results[1],
        temporal_results[1],
        group_results[1]
    ],

    "Recall": [
        random_results[2],
        temporal_results[2],
        group_results[2]
    ],

    "F1": [
        random_results[3],
        temporal_results[3],
        group_results[3]
    ],

    "ROC-AUC": [
        random_results[4],
        temporal_results[4],
        group_results[4]
    ]
})

print(
    comparison.round(4).to_string(index=False)
)


# ==========================================
# SAVE RESULTS
# ==========================================

comparison.to_csv(
    "../experiments/validation_results.csv",
    index=False
)

print("\nResults saved to:")
print("experiments/validation_results.csv")


print("\n==========================================")
print("INTEGRATED VALIDATION COMPLETED")
print("==========================================")