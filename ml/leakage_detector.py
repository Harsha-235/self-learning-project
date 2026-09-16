import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# ==========================================
# STEP 4 - LEAKAGE DETECTOR
# ==========================================

# 1. Load the leaky dataset
df = pd.read_csv("../data/machine_data_leaky.csv")

print("========== LEAKAGE DETECTION ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# 2. Target
y = df["failure"]


# 3. Features
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


# 4. Check every feature individually
results = []

for feature in feature_columns:

    X = df[[feature]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probability)

    results.append({
        "feature": feature,
        "roc_auc": auc
    })


# 5. Display results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
)


print("\n========== FEATURE LEAKAGE SCORES ==========")
print(results_df.to_string(index=False))


# 6. Identify suspicious features
print("\n========== SUSPICIOUS FEATURES ==========")

suspicious = results_df[
    results_df["roc_auc"] >= 0.95
]

if len(suspicious) == 0:

    print("No highly suspicious features detected.")

else:

    for _, row in suspicious.iterrows():

        print(
            "WARNING:",
            row["feature"],
            "has ROC-AUC =",
            round(row["roc_auc"], 4)
        )


print("\n========================================")
print("LEAKAGE DETECTION COMPLETED")
print("========================================")