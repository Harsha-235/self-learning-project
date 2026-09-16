import os
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# CONTROLLED INDIRECT TARGET LEAKAGE EXPERIMENT
# ============================================================

print("=" * 65)
print(" CONTROLLED INDIRECT TARGET LEAKAGE EXPERIMENT")
print("=" * 65)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "machine_data_v2.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# REQUIRED FEATURES
# ============================================================

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]

TARGET = "failure"


for column in FEATURES + [TARGET]:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET]
).reset_index(drop=True)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_index, test_index = train_test_split(

    df.index,

    test_size=0.20,

    random_state=42,

    stratify=df[TARGET]
)

train_index = sorted(train_index)
test_index = sorted(test_index)


X_train = df.loc[
    train_index,
    FEATURES
].copy()

X_test = df.loc[
    test_index,
    FEATURES
].copy()

y_train = df.loc[
    train_index,
    TARGET
].astype(int)

y_test = df.loc[
    test_index,
    TARGET
].astype(int)


print("\n========== DATA SPLIT ==========")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# MODEL PARAMETERS
# SAME AS FINAL MODEL
# ============================================================

MODEL_PARAMETERS = {

    "n_estimators": 100,

    "learning_rate": 0.05,

    "max_depth": 3,

    "random_state": 42
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return results


# ============================================================
# 1. CLEAN BASELINE MODEL
# ============================================================

print(
    "\n========== 1. CLEAN BASELINE MODEL =========="
)

baseline_model = GradientBoostingClassifier(
    **MODEL_PARAMETERS
)

baseline_results = evaluate_model(

    baseline_model,

    X_train,
    X_test,

    y_train,
    y_test
)


print(
    f"Accuracy : {baseline_results['accuracy']:.4f}"
)

print(
    f"Precision: {baseline_results['precision']:.4f}"
)

print(
    f"Recall   : {baseline_results['recall']:.4f}"
)

print(
    f"F1 Score : {baseline_results['f1']:.4f}"
)

print(
    f"ROC-AUC  : {baseline_results['roc_auc']:.4f}"
)


# ============================================================
# 2. CREATE CONTROLLED INDIRECT LEAKAGE
# ============================================================

print(
    "\n========== 2. CREATE CONTROLLED INDIRECT LEAKAGE =========="
)

print(
    "Creating synthetic derived feature:"
)

print(
    "maintenance_risk_score"
)

print(
    "The feature is NOT equal to failure."
)

print(
    "It is a transformed target-derived score with noise."
)


# ------------------------------------------------------------
# IMPORTANT:
#
# We deliberately create a feature that indirectly encodes
# the target.
#
# It is NOT:
#
# maintenance_risk_score = failure
#
# Instead:
#
# failure
#    ↓
# transformed risk value
#    ↓
# random noise
#    ↓
# maintenance_risk_score
#
# A real deployment system would NOT have access to the
# target when calculating this feature.
#
# Therefore this is a controlled leakage experiment.
# ------------------------------------------------------------

rng = np.random.default_rng(42)


# Create a continuous risk score.

risk_score = (
    0.20
    +
    0.65 * df[TARGET].astype(float)
    +
    rng.normal(
        0,
        0.08,
        len(df)
    )
)


# Keep values inside a realistic 0-1 range.

risk_score = np.clip(
    risk_score,
    0,
    1
)


df["maintenance_risk_score"] = risk_score


LEAKAGE_FEATURES = FEATURES + [
    "maintenance_risk_score"
]


# ============================================================
# DISPLAY FEATURE INFORMATION
# ============================================================

print(
    "\nControlled leakage feature statistics:"
)

print(
    df[
        "maintenance_risk_score"
    ].describe()
)


# ============================================================
# SPLIT LEAKAGE DATA
# ============================================================

X_leak_train = df.loc[
    train_index,
    LEAKAGE_FEATURES
].copy()

X_leak_test = df.loc[
    test_index,
    LEAKAGE_FEATURES
].copy()


# ============================================================
# 3. FEATURE-LEVEL LEAKAGE TEST
# ============================================================

print(
    "\n========== 3. FEATURE-LEVEL LEAKAGE TEST =========="
)

feature_auc = roc_auc_score(

    y_test,

    X_leak_test[
        "maintenance_risk_score"
    ]
)


print(
    "maintenance_risk_score ROC-AUC:",
    round(
        feature_auc,
        4
    )
)


# ============================================================
# 4. LEAKAGE MODEL
# ============================================================

print(
    "\n========== 4. MODEL WITH INDIRECT LEAKAGE =========="
)

leaky_model = GradientBoostingClassifier(
    **MODEL_PARAMETERS
)

leaky_results = evaluate_model(

    leaky_model,

    X_leak_train,
    X_leak_test,

    y_train,
    y_test
)


print(
    f"Accuracy : {leaky_results['accuracy']:.4f}"
)

print(
    f"Precision: {leaky_results['precision']:.4f}"
)

print(
    f"Recall   : {leaky_results['recall']:.4f}"
)

print(
    f"F1 Score : {leaky_results['f1']:.4f}"
)

print(
    f"ROC-AUC  : {leaky_results['roc_auc']:.4f}"
)


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

print(
    "\n========== 5. FEATURE IMPORTANCE =========="
)

importance_df = pd.DataFrame({

    "feature":
        LEAKAGE_FEATURES,

    "importance":
        leaky_model.feature_importances_

})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print(
    importance_df.to_string(
        index=False
    )
)


leakage_importance = importance_df.loc[

    importance_df["feature"]
    == "maintenance_risk_score",

    "importance"

].iloc[0]


print(
    "\nmaintenance_risk_score importance:",
    round(
        leakage_importance,
        6
    )
)


# ============================================================
# 6. LEAKAGE DECISION
# ============================================================

print(
    "\n========== 6. LEAKAGE DETECTION =========="
)


if feature_auc >= 0.80:

    leakage_status = "DETECTED"

    print(
        "WARNING: Indirect target leakage detected."
    )

    print(
        "The derived feature strongly encodes the target."
    )

    print(
        "SAFE RESPONSE: Reject the feature."
    )

    print(
        "The feature must not be used for deployment."
    )

else:

    leakage_status = "NOT_DETECTED"

    print(
        "WARNING: Leakage threshold not reached."
    )


# ============================================================
# 7. REMOVE LEAKAGE FEATURE
# ============================================================

print(
    "\n========== 7. REMOVE LEAKAGE FEATURE =========="
)

X_train_clean = X_train.copy()

X_test_clean = X_test.copy()


print(
    "Removed feature:"
)

print(
    "maintenance_risk_score"
)


# ============================================================
# 8. RETRAIN CLEAN MODEL
# ============================================================

print(
    "\n========== 8. RETRAIN LEAKAGE-RESISTANT MODEL =========="
)

clean_model = GradientBoostingClassifier(
    **MODEL_PARAMETERS
)

clean_results = evaluate_model(

    clean_model,

    X_train_clean,
    X_test_clean,

    y_train,
    y_test
)


print(
    f"Accuracy : {clean_results['accuracy']:.4f}"
)

print(
    f"Precision: {clean_results['precision']:.4f}"
)

print(
    f"Recall   : {clean_results['recall']:.4f}"
)

print(
    f"F1 Score : {clean_results['f1']:.4f}"
)

print(
    f"ROC-AUC  : {clean_results['roc_auc']:.4f}"
)


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

print(
    "\n========== 9. FINAL COMPARISON =========="
)

comparison = pd.DataFrame([

    {

        "model":
            "Clean Baseline",

        "accuracy":
            baseline_results["accuracy"],

        "precision":
            baseline_results["precision"],

        "recall":
            baseline_results["recall"],

        "f1":
            baseline_results["f1"],

        "roc_auc":
            baseline_results["roc_auc"]

    },

    {

        "model":
            "Indirect Leakage",

        "accuracy":
            leaky_results["accuracy"],

        "precision":
            leaky_results["precision"],

        "recall":
            leaky_results["recall"],

        "f1":
            leaky_results["f1"],

        "roc_auc":
            leaky_results["roc_auc"]

    },

    {

        "model":
            "After Leakage Removal",

        "accuracy":
            clean_results["accuracy"],

        "precision":
            clean_results["precision"],

        "recall":
            clean_results["recall"],

        "f1":
            clean_results["f1"],

        "roc_auc":
            clean_results["roc_auc"]

    }

])


print(
    comparison.round(4).to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

comparison_path = os.path.join(

    RESULT_DIR,

    "controlled_indirect_leakage_results.csv"

)

comparison.to_csv(

    comparison_path,

    index=False
)


importance_path = os.path.join(

    RESULT_DIR,

    "controlled_indirect_leakage_feature_importance.csv"

)

importance_df.to_csv(

    importance_path,

    index=False
)


detection_report = pd.DataFrame({

    "feature": [
        "maintenance_risk_score"
    ],

    "feature_roc_auc": [
        feature_auc
    ],

    "model_importance": [
        leakage_importance
    ],

    "leakage_status": [
        leakage_status
    ],

    "safe_response": [
        "Reject feature and retrain without it."
    ]

})


report_path = os.path.join(

    RESULT_DIR,

    "controlled_indirect_leakage_report.csv"

)

detection_report.to_csv(

    report_path,

    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========== EXPERIMENT SUMMARY =========="
)

print(
    "Indirect leakage feature:",
    "maintenance_risk_score"
)

print(
    "Feature ROC-AUC:",
    round(
        feature_auc,
        4
    )
)

print(
    "Feature importance:",
    round(
        leakage_importance,
        4
    )
)

print(
    "Leakage status:",
    leakage_status
)


if leakage_status == "DETECTED":

    print(
        "Safe response: FEATURE REJECTED"
    )

    print(
        "Production model uses only leakage-free features."
    )


print(
    "\nResults saved to:"
)

print(
    comparison_path
)

print(
    importance_path
)

print(
    report_path
)


print(
    "\n" + "=" * 65
)

print(
    " CONTROLLED INDIRECT LEAKAGE EXPERIMENT COMPLETED"
)

print(
    "=" * 65
)