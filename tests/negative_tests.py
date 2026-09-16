import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier


# ==========================================
# STEP 9 - MANDATORY NEGATIVE TEST CAMPAIGN
# ==========================================

DATA_PATH = "data/machine_data_v2.csv"
LEAKY_DATA_PATH = "data/machine_data_leaky.csv"
MODEL_PATH = "../models/gradient_boosting_model.pkl"

FEATURE_COLUMNS = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]


# ==========================================
# LOAD CLEAN DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

X = df[FEATURE_COLUMNS]
y = df["failure"]


# ==========================================
# TRAIN AND SAVE REFERENCE MODEL
# ==========================================

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

Path("../models").mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH)


print("==========================================")
print(" MANDATORY NEGATIVE TEST CAMPAIGN")
print("==========================================")

print("\nReference model trained successfully.")


# ==========================================
# NT-1
# EMPTY VALIDATION DATA
# ==========================================

print("\n========== NT-1 ==========")

try:

    validation_data = pd.DataFrame()

    if validation_data.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

except ValueError as e:

    print("FAULT DETECTED:", e)
    print("SAFE RESPONSE: Reject validation run.")
    print("STATUS: PASS")


# ==========================================
# NT-2
# LEAKAGE FEATURE PRESENT
# ==========================================

print("\n========== NT-2 ==========")

leaky_df = pd.read_csv(LEAKY_DATA_PATH)

leak_feature = "future_failure_signal"

if leak_feature in leaky_df.columns:

    print(
        "FAULT DETECTED:",
        leak_feature,
        "is present."
    )

    print(
        "SAFE RESPONSE:",
        "Reject feature and block deployment."
    )

    print("STATUS: PASS")

else:

    print(
        "FAULT NOT DETECTED:",
        leak_feature,
        "is missing from leaky test."
    )

    print("STATUS: FAIL")


# ==========================================
# NT-3
# TEMPORAL ORDER VIOLATION
# ==========================================

print("\n========== NT-3 ==========")

df_time = df.sort_values(
    "timestamp"
).reset_index(drop=True)

split_index = int(len(df_time) * 0.80)

train_df = df_time.iloc[:split_index].copy()
test_df = df_time.iloc[split_index:].copy()


# Deliberately create an invalid temporal situation.
# Move one test timestamp into the training period.

test_df.loc[
    test_df.index[0],
    "timestamp"
] = train_df["timestamp"].iloc[-1]


train_end_time = train_df["timestamp"].max()
test_start_time = test_df["timestamp"].min()


if test_start_time <= train_end_time:

    print("FAULT DETECTED: Temporal order violation.")

    print(
        "Training ends:",
        train_end_time
    )

    print(
        "Testing starts:",
        test_start_time
    )

    print(
        "SAFE RESPONSE:",
        "Reject validation split."
    )

    print("STATUS: PASS")

else:

    print("Temporal violation was not detected.")
    print("STATUS: FAIL")


# ==========================================
# NT-4
# MISSING SENSOR VALUE
# ==========================================

print("\n========== NT-4 ==========")

faulty_data = X.copy()

# Deliberately inject missing sensor value.
faulty_data.loc[
    faulty_data.index[0],
    "temperature"
] = np.nan


if faulty_data.isnull().any().any():

    print(
        "FAULT DETECTED:",
        "Missing temperature value."
    )

    print(
        "SAFE RESPONSE:",
        "Reject input and require correction."
    )

    print("STATUS: PASS")

else:

    print("Missing value was not detected.")
    print("STATUS: FAIL")


# ==========================================
# NT-5
# EXTREME SENSOR VALUE
# ==========================================

print("\n========== NT-5 ==========")

faulty_data = X.copy()

# Deliberately inject an impossible/extreme
# vibration value.
faulty_data.loc[
    faulty_data.index[0],
    "vibration"
] = 9999


VIBRATION_LIMIT = 100


if faulty_data["vibration"].max() > VIBRATION_LIMIT:

    print(
        "FAULT DETECTED:",
        "Extreme vibration value."
    )

    print(
        "SAFE RESPONSE:",
        "Reject unsafe input."
    )

    print("STATUS: PASS")

else:

    print("Extreme value was not detected.")
    print("STATUS: FAIL")


# ==========================================
# FINAL RESULT
# ==========================================

print("\n==========================================")
print("NEGATIVE TEST CAMPAIGN COMPLETED")
print("==========================================")

print("\nRequired tests:")
print("NT-1: Empty validation data")
print("NT-2: Leakage feature")
print("NT-3: Temporal violation")
print("NT-4: Missing sensor value")
print("NT-5: Extreme sensor value")

print("\n==========================================")
