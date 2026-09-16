import pandas as pd

# ==========================================
# STEP 2 - DATA PREPROCESSING
# ==========================================

# 1. Load dataset
df = pd.read_csv("data/machine_data.csv")

print("========== ORIGINAL DATA ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# 2. Convert timestamp from text to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nTimestamp converted successfully.")
print("Timestamp type:", df["timestamp"].dtype)


# 3. Sort data by time
df = df.sort_values("timestamp").reset_index(drop=True)

print("\nData sorted by timestamp.")


# 4. Define target
y = df["failure"]

print("\n========== TARGET ==========")
print("Target column: failure")
print("Target values:")
print(y.value_counts())


# 5. Define ML features
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


# 6. Keep validation information separately
validation_metadata = df[
    ["timestamp", "machine_id"]
]


# 7. Display information
print("\n========== ML FEATURES ==========")
print(X.head())

print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeature shape:", X.shape)


print("\n========== VALIDATION METADATA ==========")
print(validation_metadata.head())

print("\nMetadata shape:", validation_metadata.shape)


# 8. Final checks
print("\n========== FINAL CHECKS ==========")

print("Missing values in X:")
print(X.isnull().sum())

print("\nMissing values in y:")
print(y.isnull().sum())

print("\nTotal X rows:", len(X))
print("Total y rows:", len(y))
print("Total metadata rows:", len(validation_metadata))

print("\n========================================")
print("STEP 2 PREPROCESSING COMPLETED")
print("========================================")