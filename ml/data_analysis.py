import pandas as pd

# ==========================================
# STEP 3B - FEATURE / TARGET ANALYSIS
# ==========================================

# Load dataset
df = pd.read_csv("../data/machine_data_v2.csv")

# Features we want to inspect
feature_columns = [
    "temperature",
    "pressure",
    "vibration",
    "rotational_speed",
    "operating_hours",
    "ambient_temperature",
    "load_percent"
]

print("========== FAILURE GROUP ANALYSIS ==========")

# Separate normal and failure records
normal = df[df["failure"] == 0]
failure = df[df["failure"] == 1]

print("\nNormal records:", len(normal))
print("Failure records:", len(failure))


# Compare averages
print("\n========== FEATURE AVERAGES ==========")

for feature in feature_columns:

    normal_average = normal[feature].mean()
    failure_average = failure[feature].mean()

    print("\nFeature:", feature)
    print("Normal average :", round(normal_average, 3))
    print("Failure average:", round(failure_average, 3))


print("\n========================================")
print("FEATURE ANALYSIS COMPLETED")
print("========================================")