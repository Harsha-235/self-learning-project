import pandas as pd

# Load our clean dataset
df = pd.read_csv("../data/machine_data_v2.csv")

# Create an intentionally leaky feature
df["future_failure_signal"] = df["failure"]

# Save as a separate leaky dataset
df.to_csv("../data/machine_data_leaky.csv", index=False)

print("Leaky dataset created successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("File: data/machine_data_leaky.csv")