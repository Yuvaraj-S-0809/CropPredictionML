
import pandas as pd
from pathlib import Path

file_path = Path("data/crop_dataset.csv")
df = pd.read_csv(file_path)

# The correct target column is CROPS
target = "CROPS"

print("\n--- DATASET SUMMARY ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Number of crop classes:", df[target].nunique())

print("\n--- CROP DISTRIBUTION ---")
print(df[target].value_counts().sort_index())

print("\n--- AVAILABLE FEATURES ---")
for column in df.columns:
    if column != target:
        print(
            f"{column}: "
            f"{df[column].nunique()} unique values | "
            f"Type: {df[column].dtype}"
        )

print("\n--- NUMERICAL FEATURE SUMMARY ---")
print(df.describe().round(2).to_string())

print("\n--- CATEGORICAL VALUES ---")
for column in ["SOIL", "SEASON", "WATER_SOURCE", "TYPE_OF_CROP"]:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts().to_string())

print("\n--- TARGET VALIDATION ---")
print("Missing crop labels:", df[target].isna().sum())
print("Blank crop labels:", df[target].astype(str).str.strip().eq("").sum())

print("\nInspection completed.")
