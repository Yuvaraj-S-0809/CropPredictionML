
import pandas as pd
from pathlib import Path

file_path = Path("data/crop_dataset.csv")
df = pd.read_csv(file_path)

target = "CROPS"

# Features provisionally selected for crop recommendation
features = [
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE",
    "SOIL_PH",
    "TEMP",
    "RELATIVE_HUMIDITY",
    "N",
    "P",
    "K",
]

print("=== DATASET AUDIT ===")

# Check required columns
required = features + [target]
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Clean whitespace in text columns
for col in df.select_dtypes(include=["str", "object"]).columns:
    df[col] = df[col].str.strip()

print("\n1. Target classes:", df[target].nunique())
print("2. Rows:", len(df))
print("3. Exact duplicates:", df.duplicated().sum())

print("\n4. Crop count range:")
counts = df[target].value_counts()
print("Minimum records per crop:", counts.min())
print("Maximum records per crop:", counts.max())

print("\n5. Selected feature missing values:")
print(df[features].isnull().sum())

print("\n6. Categorical feature values:")
for col in ["SOIL", "SEASON", "SOWN", "WATER_SOURCE"]:
    print(f"\n{col}: {sorted(df[col].dropna().unique().tolist())}")

print("\n7. Selected feature uniqueness:")
print(df[features].nunique())

print("\n8. Duplicate selected input combinations:")
print(df.duplicated(subset=features).sum())

print("\n9. Number of distinct crops for each input combination:")
crop_counts = df.groupby(features, dropna=False)[target].nunique()
print(crop_counts.value_counts().sort_index())

print("\n10. Features correlated with target metadata?")
print(
    "TYPE_OF_CROP is excluded from model inputs to prevent "
    "target leakage."
)

print("\nAudit completed.")
