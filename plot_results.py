
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read the actual saved model comparison results
results_path = Path("outputs/model_comparison.csv")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

if not results_path.exists():
    raise FileNotFoundError(
        "outputs/model_comparison.csv was not found. "
        "Run train_models.py first."
    )

df = pd.read_csv(results_path)

required_columns = [
    "Model",
    "CV Accuracy Mean",
    "CV Accuracy Std",
    "CV Macro F1 Mean",
    "Test Accuracy",
    "Test Macro Precision",
    "Test Macro Recall",
    "Test Macro F1",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}")

# Convert scores to percentages for readable charts
metric_columns = [
    "Test Accuracy",
    "Test Macro Precision",
    "Test Macro Recall",
    "Test Macro F1",
]

# Chart 1: Test metrics comparison
ax = (
    df.set_index("Model")[metric_columns]
    .plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
)

ax.set_title("Crop Recommendation: Test Metrics Comparison")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Score (0 to 1)")
ax.legend(
    ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    loc="lower right",
)
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(output_dir / "test_metrics_comparison.png", dpi=300)
plt.close()

# Chart 2: Cross-validation performance
cv_columns = ["CV Accuracy Mean", "CV Macro F1 Mean"]

ax = (
    df.set_index("Model")[cv_columns]
    .plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))
)

ax.set_title("Five-Fold Cross-Validation Performance")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Mean Score (0 to 1)")
ax.legend(["CV Accuracy", "CV Macro F1"])
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(output_dir / "cross_validation_comparison.png", dpi=300)
plt.close()

# Chart 3: CV accuracy stability
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    df["Model"],
    df["CV Accuracy Mean"],
    yerr=df["CV Accuracy Std"],
    capsize=6,
)

ax.set_title("Cross-Validation Accuracy and Standard Deviation")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Accuracy (0 to 1)")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(output_dir / "cross_validation_stability.png", dpi=300)
plt.close()

print("Charts generated successfully:")
print(output_dir / "test_metrics_comparison.png")
print(output_dir / "cross_validation_comparison.png")
print(output_dir / "cross_validation_stability.png")
print("\nActual comparison data:")
print(df.round(4).to_string(index=False))
