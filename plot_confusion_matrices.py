
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

models = {
    "Decision Tree": "decision_tree_confusion_matrix.csv",
    "Random Forest": "random_forest_confusion_matrix.csv",
    "Linear SVM": "linear_svm_confusion_matrix.csv",
}

for model_name, filename in models.items():
    csv_path = OUTPUT_DIR / filename

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        print("Check the filename in your outputs folder.")
        continue

    # Load the saved confusion matrix
    cm = pd.read_csv(csv_path, index_col=0)

    # Ensure the values are numeric
    cm = cm.apply(pd.to_numeric)

    # 1. Raw confusion matrix: number of predictions
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=False,
        linewidths=0,
        cbar_kws={"label": "Number of records"},
    )

    plt.title(
        f"{model_name} — Confusion Matrix (Counts)",
        fontsize=17,
        pad=15,
    )
    plt.xlabel("Predicted Crop", fontsize=12)
    plt.ylabel("Actual Crop", fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    raw_path = OUTPUT_DIR / (
        f"{model_name.lower().replace(' ', '_')}_confusion_counts.png"
    )
    plt.savefig(raw_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Row-normalized matrix: percentage for each actual crop
    row_totals = cm.sum(axis=1).replace(0, 1)
    normalized_cm = cm.div(row_totals, axis=0) * 100

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        normalized_cm,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot=False,
        linewidths=0,
        cbar_kws={"label": "Percentage (%)"},
    )

    plt.title(
        f"{model_name} — Normalized Confusion Matrix (%)",
        fontsize=17,
        pad=15,
    )
    plt.xlabel("Predicted Crop", fontsize=12)
    plt.ylabel("Actual Crop", fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    normalized_path = OUTPUT_DIR / (
        f"{model_name.lower().replace(' ', '_')}_confusion_normalized.png"
    )
    plt.savefig(normalized_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Summarize the saved matrix
    total = cm.to_numpy().sum()
    correct = cm.to_numpy().diagonal().sum()
    accuracy = correct / total if total else 0

    print(f"\n{model_name}")
    print(f"Records represented: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy from matrix: {accuracy:.4%}")
    print(f"Saved: {raw_path}")
    print(f"Saved: {normalized_path}")

print("\nConfusion-matrix visualization completed.")
