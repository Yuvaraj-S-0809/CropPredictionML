
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 1. File locations
DATA_PATH = Path("data/crop_dataset.csv")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

target = "CROPS"

categorical_features = [
    "SOIL",
    "SEASON",
    "SOWN",
    "WATER_SOURCE",
]

numerical_features = [
    "SOIL_PH",
    "TEMP",
    "RELATIVE_HUMIDITY",
    "N",
    "P",
    "K",
]

features = categorical_features + numerical_features

# Clean whitespace in text columns
for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].str.replace("\u00a0", " ", regex=False)
    df[column] = df[column].str.strip()

X = df[features]
y = df[target]

# 3. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Crop classes:", y.nunique())

# 4. Prepare categorical and numerical inputs
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
    ]
)

# 5. Define three models
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "Linear SVM": LinearSVC(
        random_state=42,
        max_iter=5000,
    ),
}

# 6. Evaluate models
results = []
trained_models = {}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for model_name, classifier in models.items():
    print(f"\n{'=' * 50}")
    print("Evaluating:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("Running 5-fold cross-validation...")

    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=1,
        error_score="raise",
    )

    print(
        "CV accuracy:",
        round(cv_scores["test_accuracy"].mean(), 4),
        "+/-",
        round(cv_scores["test_accuracy"].std(), 4),
    )
    print(
        "CV macro F1:",
        round(cv_scores["test_macro_f1"].mean(), 4),
    )

    # Train on the complete training split
    pipeline.fit(X_train, y_train)

    # Evaluate on the untouched test split
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print("\nTest accuracy:", round(accuracy, 4))
    print("Test macro precision:", round(precision, 4))
    print("Test macro recall:", round(recall, 4))
    print("Test macro F1:", round(macro_f1, 4))

    results.append({
        "Model": model_name,
        "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
        "CV Accuracy Std": cv_scores["test_accuracy"].std(),
        "CV Macro F1 Mean": cv_scores["test_macro_f1"].mean(),
        "Test Accuracy": accuracy,
        "Test Macro Precision": precision,
        "Test Macro Recall": recall,
        "Test Macro F1": macro_f1,
    })

    trained_models[model_name] = pipeline

    # Save detailed test results
    safe_name = model_name.lower().replace(" ", "_")
    pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        OUTPUT_DIR / f"{safe_name}_classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(
            y_test,
            predictions,
            labels=sorted(y.unique()),
        ),
        index=sorted(y.unique()),
        columns=sorted(y.unique()),
    ).to_csv(OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv")

# 7. Compare all models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="CV Macro F1 Mean",
    ascending=False,
)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 8. Save the best model based on cross-validation macro F1
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

import joblib

joblib.dump(best_model, MODEL_DIR / "best_crop_model.joblib", compress=3)

with open(OUTPUT_DIR / "best_model_info.json", "w") as file:
    json.dump(
        {
            "best_model": best_name,
            "selection_metric": "5-fold CV Macro F1",
            "selection_score": float(
                results_df.iloc[0]["CV Macro F1 Mean"]
            ),
            "test_accuracy": float(
                results_df.iloc[0]["Test Accuracy"]
            ),
        },
        file,
        indent=4,
    )

print("\nSelected model:", best_name)
print("Saved model to:", MODEL_DIR / "best_crop_model.joblib")
print("Saved comparison to:", OUTPUT_DIR / "model_comparison.csv")
print("\nTraining and evaluation completed.")
