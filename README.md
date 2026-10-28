# 🌱 CropVibe AI | Intelligent Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-F7931E.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-98.40%25-10B981.svg)]()
[![Design Reference](https://img.shields.io/badge/UI%2FUX%20Reference-PixVibe.ai-7C3AED.svg)](https://pixvibe.ai/)

> **CropVibe AI** is a machine learning-driven agronomic intelligence platform designed with modern, high-performance UI/UX inspired by [PixVibe.ai](https://pixvibe.ai/). It analyzes multi-factor soil chemistry, climate conditions, and macronutrient balances to recommend optimal crop varieties with 98.4% accuracy.

---

## ✨ Features

- **🔮 AI Crop Recommendation Studio**: Interactive parameter deck for field environment, soil pH bioavailability, ambient temperature, relative humidity, and N-P-K macronutrient profiling.
- **⚡ Quick Agronomic Presets**: 1-click scenario presets (*Kharif Paddy*, *Highland Maize*, *Rabi Wheat*, *Semi-Arid Pulses*, *Commercial Horticulture*).
- **📊 Multi-Model Benchmarking**: Evaluates and compares Random Forest, Decision Tree, and Linear SVM classifiers across Stratified 5-Fold Cross-Validation, Accuracy, Precision, Recall, and F1 score.
- **📂 Batch CSV Processing Engine**: Bulk score agricultural field surveys locally in milliseconds with instant CSV export.
- **🧪 Soil & Nutrient Diagnostics**: Educational spectrum of soil pH bioavailability ($CaCO_3$ / Gypsum remedies) and N-P-K nutrient dynamics.
- **📘 AI Agent / Developer Architecture**: `/llms.txt`-inspired machine-readable JSON schemas and Python snippets for autonomous LLM agents and pipeline automation.

---

## 🏆 Model Performance

Trained on **57,000 verified agronomic field records** across **57 unique crop varieties**:

| Model | Test Accuracy | Test Macro F1 | CV Accuracy Mean | CV Macro F1 Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **98.40%** | **98.40%** | **98.39%** | **98.39%** |
| Decision Tree | 97.87% | 97.86% | 97.69% | 97.69% |
| Linear SVM | 94.59% | 94.53% | 94.51% | 94.46% |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Yuvaraj-S-0809/CropPredictionML.git
cd CropPredictionML
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser to experience CropVibe AI.

---

## 📁 Repository Structure

```
CropPredictionML/
├── app.py                     # Main Streamlit application (PixVibe design system)
├── train_models.py            # Training pipeline with 5-fold cross-validation
├── inspect_data.py            # Dataset inspection script
├── audit_dataset.py           # Dataset distribution & balance auditor
├── plot_results.py            # Cross-validation comparison visualization
├── plot_confusion_matrices.py # Confusion matrix generator
├── requirements.txt           # Project dependencies
├── .gitignore                 # Standard exclusions
├── data/
│   └── crop_dataset.csv       # 57,000 agronomic field records
├── models/
│   └── best_crop_model.joblib # Trained Random Forest model (compressed)
└── outputs/
    ├── best_model_info.json
    ├── model_comparison.csv
    ├── *.png                  # Confusion matrices & stability plots
    └── *.csv                  # Per-class classification reports
```

---

## 🔒 Privacy & Architecture

- **100% Local-First Inference**: Predictions run in your Python execution environment with zero remote data telemetry.
- **Sub-millisecond latency**: Optimized pipeline utilizing Scikit-Learn transformers and compressed model serialization.

---

## 📜 Disclaimer
*CropVibe AI is intended for educational, decision support, and research purposes. Soil health recommendations should be cross-referenced with regional agricultural extension services and certified laboratory soil testing.*
