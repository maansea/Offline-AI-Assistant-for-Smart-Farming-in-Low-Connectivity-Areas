# 🌾 Offline AI Assistant for Smart Farming in Low-Connectivity Areas

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

## 📌 Project Overview

Farmers in rural areas often lack reliable internet access, limiting their ability to use modern AI tools for crop planning and fertilizer decisions. This project provides a **lightweight, offline-first AI assistant** that helps farmers make smart crop and fertilizer decisions using voice or text input (Hindi/English) — without requiring internet access.

---

## 👥 Team

| Member | Role |
|--------|------|
| Member 1 | AI/ML — Data, Models, Explainability |
| Member 2 | Backend, Voice Assistant & UI (Flask, Vosk, SQLite) |

---

## 🧠 Member 1 Responsibilities (This Repo)

- 📦 Dataset collection and preprocessing
- 🌾 Crop recommendation ML model
- 🧪 Fertilizer recommendation ML model
- 🔍 Explainability module (Rule-based + SHAP)
- 📊 Model evaluation and testing
- 🔗 Integration support with backend

---

## 📁 Project Structure

```
Offline-AI-Assistant-for-Smart-Farming/
│
├── data/
│   ├── raw/                        # Original Kaggle datasets
│   └── processed/                  # Cleaned & encoded datasets
│
├── models/                         # Saved trained models (.pkl)
│   ├── crop_model.pkl
│   └── fertilizer_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_crop_training.ipynb      # Crop model training
│   ├── 03_fertilizer_training.ipynb# Fertilizer model training
│   └── 04_model_evaluation.ipynb  # Evaluation & comparison
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Data cleaning & encoding
│   ├── crop_model.py               # Crop recommendation model
│   ├── fertilizer_model.py         # Fertilizer recommendation model
│   ├── recommendation_logic.py     # ML + rule-based hybrid logic
│   └── explainability.py           # SHAP + rule-based explanations
│
├── tests/
│   ├── __init__.py
│   └── test_models.py              # Unit tests for models
│
├── requirements.txt
└── README.md
```

---

## 📊 Datasets Used

| Dataset | Source | Description |
|---------|--------|-------------|
| Crop Recommendation | [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | N, P, K, temperature, humidity, pH, rainfall → crop |
| Fertilizer Recommendation | [Kaggle](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction) | Soil nutrients, crop type → fertilizer |

---

## 🤖 ML Models

| Model | Algorithm | Accuracy |
|-------|-----------|----------|
| Crop Recommendation | Random Forest + XGBoost | ~99% |
| Fertilizer Recommendation | Random Forest + XGBoost | ~95% |

---

## 🔍 Explainability

Every recommendation comes with a clear explanation:

- **Rule-based:** *"Nitrogen level is low. Urea is recommended to boost nitrogen content."*
- **SHAP values:** Feature importance scores showing which inputs influenced the decision most.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Mansi2801-dev/Offline-AI-Assistant-for-Smart-Farming-in-Low-Connectivity-Areas.git
cd Offline-AI-Assistant-for-Smart-Farming-in-Low-Connectivity-Areas
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
- Download datasets from Kaggle links above
- Place them in `data/raw/` folder

### 4. Run preprocessing
```bash
python src/preprocess.py
```

### 5. Train models
```bash
python src/crop_model.py
python src/fertilizer_model.py
```

### 6. Run tests
```bash
pytest tests/
```

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn** — Machine Learning
- **XGBoost** — Gradient Boosting
- **SHAP** — Explainability
- **Pandas / NumPy** — Data Processing
- **Matplotlib / Seaborn** — Visualization
- **Joblib** — Model Serialization
- **Pytest** — Testing

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Kaggle datasets for agricultural data
- Scikit-learn and XGBoost communities
- SHAP library for model explainability
