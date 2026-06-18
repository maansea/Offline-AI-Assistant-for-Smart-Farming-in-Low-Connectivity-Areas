"""
preprocess.py
Member 1 - AI/ML
Data cleaning, encoding, and preparation for crop and fertilizer recommendation models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MODELS_PATH = "models"


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)


# ─────────────────────────────────────────────
# CROP DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_crop_data(filename="crop_recommendation.csv"):
    """
    Load and preprocess the crop recommendation dataset.

    Features: N, P, K, temperature, humidity, ph, rainfall
    Target: label (crop name)

    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    print("🌾 Loading crop recommendation dataset...")
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(filepath)

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Missing values:\n{df.isnull().sum()}")

    # ── Drop duplicates ──
    df.drop_duplicates(inplace=True)
    print(f"   After dropping duplicates: {df.shape}")

    # ── Features and target ──
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols]
    y = df[target_col]

    # ── Encode target labels ──
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"   Crops found: {list(le.classes_)}")

    # ── Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"   Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ── Save processed data ──
    ensure_dirs()
    X_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "crop_X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "crop_X_test.csv"), index=False)
    pd.Series(y_train).to_csv(os.path.join(PROCESSED_DATA_PATH, "crop_y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(PROCESSED_DATA_PATH, "crop_y_test.csv"), index=False)

    # ── Save scaler and encoder ──
    joblib.dump(scaler, os.path.join(MODELS_PATH, "crop_scaler.pkl"))
    joblib.dump(le, os.path.join(MODELS_PATH, "crop_label_encoder.pkl"))

    print("✅ Crop data preprocessed and saved!")
    return X_train, X_test, y_train, y_test, scaler, le


# ─────────────────────────────────────────────
# FERTILIZER DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_fertilizer_data(filename="fertilizer_recommendation.csv"):
    """
    Load and preprocess the fertilizer recommendation dataset.

    Features: Temparature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
    Target: Fertilizer Name

    Returns:
        X_train, X_test, y_train, y_test, scaler, encoders
    """
    print("\n🧪 Loading fertilizer recommendation dataset...")
    filepath = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(filepath)

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Missing values:\n{df.isnull().sum()}")

    # ── Drop duplicates ──
    df.drop_duplicates(inplace=True)
    print(f"   After dropping duplicates: {df.shape}")

    # ── Standardize column names ──
    df.columns = df.columns.str.strip()

    # ── Encode categorical columns ──
    encoders = {}
    categorical_cols = ['Soil Type', 'Crop Type']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"   Encoded '{col}': {list(le.classes_)}")

    # ── Features and target ──
    feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type',
                    'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    target_col = 'Fertilizer Name'

    # Keep only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    X = df[feature_cols]
    y = df[target_col]

    # ── Encode target ──
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    encoders['Fertilizer Name'] = le_target
    print(f"   Fertilizers found: {list(le_target.classes_)}")

    # ── Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"   Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ── Save processed data ──
    ensure_dirs()
    X_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "fertilizer_X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "fertilizer_X_test.csv"), index=False)
    pd.Series(y_train).to_csv(os.path.join(PROCESSED_DATA_PATH, "fertilizer_y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(PROCESSED_DATA_PATH, "fertilizer_y_test.csv"), index=False)

    # ── Save scaler and encoders ──
    joblib.dump(scaler, os.path.join(MODELS_PATH, "fertilizer_scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_PATH, "fertilizer_encoders.pkl"))

    print("✅ Fertilizer data preprocessed and saved!")
    return X_train, X_test, y_train, y_test, scaler, encoders


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    preprocess_crop_data()
    preprocess_fertilizer_data()
    print("\n🎉 All preprocessing complete!")
    print(f"   Processed data saved to: {PROCESSED_DATA_PATH}/")
    print(f"   Scalers/Encoders saved to: {MODELS_PATH}/")
