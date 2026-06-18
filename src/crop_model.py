"""
crop_model.py
Member 1 - AI/ML
Crop recommendation model using Random Forest and XGBoost.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MODELS_PATH = "models"


class CropRecommendationModel:
    """
    Crop Recommendation Model
    Trains Random Forest and XGBoost classifiers.
    Saves the best model for offline use.
    """

    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = None
        self.scaler = None

    # ─────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────
    def load_data(self):
        """Load preprocessed train/test data."""
        print("📂 Loading preprocessed crop data...")
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "crop_X_train.csv"))
        X_test  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "crop_X_test.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "crop_y_train.csv")).values.ravel()
        y_test  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "crop_y_test.csv")).values.ravel()

        self.label_encoder = joblib.load(os.path.join(MODELS_PATH, "crop_label_encoder.pkl"))
        self.scaler        = joblib.load(os.path.join(MODELS_PATH, "crop_scaler.pkl"))

        print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ─────────────────────────────────────────────
    # TRAIN RANDOM FOREST
    # ─────────────────────────────────────────────
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier."""
        print("\n🌲 Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        print("   ✅ Random Forest trained!")
        return self.rf_model

    # ─────────────────────────────────────────────
    # TRAIN XGBOOST
    # ─────────────────────────────────────────────
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost classifier."""
        print("\n⚡ Training XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        self.xgb_model.fit(X_train, y_train)
        print("   ✅ XGBoost trained!")
        return self.xgb_model

    # ─────────────────────────────────────────────
    # EVALUATE
    # ─────────────────────────────────────────────
    def evaluate(self, X_test, y_test):
        """Evaluate both models and select the best one."""
        print("\n📊 Evaluating models...")

        results = {}

        for name, model in [("Random Forest", self.rf_model), ("XGBoost", self.xgb_model)]:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {"model": model, "accuracy": acc, "predictions": y_pred}

            print(f"\n── {name} ──")
            print(f"   Accuracy: {acc * 100:.2f}%")
            print(f"   Classification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_
            ))

        # Select best model
        best_name = max(results, key=lambda k: results[k]["accuracy"])
        self.best_model = results[best_name]["model"]
        self.best_model_name = best_name
        print(f"\n🏆 Best Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")
        return results

    # ─────────────────────────────────────────────
    # SAVE MODELS
    # ─────────────────────────────────────────────
    def save_models(self):
        """Save all trained models."""
        os.makedirs(MODELS_PATH, exist_ok=True)
        joblib.dump(self.rf_model,   os.path.join(MODELS_PATH, "crop_rf_model.pkl"))
        joblib.dump(self.xgb_model,  os.path.join(MODELS_PATH, "crop_xgb_model.pkl"))
        joblib.dump(self.best_model, os.path.join(MODELS_PATH, "crop_model.pkl"))

        # Save best model name
        with open(os.path.join(MODELS_PATH, "crop_best_model.txt"), "w") as f:
            f.write(self.best_model_name)

        print(f"\n💾 Models saved to '{MODELS_PATH}/'")
        print(f"   crop_rf_model.pkl")
        print(f"   crop_xgb_model.pkl")
        print(f"   crop_model.pkl  ← best model (used in production)")

    # ─────────────────────────────────────────────
    # PREDICT (used by Flask backend)
    # ─────────────────────────────────────────────
    def predict(self, input_data: dict) -> dict:
        """
        Predict crop from input features.

        Args:
            input_data (dict): Keys — N, P, K, temperature, humidity, ph, rainfall

        Returns:
            dict: crop name, confidence, top 3 crops
        """
        if self.best_model is None:
            self.best_model = joblib.load(os.path.join(MODELS_PATH, "crop_model.pkl"))
        if self.label_encoder is None:
            self.label_encoder = joblib.load(os.path.join(MODELS_PATH, "crop_label_encoder.pkl"))
        if self.scaler is None:
            self.scaler = joblib.load(os.path.join(MODELS_PATH, "crop_scaler.pkl"))

        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([input_data])[feature_cols]
        input_scaled = self.scaler.transform(input_df)

        prediction = self.best_model.predict(input_scaled)[0]
        probabilities = self.best_model.predict_proba(input_scaled)[0]

        crop_name = self.label_encoder.inverse_transform([prediction])[0]
        confidence = round(float(probabilities[prediction]) * 100, 2)

        # Top 3 crops
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3_crops = [
            {
                "crop": self.label_encoder.inverse_transform([i])[0],
                "confidence": round(float(probabilities[i]) * 100, 2)
            }
            for i in top3_indices
        ]

        return {
            "recommended_crop": crop_name,
            "confidence": confidence,
            "top_3_crops": top3_crops
        }

    # ─────────────────────────────────────────────
    # FULL PIPELINE
    # ─────────────────────────────────────────────
    def run_pipeline(self):
        """Run full training pipeline."""
        X_train, X_test, y_train, y_test = self.load_data()
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        results = self.evaluate(X_test, y_test)
        self.save_models()
        return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  CROP RECOMMENDATION MODEL TRAINING")
    print("=" * 50)
    model = CropRecommendationModel()
    model.run_pipeline()
    print("\n🎉 Crop model training complete!")

    # Test prediction
    print("\n🔍 Test Prediction:")
    test_input = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.87, "humidity": 82.0,
        "ph": 6.5, "rainfall": 202.93
    }
    result = model.predict(test_input)
    print(f"   Input: {test_input}")
    print(f"   Recommended Crop: {result['recommended_crop']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Top 3: {result['top_3_crops']}")
