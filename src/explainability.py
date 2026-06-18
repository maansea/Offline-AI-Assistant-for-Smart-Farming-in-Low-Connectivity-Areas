"""
explainability.py
Member 1 - AI/ML
Explainability module combining rule-based text explanations with SHAP feature importance.
Every recommendation comes with a clear, simple reason understandable by rural farmers.
"""

import numpy as np
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────
# CROP EXPLANATION TEMPLATES
# ─────────────────────────────────────────────
CROP_EXPLANATIONS = {
    "rice":       "Rice grows well in high humidity and heavy rainfall areas with neutral soil pH.",
    "wheat":      "Wheat thrives in moderate temperature with low humidity and well-drained soil.",
    "maize":      "Maize needs warm temperature, moderate rainfall, and good nitrogen levels.",
    "cotton":     "Cotton requires high temperature, low humidity, and well-drained black soil.",
    "sugarcane":  "Sugarcane grows best in tropical climate with high rainfall and fertile soil.",
    "jute":       "Jute requires high humidity, warm temperature, and waterlogged conditions.",
    "coffee":     "Coffee grows in high humidity, acidic soil with moderate temperature.",
    "tea":        "Tea thrives in acidic soil, high rainfall, and cool temperatures.",
    "banana":     "Banana needs tropical climate, high humidity, and well-drained fertile soil.",
    "mango":      "Mango grows well in tropical climate with dry season for flowering.",
    "grapes":     "Grapes require dry climate, well-drained soil, and warm temperatures.",
    "watermelon": "Watermelon needs warm temperature, sandy soil, and low humidity.",
    "muskmelon":  "Muskmelon grows in warm, dry conditions with well-drained sandy soil.",
    "apple":      "Apple requires cold climate with moderate rainfall and well-drained soil.",
    "orange":     "Orange grows in subtropical climate with moderate temperature and rainfall.",
    "papaya":     "Papaya needs tropical climate, warm temperature, and well-drained fertile soil.",
    "coconut":    "Coconut thrives in coastal areas with high humidity and sandy soil.",
    "groundnuts": "Groundnuts grow best in sandy loam soil with moderate rainfall.",
    "blackgram":  "Blackgram requires warm and humid climate with moderate rainfall.",
    "mungbean":   "Mungbean grows in warm climate with moderate rainfall and well-drained soil.",
    "mothbeans":  "Mothbeans are drought-resistant and grow in dry, sandy soil.",
    "pigeonpeas": "Pigeonpeas tolerate dry conditions and grow in various soil types.",
    "kidneybeans":"Kidneybeans prefer cool climate with moderate rainfall and loamy soil.",
    "chickpea":   "Chickpea grows in cool, dry conditions with well-drained soil.",
    "lentil":     "Lentil needs cool temperature, moderate rainfall, and well-drained loamy soil.",
    "pomegranate":"Pomegranate grows in semi-arid climate with well-drained soil.",
}

# ─────────────────────────────────────────────
# FERTILIZER EXPLANATION TEMPLATES
# ─────────────────────────────────────────────
FERTILIZER_EXPLANATIONS = {
    "Urea":        "Urea is a nitrogen-rich fertilizer. It promotes leaf and stem growth.",
    "DAP":         "DAP (Di-Ammonium Phosphate) is rich in phosphorus and nitrogen. It supports root development and early plant growth.",
    "MOP":         "MOP (Muriate of Potash) supplies potassium. It improves crop quality, disease resistance, and water efficiency.",
    "14-35-14 NPK":"This balanced NPK fertilizer supports overall plant growth with emphasis on phosphorus.",
    "10-26-26 NPK":"This NPK fertilizer is high in phosphorus and potassium, ideal for root and fruit development.",
    "17-17-17 NPK":"This balanced NPK fertilizer provides equal amounts of nitrogen, phosphorus, and potassium.",
    "20-20 NPK":   "This NPK blend supports vegetative and root growth equally.",
    "28-28 NPK":   "A concentrated NPK fertilizer for fast-growing crops needing quick nutrition.",
    "TSP":         "Triple Super Phosphate supplies high phosphorus for root and flowering development.",
    "SSP":         "Single Super Phosphate provides phosphorus and sulfur for overall crop health.",
}

# ─────────────────────────────────────────────
# FEATURE LABELS (human-readable)
# ─────────────────────────────────────────────
CROP_FEATURE_LABELS = {
    "N":           "Nitrogen level",
    "P":           "Phosphorus level",
    "K":           "Potassium level",
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "ph":          "Soil pH",
    "rainfall":    "Rainfall",
}

FERTILIZER_FEATURE_LABELS = {
    "Temparature":  "Temperature",
    "Humidity":     "Humidity",
    "Moisture":     "Soil Moisture",
    "Soil Type":    "Soil Type",
    "Crop Type":    "Crop Type",
    "Nitrogen":     "Nitrogen level",
    "Potassium":    "Potassium level",
    "Phosphorous":  "Phosphorous level",
}


class ExplainabilityModule:
    """
    Explainability Module
    Provides:
    1. Simple rule-based text explanations for farmers
    2. SHAP-based feature importance for technical reports
    """

    def __init__(self):
        self.crop_model_path       = os.path.join("models", "crop_model.pkl")
        self.fertilizer_model_path = os.path.join("models", "fertilizer_model.pkl")
        self.crop_scaler_path      = os.path.join("models", "crop_scaler.pkl")
        self.fertilizer_scaler_path= os.path.join("models", "fertilizer_scaler.pkl")

    # ─────────────────────────────────────────────
    # RULE-BASED EXPLANATION: CROP
    # ─────────────────────────────────────────────
    def _rule_based_crop_explanation(self, input_data: dict, crop_name: str) -> str:
        """Generate simple human-readable explanation for crop recommendation."""
        reasons = []

        # Nitrogen
        n = input_data.get("N", 0)
        if n > 60:
            reasons.append(f"Nitrogen level is high ({n}), suitable for leafy crops like {crop_name}.")
        elif n < 20:
            reasons.append(f"Nitrogen level is low ({n}). {crop_name} can grow with less nitrogen.")

        # Rainfall
        rain = input_data.get("rainfall", 0)
        if rain > 150:
            reasons.append(f"High rainfall ({rain:.1f}mm) favors water-intensive crops like {crop_name}.")
        elif rain < 60:
            reasons.append(f"Low rainfall ({rain:.1f}mm) — {crop_name} is drought-tolerant.")

        # Temperature
        temp = input_data.get("temperature", 0)
        if temp > 30:
            reasons.append(f"High temperature ({temp:.1f}°C) is ideal for tropical crops like {crop_name}.")
        elif temp < 15:
            reasons.append(f"Cool temperature ({temp:.1f}°C) suits crops like {crop_name}.")

        # Humidity
        hum = input_data.get("humidity", 0)
        if hum > 70:
            reasons.append(f"High humidity ({hum:.1f}%) supports moisture-loving crops like {crop_name}.")

        # pH
        ph = input_data.get("ph", 7)
        if ph < 5.5:
            reasons.append(f"Acidic soil (pH {ph:.1f}) is suitable for crops like {crop_name}.")
        elif ph > 7.5:
            reasons.append(f"Alkaline soil (pH {ph:.1f}) suits crops like {crop_name}.")
        else:
            reasons.append(f"Neutral soil pH ({ph:.1f}) is ideal for most crops including {crop_name}.")

        # Add general crop knowledge
        general = CROP_EXPLANATIONS.get(crop_name.lower(), "")
        if general:
            reasons.append(general)

        return " ".join(reasons) if reasons else f"{crop_name} is recommended based on your soil and climate conditions."

    # ─────────────────────────────────────────────
    # RULE-BASED EXPLANATION: FERTILIZER
    # ─────────────────────────────────────────────
    def _rule_based_fertilizer_explanation(self, input_data: dict, fertilizer_name: str) -> str:
        """Generate simple human-readable explanation for fertilizer recommendation."""
        reasons = []

        nitrogen    = input_data.get("Nitrogen",    0)
        phosphorous = input_data.get("Phosphorous", 0)
        potassium   = input_data.get("Potassium",   0)

        if nitrogen < 20:
            reasons.append(f"Nitrogen is very low ({nitrogen}). Your crop needs nitrogen to grow healthy leaves.")
        elif nitrogen < 40:
            reasons.append(f"Nitrogen is moderate ({nitrogen}). Some nitrogen supplement is recommended.")

        if phosphorous < 15:
            reasons.append(f"Phosphorous is low ({phosphorous}). Roots and flowers need more phosphorous.")

        if potassium < 10:
            reasons.append(f"Potassium is low ({potassium}). Potassium helps the crop fight disease and drought.")

        # Add general fertilizer knowledge
        general = FERTILIZER_EXPLANATIONS.get(fertilizer_name, "")
        if general:
            reasons.append(general)

        return " ".join(reasons) if reasons else f"{fertilizer_name} is recommended based on your soil nutrient levels."

    # ─────────────────────────────────────────────
    # SHAP EXPLANATION: CROP
    # ─────────────────────────────────────────────
    def _shap_crop_explanation(self, input_data: dict) -> dict:
        """Generate SHAP feature importance for crop recommendation."""
        try:
            import shap
            model  = joblib.load(self.crop_model_path)
            scaler = joblib.load(self.crop_scaler_path)

            feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            input_df     = pd.DataFrame([input_data])[feature_cols]
            input_scaled = scaler.transform(input_df)

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)

            # Get prediction class index
            pred_class  = model.predict(input_scaled)[0]
            shap_for_pred = shap_values[pred_class][0] if isinstance(shap_values, list) else shap_values[0]

            feature_importance = {
                CROP_FEATURE_LABELS.get(col, col): round(float(val), 4)
                for col, val in zip(feature_cols, shap_for_pred)
            }

            # Sort by absolute importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            )

            top_feature = list(sorted_importance.keys())[0]
            return {
                "shap_values":   sorted_importance,
                "top_feature":   top_feature,
                "shap_summary":  f"The most influential factor was '{top_feature}'.",
            }

        except Exception as e:
            return {"shap_values": {}, "top_feature": "N/A", "shap_summary": f"SHAP unavailable: {str(e)}"}

    # ─────────────────────────────────────────────
    # SHAP EXPLANATION: FERTILIZER
    # ─────────────────────────────────────────────
    def _shap_fertilizer_explanation(self, input_data: dict) -> dict:
        """Generate SHAP feature importance for fertilizer recommendation."""
        try:
            import shap
            model    = joblib.load(self.fertilizer_model_path)
            scaler   = joblib.load(self.fertilizer_scaler_path)
            encoders = joblib.load(os.path.join("models", "fertilizer_encoders.pkl"))

            input_copy = input_data.copy()
            for col in ['Soil Type', 'Crop Type']:
                if col in encoders and col in input_copy:
                    input_copy[col] = encoders[col].transform([str(input_copy[col])])[0]

            feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type',
                            'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
            feature_cols = [col for col in feature_cols if col in input_copy]

            input_df     = pd.DataFrame([input_copy])[feature_cols]
            input_scaled = scaler.transform(input_df)

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)

            pred_class    = model.predict(input_scaled)[0]
            shap_for_pred = shap_values[pred_class][0] if isinstance(shap_values, list) else shap_values[0]

            feature_importance = {
                FERTILIZER_FEATURE_LABELS.get(col, col): round(float(val), 4)
                for col, val in zip(feature_cols, shap_for_pred)
            }

            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            )

            top_feature = list(sorted_importance.keys())[0]
            return {
                "shap_values":  sorted_importance,
                "top_feature":  top_feature,
                "shap_summary": f"The most influential factor was '{top_feature}'.",
            }

        except Exception as e:
            return {"shap_values": {}, "top_feature": "N/A", "shap_summary": f"SHAP unavailable: {str(e)}"}

    # ─────────────────────────────────────────────
    # PUBLIC: EXPLAIN CROP
    # ─────────────────────────────────────────────
    def explain_crop(self, input_data: dict, crop_name: str, ml_result: dict) -> dict:
        """
        Full explanation for crop recommendation.
        Returns both simple (farmer-friendly) and technical (SHAP) explanation.
        """
        simple = self._rule_based_crop_explanation(input_data, crop_name)
        shap   = self._shap_crop_explanation(input_data)

        return {
            "simple_explanation": simple,
            "shap_explanation":   shap,
            "confidence":         ml_result.get("confidence", 0),
            "top_3_crops":        ml_result.get("top_3_crops", []),
        }

    # ─────────────────────────────────────────────
    # PUBLIC: EXPLAIN FERTILIZER
    # ─────────────────────────────────────────────
    def explain_fertilizer(self, input_data: dict, fertilizer_name: str, ml_result: dict) -> dict:
        """
        Full explanation for fertilizer recommendation.
        Returns both simple (farmer-friendly) and technical (SHAP) explanation.
        """
        simple = self._rule_based_fertilizer_explanation(input_data, fertilizer_name)
        shap   = self._shap_fertilizer_explanation(input_data)

        return {
            "simple_explanation":  simple,
            "shap_explanation":    shap,
            "confidence":          ml_result.get("confidence", 0),
            "top_3_fertilizers":   ml_result.get("top_3_fertilizers", []),
        }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  EXPLAINABILITY MODULE TEST")
    print("=" * 50)

    explainer = ExplainabilityModule()

    # Test crop explanation
    crop_input = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.87, "humidity": 82.0,
        "ph": 6.5, "rainfall": 202.93
    }
    crop_explanation = explainer._rule_based_crop_explanation(crop_input, "rice")
    print(f"\n🌾 Crop Explanation:\n   {crop_explanation}")

    # Test fertilizer explanation
    fert_input = {
        "Temparature": 26, "Humidity": 52, "Moisture": 38,
        "Soil Type": "Sandy", "Crop Type": "Maize",
        "Nitrogen": 10, "Potassium": 5, "Phosphorous": 8
    }
    fert_explanation = explainer._rule_based_fertilizer_explanation(fert_input, "Urea")
    print(f"\n🧪 Fertilizer Explanation:\n   {fert_explanation}")
