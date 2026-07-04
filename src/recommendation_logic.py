"""
recommendation_logic.py
Member 1 - AI/ML
Hybrid recommendation engine combining ML predictions with rule-based agricultural knowledge.
This is the main entry point used by Member 2's Flask backend.
"""

import os
import joblib
from pathlib import Path

from src.crop_model import CropRecommendationModel
from src.fertilizer_model import FertilizerRecommendationModel
from src.explainability import ExplainabilityModule

# ─────────────────────────────────────────────
# RULE-BASED KNOWLEDGE BASE
# ─────────────────────────────────────────────

# Soil type to suitable crops mapping
SOIL_CROP_RULES = {
    "Sandy":    ["groundnut", "watermelon", "carrot", "potato"],
    "Loamy":    ["wheat", "maize", "rice", "sugarcane", "cotton"],
    "Clayey":   ["rice", "jute", "sugarcane"],
    "Black":    ["cotton", "soybean", "sunflower", "jowar"],
    "Red":      ["groundnut", "millets", "tobacco", "potato"],
    "Alluvial": ["wheat", "rice", "sugarcane", "maize", "pulses"],
}

# Season to suitable crops mapping
SEASON_CROP_RULES = {
    "Kharif":  ["rice", "maize", "cotton", "sugarcane", "groundnut", "soybean"],
    "Rabi":    ["wheat", "barley", "mustard", "pea", "lentil"],
    "Zaid":    ["watermelon", "muskmelon", "cucumber", "moong"],
    "Summer":  ["watermelon", "cucumber", "moong", "sunflower"],
    "Winter":  ["wheat", "mustard", "pea", "potato", "carrot"],
    "Monsoon": ["rice", "maize", "soybean", "cotton", "groundnut"],
}

# Nutrient deficiency rules for fertilizer
NUTRIENT_RULES = {
    "low_nitrogen":    {"fertilizer": "Urea",        "reason": "Nitrogen is low. Urea boosts nitrogen for leaf and stem growth."},
    "low_phosphorous": {"fertilizer": "DAP",          "reason": "Phosphorous is low. DAP supports root development and flowering."},
    "low_potassium":   {"fertilizer": "MOP",          "reason": "Potassium is low. MOP improves crop quality and disease resistance."},
    "low_npk":         {"fertilizer": "10-26-26 NPK", "reason": "All nutrients are low. Balanced NPK fertilizer is recommended."},
    "balanced":        {"fertilizer": "14-35-14 NPK", "reason": "Nutrients are moderately balanced. Use NPK for maintenance."},
}

# Thresholds for nutrient levels
NUTRIENT_THRESHOLDS = {
    "nitrogen":    {"low": 20,  "high": 80},
    "phosphorous": {"low": 15,  "high": 60},
    "potassium":   {"low": 10,  "high": 50},
}


class RecommendationEngine:
    """
    Hybrid Recommendation Engine
    Combines ML model predictions with rule-based agricultural knowledge.
    Returns recommendations + explanations for use by Flask backend.
    """

    def __init__(self):
        self.crop_model       = CropRecommendationModel()
        self.fertilizer_model = FertilizerRecommendationModel()
        self.explainer        = ExplainabilityModule()
        self._models_loaded   = False

    # ─────────────────────────────────────────────
    # LOAD MODELS
    # ─────────────────────────────────────────────
    def load_models(self):
        """Load saved ML models when available, otherwise fall back to rule-based recommendations."""
        if self._models_loaded:
            return

        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "models"

        print("🔄 Loading models...")
        model_files = {
            "crop_model": model_dir / "crop_model.pkl",
            "crop_label_encoder": model_dir / "crop_label_encoder.pkl",
            "crop_scaler": model_dir / "crop_scaler.pkl",
            "fertilizer_model": model_dir / "fertilizer_model.pkl",
            "fertilizer_encoders": model_dir / "fertilizer_encoders.pkl",
            "fertilizer_scaler": model_dir / "fertilizer_scaler.pkl",
        }

        if all(path.exists() for path in model_files.values()):
            self.crop_model.best_model = joblib.load(model_files["crop_model"])
            self.crop_model.label_encoder = joblib.load(model_files["crop_label_encoder"])
            self.crop_model.scaler = joblib.load(model_files["crop_scaler"])
            self.fertilizer_model.best_model = joblib.load(model_files["fertilizer_model"])
            self.fertilizer_model.encoders = joblib.load(model_files["fertilizer_encoders"])
            self.fertilizer_model.scaler = joblib.load(model_files["fertilizer_scaler"])
            self._models_loaded = True
            print("✅ Models loaded!")
        else:
            self._models_loaded = True
            print("⚠️ Model files are not available. Falling back to rule-based recommendations.")

    # ─────────────────────────────────────────────
    # RULE-BASED CROP VALIDATION
    # ─────────────────────────────────────────────
    def _apply_crop_rules(self, ml_crop: str, soil_type: str = None, season: str = None) -> dict:
        """
        Validate ML crop prediction against agricultural rules.
        Returns rule-based confirmation or alternative suggestion.
        """
        rule_flags = []
        rule_warnings = []

        if soil_type and soil_type in SOIL_CROP_RULES:
            suitable = SOIL_CROP_RULES[soil_type]
            if ml_crop.lower() in [c.lower() for c in suitable]:
                rule_flags.append(f"✅ {ml_crop} is well-suited for {soil_type} soil.")
            else:
                rule_warnings.append(
                    f"⚠️ {ml_crop} may not be ideal for {soil_type} soil. "
                    f"Consider: {', '.join(suitable[:3])}."
                )

        if season and season in SEASON_CROP_RULES:
            suitable = SEASON_CROP_RULES[season]
            if ml_crop.lower() in [c.lower() for c in suitable]:
                rule_flags.append(f"✅ {ml_crop} is a good {season} season crop.")
            else:
                rule_warnings.append(
                    f"⚠️ {ml_crop} is not typically grown in {season} season. "
                    f"Consider: {', '.join(suitable[:3])}."
                )

        return {"confirmations": rule_flags, "warnings": rule_warnings}

    # ─────────────────────────────────────────────
    # RULE-BASED FERTILIZER VALIDATION
    # ─────────────────────────────────────────────
    def _apply_fertilizer_rules(self, nitrogen: float, phosphorous: float, potassium: float) -> dict:
        """
        Apply nutrient threshold rules to validate/supplement ML fertilizer prediction.
        """
        n_low  = nitrogen    < NUTRIENT_THRESHOLDS["nitrogen"]["low"]
        p_low  = phosphorous < NUTRIENT_THRESHOLDS["phosphorous"]["low"]
        k_low  = potassium   < NUTRIENT_THRESHOLDS["potassium"]["low"]

        rule_suggestions = []

        if n_low and p_low and k_low:
            rule_suggestions.append(NUTRIENT_RULES["low_npk"])
        elif n_low:
            rule_suggestions.append(NUTRIENT_RULES["low_nitrogen"])
        elif p_low:
            rule_suggestions.append(NUTRIENT_RULES["low_phosphorous"])
        elif k_low:
            rule_suggestions.append(NUTRIENT_RULES["low_potassium"])
        else:
            rule_suggestions.append(NUTRIENT_RULES["balanced"])

        return {"rule_based_suggestions": rule_suggestions}

    # ─────────────────────────────────────────────
    # MAIN: GET CROP RECOMMENDATION
    # ─────────────────────────────────────────────
    def get_crop_recommendation(self, input_data: dict) -> dict:
        """
        Full crop recommendation pipeline.

        Args:
            input_data (dict): {
                N, P, K, temperature, humidity, ph, rainfall,
                soil_type (optional), season (optional)
            }

        Returns:
            dict: Full recommendation with ML result + rules + explanation
        """
        self.load_models()

        # Extract optional rule-based inputs
        soil_type = input_data.pop("soil_type", None)
        season = input_data.pop("season", None)

        if all(path.exists() for path in [
            Path(__file__).resolve().parent.parent / "models" / "crop_model.pkl",
            Path(__file__).resolve().parent.parent / "models" / "crop_label_encoder.pkl",
            Path(__file__).resolve().parent.parent / "models" / "crop_scaler.pkl",
        ]):
            ml_result = self.crop_model.predict(input_data)
            crop_name = ml_result["recommended_crop"]
            confidence = ml_result["confidence"]
            top_3_crops = ml_result["top_3_crops"]
            explanation = self.explainer.explain_crop(input_data, crop_name, ml_result)
        else:
            crop_name = self._fallback_crop_recommendation(soil_type=soil_type, season=season)
            confidence = 70.0
            top_3_crops = [{"crop": crop_name, "confidence": confidence}]
            explanation = {
                "simple_explanation": f"Using offline rule-based guidance, {crop_name} is suggested for the provided conditions.",
                "feature_importance": {},
            }

        rules = self._apply_crop_rules(crop_name, soil_type, season)

        return {
            "type": "crop",
            "recommended_crop": crop_name,
            "confidence": confidence,
            "top_3_crops": top_3_crops,
            "rule_confirmations": rules["confirmations"],
            "rule_warnings": rules["warnings"],
            "explanation": explanation,
        }

    # ─────────────────────────────────────────────
    # MAIN: GET FERTILIZER RECOMMENDATION
    # ─────────────────────────────────────────────
    def get_fertilizer_recommendation(self, input_data: dict) -> dict:
        """
        Full fertilizer recommendation pipeline.

        Args:
            input_data (dict): {
                Temparature, Humidity, Moisture,
                Soil Type, Crop Type,
                Nitrogen, Potassium, Phosphorous
            }

        Returns:
            dict: Full recommendation with ML result + rules + explanation
        """
        self.load_models()

        model_dir = Path(__file__).resolve().parent.parent / "models"
        has_fertilizer_models = all(
            (model_dir / filename).exists()
            for filename in ["fertilizer_model.pkl", "fertilizer_encoders.pkl", "fertilizer_scaler.pkl"]
        )

        if has_fertilizer_models:
            ml_result = self.fertilizer_model.predict(input_data)
            fertilizer_name = ml_result["recommended_fertilizer"]
            confidence = ml_result["confidence"]
            top_3_fertilizers = ml_result["top_3_fertilizers"]
            explanation = self.explainer.explain_fertilizer(input_data, fertilizer_name, ml_result)
        else:
            fertilizer_name = self._fallback_fertilizer_recommendation(input_data)
            confidence = 70.0
            top_3_fertilizers = [{"fertilizer": fertilizer_name, "confidence": confidence}]
            explanation = {
                "simple_explanation": f"Using offline rule-based guidance, {fertilizer_name} is recommended for the current soil and crop conditions.",
                "feature_importance": {},
            }

        rules = self._apply_fertilizer_rules(
            nitrogen=input_data.get("Nitrogen", 0),
            phosphorous=input_data.get("Phosphorous", 0),
            potassium=input_data.get("Potassium", 0),
        )

        return {
            "type": "fertilizer",
            "recommended_fertilizer": fertilizer_name,
            "confidence": confidence,
            "top_3_fertilizers": top_3_fertilizers,
            "rule_based_suggestions": rules["rule_based_suggestions"],
            "explanation": explanation,
        }

    # ─────────────────────────────────────────────
    # COMBINED: FULL FARM RECOMMENDATION
    # ─────────────────────────────────────────────
    def get_full_recommendation(self, crop_input: dict, fertilizer_input: dict) -> dict:
        """
        Get both crop and fertilizer recommendations in one call.
        Used by Flask backend for complete farm advisory.
        """
        crop_result       = self.get_crop_recommendation(crop_input)
        fertilizer_result = self.get_fertilizer_recommendation(fertilizer_input)

        return {
            "crop_recommendation": crop_result,
            "fertilizer_recommendation": fertilizer_result,
        }

    def _fallback_crop_recommendation(self, soil_type: str = None, season: str = None) -> str:
        """Suggest a crop from rule-based knowledge when no ML model is available."""
        if soil_type and soil_type in SOIL_CROP_RULES:
            candidates = SOIL_CROP_RULES[soil_type]
            if season and season in SEASON_CROP_RULES:
                overlap = [c for c in candidates if c.lower() in [s.lower() for s in SEASON_CROP_RULES[season]]]
                if overlap:
                    return overlap[0].capitalize()
            return candidates[0].capitalize()
        if season and season in SEASON_CROP_RULES:
            return SEASON_CROP_RULES[season][0].capitalize()
        return "Maize"

    def _fallback_fertilizer_recommendation(self, input_data: dict) -> str:
        """Suggest a fertilizer from simple nutrient rules when no ML model is available."""
        nitrogen = input_data.get("Nitrogen", 0)
        phosphorous = input_data.get("Phosphorous", 0)
        potassium = input_data.get("Potassium", 0)

        if nitrogen < 20 and phosphorous < 15 and potassium < 10:
            return "10-26-26 NPK"
        if nitrogen < 20:
            return "Urea"
        if phosphorous < 15:
            return "DAP"
        if potassium < 10:
            return "MOP"
        return "14-35-14 NPK"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  RECOMMENDATION ENGINE TEST")
    print("=" * 50)

    engine = RecommendationEngine()

    # Test crop recommendation
    crop_input = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.87, "humidity": 82.0,
        "ph": 6.5, "rainfall": 202.93,
        "soil_type": "Loamy", "season": "Kharif"
    }

    print("\n🌾 CROP RECOMMENDATION:")
    crop_result = engine.get_crop_recommendation(crop_input)
    print(f"   Crop       : {crop_result['recommended_crop']}")
    print(f"   Confidence : {crop_result['confidence']}%")
    print(f"   Explanation: {crop_result['explanation']['simple_explanation']}")
    for c in crop_result['rule_confirmations']:
        print(f"   {c}")
    for w in crop_result['rule_warnings']:
        print(f"   {w}")

    # Test fertilizer recommendation
    fert_input = {
        "Temparature": 26, "Humidity": 52, "Moisture": 38,
        "Soil Type": "Sandy", "Crop Type": "Maize",
        "Nitrogen": 10, "Potassium": 5, "Phosphorous": 8
    }

    print("\n🧪 FERTILIZER RECOMMENDATION:")
    fert_result = engine.get_fertilizer_recommendation(fert_input)
    print(f"   Fertilizer : {fert_result['recommended_fertilizer']}")
    print(f"   Confidence : {fert_result['confidence']}%")
    print(f"   Explanation: {fert_result['explanation']['simple_explanation']}")
    for s in fert_result['rule_based_suggestions']:
        print(f"   Rule: {s['reason']}")
