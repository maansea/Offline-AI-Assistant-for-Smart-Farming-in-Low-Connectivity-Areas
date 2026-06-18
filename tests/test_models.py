"""
test_models.py
Member 1 - AI/ML
Unit tests for crop and fertilizer recommendation models.
Run with: pytest tests/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import joblib
import numpy as np
import pandas as pd

from src.crop_model import CropRecommendationModel
from src.fertilizer_model import FertilizerRecommendationModel
from src.recommendation_logic import RecommendationEngine
from src.explainability import ExplainabilityModule

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def crop_model():
    model = CropRecommendationModel()
    return model

@pytest.fixture
def fertilizer_model():
    model = FertilizerRecommendationModel()
    return model

@pytest.fixture
def engine():
    return RecommendationEngine()

@pytest.fixture
def explainer():
    return ExplainabilityModule()

@pytest.fixture
def sample_crop_input():
    return {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.87, "humidity": 82.0,
        "ph": 6.5, "rainfall": 202.93
    }

@pytest.fixture
def sample_fertilizer_input():
    return {
        "Temparature": 26, "Humidity": 52, "Moisture": 38,
        "Soil Type": "Sandy", "Crop Type": "Maize",
        "Nitrogen": 10, "Potassium": 5, "Phosphorous": 8
    }

# ─────────────────────────────────────────────
# TESTS: MODEL FILES EXIST
# ─────────────────────────────────────────────

class TestModelFilesExist:
    """Check all saved model files exist after training."""

    def test_crop_model_exists(self):
        assert os.path.exists("models/crop_model.pkl"), \
            "crop_model.pkl not found. Run crop_model.py first."

    def test_fertilizer_model_exists(self):
        assert os.path.exists("models/fertilizer_model.pkl"), \
            "fertilizer_model.pkl not found. Run fertilizer_model.py first."

    def test_crop_scaler_exists(self):
        assert os.path.exists("models/crop_scaler.pkl"), \
            "crop_scaler.pkl not found."

    def test_fertilizer_scaler_exists(self):
        assert os.path.exists("models/fertilizer_scaler.pkl"), \
            "fertilizer_scaler.pkl not found."

    def test_crop_label_encoder_exists(self):
        assert os.path.exists("models/crop_label_encoder.pkl"), \
            "crop_label_encoder.pkl not found."

    def test_fertilizer_encoders_exist(self):
        assert os.path.exists("models/fertilizer_encoders.pkl"), \
            "fertilizer_encoders.pkl not found."


# ─────────────────────────────────────────────
# TESTS: CROP MODEL
# ─────────────────────────────────────────────

class TestCropModel:
    """Tests for crop recommendation model."""

    def test_crop_predict_returns_dict(self, crop_model, sample_crop_input):
        result = crop_model.predict(sample_crop_input)
        assert isinstance(result, dict), "Prediction should return a dict."

    def test_crop_predict_has_required_keys(self, crop_model, sample_crop_input):
        result = crop_model.predict(sample_crop_input)
        assert "recommended_crop" in result
        assert "confidence" in result
        assert "top_3_crops" in result

    def test_crop_name_is_string(self, crop_model, sample_crop_input):
        result = crop_model.predict(sample_crop_input)
        assert isinstance(result["recommended_crop"], str), \
            "Crop name should be a string."

    def test_crop_confidence_is_valid(self, crop_model, sample_crop_input):
        result = crop_model.predict(sample_crop_input)
        assert 0 <= result["confidence"] <= 100, \
            "Confidence should be between 0 and 100."

    def test_crop_top3_has_3_items(self, crop_model, sample_crop_input):
        result = crop_model.predict(sample_crop_input)
        assert len(result["top_3_crops"]) == 3, \
            "Top 3 crops should contain exactly 3 items."

    def test_crop_top3_confidence_descending(self, crop_model, sample_crop_input):
        result = crop_model.predict(sample_crop_input)
        confidences = [c["confidence"] for c in result["top_3_crops"]]
        assert confidences == sorted(confidences, reverse=True), \
            "Top 3 crops should be sorted by confidence descending."

    def test_crop_predict_different_inputs(self, crop_model):
        """Test with multiple different inputs."""
        inputs = [
            {"N": 20, "P": 15, "K": 20, "temperature": 25, "humidity": 60, "ph": 7.0, "rainfall": 80},
            {"N": 100, "P": 80, "K": 60, "temperature": 30, "humidity": 90, "ph": 5.5, "rainfall": 300},
            {"N": 5, "P": 5, "K": 5, "temperature": 15, "humidity": 40, "ph": 8.0, "rainfall": 30},
        ]
        for inp in inputs:
            result = crop_model.predict(inp)
            assert "recommended_crop" in result
            assert result["confidence"] > 0


# ─────────────────────────────────────────────
# TESTS: FERTILIZER MODEL
# ─────────────────────────────────────────────

class TestFertilizerModel:
    """Tests for fertilizer recommendation model."""

    def test_fertilizer_predict_returns_dict(self, fertilizer_model, sample_fertilizer_input):
        result = fertilizer_model.predict(sample_fertilizer_input)
        assert isinstance(result, dict), "Prediction should return a dict."

    def test_fertilizer_predict_has_required_keys(self, fertilizer_model, sample_fertilizer_input):
        result = fertilizer_model.predict(sample_fertilizer_input)
        assert "recommended_fertilizer" in result
        assert "confidence" in result
        assert "top_3_fertilizers" in result

    def test_fertilizer_name_is_string(self, fertilizer_model, sample_fertilizer_input):
        result = fertilizer_model.predict(sample_fertilizer_input)
        assert isinstance(result["recommended_fertilizer"], str)

    def test_fertilizer_confidence_is_valid(self, fertilizer_model, sample_fertilizer_input):
        result = fertilizer_model.predict(sample_fertilizer_input)
        assert 0 <= result["confidence"] <= 100

    def test_fertilizer_top3_has_3_items(self, fertilizer_model, sample_fertilizer_input):
        result = fertilizer_model.predict(sample_fertilizer_input)
        assert len(result["top_3_fertilizers"]) == 3

    def test_fertilizer_different_soil_types(self, fertilizer_model):
        """Test with different soil types."""
        soil_types = ["Sandy", "Loamy", "Clayey", "Black", "Red"]
        for soil in soil_types:
            inp = {
                "Temparature": 26, "Humidity": 52, "Moisture": 38,
                "Soil Type": soil, "Crop Type": "Wheat",
                "Nitrogen": 30, "Potassium": 20, "Phosphorous": 15
            }
            result = fertilizer_model.predict(inp)
            assert "recommended_fertilizer" in result


# ─────────────────────────────────────────────
# TESTS: RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

class TestRecommendationEngine:
    """Tests for the hybrid recommendation engine."""

    def test_crop_recommendation_returns_dict(self, engine, sample_crop_input):
        result = engine.get_crop_recommendation(sample_crop_input)
        assert isinstance(result, dict)

    def test_crop_recommendation_has_explanation(self, engine, sample_crop_input):
        result = engine.get_crop_recommendation(sample_crop_input)
        assert "explanation" in result
        assert "simple_explanation" in result["explanation"]

    def test_crop_recommendation_with_soil_and_season(self, engine):
        inp = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.87, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.93,
            "soil_type": "Loamy", "season": "Kharif"
        }
        result = engine.get_crop_recommendation(inp)
        assert "rule_confirmations" in result
        assert "rule_warnings" in result

    def test_fertilizer_recommendation_returns_dict(self, engine, sample_fertilizer_input):
        result = engine.get_fertilizer_recommendation(sample_fertilizer_input)
        assert isinstance(result, dict)

    def test_fertilizer_recommendation_has_explanation(self, engine, sample_fertilizer_input):
        result = engine.get_fertilizer_recommendation(sample_fertilizer_input)
        assert "explanation" in result
        assert "simple_explanation" in result["explanation"]

    def test_fertilizer_recommendation_has_rule_suggestions(self, engine, sample_fertilizer_input):
        result = engine.get_fertilizer_recommendation(sample_fertilizer_input)
        assert "rule_based_suggestions" in result
        assert len(result["rule_based_suggestions"]) > 0


# ─────────────────────────────────────────────
# TESTS: EXPLAINABILITY
# ─────────────────────────────────────────────

class TestExplainability:
    """Tests for the explainability module."""

    def test_crop_explanation_returns_string(self, explainer, sample_crop_input):
        explanation = explainer._rule_based_crop_explanation(sample_crop_input, "rice")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_fertilizer_explanation_returns_string(self, explainer, sample_fertilizer_input):
        explanation = explainer._rule_based_fertilizer_explanation(sample_fertilizer_input, "Urea")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_crop_explanation_mentions_crop(self, explainer, sample_crop_input):
        crop_name   = "rice"
        explanation = explainer._rule_based_crop_explanation(sample_crop_input, crop_name)
        assert crop_name.lower() in explanation.lower(), \
            "Explanation should mention the crop name."

    def test_fertilizer_explanation_low_nitrogen(self, explainer):
        """Low nitrogen input should mention nitrogen in explanation."""
        inp = {
            "Temparature": 26, "Humidity": 52, "Moisture": 38,
            "Soil Type": "Sandy", "Crop Type": "Maize",
            "Nitrogen": 5, "Potassium": 5, "Phosphorous": 5
        }
        explanation = explainer._rule_based_fertilizer_explanation(inp, "Urea")
        assert "nitrogen" in explanation.lower(), \
            "Low nitrogen should be mentioned in explanation."

    def test_explain_crop_full(self, explainer, sample_crop_input):
        ml_result = {"confidence": 95.0, "top_3_crops": []}
        result    = explainer.explain_crop(sample_crop_input, "rice", ml_result)
        assert "simple_explanation" in result
        assert "shap_explanation" in result

    def test_explain_fertilizer_full(self, explainer, sample_fertilizer_input):
        ml_result = {"confidence": 90.0, "top_3_fertilizers": []}
        result    = explainer.explain_fertilizer(sample_fertilizer_input, "Urea", ml_result)
        assert "simple_explanation" in result
        assert "shap_explanation" in result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
