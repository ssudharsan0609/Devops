import os
import joblib
import logging
import pandas as pd
from typing import Dict, Any, Tuple
from app.config import settings

logger = logging.getLogger("mlops_app")


class ModelPredictor:
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.class_labels = {0: "Malignant", 1: "Benign"}

    def load_models(self):
        """Loads regression and classification model pipelines from disk."""
        # Load Regression Model
        if os.path.exists(settings.REGRESSION_MODEL_PATH):
            try:
                self.regression_model = joblib.load(settings.REGRESSION_MODEL_PATH)
                logger.info(f"Successfully loaded regression model from: {settings.REGRESSION_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load regression model: {e}")
        else:
            logger.warning(f"Regression model not found at path: {settings.REGRESSION_MODEL_PATH}")

        # Load Classification Model
        if os.path.exists(settings.CLASSIFICATION_MODEL_PATH):
            try:
                self.classification_model = joblib.load(settings.CLASSIFICATION_MODEL_PATH)
                logger.info(f"Successfully loaded classification model from: {settings.CLASSIFICATION_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load classification model: {e}")
        else:
            logger.warning(f"Classification model not found at path: {settings.CLASSIFICATION_MODEL_PATH}")

    def predict_regression(self, input_data: Dict[str, float]) -> float:
        """Runs inference using the loaded Regression model pipeline."""
        if self.regression_model is None:
            raise RuntimeError("Regression model is not loaded.")

        df = pd.DataFrame([input_data])
        prediction = self.regression_model.predict(df)[0]
        return float(round(prediction, 4))

    def predict_classification(self, input_data: Dict[str, float]) -> Tuple[int, str, float, Dict[str, float]]:
        """Runs inference using the loaded Classification model pipeline."""
        if self.classification_model is None:
            raise RuntimeError("Classification model is not loaded.")

        # Convert underscore JSON keys (mean_radius) to dataset feature names (mean radius)
        mapped_input = {k.replace("_", " "): v for k, v in input_data.items()}
        df = pd.DataFrame([mapped_input])
        prediction_id = int(self.classification_model.predict(df)[0])
        probabilities = self.classification_model.predict_proba(df)[0]

        pred_name = self.class_labels.get(prediction_id, "Unknown")
        confidence = float(round(max(probabilities), 4))

        prob_dict = {
            "Malignant": float(round(probabilities[0], 4)),
            "Benign": float(round(probabilities[1], 4))
        }

        return prediction_id, pred_name, confidence, prob_dict



# Global predictor instance
predictor = ModelPredictor()
