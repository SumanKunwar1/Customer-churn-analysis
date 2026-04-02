"""
Production Prediction Module for Customer Churn.

Features:
- Load saved model + preprocessing artifacts
- Predict on new data (single or batch)
- Structured logging
- SHAP explanations for each prediction
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def setup_logging(level=logging.INFO):
    """Configure structured logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)


class ChurnPredictor:
    """Production-ready churn prediction service."""

    def __init__(self, model_path=None, artifacts_path=None):
        self.model_path = model_path or MODEL_DIR / "best_model.joblib"
        self.artifacts_path = artifacts_path or MODEL_DIR / "preprocessing_artifacts.joblib"
        self._load_model()

    def _load_model(self):
        """Load trained model and preprocessing artifacts."""
        logger.info("Loading model from %s", self.model_path)
        model_artifact = joblib.load(self.model_path)
        self.pipeline = model_artifact["pipeline"]
        self.metadata = model_artifact["metadata"]
        self.feature_names = self.metadata.get("feature_names", [])

        logger.info("Loading preprocessing artifacts from %s", self.artifacts_path)
        artifacts = joblib.load(self.artifacts_path)
        self.feature_engineer = artifacts["feature_engineer"]
        self.encoder = artifacts["encoder"]
        self.missing_handler = artifacts["missing_handler"]
        self.scaler = artifacts["scaler"]

        # Initialize SHAP explainer
        try:
            model = self.pipeline.named_steps.get("model", self.pipeline)
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = None
            logger.warning("Could not initialize SHAP explainer")

        logger.info("Model loaded: %s (AUC=%.4f)",
                     self.metadata.get("model_name", "unknown"),
                     self.metadata.get("metrics", {}).get("roc_auc", 0))

    def preprocess(self, df):
        """Apply full preprocessing pipeline to raw input data."""
        df = self.missing_handler.transform(df)
        df = self.feature_engineer.transform(df)
        df = self.encoder.transform(df)
        df = self.scaler.transform(df)

        # Align columns with training data
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        return df

    def predict(self, df, threshold=0.5):
        """Predict churn for input dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Raw customer data (before preprocessing).
        threshold : float
            Classification threshold.

        Returns
        -------
        pd.DataFrame with columns: churn_probability, churn_prediction, risk_level
        """
        logger.info("Predicting for %d customers", len(df))

        X = self.preprocess(df)
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        result = pd.DataFrame({
            "churn_probability": np.round(probabilities, 4),
            "churn_prediction": predictions,
            "risk_level": pd.cut(
                probabilities,
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=["Low", "Medium", "High", "Critical"],
            ),
        })

        logger.info("Predictions: %d high-risk, %d medium, %d low",
                     (result["risk_level"].isin(["High", "Critical"])).sum(),
                     (result["risk_level"] == "Medium").sum(),
                     (result["risk_level"] == "Low").sum())
        return result

    def predict_single(self, customer_data: dict, threshold=0.5):
        """Predict churn for a single customer.

        Parameters
        ----------
        customer_data : dict
            Customer attributes.

        Returns
        -------
        dict with probability, prediction, risk_level, and top_drivers.
        """
        df = pd.DataFrame([customer_data])
        X = self.preprocess(df)

        proba = self.pipeline.predict_proba(X)[:, 1][0]
        prediction = int(proba >= threshold)

        risk_level = (
            "Critical" if proba >= 0.75
            else "High" if proba >= 0.5
            else "Medium" if proba >= 0.25
            else "Low"
        )

        result = {
            "churn_probability": round(float(proba), 4),
            "churn_prediction": prediction,
            "risk_level": risk_level,
        }

        # SHAP explanation
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_vals = shap_values[0]
                feature_impact = sorted(
                    zip(self.feature_names, shap_vals),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:5]
                result["top_drivers"] = [
                    {"feature": f, "impact": round(float(v), 4)} for f, v in feature_impact
                ]
            except Exception as e:
                logger.warning("SHAP explanation failed: %s", e)

        logger.info("Single prediction: prob=%.4f, risk=%s", proba, risk_level)
        return result

    def batch_predict(self, filepath, output_path=None, threshold=0.5):
        """Run batch predictions on a CSV file."""
        logger.info("Batch prediction from %s", filepath)
        df = pd.read_csv(filepath)

        # Preserve identifiers
        id_col = None
        if "customerID" in df.columns:
            id_col = df["customerID"]
            df = df.drop(columns=["customerID"])
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])

        result = self.predict(df, threshold)

        if id_col is not None:
            result.insert(0, "customerID", id_col.values)

        if output_path:
            result.to_csv(output_path, index=False)
            logger.info("Batch results saved to %s", output_path)

        return result


if __name__ == "__main__":
    setup_logging()
    predictor = ChurnPredictor()

    # Example: batch prediction
    result = predictor.batch_predict(
        str(PROJECT_ROOT / "data" / "telco_churn.csv"),
        output_path=str(MODEL_DIR / "batch_predictions.csv"),
    )
    print(f"\nBatch prediction complete: {len(result)} customers scored")
    print(result["risk_level"].value_counts())
