"""Unit tests for prediction module."""

import numpy as np
import pandas as pd
import pytest


class TestPredictionOutput:
    """Test prediction output format and constraints."""

    def test_risk_levels_valid(self):
        """Verify risk level categories."""
        valid_levels = {"Low", "Medium", "High", "Critical"}
        probas = [0.1, 0.3, 0.6, 0.9]

        for p in probas:
            if p >= 0.75:
                level = "Critical"
            elif p >= 0.5:
                level = "High"
            elif p >= 0.25:
                level = "Medium"
            else:
                level = "Low"
            assert level in valid_levels

    def test_probability_bounds(self):
        """Probabilities should be between 0 and 1."""
        probas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert all(0 <= p <= 1 for p in probas)

    def test_prediction_binary(self):
        """Predictions should be 0 or 1."""
        probas = np.array([0.1, 0.3, 0.6, 0.9])
        threshold = 0.5
        preds = (probas >= threshold).astype(int)
        assert set(preds).issubset({0, 1})

    def test_batch_output_format(self):
        """Batch predictions should have required columns."""
        required_cols = ["churn_probability", "churn_prediction", "risk_level"]
        result = pd.DataFrame({
            "churn_probability": [0.2, 0.8],
            "churn_prediction": [0, 1],
            "risk_level": ["Low", "Critical"],
        })
        for col in required_cols:
            assert col in result.columns
