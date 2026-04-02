"""Unit tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "tenure": np.random.randint(0, 72, n),
        "MonthlyCharges": np.random.uniform(18, 120, n),
        "TotalCharges": np.random.uniform(18, 8000, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "InternetService": np.random.choice(["Fiber optic", "DSL", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
    })


class TestFeatureEngineer:
    def test_creates_expected_features(self, sample_df):
        fe = FeatureEngineer(n_segments=3)
        result = fe.fit_transform(sample_df)

        expected_cols = [
            "is_new_customer", "is_loyal_customer", "tenure_squared",
            "charge_per_month", "engagement_score", "num_services",
            "risk_score", "behavior_segment", "high_value",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_engagement_score_range(self, sample_df):
        fe = FeatureEngineer(n_segments=3)
        result = fe.fit_transform(sample_df)
        assert result["engagement_score"].min() >= 0
        assert result["engagement_score"].max() <= 1

    def test_risk_score_range(self, sample_df):
        fe = FeatureEngineer(n_segments=3)
        result = fe.fit_transform(sample_df)
        assert result["risk_score"].min() >= 0
        assert result["risk_score"].max() <= 1

    def test_behavior_segments(self, sample_df):
        fe = FeatureEngineer(n_segments=4)
        result = fe.fit_transform(sample_df)
        assert result["behavior_segment"].nunique() <= 4

    def test_binary_flags(self, sample_df):
        fe = FeatureEngineer(n_segments=3)
        result = fe.fit_transform(sample_df)
        assert set(result["is_new_customer"].unique()).issubset({0, 1})
        assert set(result["is_loyal_customer"].unique()).issubset({0, 1})
        assert set(result["is_month_to_month"].unique()).issubset({0, 1})
