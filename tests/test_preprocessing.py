"""Unit tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    CategoricalEncoder,
    FeatureScaler,
    load_and_clean_data,
)


@pytest.fixture
def sample_df():
    """Create a sample dataframe mimicking telco churn data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "tenure": np.random.randint(0, 72, n),
        "MonthlyCharges": np.random.uniform(18, 120, n),
        "TotalCharges": np.random.uniform(18, 8000, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
    })


@pytest.fixture
def df_with_missing(sample_df):
    """Inject missing values into sample data."""
    df = sample_df.copy()
    df.loc[0:5, "TotalCharges"] = np.nan
    df.loc[10:12, "MonthlyCharges"] = np.nan
    df.loc[20:22, "Contract"] = np.nan
    return df


class TestMissingValueHandler:
    def test_median_imputation(self, df_with_missing):
        handler = MissingValueHandler(strategy="median")
        result = handler.fit_transform(df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_mean_imputation(self, df_with_missing):
        handler = MissingValueHandler(strategy="mean")
        result = handler.fit_transform(df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_knn_imputation(self, df_with_missing):
        handler = MissingValueHandler(strategy="knn", knn_neighbors=3)
        result = handler.fit_transform(df_with_missing)
        assert result.select_dtypes(include=[np.number]).isnull().sum().sum() == 0

    def test_shape_preserved(self, df_with_missing):
        handler = MissingValueHandler(strategy="median")
        result = handler.fit_transform(df_with_missing)
        assert result.shape == df_with_missing.shape


class TestOutlierHandler:
    def test_outliers_capped(self, sample_df):
        # Inject extreme outliers
        df = sample_df.copy()
        df.loc[0, "MonthlyCharges"] = 99999
        df.loc[1, "MonthlyCharges"] = -9999

        handler = OutlierHandler(factor=1.5)
        result = handler.fit_transform(df)
        assert result["MonthlyCharges"].max() < 99999
        assert result["MonthlyCharges"].min() > -9999

    def test_shape_preserved(self, sample_df):
        handler = OutlierHandler(factor=1.5)
        result = handler.fit_transform(sample_df)
        assert result.shape == sample_df.shape


class TestCategoricalEncoder:
    def test_onehot_encoding(self, sample_df):
        y = sample_df["Churn"]
        X = sample_df.drop(columns=["Churn"])
        encoder = CategoricalEncoder(method="onehot")
        encoder.fit(X, y)
        result = encoder.transform(X)
        # Should have more columns after one-hot encoding
        assert result.shape[1] > X.shape[1] - 2  # -2 cat cols + new dummies
        # No object columns should remain
        assert len(result.select_dtypes(include=["object"]).columns) == 0

    def test_no_object_columns_remain(self, sample_df):
        y = sample_df["Churn"]
        X = sample_df.drop(columns=["Churn"])
        encoder = CategoricalEncoder(method="onehot")
        encoder.fit(X, y)
        result = encoder.transform(X)
        assert result.select_dtypes(include=["object"]).shape[1] == 0


class TestFeatureScaler:
    def test_standard_scaling(self, sample_df):
        scaler = FeatureScaler(method="standard")
        result = scaler.fit_transform(sample_df)
        # Scaled numeric columns should have ~0 mean
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(result[col].mean()) < 0.5  # Approximately 0

    def test_robust_scaling(self, sample_df):
        scaler = FeatureScaler(method="robust")
        result = scaler.fit_transform(sample_df)
        assert result.shape == sample_df.shape


class TestLoadAndCleanData:
    def test_loads_successfully(self, tmp_path):
        # Create a minimal CSV
        df = pd.DataFrame({
            "customerID": ["C001", "C002", "C003"],
            "tenure": [1, 24, 60],
            "MonthlyCharges": [50.0, 70.0, 90.0],
            "TotalCharges": ["50", "1680", " "],
            "Churn": ["Yes", "No", "No"],
        })
        filepath = tmp_path / "test.csv"
        df.to_csv(filepath, index=False)

        result = load_and_clean_data(str(filepath))
        assert "customerID" not in result.columns
        assert result["Churn"].dtype in [np.int64, np.int32, int]
        assert result["TotalCharges"].dtype == np.float64
