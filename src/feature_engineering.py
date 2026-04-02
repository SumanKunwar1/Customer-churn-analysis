"""
Advanced Feature Engineering for Customer Churn Prediction.

Creates behavioral, temporal, and engagement features that go beyond
basic column transformations.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer advanced features from raw telco churn data.

    Features created:
    - Customer lifetime & tenure metrics
    - Engagement score (composite)
    - Recency-frequency metrics
    - Behavioral segmentation via KMeans clustering
    - Interaction features
    """

    def __init__(self, n_segments=4):
        self.n_segments = n_segments

    def fit(self, X, y=None):
        # Fit KMeans for behavioral segmentation
        seg_features = self._get_segmentation_features(X)
        if seg_features is not None:
            self._seg_scaler = StandardScaler()
            scaled = self._seg_scaler.fit_transform(seg_features)
            self._kmeans = KMeans(
                n_clusters=self.n_segments, random_state=42, n_init=10
            )
            self._kmeans.fit(scaled)
        else:
            self._kmeans = None
        return self

    def transform(self, X):
        X = X.copy()

        # --- Tenure-based features ---
        X["tenure_months"] = X["tenure"].clip(lower=0)
        X["is_new_customer"] = (X["tenure"] <= 6).astype(int)
        X["is_loyal_customer"] = (X["tenure"] >= 48).astype(int)
        X["tenure_squared"] = X["tenure"] ** 2

        # Tenure groups (more granular)
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 6, 12, 24, 36, 48, 60, 72],
            labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
            include_lowest=True,
        )

        # --- Financial features ---
        X["charge_per_month"] = X["TotalCharges"] / (X["tenure"] + 1)
        X["monthly_charge_ratio"] = X["MonthlyCharges"] / (X["MonthlyCharges"].mean() + 1e-8)
        X["total_charge_ratio"] = X["TotalCharges"] / (X["TotalCharges"].mean() + 1e-8)
        X["charge_trend"] = X["MonthlyCharges"] - X["charge_per_month"]
        X["high_value"] = (X["MonthlyCharges"] > X["MonthlyCharges"].quantile(0.75)).astype(int)
        X["low_tenure_high_charge"] = (
            (X["tenure"] <= 12) & (X["MonthlyCharges"] > X["MonthlyCharges"].median())
        ).astype(int)

        # --- Service engagement score ---
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        existing_svc = [c for c in service_cols if c in X.columns]
        if existing_svc:
            X["num_services"] = sum(
                (X[col].isin(["Yes", 1, "1"])).astype(int) for col in existing_svc
            )
            X["engagement_score"] = X["num_services"] / len(existing_svc)
            X["has_premium_support"] = (
                (X.get("OnlineSecurity", "No").isin(["Yes", 1, "1"]))
                & (X.get("TechSupport", "No").isin(["Yes", 1, "1"]))
            ).astype(int)
        else:
            X["num_services"] = 0
            X["engagement_score"] = 0
            X["has_premium_support"] = 0

        # --- Contract & billing risk features ---
        if "Contract" in X.columns:
            X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        if "PaperlessBilling" in X.columns:
            X["paperless_billing"] = (X["PaperlessBilling"].isin(["Yes", 1, "1"])).astype(int)
        if "PaymentMethod" in X.columns:
            X["is_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)

        # --- Composite risk score ---
        X["risk_score"] = (
            X.get("is_month_to_month", 0) * 0.3
            + X.get("is_new_customer", 0) * 0.25
            + X.get("is_electronic_check", 0) * 0.15
            + (1 - X.get("engagement_score", 0)) * 0.2
            + X.get("high_value", 0) * 0.1
        )

        # --- Behavioral segmentation ---
        seg_features = self._get_segmentation_features(X)
        if self._kmeans is not None and seg_features is not None:
            scaled = self._seg_scaler.transform(seg_features)
            X["behavior_segment"] = self._kmeans.predict(scaled)

        logger.info("Engineered %d new features", 15)
        return X

    def _get_segmentation_features(self, X):
        """Extract features used for behavioral segmentation."""
        required = ["tenure", "MonthlyCharges", "TotalCharges"]
        if all(c in X.columns for c in required):
            df = X[required].copy()
            df = df.fillna(df.median())
            return df
        return None


def get_feature_names():
    """Return list of all engineered feature names for documentation."""
    return [
        "tenure_months", "is_new_customer", "is_loyal_customer", "tenure_squared",
        "tenure_group", "charge_per_month", "monthly_charge_ratio",
        "total_charge_ratio", "charge_trend", "high_value",
        "low_tenure_high_charge", "num_services", "engagement_score",
        "has_premium_support", "is_month_to_month", "paperless_billing",
        "is_electronic_check", "risk_score", "behavior_segment",
    ]
