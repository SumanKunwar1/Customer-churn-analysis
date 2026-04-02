"""
Advanced Data Preprocessing Pipeline for Customer Churn Prediction.

Handles missing values (multiple strategies), outlier detection/treatment,
categorical encoding (target encoding, one-hot), and feature scaling.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from category_encoders import TargetEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Missing Value Handlers
# ---------------------------------------------------------------------------

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values using multiple strategies.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'knn', 'most_frequent'.
    knn_neighbors : int
        Number of neighbors for KNN imputation.
    """

    def __init__(self, strategy="median", knn_neighbors=5):
        self.strategy = strategy
        self.knn_neighbors = knn_neighbors

    def fit(self, X, y=None):
        if self.strategy == "knn":
            self._imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self._imputer = SimpleImputer(strategy=self.strategy)

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self._numeric_cols = numeric_cols
        self._imputer.fit(X[numeric_cols])

        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        self._cat_cols = cat_cols
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        if len(cat_cols) > 0:
            self._cat_imputer.fit(X[cat_cols])

        return self

    def transform(self, X):
        X = X.copy()
        if len(self._numeric_cols) > 0:
            X[self._numeric_cols] = self._imputer.transform(X[self._numeric_cols])
        if len(self._cat_cols) > 0:
            X[self._cat_cols] = self._cat_imputer.transform(X[self._cat_cols])
        logger.info("Missing values handled with strategy='%s'", self.strategy)
        return X


# ---------------------------------------------------------------------------
# Outlier Detection & Treatment
# ---------------------------------------------------------------------------

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Detect and cap outliers using IQR method.

    Parameters
    ----------
    factor : float
        IQR multiplier (default 1.5).
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self._numeric_cols = numeric_cols
        self._bounds = {}
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self._bounds[col] = (Q1 - self.factor * IQR, Q3 + self.factor * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self._bounds.items():
            if col in X.columns:
                outliers = ((X[col] < lower) | (X[col] > upper)).sum()
                X[col] = X[col].clip(lower, upper)
                if outliers > 0:
                    logger.info("Capped %d outliers in '%s'", outliers, col)
        return X


# ---------------------------------------------------------------------------
# Categorical Encoding
# ---------------------------------------------------------------------------

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables.

    Parameters
    ----------
    method : str
        'onehot' or 'target'.
    target_encode_cols : list or None
        Columns for target encoding (high-cardinality). Others get one-hot.
    """

    def __init__(self, method="onehot", target_encode_cols=None):
        self.method = method
        self.target_encode_cols = target_encode_cols or []

    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._cat_cols = cat_cols

        if self.method == "target" and y is not None and self.target_encode_cols:
            te_cols = [c for c in self.target_encode_cols if c in cat_cols]
            oh_cols = [c for c in cat_cols if c not in te_cols]
        else:
            te_cols = []
            oh_cols = cat_cols

        self._te_cols = te_cols
        self._oh_cols = oh_cols

        if te_cols:
            self._target_encoder = TargetEncoder(cols=te_cols, smoothing=0.3)
            self._target_encoder.fit(X[te_cols], y)

        if oh_cols:
            self._oh_encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
            self._oh_encoder.fit(X[oh_cols])
            self._oh_feature_names = self._oh_encoder.get_feature_names_out(oh_cols)

        return self

    def transform(self, X):
        X = X.copy()

        if self._te_cols:
            X[self._te_cols] = self._target_encoder.transform(X[self._te_cols])

        if self._oh_cols:
            oh_encoded = self._oh_encoder.transform(X[self._oh_cols])
            oh_df = pd.DataFrame(oh_encoded, columns=self._oh_feature_names, index=X.index)
            X = X.drop(columns=self._oh_cols)
            X = pd.concat([X, oh_df], axis=1)

        logger.info("Encoded %d categorical columns", len(self._cat_cols))
        return X


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------

class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scale numerical features.

    Parameters
    ----------
    method : str
        'standard' or 'robust'.
    """

    def __init__(self, method="robust"):
        self.method = method

    def fit(self, X, y=None):
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.method == "robust":
            self._scaler = RobustScaler()
        else:
            self._scaler = StandardScaler()
        self._scaler.fit(X[self._numeric_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self._numeric_cols] = self._scaler.transform(X[self._numeric_cols])
        return X


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load the Telco churn dataset and perform initial cleaning."""
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset: %s rows, %s columns", *df.shape)

    # Fix TotalCharges - coerce errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID (non-predictive)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target
    if df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    logger.info("Churn distribution:\n%s", df["Churn"].value_counts(normalize=True))
    return df


def build_preprocessing_pipeline(strategy="median", scaling="robust", encoding="onehot"):
    """Return ordered list of preprocessing transformers."""
    from sklearn.pipeline import Pipeline

    steps = [
        ("missing", MissingValueHandler(strategy=strategy)),
        ("outliers", OutlierHandler(factor=1.5)),
        ("scaler", FeatureScaler(method=scaling)),
    ]
    return Pipeline(steps)
