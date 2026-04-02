"""
Advanced Model Training Pipeline for Customer Churn Prediction.

Features:
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Hyperparameter tuning with Optuna
- Stratified K-Fold cross-validation
- SMOTE for class imbalance
- sklearn Pipeline integration
- MLflow experiment tracking
"""

import logging
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

from src.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    CategoricalEncoder,
    FeatureScaler,
    load_and_clean_data,
)
from src.feature_engineering import FeatureEngineer
from src.evaluate import ModelEvaluator

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
    ),
    "xgboost": xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=3, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
    "lightgbm": lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    ),
}


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------

def optuna_xgb_objective(trial, X, y, cv_folds=5):
    """Optuna objective for XGBoost hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }

    model = xgb.XGBClassifier(
        **params, use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def optuna_lgbm_objective(trial, X, y, cv_folds=5):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(
        **params, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def tune_model(model_name, X_train, y_train, n_trials=50):
    """Run Optuna hyperparameter search and return best model."""
    logger.info("Tuning %s with %d Optuna trials...", model_name, n_trials)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if model_name == "xgboost":
        study = optuna.create_study(direction="maximize", study_name="xgb_tune")
        study.optimize(
            lambda trial: optuna_xgb_objective(trial, X_train, y_train),
            n_trials=n_trials,
        )
        best_params = study.best_params
        best_model = xgb.XGBClassifier(
            **best_params, use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    elif model_name == "lightgbm":
        study = optuna.create_study(direction="maximize", study_name="lgbm_tune")
        study.optimize(
            lambda trial: optuna_lgbm_objective(trial, X_train, y_train),
            n_trials=n_trials,
        )
        best_params = study.best_params
        best_model = lgb.LGBMClassifier(
            **best_params, class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )
    else:
        raise ValueError(f"Tuning not implemented for {model_name}")

    logger.info("Best params for %s: %s", model_name, best_params)
    logger.info("Best CV ROC-AUC: %.4f", study.best_value)
    return best_model, best_params, study.best_value


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_model(model, X, y, n_splits=5):
    """Perform stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"roc_auc": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        y_proba = model.predict_proba(X_val_cv)[:, 1]

        metrics["roc_auc"].append(roc_auc_score(y_val_cv, y_proba))
        metrics["f1"].append(f1_score(y_val_cv, y_pred))
        metrics["precision"].append(precision_score(y_val_cv, y_pred))
        metrics["recall"].append(recall_score(y_val_cv, y_pred))

        logger.info("Fold %d - AUC: %.4f, F1: %.4f", fold, metrics["roc_auc"][-1], metrics["f1"][-1])

    summary = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metrics.items()}
    return summary


# ---------------------------------------------------------------------------
# Training Pipeline with SMOTE
# ---------------------------------------------------------------------------

def build_training_pipeline(model, use_smote=True):
    """Build an imblearn pipeline with optional SMOTE."""
    if use_smote:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42, sampling_strategy=0.8)),
            ("model", model),
        ])
    else:
        pipeline = Pipeline([("model", model)])
    return pipeline


# ---------------------------------------------------------------------------
# Main Training Orchestrator
# ---------------------------------------------------------------------------

def train_all_models(
    X_train, y_train, X_test, y_test,
    tune=False, n_trials=50, use_smote=True, track_mlflow=True,
):
    """Train all models, evaluate, and return results."""
    results = {}

    if track_mlflow:
        mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
        mlflow.set_experiment("churn_prediction")

    for name, base_model in MODELS.items():
        logger.info("=" * 60)
        logger.info("Training: %s", name)

        # Optuna tuning for gradient boosting models
        if tune and name in ("xgboost", "lightgbm"):
            model, best_params, best_cv_auc = tune_model(name, X_train, y_train, n_trials)
        else:
            model = base_model
            best_params = model.get_params()

        # Build pipeline with SMOTE
        pipeline = build_training_pipeline(model, use_smote=use_smote)
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results[name] = {
            "pipeline": pipeline,
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": {
                "roc_auc": round(auc, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            },
            "params": best_params,
        }

        logger.info("%s - AUC: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f",
                     name, auc, f1, precision, recall)

        # MLflow tracking
        if track_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({k: str(v)[:250] for k, v in best_params.items()})
                mlflow.log_metrics({
                    "roc_auc": auc, "f1": f1,
                    "precision": precision, "recall": recall,
                })
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return results


def select_best_model(results):
    """Select the model with the highest ROC-AUC."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best = results[best_name]
    logger.info("Best model: %s (AUC=%.4f)", best_name, best["metrics"]["roc_auc"])
    return best_name, best


def save_model(pipeline, metadata, filepath=None):
    """Save model pipeline and metadata with joblib."""
    if filepath is None:
        filepath = MODEL_DIR / "best_model.joblib"

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }
    joblib.dump(artifact, filepath)
    logger.info("Model saved to %s", filepath)

    # Save metrics JSON
    metrics_path = MODEL_DIR / "results.json"
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)
    return filepath


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from sklearn.model_selection import train_test_split

    # Load and preprocess
    df = load_and_clean_data(str(DATA_PATH))
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Feature engineering
    fe = FeatureEngineer(n_segments=4)
    X = fe.fit_transform(X)

    # Encode categoricals
    encoder = CategoricalEncoder(method="onehot")
    encoder.fit(X, y)
    X = encoder.transform(X)

    # Handle missing values & scale
    from src.preprocessing import MissingValueHandler, FeatureScaler
    mv = MissingValueHandler(strategy="median")
    X = mv.fit_transform(X)
    scaler = FeatureScaler(method="robust")
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # Train all models
    results = train_all_models(
        X_train, y_train, X_test, y_test,
        tune=True, n_trials=30, use_smote=True, track_mlflow=True,
    )

    # Select best
    best_name, best = select_best_model(results)

    # Cross-validate best
    cv_results = cross_validate_model(best["model"], X_train, y_train)
    logger.info("CV Results: %s", cv_results)

    # Evaluate
    evaluator = ModelEvaluator(output_dir=str(PROJECT_ROOT / "models" / "plots"))
    evaluator.full_evaluation(y_test, best["y_pred"], best["y_proba"], model_name=best_name)

    # Save
    metadata = {
        "model_name": best_name,
        "metrics": best["metrics"],
        "cv_results": {k: v["mean"] for k, v in cv_results.items()},
        "feature_names": list(X_train.columns),
    }
    save_model(best["pipeline"], metadata)

    # Save preprocessing artifacts
    artifacts = {
        "feature_engineer": fe,
        "encoder": encoder,
        "missing_handler": mv,
        "scaler": scaler,
    }
    joblib.dump(artifacts, MODEL_DIR / "preprocessing_artifacts.joblib")

    logger.info("Training complete!")
