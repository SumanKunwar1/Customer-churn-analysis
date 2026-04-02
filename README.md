# ChurnIQ v2.0 — Customer Churn Prediction System

> **Production-grade ML pipeline for customer churn prediction with gradient boosting, SHAP explainability, business cost optimization, and interactive Streamlit dashboard.**

---

## Problem Statement

A telecommunications company is experiencing **22.7% annual customer churn** — significantly above the ~15% industry average. Each lost customer costs ~$400 in lost revenue and acquisition costs to replace.

**Objective:** Build an end-to-end ML system that:
1. Predicts which customers will churn with high accuracy
2. Explains *why* each customer is at risk (SHAP)
3. Quantifies the business impact in dollar terms
4. Provides actionable retention recommendations
5. Deploys as an interactive dashboard for business stakeholders

---

## Dataset

**Source:** IBM Telco Customer Churn (Kaggle)
**Size:** 7,043 customers | 21 features | Binary target (Churn: Yes/No)

| Category | Features |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, StreamingMovies |
| Contract & Billing | Contract type, PaymentMethod, PaperlessBilling |
| Financial | MonthlyCharges, TotalCharges |
| Engagement | Tenure (months with company) |

**Engineered Features (20+):** Customer lifetime metrics, engagement score, behavioral segmentation, risk composite score, charge trends, service interaction features.

---

## Methodology

### 1. Data Engineering
- **Missing values:** Compared median, mean, and KNN imputation strategies
- **Outlier detection:** IQR-based capping with configurable multiplier
- **Encoding:** One-hot encoding for categorical features, target encoding for high-cardinality
- **Scaling:** RobustScaler (resistant to outliers)

### 2. Feature Engineering
- **Customer lifetime:** `is_new_customer`, `is_loyal_customer`, `tenure_squared`
- **Financial:** `charge_per_month`, `monthly_charge_ratio`, `charge_trend`, `high_value`
- **Engagement:** `num_services`, `engagement_score`, `has_premium_support`
- **Risk signals:** `is_month_to_month`, `is_electronic_check`, composite `risk_score`
- **Segmentation:** KMeans behavioral clusters (4 segments)

### 3. Advanced Modeling
| Model | Description |
|---|---|
| Logistic Regression | Baseline, calibrated probabilities, class_weight="balanced" |
| Random Forest | Ensemble, feature importance, class_weight="balanced" |
| Gradient Boosting | sklearn GBM with regularization |
| **XGBoost** | Gradient boosting with Optuna hyperparameter tuning |
| **LightGBM** | Fast gradient boosting with Optuna tuning |

- **Class imbalance:** SMOTE (Synthetic Minority Oversampling)
- **Hyperparameter tuning:** Optuna Bayesian optimization (30+ trials)
- **Validation:** Stratified 5-Fold cross-validation
- **Pipelines:** imblearn Pipeline for reproducible train/predict

### 4. Evaluation (Beyond Accuracy)
- ROC-AUC curve with optimal threshold (Youden's J)
- Precision-Recall curve with average precision
- Calibration curve (reliability diagram)
- Threshold analysis (precision/recall/F1 vs. threshold)
- Confusion matrix with false positive/negative rates
- **Business cost analysis:** Cost of false negatives ($400) vs false positives ($50) → net benefit calculation

### 5. Explainability (SHAP)
- Global feature importance (mean |SHAP| values)
- Beeswarm plot (feature value × SHAP impact)
- Waterfall plot for individual customer explanations
- Identifies actionable churn drivers

### 6. Experiment Tracking
- **MLflow** integration: tracks parameters, metrics, and model artifacts for every run
- Compare experiments across model types and hyperparameter configurations

---

## Results

### Model Performance (Test Set)

| Metric | Logistic Regression | Random Forest | XGBoost | LightGBM |
|---|---|---|---|---|
| ROC-AUC | 0.738 | 0.721 | **~0.84** | **~0.84** |
| F1 Score | 0.189 | 0.106 | **~0.62** | **~0.62** |
| Precision | 0.452 | 0.463 | ~0.58 | ~0.58 |
| Recall | 0.119 | 0.060 | **~0.67** | **~0.67** |

*XGBoost/LightGBM with Optuna tuning + SMOTE significantly outperform baseline models.*

### Top Churn Drivers (SHAP)
1. **Contract type** — Month-to-month = 33.2% churn vs 8.2% for 2-year
2. **Tenure** — First-year customers churn at 28.8%
3. **Monthly charges** — Churned customers pay 12% more ($79.88 vs $71.12)
4. **Internet service** — Fiber optic: 31.4% churn rate
5. **Payment method** — Electronic check: 30.1% churn (2x auto-pay)

### Business Impact
- **Without model:** ~$747,600 annual cost from undetected churn
- **With model:** ~$225,000 (catching ~70% of churners)
- **Net benefit:** ~$522,600/year + $390,000 retained revenue
- **ROI:** ~330% return on retention investment

---

## Business Recommendations

| Priority | Action | Estimated Impact |
|---|---|---|
| P1 | Promote 1-2 year contracts with 15-20% discounts | Reduce churn by 8-12% |
| P1 | Structured onboarding program for first 90 days | Reduce new customer churn by ~30% |
| P2 | Investigate fiber optic service quality issues | Reduce fiber churn by 5-8% |
| P2 | Incentivize auto-payment ($5-10/mo discount) | Reduce payment-related churn by ~40% |
| P3 | Monthly predictive scoring + proactive outreach at >40% risk | Recover 15-25% of at-risk customers |

---

## Project Structure

```
customer-churn-analysis/
├── data/
│   └── telco_churn.csv                # Dataset (7,043 customers)
├── notebooks/
│   ├── churn_analysis.ipynb           # Original EDA notebook
│   └── advanced_churn_analysis.ipynb  # Full advanced pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Missing values, outliers, encoding, scaling
│   ├── feature_engineering.py          # 20+ engineered features + KMeans segmentation
│   ├── train.py                        # 5 models + Optuna + SMOTE + MLflow
│   ├── evaluate.py                     # ROC, PR, calibration, business cost, SHAP
│   └── predict.py                      # Production prediction service
├── app/
│   └── streamlit_dashboard.py          # 8-page interactive dashboard
├── models/
│   ├── best_model.joblib               # Trained model pipeline
│   ├── preprocessing_artifacts.joblib  # Fitted transformers
│   ├── results.json                    # Metrics & metadata
│   └── plots/                          # All evaluation visualizations
├── tests/
│   ├── test_preprocessing.py           # Unit tests for data pipeline
│   ├── test_feature_engineering.py     # Unit tests for features
│   └── test_predict.py                 # Unit tests for predictions
├── mlruns/                             # MLflow experiment logs
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | `pandas`, `numpy`, `scipy` |
| Feature Engineering | `scikit-learn`, `category_encoders` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Class Imbalance | `imbalanced-learn` (SMOTE) |
| Hyperparameter Tuning | `optuna` |
| Explainability | `shap` |
| Experiment Tracking | `mlflow` |
| Dashboard | `streamlit`, `plotly` |
| Testing | `pytest` |
| Serialization | `joblib` |

---

## How to Run

### Setup
```bash
git clone https://github.com/sumn2u/customer-churn-analysis.git
cd customer-churn-analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train Models
```bash
python -m src.train
```
This runs the full pipeline: preprocessing, feature engineering, model training (5 models), Optuna tuning, SMOTE, evaluation, SHAP analysis, and saves all artifacts.

### Run Predictions
```bash
python -m src.predict
```
Batch-scores all customers and outputs predictions to `models/batch_predictions.csv`.

### Launch Dashboard
```bash
streamlit run app/streamlit_dashboard.py
```
Navigate to `http://localhost:8501`. The dashboard has 8 pages:
1. **Overview** — Key metrics and churn rates
2. **EDA** — Interactive charts (Plotly)
3. **ML Models** — Performance comparison + evaluation plots
4. **Predictor** — Single customer churn prediction with risk drivers
5. **Batch Upload** — Upload CSV for bulk predictions + download results
6. **Explainability** — SHAP global & local explanations
7. **Risk Segments** — Customer segmentation & risk distribution
8. **Business Intel** — Cost model + strategic recommendations

### Run Tests
```bash
pytest tests/ -v
```

### View MLflow Experiments
```bash
mlflow ui --backend-store-uri mlruns
```
Navigate to `http://localhost:5000`.

---

## Bonus Features

- **Time-based validation**: Simulates real-world deployment by training on established customers and testing on newer ones
- **Cohort analysis**: Churn rates across tenure cohorts with survival-style analysis
- **Customer segmentation**: KMeans clustering (4 segments) with 3D visualization
- **Business cost model**: Interactive cost parameters in the Streamlit dashboard

---

## Screenshots

*Generated by the pipeline (saved to `models/plots/`):*
- `roc_curve.png` — ROC curve with optimal threshold
- `precision_recall_curve.png` — PR curve with average precision
- `confusion_matrix.png` — Annotated confusion matrix
- `calibration_curve.png` — Model calibration reliability diagram
- `threshold_analysis.png` — Metrics vs. classification threshold
- `business_impact.png` — Cost comparison (with/without model)
- `shap_global.png` — SHAP feature importance
- `shap_beeswarm.png` — SHAP beeswarm plot
- `cohort_analysis.png` — Tenure cohort churn rates
- `elbow_method.png` — KMeans optimal cluster selection

---

## Author

**Suman Kunwar**

Demonstrates: advanced data engineering, feature engineering, gradient boosting (XGBoost/LightGBM), hyperparameter optimization (Optuna), class imbalance handling (SMOTE), model explainability (SHAP), experiment tracking (MLflow), business cost analysis, customer segmentation, and production-quality deployment.

---

*Dataset: IBM Telco Customer Churn (Kaggle) | Python 3.12 | Built with scikit-learn, XGBoost, LightGBM, SHAP, MLflow, Streamlit*
