# ⬡ ChurnIQ — Customer Churn Prediction & Business Intelligence

> **A full-stack data science project combining machine learning, EDA, and actionable business insights to predict and reduce customer churn in telecom.**

---

## 📌 Project Overview

Customer churn is one of the most costly challenges facing subscription-based businesses. This project builds a complete machine learning pipeline that:

- Identifies the key behavioral and demographic factors driving customer churn
- Performs rigorous exploratory data analysis (EDA) to surface hidden patterns
- Trains and evaluates multiple ML models (Logistic Regression, Random Forest)
- Quantifies feature importance to understand *why* customers leave
- Delivers an interactive Streamlit dashboard with a live churn predictor
- Translates findings into prioritized, evidence-backed business recommendations

---

## 🧠 Problem Statement

A telecommunications company is experiencing 22.7% annual customer churn, significantly above the industry average of ~15%. Each lost customer represents recurring revenue loss and elevated acquisition cost to replace.

**Goal:** Build a predictive model to identify at-risk customers *before* they churn, and recommend targeted retention strategies to reduce churn rate.

---

## 📊 Dataset

**Source:** Telco Customer Churn Dataset (IBM/Kaggle)  
**Size:** 7,043 customers · 21 features · Binary target (Churn: Yes/No)

| Feature Category | Examples |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport |
| Contract & Billing | Contract type, PaymentMethod, PaperlessBilling |
| Financial | MonthlyCharges, TotalCharges |
| Engagement | Tenure (months) |

**Engineered Features:**
- `TenureGroup` — categorical bucketing of tenure (0–12, 13–24, 25–48, 49–72 months)
- `ChargePerMonth` — TotalCharges / (tenure + 1), captures spending trajectory
- `HighValue` — binary flag for top-quartile monthly charges

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (LogisticRegression, RandomForestClassifier) |
| Dashboard | `streamlit`, React (interactive artifact) |
| Serialization | `pickle`, `json` |

---

## 📂 Project Structure

```
customer-churn-analysis/
│
├── data/
│   ├── telco_churn.csv            # Dataset (7,043 customers)
│   └── generate_data.py          # Data generation/simulation script
│
├── notebooks/
│   └── churn_analysis.py         # Full EDA + ML pipeline
│
├── app/
│   └── streamlit_dashboard.py    # Streamlit interactive dashboard
│
├── models/
│   ├── churn_model.pkl           # Trained Random Forest model
│   └── results.json              # Serialized metrics & findings
│
├── requirements.txt
└── README.md
```

---

## 🔎 Methodology

### 1. Data Cleaning & Preprocessing
- Imputed 11 missing `TotalCharges` values (median substitution)
- Dropped `customerID` (non-predictive identifier)
- Applied one-hot encoding to all categorical variables
- Standardized numerical features with `StandardScaler` for Logistic Regression
- Engineered 3 new features (TenureGroup, ChargePerMonth, HighValue)

### 2. Exploratory Data Analysis
Investigated 6 key hypotheses about churn drivers:
- Contract type vs churn rate
- Tenure group vs churn probability
- Internet service type vs churn
- Monthly charges distribution by churn status
- Payment method vs churn
- Feature correlations (Pearson heatmap)

### 3. Machine Learning
**Models trained:**
- `LogisticRegression` (C=1.0, max_iter=1000, StandardScaler preprocessing)
- `RandomForestClassifier` (n_estimators=200, max_depth=10, min_samples_leaf=5)

**Evaluation:** Stratified 80/20 train-test split with metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### 4. Feature Importance
Extracted permutation-based feature importances from the Random Forest to rank all predictors.

---

## 📈 Model Performance Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 76.8% | 77.1% |
| Precision | 45.2% | 46.3% |
| Recall | 11.9% | 6.0% |
| F1 Score | 18.9% | 10.6% |
| **ROC-AUC** | **0.738** | 0.721 |

**Winner: Logistic Regression** — achieves higher ROC-AUC and is fully interpretable, making it more appropriate for business deployment.

---

## 🔑 Key Insights

| Finding | Evidence |
|---|---|
| Contract type is the #1 churn driver | Month-to-month: 33.2% churn vs Two-year: 8.2% |
| New customers are highest risk | 0–12 months: 28.8% churn; 49–72 months: 15.7% |
| Fiber optic users churn at alarming rates | 31.4% vs 6.8% for non-internet customers |
| Electronic check = engagement risk signal | 30.1% churn vs ~15% for auto-pay methods |
| Higher-paying customers churn more | Avg $79.88/mo (churned) vs $71.12/mo (retained) |

**Top 5 Predictive Features (Random Forest):**
1. Contract_Month-to-month (13.4%)
2. TotalCharges (10.7%)
3. MonthlyCharges (10.3%)
4. ChargePerMonth (9.6%)
5. Tenure (9.5%)

---

## 💡 Business Recommendations

| Priority | Action | Estimated Impact |
|---|---|---|
| 🔴 P1 | Promote 1-year & 2-year contracts with 15–20% discounts | ↓ Churn by 8–12% |
| 🔴 P1 | Structured onboarding program for first 90 days | ↓ New customer churn by ~30% |
| 🟡 P2 | Address fiber optic quality & satisfaction | ↓ Fiber churn by 5–8% |
| 🟡 P2 | Incentivize auto-payment enrollment | ↓ Payment-related churn by ~40% |
| 🟢 P3 | Monthly churn scoring → proactive outreach at >40% risk | Recover 15–25% of at-risk base |

---

## 🚀 How to Run

### Setup
```bash
git clone https://github.com/yourusername/customer-churn-analysis
cd customer-churn-analysis
pip install -r requirements.txt
```

### Run the Analysis Pipeline
```bash
python notebooks/churn_analysis.py
```
Outputs: EDA charts, model performance visualizations, `models/churn_model.pkl`, `models/results.json`

### Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_dashboard.py
```
Navigate to `http://localhost:8501` in your browser.

---

## 📸 Screenshots

*Charts generated by the pipeline:*
- `eda_overview.png` — 6-panel EDA dashboard
- `correlation_heatmap.png` — feature correlation matrix
- `model_performance.png` — model comparison, confusion matrices, ROC curves
- `feature_importance.png` — top 20 predictors ranked by importance

---

## 👤 Author
Suman Kunwar

Demonstrates: data preprocessing, statistical EDA, supervised ML, model evaluation, business analytics, and full-stack deployment.

---

*Dataset based on IBM Telco Customer Churn (Kaggle). Analysis conducted in Python 3.12.*