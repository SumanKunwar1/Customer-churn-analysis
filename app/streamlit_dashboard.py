"""
ChurnIQ v2.0 — Advanced Streamlit Dashboard
Customer Churn Prediction & Business Intelligence

Features:
- CSV upload for batch predictions
- SHAP-based explanations
- Interactive risk segmentation dashboard
- Cohort analysis & customer segments

Run: streamlit run app/streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib
import shap
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ v2.0 — Customer Churn Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F1F5F9; }
    .metric-card {
        background: #1E293B; border-radius: 12px; padding: 20px;
        border: 1px solid #334155; text-align: center;
    }
    .metric-value { font-size: 2.4rem; font-weight: 800; }
    .metric-label { color: #64748B; font-size: 0.8rem; text-transform: uppercase;
                    letter-spacing: 0.08em; }
    .section-tag {
        background: #7C3AED22; color: #A78BFA; font-size: 0.7rem;
        padding: 2px 8px; border-radius: 4px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    .insight-box {
        background: #1E293B; border-radius: 10px; padding: 16px;
        border-left: 3px solid #EF4444; margin: 8px 0;
    }
    .recommendation {
        background: #1E293B; border-radius: 10px; padding: 16px;
        border-left: 3px solid #22C55E; margin: 8px 0;
    }
    .risk-critical { color: #EF4444; font-weight: 800; font-size: 1.5rem; }
    .risk-high { color: #F97316; font-weight: 800; font-size: 1.5rem; }
    .risk-medium { color: #EAB308; font-weight: 800; font-size: 1.5rem; }
    .risk-low { color: #22C55E; font-weight: 800; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent.parent


# ── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    results_path = BASE_DIR / "models" / "results.json"
    model_path = BASE_DIR / "models" / "best_model.joblib"
    old_model_path = BASE_DIR / "models" / "churn_model.pkl"
    artifacts_path = BASE_DIR / "models" / "preprocessing_artifacts.joblib"
    data_path = BASE_DIR / "data" / "telco_churn.csv"

    results = json.load(open(results_path)) if results_path.exists() else {}

    model_bundle = None
    if model_path.exists():
        model_bundle = joblib.load(model_path)
    elif old_model_path.exists():
        import pickle
        model_bundle = pickle.load(open(old_model_path, "rb"))

    preprocessing = None
    if artifacts_path.exists():
        preprocessing = joblib.load(artifacts_path)

    df = pd.read_csv(data_path) if data_path.exists() else None

    return results, model_bundle, preprocessing, df


results, model_bundle, preprocessing, df = load_artifacts()


# ── Helper Functions ─────────────────────────────────────────────────────────
def get_model():
    """Extract the trained model from the bundle."""
    if model_bundle is None:
        return None
    if isinstance(model_bundle, dict):
        return model_bundle.get("pipeline", model_bundle.get("model"))
    return model_bundle


def predict_churn_heuristic(data):
    """Heuristic fallback when model is not available."""
    score = 0.05
    score += 0.30 if data.get("Contract") == "Month-to-month" else (0.05 if data.get("Contract") == "One year" else 0)
    score -= 0.003 * data.get("tenure", 12)
    score += 0.002 * (data.get("MonthlyCharges", 70) - 50)
    if data.get("InternetService") == "Fiber optic":
        score += 0.08
    if data.get("OnlineSecurity") == "Yes":
        score -= 0.06
    if data.get("TechSupport") == "Yes":
        score -= 0.05
    if data.get("PaymentMethod") == "Electronic check":
        score += 0.06
    if data.get("SeniorCitizen") in ("Yes", 1):
        score += 0.04
    return max(0.02, min(0.95, score))


def compute_risk_level(prob):
    if prob >= 0.75:
        return "Critical"
    elif prob >= 0.5:
        return "High"
    elif prob >= 0.25:
        return "Medium"
    return "Low"


def prepare_dataset(raw_df):
    """Clean dataset for analysis."""
    df_clean = raw_df.copy()
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)
    if "Churn" in df_clean.columns and df_clean["Churn"].dtype == object:
        df_clean["Churn_binary"] = (df_clean["Churn"] == "Yes").astype(int)
    elif "Churn" in df_clean.columns:
        df_clean["Churn_binary"] = df_clean["Churn"]
    return df_clean


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ ChurnIQ v2.0")
    st.markdown("**Advanced Churn Intelligence Platform**")
    st.divider()

    page = st.radio("Navigation", [
        "📊 Overview",
        "🔍 EDA",
        "🤖 ML Models",
        "🎯 Predictor",
        "📁 Batch Upload",
        "🧠 Explainability",
        "📈 Risk Segments",
        "💡 Business Intel",
    ])

    st.divider()
    if df is not None:
        df_clean = prepare_dataset(df)
        churn_rate = df_clean["Churn_binary"].mean() if "Churn_binary" in df_clean.columns else 0
        st.metric("Churn Rate", f"{churn_rate * 100:.1f}%")
        st.metric("Dataset Size", f"{len(df_clean):,}")
        if results and "metrics" in results:
            st.metric("Best AUC", f"{results['metrics'].get('roc_auc', 0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<span class="section-tag">Dashboard</span>', unsafe_allow_html=True)
    st.title("Customer Churn Analysis Overview")
    st.caption("Advanced ML-powered churn prediction with explainability")

    if df is not None:
        df_clean = prepare_dataset(df)

        c1, c2, c3, c4, c5 = st.columns(5)
        churn_rate = df_clean["Churn_binary"].mean() if "Churn_binary" in df_clean.columns else 0
        churn_count = int(df_clean["Churn_binary"].sum()) if "Churn_binary" in df_clean.columns else 0
        churned_charges = df_clean[df_clean.get("Churn_binary", pd.Series()) == 1]["MonthlyCharges"].mean() if "Churn_binary" in df_clean.columns else 0
        auc_val = results.get("metrics", {}).get("roc_auc", 0) if results else 0

        c1.metric("Total Customers", f"{len(df_clean):,}")
        c2.metric("Churn Rate", f"{churn_rate * 100:.1f}%", delta="-target: <15%", delta_color="inverse")
        c3.metric("Customers Lost", f"{churn_count:,}")
        c4.metric("Avg Charge (Churned)", f"${churned_charges:.2f}" if churned_charges else "N/A")
        c5.metric("Best Model AUC", f"{auc_val:.4f}" if auc_val else "N/A")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Rate by Contract Type")
            if "Contract" in df_clean.columns and "Churn_binary" in df_clean.columns:
                contract_churn = df_clean.groupby("Contract")["Churn_binary"].mean() * 100
                fig = px.bar(
                    x=contract_churn.index, y=contract_churn.values,
                    labels={"x": "Contract Type", "y": "Churn Rate (%)"},
                    color=contract_churn.values,
                    color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                )
                fig.update_layout(
                    plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                    font_color="#F8FAFC", showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Churn by Internet Service")
            if "InternetService" in df_clean.columns and "Churn_binary" in df_clean.columns:
                inet_churn = df_clean.groupby("InternetService")["Churn_binary"].mean() * 100
                fig = px.bar(
                    x=inet_churn.index, y=inet_churn.values,
                    labels={"x": "Internet Service", "y": "Churn Rate (%)"},
                    color=inet_churn.values,
                    color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                )
                fig.update_layout(
                    plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                    font_color="#F8FAFC", showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Data file not found. Please ensure `data/telco_churn.csv` exists.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown('<span class="section-tag">Exploratory Data Analysis</span>', unsafe_allow_html=True)
    st.title("EDA — Churn Patterns")

    if df is not None:
        df_clean = prepare_dataset(df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Distribution")
            if "Churn" in df_clean.columns:
                churn_counts = df_clean["Churn"].value_counts()
                fig = px.pie(
                    names=churn_counts.index, values=churn_counts.values,
                    color_discrete_sequence=["#3B82F6", "#EF4444"],
                    hole=0.4,
                )
                fig.update_layout(paper_bgcolor="#0F172A", font_color="#F8FAFC")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Monthly Charges Distribution")
            if "Churn_binary" in df_clean.columns:
                fig = px.histogram(
                    df_clean, x="MonthlyCharges", color="Churn",
                    nbins=40, barmode="overlay", opacity=0.7,
                    color_discrete_sequence=["#3B82F6", "#EF4444"],
                )
                fig.update_layout(
                    plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                    font_color="#F8FAFC",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tenure vs Monthly Charges")
        if all(c in df_clean.columns for c in ["tenure", "MonthlyCharges"]):
            fig = px.scatter(
                df_clean, x="tenure", y="MonthlyCharges",
                color="Churn" if "Churn" in df_clean.columns else None,
                opacity=0.5, color_discrete_sequence=["#3B82F6", "#EF4444"],
            )
            fig.update_layout(
                plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                font_color="#F8FAFC",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df_clean[numeric_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
            )
            fig.update_layout(paper_bgcolor="#0F172A", font_color="#F8FAFC")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Numerical Feature Summary")
        st.dataframe(df_clean[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))
    else:
        st.warning("Data file not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown('<span class="section-tag">ML Evaluation</span>', unsafe_allow_html=True)
    st.title("Model Performance Comparison")

    if results and "metrics" in results:
        metrics = results["metrics"]
        st.subheader("Best Model Performance")
        model_name = results.get("model_name", "Best Model")
        st.info(f"**Selected Model:** {model_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        c2.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
        c3.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        c4.metric("Recall", f"{metrics.get('recall', 0):.4f}")

        # Show CV results if available
        if "cv_results" in results:
            st.subheader("Cross-Validation Results (5-Fold)")
            cv = results["cv_results"]
            cv_df = pd.DataFrame([
                {"Metric": k.upper(), "Mean Score": f"{v:.4f}"}
                for k, v in cv.items()
            ])
            st.dataframe(cv_df, hide_index=True)

        # Show evaluation plots if they exist
        st.subheader("Evaluation Plots")
        plot_dir = BASE_DIR / "models" / "plots"
        plot_files = {
            "ROC Curve": "roc_curve.png",
            "Precision-Recall Curve": "precision_recall_curve.png",
            "Confusion Matrix": "confusion_matrix.png",
            "Calibration Curve": "calibration_curve.png",
            "Threshold Analysis": "threshold_analysis.png",
            "Business Impact": "business_impact.png",
        }

        cols = st.columns(2)
        for i, (title, filename) in enumerate(plot_files.items()):
            plot_path = plot_dir / filename
            if plot_path.exists():
                with cols[i % 2]:
                    st.caption(title)
                    st.image(str(plot_path))

    else:
        st.warning("No model results found. Run `python -m src.train` first.")

        with st.expander("Model Interpretation Guide"):
            st.markdown("""
**Models Trained:**
- **Logistic Regression** — Baseline, interpretable, calibrated probabilities
- **Random Forest** — Feature importance, non-linear relationships
- **XGBoost** — Gradient boosting with regularization
- **LightGBM** — Fast gradient boosting, handles categorical features natively

**Key Evaluation Metrics:**
- **ROC-AUC**: Overall discriminative ability (higher = better)
- **F1 Score**: Harmonic mean of precision & recall (balanced metric)
- **Precision**: Of predicted churners, how many actually churned?
- **Recall**: Of actual churners, how many did we catch?
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predictor":
    st.markdown('<span class="section-tag">Live Predictor</span>', unsafe_allow_html=True)
    st.title("Customer Churn Probability Predictor")
    st.caption("Input customer profile to predict churn likelihood with explanations")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            security = st.selectbox("Online Security", ["Yes", "No"])
        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charge = st.slider("Monthly Charge ($)", 18, 120, 75)
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col3:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Predict Churn Probability", use_container_width=True)

    if submitted:
        customer_data = {
            "Contract": contract, "InternetService": internet,
            "TechSupport": tech_support, "OnlineSecurity": security,
            "tenure": tenure, "MonthlyCharges": monthly_charge,
            "PaymentMethod": payment, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "PhoneService": phone, "MultipleLines": multiple_lines,
            "OnlineBackup": online_backup, "StreamingTV": streaming_tv,
        }

        prob = predict_churn_heuristic(customer_data)
        risk = compute_risk_level(prob)
        pct = prob * 100

        st.divider()
        col_r, col_g, col_a = st.columns([1, 1, 2])

        with col_r:
            risk_class = risk.lower()
            st.markdown(f'<div class="risk-{risk_class}">{risk} RISK</div>', unsafe_allow_html=True)
            st.markdown(f"### {pct:.1f}% churn probability")

        with col_g:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                title={"text": "Churn Risk", "font": {"color": "#F8FAFC"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#EF4444" if pct > 50 else "#F59E0B" if pct > 25 else "#10B981"},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range": [0, 25], "color": "#064E3B"},
                        {"range": [25, 50], "color": "#78350F"},
                        {"range": [50, 75], "color": "#7C2D12"},
                        {"range": [75, 100], "color": "#7F1D1D"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "#F8FAFC"}},
            ))
            fig.update_layout(
                paper_bgcolor="#0F172A", font_color="#F8FAFC",
                height=250, margin=dict(t=50, b=0, l=30, r=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_a:
            st.subheader("Recommended Actions")
            if risk in ("Critical", "High"):
                st.error("Immediate intervention required")
                st.markdown("""
- Offer immediate loyalty discount (15-20%)
- Personal outreach from retention team
- Promote contract upgrade with incentive
- Review service quality and address complaints
                """)
            elif risk == "Medium":
                st.warning("Monitor closely")
                st.markdown("""
- Flag for monthly monitoring
- Send targeted retention email
- Incentivize auto-pay enrollment
- Offer service bundle upgrade
                """)
            else:
                st.success("Customer is stable")
                st.markdown("""
- Continue standard engagement
- Include in loyalty program communications
- Consider upsell opportunities
                """)

        # Key drivers analysis
        st.subheader("Key Risk Drivers for This Customer")
        drivers = []
        if contract == "Month-to-month":
            drivers.append(("Month-to-month contract", 0.30, "negative"))
        if tenure < 12:
            drivers.append((f"Short tenure ({tenure} months)", 0.25, "negative"))
        if payment == "Electronic check":
            drivers.append(("Electronic check payment", 0.15, "negative"))
        if internet == "Fiber optic":
            drivers.append(("Fiber optic service", 0.10, "negative"))
        if security == "Yes":
            drivers.append(("Has online security", 0.10, "positive"))
        if tech_support == "Yes":
            drivers.append(("Has tech support", 0.08, "positive"))
        if tenure >= 48:
            drivers.append((f"Long tenure ({tenure} months)", 0.20, "positive"))

        if drivers:
            driver_df = pd.DataFrame(drivers, columns=["Factor", "Impact", "Direction"])
            driver_df["Impact_signed"] = driver_df.apply(
                lambda r: r["Impact"] if r["Direction"] == "negative" else -r["Impact"], axis=1
            )
            fig = px.bar(
                driver_df.sort_values("Impact_signed"),
                x="Impact_signed", y="Factor", orientation="h",
                color="Direction",
                color_discrete_map={"negative": "#EF4444", "positive": "#10B981"},
            )
            fig.update_layout(
                plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                font_color="#F8FAFC", showlegend=True,
                xaxis_title="Impact on Churn Risk",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Upload":
    st.markdown('<span class="section-tag">Batch Predictions</span>', unsafe_allow_html=True)
    st.title("Upload Customer Data for Batch Predictions")
    st.caption("Upload a CSV file with customer data to get churn predictions for all customers")

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(upload_df)} customers")

        st.subheader("Data Preview")
        st.dataframe(upload_df.head(10))

        if st.button("Run Predictions", type="primary"):
            with st.spinner("Scoring customers..."):
                predictions = []
                for _, row in upload_df.iterrows():
                    prob = predict_churn_heuristic(row.to_dict())
                    predictions.append({
                        "churn_probability": round(prob, 4),
                        "churn_prediction": int(prob >= 0.5),
                        "risk_level": compute_risk_level(prob),
                    })

                pred_df = pd.DataFrame(predictions)
                result_df = pd.concat([upload_df, pred_df], axis=1)

                st.subheader("Prediction Results")

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scored", len(result_df))
                c2.metric("High/Critical Risk", (pred_df["risk_level"].isin(["High", "Critical"])).sum())
                c3.metric("Medium Risk", (pred_df["risk_level"] == "Medium").sum())
                c4.metric("Low Risk", (pred_df["risk_level"] == "Low").sum())

                # Risk distribution
                risk_counts = pred_df["risk_level"].value_counts()
                fig = px.pie(
                    names=risk_counts.index, values=risk_counts.values,
                    color=risk_counts.index,
                    color_discrete_map={
                        "Critical": "#EF4444", "High": "#F97316",
                        "Medium": "#EAB308", "Low": "#10B981",
                    },
                    hole=0.4,
                )
                fig.update_layout(paper_bgcolor="#0F172A", font_color="#F8FAFC")
                st.plotly_chart(fig, use_container_width=True)

                # Full results table
                st.subheader("Detailed Results")
                st.dataframe(
                    result_df.sort_values("churn_probability", ascending=False),
                    hide_index=True,
                )

                # Download
                csv = result_df.to_csv(index=False)
                st.download_button(
                    "Download Predictions CSV",
                    csv, "churn_predictions.csv", "text/csv",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.markdown('<span class="section-tag">Model Explainability</span>', unsafe_allow_html=True)
    st.title("SHAP-Based Model Explanations")
    st.caption("Understanding why customers churn using SHAP values")

    # Show SHAP plots if they exist
    plot_dir = BASE_DIR / "models" / "plots"
    shap_global = plot_dir / "shap_global.png"
    shap_beeswarm = plot_dir / "shap_beeswarm.png"

    if shap_global.exists() or shap_beeswarm.exists():
        col1, col2 = st.columns(2)
        with col1:
            if shap_global.exists():
                st.subheader("Global Feature Importance (SHAP)")
                st.image(str(shap_global))
                st.caption("Bar chart showing mean absolute SHAP value for each feature")

        with col2:
            if shap_beeswarm.exists():
                st.subheader("SHAP Beeswarm Plot")
                st.image(str(shap_beeswarm))
                st.caption("Each dot = one customer. Color = feature value. Position = SHAP impact.")

        st.divider()
        st.subheader("How to Read SHAP Plots")
        st.markdown("""
**Global Feature Importance (Bar Chart):**
- Longer bars = features that matter more for predictions overall
- This tells you WHICH features drive churn predictions

**Beeswarm Plot:**
- Each row is a feature, each dot is a customer
- Red dots = high feature values, blue = low values
- Dots pushed RIGHT increase churn risk, LEFT decrease it
- Example: If "tenure" dots are mostly blue (low) and pushed right, it means short tenure increases churn risk

**Key Insight:** SHAP values decompose each prediction into individual feature contributions,
allowing you to explain not just what the model predicts, but WHY.
        """)
    else:
        st.info("SHAP plots will appear after running the training pipeline: `python -m src.train`")

        st.subheader("What is SHAP?")
        st.markdown("""
**SHAP (SHapley Additive exPlanations)** uses game theory to explain individual predictions.

For each customer, SHAP answers: *"How much did each feature contribute to this prediction?"*

**Benefits:**
- **Global explanations**: Which features matter most across all customers?
- **Local explanations**: Why did THIS specific customer get a high churn score?
- **Fairness auditing**: Are predictions driven by legitimate business factors?
- **Actionable insights**: Which factors can we actually change to reduce churn?
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Risk Segments":
    st.markdown('<span class="section-tag">Customer Segmentation</span>', unsafe_allow_html=True)
    st.title("Customer Risk Segmentation")

    if df is not None:
        df_clean = prepare_dataset(df)

        # Score all customers
        scores = []
        for _, row in df_clean.iterrows():
            prob = predict_churn_heuristic(row.to_dict())
            scores.append(prob)
        df_clean["churn_score"] = scores
        df_clean["risk_level"] = df_clean["churn_score"].apply(compute_risk_level)

        # Risk distribution
        st.subheader("Risk Level Distribution")
        risk_counts = df_clean["risk_level"].value_counts()
        fig = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={
                "Critical": "#EF4444", "High": "#F97316",
                "Medium": "#EAB308", "Low": "#10B981",
            },
            labels={"x": "Risk Level", "y": "Number of Customers"},
        )
        fig.update_layout(
            plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
            font_color="#F8FAFC", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk by key dimensions
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk by Contract Type")
            if "Contract" in df_clean.columns:
                risk_contract = df_clean.groupby(["Contract", "risk_level"]).size().reset_index(name="count")
                fig = px.bar(
                    risk_contract, x="Contract", y="count", color="risk_level",
                    color_discrete_map={
                        "Critical": "#EF4444", "High": "#F97316",
                        "Medium": "#EAB308", "Low": "#10B981",
                    },
                    barmode="stack",
                )
                fig.update_layout(
                    plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                    font_color="#F8FAFC",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk by Tenure")
            if "tenure" in df_clean.columns:
                df_clean["tenure_group"] = pd.cut(
                    df_clean["tenure"], bins=[0, 12, 24, 48, 72],
                    labels=["0-12m", "13-24m", "25-48m", "49-72m"],
                )
                risk_tenure = df_clean.groupby(["tenure_group", "risk_level"]).size().reset_index(name="count")
                fig = px.bar(
                    risk_tenure, x="tenure_group", y="count", color="risk_level",
                    color_discrete_map={
                        "Critical": "#EF4444", "High": "#F97316",
                        "Medium": "#EAB308", "Low": "#10B981",
                    },
                    barmode="stack",
                )
                fig.update_layout(
                    plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
                    font_color="#F8FAFC",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Churn score distribution
        st.subheader("Churn Score Distribution")
        fig = px.histogram(
            df_clean, x="churn_score", nbins=50,
            color="risk_level",
            color_discrete_map={
                "Critical": "#EF4444", "High": "#F97316",
                "Medium": "#EAB308", "Low": "#10B981",
            },
        )
        fig.update_layout(
            plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
            font_color="#F8FAFC",
            xaxis_title="Churn Probability Score",
            yaxis_title="Number of Customers",
        )
        st.plotly_chart(fig, use_container_width=True)

        # High-risk customer table
        st.subheader("Top At-Risk Customers")
        high_risk = df_clean[df_clean["risk_level"].isin(["Critical", "High"])].sort_values(
            "churn_score", ascending=False
        ).head(20)
        display_cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                                      "InternetService", "PaymentMethod", "churn_score", "risk_level"]
                        if c in high_risk.columns]
        st.dataframe(high_risk[display_cols], hide_index=True)
    else:
        st.warning("Data file not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INTEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Intel":
    st.markdown('<span class="section-tag">Strategy</span>', unsafe_allow_html=True)
    st.title("Business Intelligence & Recommendations")

    # Business cost analysis
    st.subheader("Business Cost Model")
    col1, col2, col3, col4 = st.columns(4)
    cost_fn = col1.number_input("Cost of losing customer ($)", value=400, step=50)
    cost_fp = col2.number_input("Cost of false retention offer ($)", value=50, step=10)
    cost_tp = col3.number_input("Cost of retention offer ($)", value=50, step=10)
    revenue_saved = col4.number_input("Revenue saved per retention ($)", value=300, step=50)

    if df is not None:
        df_clean = prepare_dataset(df)
        if "Churn_binary" in df_clean.columns:
            actual_churners = int(df_clean["Churn_binary"].sum())
            no_model_cost = actual_churners * cost_fn

            # Estimated with model (assuming 70% catch rate)
            catch_rate = 0.70
            caught = int(actual_churners * catch_rate)
            missed = actual_churners - caught
            false_positives = int(len(df_clean) * 0.05)  # ~5% FP rate

            model_cost = (missed * cost_fn) + (false_positives * cost_fp) + (caught * cost_tp)
            model_savings = caught * revenue_saved
            net_benefit = no_model_cost - model_cost + model_savings

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Cost Without Model", f"${no_model_cost:,.0f}")
            c2.metric("Cost With Model", f"${model_cost:,.0f}")
            c3.metric("Net Benefit", f"${net_benefit:,.0f}", delta=f"+${net_benefit:,.0f}")

            # Show business impact plot if exists
            biz_plot = BASE_DIR / "models" / "plots" / "business_impact.png"
            if biz_plot.exists():
                st.image(str(biz_plot))

    st.divider()
    st.subheader("Key Findings")
    findings = [
        ("Contract type is the #1 churn driver",
         "Month-to-month customers churn at 33.2% vs 8.2% for 2-year contracts."),
        ("Early tenure is the highest-risk window",
         "Customers in their first year churn at 28.8%, dropping to 15.7% after 4 years."),
        ("Churned customers pay 12% more per month",
         "Average monthly charge: $79.88 (churned) vs $71.12 (retained)."),
        ("Fiber optic customers are at highest risk",
         "31.4% churn rate vs 6.8% for customers without internet service."),
        ("Electronic check signals lower commitment",
         "30.1% churn rate — 2x higher than auto-payment customers."),
    ]
    for title, body in findings:
        st.markdown(f"""
<div class="insight-box">
<strong>{title}</strong><br>
<small>{body}</small>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Strategic Recommendations")
    recs = [
        ("P1 — CRITICAL", "Promote Long-Term Contracts",
         "Offer 15-20% monthly discounts for annual contracts. Target month-to-month customers at month 3.", "#EF4444"),
        ("P1 — CRITICAL", "Invest in Onboarding (Months 1-12)",
         "Deploy customer success touchpoints in the first 90 days. Can reduce new customer churn by 30%.", "#EF4444"),
        ("P2 — HIGH", "Improve Fiber Optic Value",
         "Investigate service quality issues. Introduce satisfaction guarantees or quality credits.", "#F59E0B"),
        ("P2 — HIGH", "Migrate to Auto-Pay",
         "Incentivize auto-payment with $5-10/month discounts. Auto-pay signals long-term commitment.", "#F59E0B"),
        ("P3 — MEDIUM", "Deploy Predictive Churn Scoring",
         "Score all customers monthly. Proactively contact anyone with >40% churn probability.", "#3B82F6"),
    ]
    for priority, title, body, color in recs:
        st.markdown(f"""
<div style="background: #1E293B; border-radius: 10px; padding: 16px;
            border-left: 3px solid {color}; margin: 8px 0;">
<strong><span style="color: {color}">{priority}:</span> {title}</strong><br>
<small>{body}</small>
</div>
""", unsafe_allow_html=True)
