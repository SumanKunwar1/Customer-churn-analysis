"""
Comprehensive Model Evaluation for Customer Churn Prediction.

Goes beyond accuracy with:
- ROC-AUC, Precision-Recall curves
- Calibration curves
- Confusion matrix analysis
- Business cost metrics
- SHAP-based explainability
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Dark theme for plots
plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#F8FAFC",
    "text.color": "#F8FAFC",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",
    "grid.alpha": 0.3,
})

ACCENT = "#EF4444"
ACCENT2 = "#3B82F6"
ACCENT3 = "#10B981"


class ModelEvaluator:
    """Comprehensive model evaluation suite."""

    def __init__(self, output_dir="models/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_all_metrics(self, y_true, y_pred, y_proba):
        """Compute all classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "avg_precision": average_precision_score(y_true, y_proba),
        }
        logger.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """Plot annotated confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
            ax=ax, linewidths=0.5,
            annot_kws={"size": 16, "weight": "bold"},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")

        # Annotate with rates
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        textstr = (
            f"True Pos: {tp} | False Pos: {fp}\n"
            f"True Neg: {tn} | False Neg: {fn}\n"
            f"FPR: {fp/(fp+tn):.2%} | FNR: {fn/(fn+tp):.2%}"
        )
        ax.text(0.02, -0.15, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", color="#94A3B8")

        plt.tight_layout()
        fig.savefig(self.output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return cm

    def plot_roc_curve(self, y_true, y_proba, model_name="Model"):
        """Plot ROC curve with AUC."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"{model_name} (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#475569", linestyle="--", linewidth=1, label="Random (AUC = 0.5)")
        ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)

        # Optimal threshold (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        ax.scatter(fpr[best_idx], tpr[best_idx], color=ACCENT3, s=100, zorder=5,
                   label=f"Optimal threshold = {thresholds[best_idx]:.3f}")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return auc

    def plot_precision_recall_curve(self, y_true, y_proba, model_name="Model"):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color=ACCENT2, linewidth=2.5,
                label=f"{model_name} (AP = {ap:.4f})")
        ax.fill_between(recall, precision, alpha=0.15, color=ACCENT2)

        # Baseline (positive class proportion)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color="#475569", linestyle="--", linewidth=1,
                    label=f"Baseline = {baseline:.3f}")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return ap

    def plot_calibration_curve(self, y_true, y_proba, model_name="Model", n_bins=10):
        """Plot calibration curve (reliability diagram)."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Calibration plot
        ax1.plot(prob_pred, prob_true, "s-", color=ACCENT, linewidth=2, markersize=8, label=model_name)
        ax1.plot([0, 1], [0, 1], "--", color="#475569", linewidth=1, label="Perfectly Calibrated")
        ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax1.set_ylabel("Fraction of Positives", fontsize=12)
        ax1.set_title("Calibration Curve", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.hist(y_proba, bins=30, color=ACCENT, alpha=0.7, edgecolor="#1E293B")
        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Prediction Distribution", fontsize=11)

        plt.tight_layout()
        fig.savefig(self.output_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_threshold_analysis(self, y_true, y_proba):
        """Analyze metrics across different classification thresholds."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        metrics_at_threshold = []

        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            metrics_at_threshold.append({
                "threshold": t,
                "precision": precision_score(y_true, y_pred_t, zero_division=0),
                "recall": recall_score(y_true, y_pred_t, zero_division=0),
                "f1": f1_score(y_true, y_pred_t, zero_division=0),
            })

        df = pd.DataFrame(metrics_at_threshold)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["threshold"], df["precision"], color=ACCENT2, linewidth=2, label="Precision")
        ax.plot(df["threshold"], df["recall"], color=ACCENT, linewidth=2, label="Recall")
        ax.plot(df["threshold"], df["f1"], color=ACCENT3, linewidth=2, label="F1 Score")

        # Mark best F1 threshold
        best_f1_idx = df["f1"].idxmax()
        best_threshold = df.loc[best_f1_idx, "threshold"]
        ax.axvline(x=best_threshold, color="#F59E0B", linestyle="--", linewidth=1.5,
                    label=f"Best F1 threshold = {best_threshold:.2f}")

        ax.set_xlabel("Classification Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metrics vs. Classification Threshold", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return best_threshold

    # -------------------------------------------------------------------
    # Business Cost Analysis
    # -------------------------------------------------------------------

    def business_cost_analysis(
        self, y_true, y_pred, y_proba,
        cost_fn=400,     # Cost of losing a customer (false negative)
        cost_fp=50,      # Cost of unnecessary retention offer (false positive)
        cost_tp=50,      # Cost of retention offer for true churn (saves the customer)
        revenue_saved=300,  # Revenue saved per correctly retained customer
    ):
        """Compute business-oriented cost metrics.

        Parameters
        ----------
        cost_fn : float - Cost of missing a churning customer
        cost_fp : float - Cost of retention offer to non-churning customer
        cost_tp : float - Cost of retention offer to churning customer
        revenue_saved : float - Revenue saved by retaining a customer
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Without model: all customers churn at the base rate
        total_customers = len(y_true)
        actual_churners = y_true.sum()
        no_model_cost = actual_churners * cost_fn

        # With model
        model_cost = (fn * cost_fn) + (fp * cost_fp) + (tp * cost_tp)
        model_revenue_saved = tp * revenue_saved
        net_benefit = no_model_cost - model_cost + model_revenue_saved

        report = {
            "total_customers": int(total_customers),
            "actual_churners": int(actual_churners),
            "true_positives_caught": int(tp),
            "false_negatives_missed": int(fn),
            "false_positives_wasted": int(fp),
            "cost_without_model": round(no_model_cost, 2),
            "cost_with_model": round(model_cost, 2),
            "revenue_saved": round(model_revenue_saved, 2),
            "net_benefit": round(net_benefit, 2),
            "roi_percent": round((net_benefit / (model_cost + 1)) * 100, 1),
            "catch_rate": round(tp / (actual_churners + 1e-8) * 100, 1),
        }

        logger.info("Business Impact Report:")
        for k, v in report.items():
            logger.info("  %s: %s", k, v)

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Cost comparison
        categories = ["Without Model", "With Model"]
        costs = [no_model_cost, model_cost]
        colors = ["#EF4444", "#10B981"]
        axes[0].bar(categories, costs, color=colors, edgecolor="#1E293B", width=0.5)
        axes[0].set_title("Cost Comparison", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Cost ($)", fontsize=12)
        for i, v in enumerate(costs):
            axes[0].text(i, v + 200, f"${v:,.0f}", ha="center", fontsize=12, fontweight="bold")

        # Churn catch breakdown
        labels = [f"Caught\n({tp})", f"Missed\n({fn})"]
        sizes = [tp, fn]
        colors_pie = [ACCENT3, ACCENT]
        axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
                     startangle=90, textprops={"color": "#F8FAFC", "fontsize": 12})
        axes[1].set_title("Churn Detection", fontsize=14, fontweight="bold")

        plt.tight_layout()
        fig.savefig(self.output_dir / "business_impact.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        return report

    # -------------------------------------------------------------------
    # SHAP Explainability
    # -------------------------------------------------------------------

    def shap_analysis(self, model, X, feature_names=None, max_display=20):
        """Generate SHAP values for global and local explanations."""
        logger.info("Computing SHAP values...")

        # Use TreeExplainer for tree-based models, otherwise KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception:
            # Fallback for non-tree models
            background = shap.sample(X, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (churn)

        # Global feature importance (bar plot)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type="bar", max_display=max_display, show=False)
        plt.title("SHAP Feature Importance (Global)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "shap_global.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          max_display=max_display, show=False)
        plt.title("SHAP Summary (Beeswarm)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("SHAP analysis complete. Plots saved.")
        return shap_values, explainer

    def explain_single_prediction(self, explainer, X_single, feature_names=None):
        """Generate SHAP waterfall for a single customer prediction."""
        shap_values = explainer.shap_values(X_single)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        explanation = shap.Explanation(
            values=shap_values[0] if shap_values.ndim > 1 else shap_values,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_single.iloc[0].values if hasattr(X_single, "iloc") else X_single[0],
            feature_names=feature_names,
        )
        return explanation

    # -------------------------------------------------------------------
    # Full Evaluation Suite
    # -------------------------------------------------------------------

    def full_evaluation(self, y_true, y_pred, y_proba, model_name="Model"):
        """Run all evaluation plots and metrics."""
        logger.info("Running full evaluation for %s...", model_name)

        metrics = self.compute_all_metrics(y_true, y_pred, y_proba)
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        self.plot_roc_curve(y_true, y_proba, model_name)
        self.plot_precision_recall_curve(y_true, y_proba, model_name)
        self.plot_calibration_curve(y_true, y_proba, model_name)
        best_threshold = self.plot_threshold_analysis(y_true, y_proba)
        business = self.business_cost_analysis(y_true, y_pred, y_proba)

        logger.info("Full evaluation complete. Best threshold: %.2f", best_threshold)
        return {
            "metrics": metrics,
            "business_impact": business,
            "optimal_threshold": best_threshold,
        }
