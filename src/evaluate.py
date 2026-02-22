"""
evaluate.py — Model evaluation with error analysis and segment breakdowns.

Produces a comprehensive evaluation report including:
  - Core metrics (PR-AUC, F1, precision, recall)
  - Error analysis (patterns in false negatives / positives)
  - Segment-level analysis (by state, category, delivery estimate)
  - Practical limitations
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_order_dataset
from src.features import (
    CATEGORICAL_FEATURES,
    compute_seller_history,
    engineer_features,
    prepare_for_training,
)
from src.train import compute_metrics, temporal_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def load_model_artifacts():
    """Load the trained model and supporting artifacts."""
    model = joblib.load(MODELS_DIR / "lgbm_late_delivery.joblib")
    threshold = joblib.load(MODELS_DIR / "optimal_threshold.joblib")
    seller_stats = joblib.load(MODELS_DIR / "seller_stats.joblib")
    return model, threshold, seller_stats


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------
def segment_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    segment_col: str,
    min_count: int = 50,
) -> pd.DataFrame:
    """Compute metrics broken down by a segment column."""
    y_pred = (y_prob >= threshold).astype(int)
    temp = df[[segment_col]].copy()
    temp["y_true"] = y_true
    temp["y_pred"] = y_pred
    temp["y_prob"] = y_prob

    rows = []
    for segment, grp in temp.groupby(segment_col):
        if len(grp) < min_count:
            continue
        yt = grp["y_true"].values
        yp = grp["y_prob"].values
        ypred = grp["y_pred"].values

        n_late = yt.sum()
        if n_late == 0 or n_late == len(yt):
            continue  # can't compute AUC

        rows.append({
            "segment": segment,
            "count": len(grp),
            "actual_late_rate": float(yt.mean()),
            "predicted_late_rate": float(ypred.mean()),
            "pr_auc": float(average_precision_score(yt, yp)),
            "f1": float(f1_score(yt, ypred, zero_division=0)),
            "recall": float(
                ((ypred == 1) & (yt == 1)).sum() / max(yt.sum(), 1)
            ),
            "precision": float(
                ((ypred == 1) & (yt == 1)).sum()
                / max((ypred == 1).sum(), 1)
            ),
        })

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=True)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Analyse false positives and false negatives."""
    y_pred = (y_prob >= threshold).astype(int)
    temp = df.copy()
    temp["y_true"] = y_true
    temp["y_pred"] = y_pred
    temp["y_prob"] = y_prob

    # Classify error types
    temp["error_type"] = "correct"
    temp.loc[(temp["y_pred"] == 1) & (temp["y_true"] == 0), "error_type"] = "false_positive"
    temp.loc[(temp["y_pred"] == 0) & (temp["y_true"] == 1), "error_type"] = "false_negative"

    fp = temp[temp["error_type"] == "false_positive"]
    fn = temp[temp["error_type"] == "false_negative"]
    tp = temp[(temp["y_pred"] == 1) & (temp["y_true"] == 1)]

    report = {
        "total_orders": len(temp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "true_positives": len(tp),
    }

    # ---- False Negative analysis (the dangerous ones: missed late orders) ----
    if len(fn) > 0 and "delivery_delay_days" in fn.columns:
        report["fn_delay_stats"] = {
            "mean_delay_days": float(fn["delivery_delay_days"].mean()),
            "median_delay_days": float(fn["delivery_delay_days"].median()),
            "pct_under_3_days_late": float(
                (fn["delivery_delay_days"] <= 3).mean()
            ),
            "pct_over_10_days_late": float(
                (fn["delivery_delay_days"] > 10).mean()
            ),
        }
        # Top states for false negatives
        if "customer_state" in fn.columns:
            report["fn_top_states"] = (
                fn["customer_state"]
                .value_counts()
                .head(5)
                .to_dict()
            )
        # Categories where we miss
        if "product_category_name_english" in fn.columns:
            report["fn_top_categories"] = (
                fn["product_category_name_english"]
                .value_counts()
                .head(5)
                .to_dict()
            )

    # ---- False Positive analysis (crying wolf) ----
    if len(fp) > 0:
        if "estimated_delivery_days" in fp.columns:
            report["fp_avg_estimated_days"] = float(
                fp["estimated_delivery_days"].mean()
            )
        if "is_same_state" in fp.columns:
            report["fp_same_state_pct"] = float(fp["is_same_state"].mean())

    return report


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def run_evaluation(data_dir: str | None = None) -> dict:
    """
    Run the complete evaluation pipeline.

    Returns a structured report dict and saves it to experiments/.
    """
    logger.info("Loading model artifacts …")
    model, threshold, seller_stats = load_model_artifacts()

    logger.info("Building dataset …")
    raw_df = build_order_dataset(data_dir)
    _, _, test_raw = temporal_split(raw_df)

    logger.info("Engineering features for test set …")
    test_feat = engineer_features(test_raw, seller_stats)
    X_test, y_test = prepare_for_training(test_feat)

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # --- Core metrics ---
    metrics = compute_metrics(y_test.values, y_prob, threshold)
    logger.info("Test PR-AUC: %.4f | F1: %.4f | Recall: %.4f | Precision: %.4f",
                metrics["pr_auc"], metrics["f1"],
                metrics["recall"], metrics["precision"])

    # --- Classification report ---
    class_report = classification_report(
        y_test, y_pred, target_names=["on_time", "late"], output_dict=True,
    )

    # --- Segment analysis ---
    segments = {}
    for seg_col in ["customer_state", "product_category_name_english", "is_same_state"]:
        if seg_col in test_feat.columns:
            seg_df = segment_analysis(
                test_feat, y_test.values, y_prob, threshold, seg_col,
            )
            segments[seg_col] = seg_df.to_dict(orient="records")
            logger.info(
                "Segment '%s' — worst PR-AUC: %.4f (%s), best: %.4f (%s)",
                seg_col,
                seg_df["pr_auc"].min(),
                seg_df.iloc[0]["segment"] if len(seg_df) > 0 else "n/a",
                seg_df["pr_auc"].max(),
                seg_df.iloc[-1]["segment"] if len(seg_df) > 0 else "n/a",
            )

    # --- Error analysis ---
    errors = error_analysis(test_feat, y_test.values, y_prob, threshold)
    logger.info(
        "Errors — FP: %d | FN: %d | TP: %d",
        errors["false_positives"],
        errors["false_negatives"],
        errors["true_positives"],
    )
    if "fn_delay_stats" in errors:
        logger.info(
            "Missed late orders: %.1f%% were ≤3 days late (close calls), "
            "%.1f%% were >10 days late (major misses)",
            errors["fn_delay_stats"]["pct_under_3_days_late"] * 100,
            errors["fn_delay_stats"]["pct_over_10_days_late"] * 100,
        )

    # --- Assemble report ---
    report = {
        "model": "LightGBM",
        "threshold": threshold,
        "test_set_size": len(y_test),
        "test_late_rate": float(y_test.mean()),
        "core_metrics": metrics,
        "classification_report": class_report,
        "segment_analysis": segments,
        "error_analysis": errors,
        "practical_limitations": [
            "Model relies on platform-set estimated delivery dates; if estimation logic changes, model needs retraining.",
            "New sellers with no history fall back to global averages — predictions less reliable for them.",
            "Sparse-data states (RR, AP, AC) may have poorly calibrated predictions.",
            "Data covers 2016–2018; post-2018 logistics changes (COVID, new carriers) are not captured.",
            f"At threshold {threshold}, the model trades off precision vs recall — "
            f"adjust threshold based on cost of false alarms vs missed late orders.",
        ],
    }

    # Save report
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPERIMENTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved → %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    report = run_evaluation()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    m = report["core_metrics"]
    print(f"  PR-AUC:              {m['pr_auc']:.4f}")
    print(f"  ROC-AUC:             {m['roc_auc']:.4f}")
    print(f"  F1 (threshold={report['threshold']}): {m['f1']:.4f}")
    print(f"  Precision:           {m['precision']:.4f}")
    print(f"  Recall:              {m['recall']:.4f}")
    print(f"  Recall@50%Prec:      {m['recall_at_50_precision']:.4f}")

    e = report["error_analysis"]
    print(f"\n  False Negatives:     {e['false_negatives']}")
    print(f"  False Positives:     {e['false_positives']}")
    if "fn_delay_stats" in e:
        print(f"  FN ≤3 days late:     {e['fn_delay_stats']['pct_under_3_days_late']*100:.1f}%")
        print(f"  FN >10 days late:    {e['fn_delay_stats']['pct_over_10_days_late']*100:.1f}%")

    print(f"\n  Report: {EXPERIMENTS_DIR / 'evaluation_report.json'}")
