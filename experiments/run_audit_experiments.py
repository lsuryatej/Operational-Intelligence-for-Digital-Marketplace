"""
run_audit_experiments.py — Systematic audit validation experiments.

Runs all 6 diagnostic experiments from the ML audit report,
logs results to experiments/audit/, and produces a summary.

Usage:
    python experiments/run_audit_experiments.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_order_dataset
from src.features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC,
    RAW_FEATURES,
    compute_seller_history,
    engineer_features,
    get_feature_columns,
    prepare_for_training,
)
from src.train import (
    compute_metrics,
    find_optimal_threshold,
    temporal_split,
    train_lightgbm,
    train_logistic_regression,
    predict_lr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

AUDIT_DIR = PROJECT_ROOT / "experiments" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def save_result(name: str, data: dict):
    path = AUDIT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved → %s", path)


def load_and_split():
    """Load data and do temporal split — shared across experiments."""
    raw_df = build_order_dataset()
    train_raw, val_raw, test_raw = temporal_split(raw_df)
    return raw_df, train_raw, val_raw, test_raw


def prepare_full(train_raw, val_raw, test_raw):
    """Full feature engineering + prepare for training."""
    train_feat = engineer_features(train_raw)
    seller_stats = compute_seller_history(train_feat)
    val_feat = engineer_features(val_raw, seller_stats)
    test_feat = engineer_features(test_raw, seller_stats)

    X_train, y_train = prepare_for_training(train_feat)
    X_val, y_val = prepare_for_training(val_feat)
    X_test, y_test = prepare_for_training(test_feat)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, label=""):
    """Train LightGBM and return metrics dict."""
    import lightgbm as lgb

    model = train_lightgbm(X_train, y_train, X_val, y_val)
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    t = find_optimal_threshold(y_val.values, prob_val)
    m_train = compute_metrics(y_train.values, prob_train, t)
    m_val = compute_metrics(y_val.values, prob_val, t)
    m_test = compute_metrics(y_test.values, prob_test, t)

    logger.info("[%s] Train PR-AUC: %.4f | Val PR-AUC: %.4f | Test PR-AUC: %.4f",
                label, m_train["pr_auc"], m_val["pr_auc"], m_test["pr_auc"])
    logger.info("[%s] Train ROC-AUC: %.4f | Val ROC-AUC: %.4f | Test ROC-AUC: %.4f",
                label, m_train["roc_auc"], m_val["roc_auc"], m_test["roc_auc"])
    logger.info("[%s] Overfit ratio (train/val PR-AUC): %.2f",
                label, m_train["pr_auc"] / m_val["pr_auc"] if m_val["pr_auc"] > 0 else float("inf"))
    return {
        "train": m_train, "val": m_val, "test": m_test,
        "threshold": t, "best_iteration": model.best_iteration_,
        "overfit_ratio": round(m_train["pr_auc"] / m_val["pr_auc"], 2) if m_val["pr_auc"] > 0 else None,
    }


# =========================================================================
# EXPERIMENT A: Remove seller history features
# =========================================================================
def experiment_a_seller_history(train_raw, val_raw, test_raw):
    """Test whether seller history features inflate train metrics (leakage)."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT A: Remove seller history features")
    logger.info("=" * 70)

    seller_features = [
        "seller_order_count", "seller_avg_review",
        "seller_historical_late_pct", "seller_avg_delivery_days",
        "seller_avg_freight", "seller_avg_price",
    ]

    # Full model (baseline)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_full(train_raw, val_raw, test_raw)
    baseline = train_and_evaluate_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, "baseline")

    # Without seller features
    cols_to_keep = [c for c in X_train.columns if c not in seller_features]
    ablated = train_and_evaluate_lgbm(
        X_train[cols_to_keep], y_train,
        X_val[cols_to_keep], y_val,
        X_test[cols_to_keep], y_test,
        "no_seller",
    )

    result = {
        "experiment": "A: Remove seller history features",
        "hypothesis": "Seller history self-referencing inflates train PR-AUC",
        "features_removed": seller_features,
        "baseline": baseline,
        "ablated": ablated,
        "train_prauc_drop": round(baseline["train"]["pr_auc"] - ablated["train"]["pr_auc"], 4),
        "val_prauc_drop": round(baseline["val"]["pr_auc"] - ablated["val"]["pr_auc"], 4),
        "test_prauc_drop": round(baseline["test"]["pr_auc"] - ablated["test"]["pr_auc"], 4),
        "conclusion": "",
    }

    # Interpret
    if result["train_prauc_drop"] > 3 * abs(result["val_prauc_drop"]):
        result["conclusion"] = "CONFIRMED: seller features inflate train much more than val — leakage signal present"
    elif abs(result["val_prauc_drop"]) > 0.005:
        result["conclusion"] = "MIXED: seller features carry some real signal but may also leak"
    else:
        result["conclusion"] = "UNLIKELY: removal had negligible impact — features are mostly noise"

    save_result("experiment_a_seller_history", result)
    return result


# =========================================================================
# EXPERIMENT B: Rolling window temporal CV
# =========================================================================
def experiment_b_rolling_cv(raw_df):
    """Expanding-window temporal CV to assess single-split robustness."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT B: Rolling window temporal CV")
    logger.info("=" * 70)

    raw_df = raw_df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    raw_df["purchase_month"] = raw_df["order_purchase_timestamp"].dt.to_period("M")
    months = sorted(raw_df["purchase_month"].unique())

    if len(months) < 6:
        logger.warning("Not enough months for rolling CV")
        return {}

    fold_results = []

    # Use expanding window: train on months[0:i], validate on months[i]
    # Start with at least 12 months of training data
    start_idx = min(12, len(months) - 3)

    for i in range(start_idx, len(months) - 1):
        train_months = months[:i]
        val_month = months[i]

        train_mask = raw_df["purchase_month"].isin(train_months)
        val_mask = raw_df["purchase_month"] == val_month

        train_df = raw_df[train_mask].copy()
        val_df = raw_df[val_mask].copy()

        if len(val_df) < 50 or val_df["is_late"].sum() < 5:
            continue

        # Feature engineering
        train_feat = engineer_features(train_df)
        seller_stats = compute_seller_history(train_feat)
        val_feat = engineer_features(val_df, seller_stats)

        X_tr, y_tr = prepare_for_training(train_feat)
        X_v, y_v = prepare_for_training(val_feat)

        # Align columns
        for col in X_tr.columns:
            if col not in X_v.columns:
                X_v[col] = 0
        X_v = X_v[X_tr.columns]

        try:
            model = train_lightgbm(X_tr, y_tr, X_v, y_v)
            prob_val = model.predict_proba(X_v)[:, 1]
            m = compute_metrics(y_v.values, prob_val, 0.1)

            fold_results.append({
                "val_month": str(val_month),
                "train_size": len(train_df),
                "val_size": len(val_df),
                "val_late_rate": round(float(val_df["is_late"].mean()), 4),
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "f1": m["f1"],
                "best_iteration": model.best_iteration_,
            })
            logger.info("[Fold %s] Val PR-AUC: %.4f | ROC-AUC: %.4f | late_rate: %.3f | n=%d",
                        val_month, m["pr_auc"], m["roc_auc"],
                        val_df["is_late"].mean(), len(val_df))
        except Exception as e:
            logger.warning("[Fold %s] Failed: %s", val_month, e)

    pr_aucs = [f["pr_auc"] for f in fold_results]
    roc_aucs = [f["roc_auc"] for f in fold_results]

    result = {
        "experiment": "B: Rolling window temporal CV",
        "hypothesis": "Single temporal split may land on anomalous boundary",
        "num_folds": len(fold_results),
        "folds": fold_results,
        "pr_auc_mean": round(float(np.mean(pr_aucs)), 4) if pr_aucs else None,
        "pr_auc_std": round(float(np.std(pr_aucs)), 4) if pr_aucs else None,
        "pr_auc_min": round(float(np.min(pr_aucs)), 4) if pr_aucs else None,
        "pr_auc_max": round(float(np.max(pr_aucs)), 4) if pr_aucs else None,
        "roc_auc_mean": round(float(np.mean(roc_aucs)), 4) if roc_aucs else None,
        "roc_auc_std": round(float(np.std(roc_aucs)), 4) if roc_aucs else None,
        "our_single_split_pr_auc": 0.141,
        "conclusion": "",
    }

    if pr_aucs:
        if result["pr_auc_std"] > 0.03:
            result["conclusion"] = f"HIGH VARIANCE: std={result['pr_auc_std']:.4f} — single split unreliable"
        elif abs(result["pr_auc_mean"] - 0.141) < 0.02:
            result["conclusion"] = f"STABLE: mean={result['pr_auc_mean']:.4f} close to single split (0.141) — split is representative"
        else:
            result["conclusion"] = f"DIVERGENT: mean={result['pr_auc_mean']:.4f} differs from single split (0.141)"

    save_result("experiment_b_rolling_cv", result)
    return result


# =========================================================================
# EXPERIMENT C: Feature distribution drift (PSI)
# =========================================================================
def experiment_c_feature_drift(X_train, X_val, X_test):
    """Compute Population Stability Index for top features."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT C: Feature distribution drift (PSI)")
    logger.info("=" * 70)

    def compute_psi(expected, actual, bins=10):
        """Population Stability Index between two distributions."""
        # Bin edges from expected
        breakpoints = np.percentile(expected.dropna(), np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0

        expected_counts = np.histogram(expected.dropna(), bins=breakpoints)[0]
        actual_counts = np.histogram(actual.dropna(), bins=breakpoints)[0]

        # Avoid division by zero
        expected_pct = (expected_counts + 1) / (expected_counts.sum() + len(expected_counts))
        actual_pct = (actual_counts + 1) / (actual_counts.sum() + len(actual_counts))

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    # Top 10 features by importance (from v2 experiment log)
    top_features = [
        "estimated_delivery_days", "purchase_month", "total_freight",
        "product_volume_cm3", "product_weight_g", "freight_price_ratio",
        "avg_price", "total_payment", "total_price", "product_description_lenght",
    ]

    numeric_features = [f for f in top_features if f in X_train.columns
                        and f not in CATEGORICAL_FEATURES]

    drift_results = []
    for feat in numeric_features:
        psi_train_val = compute_psi(X_train[feat], X_val[feat])
        psi_train_test = compute_psi(X_train[feat], X_test[feat])

        # KS test
        ks_val, p_val = stats.ks_2samp(
            X_train[feat].dropna().values[:5000],
            X_val[feat].dropna().values[:5000],
        )
        ks_test, p_test = stats.ks_2samp(
            X_train[feat].dropna().values[:5000],
            X_test[feat].dropna().values[:5000],
        )

        severity = "stable"
        if psi_train_val > 0.25 or psi_train_test > 0.25:
            severity = "high_drift"
        elif psi_train_val > 0.10 or psi_train_test > 0.10:
            severity = "moderate_drift"

        drift_results.append({
            "feature": feat,
            "psi_train_val": round(psi_train_val, 4),
            "psi_train_test": round(psi_train_test, 4),
            "ks_stat_val": round(float(ks_val), 4),
            "ks_pval_val": round(float(p_val), 6),
            "ks_stat_test": round(float(ks_test), 4),
            "ks_pval_test": round(float(p_test), 6),
            "severity": severity,
        })
        logger.info("[%s] PSI(val)=%.4f PSI(test)=%.4f KS(val)=%.4f KS(test)=%.4f → %s",
                    feat, psi_train_val, psi_train_test, ks_val, ks_test, severity)

    unstable = [d for d in drift_results if d["severity"] != "stable"]

    result = {
        "experiment": "C: Feature distribution drift (PSI)",
        "hypothesis": "Key features shift distribution across temporal splits",
        "features_analyzed": len(drift_results),
        "features_with_drift": len(unstable),
        "drift_details": drift_results,
        "unstable_features": [d["feature"] for d in unstable],
        "conclusion": "",
    }

    if unstable:
        result["conclusion"] = (
            f"DRIFT DETECTED in {len(unstable)} features: {result['unstable_features']}. "
            "Consider removing or engineering more robust versions."
        )
    else:
        result["conclusion"] = "STABLE: all top features have PSI < 0.10 across splits"

    save_result("experiment_c_feature_drift", result)
    return result


# =========================================================================
# EXPERIMENT D: Feature importance stability
# =========================================================================
def experiment_d_importance_stability(raw_df):
    """Train on sliding windows and compare importance rankings."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT D: Feature importance stability across time")
    logger.info("=" * 70)

    raw_df = raw_df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    raw_df["purchase_quarter"] = raw_df["order_purchase_timestamp"].dt.to_period("Q")
    quarters = sorted(raw_df["purchase_quarter"].unique())

    # Need at least 3 quarters
    if len(quarters) < 3:
        return {}

    window_importances = []

    for q in quarters:
        q_data = raw_df[raw_df["purchase_quarter"] == q].copy()
        if len(q_data) < 500 or q_data["is_late"].sum() < 20:
            continue

        # Train/val split within the quarter (80/20)
        split_idx = int(len(q_data) * 0.8)
        tr = q_data.iloc[:split_idx]
        va = q_data.iloc[split_idx:]

        if va["is_late"].sum() < 3:
            continue

        try:
            tr_feat = engineer_features(tr)
            seller_st = compute_seller_history(tr_feat)
            va_feat = engineer_features(va, seller_st)

            X_tr, y_tr = prepare_for_training(tr_feat)
            X_va, y_va = prepare_for_training(va_feat)

            for col in X_tr.columns:
                if col not in X_va.columns:
                    X_va[col] = 0
            X_va = X_va[X_tr.columns]

            model = train_lightgbm(X_tr, y_tr, X_va, y_va)
            imp = dict(zip(X_tr.columns, model.feature_importances_))

            window_importances.append({
                "quarter": str(q),
                "n_orders": len(q_data),
                "late_rate": round(float(q_data["is_late"].mean()), 4),
                "importance": imp,
                "top5": sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5],
            })
            logger.info("[%s] n=%d, late_rate=%.3f, top=%s",
                        q, len(q_data), q_data["is_late"].mean(),
                        [x[0] for x in window_importances[-1]["top5"]])
        except Exception as e:
            logger.warning("[%s] Failed: %s", q, e)

    # Compute Spearman rank correlations between consecutive windows
    correlations = []
    for i in range(len(window_importances) - 1):
        imp1 = window_importances[i]["importance"]
        imp2 = window_importances[i + 1]["importance"]
        common_features = sorted(set(imp1.keys()) & set(imp2.keys()))
        if len(common_features) < 5:
            continue
        ranks1 = [imp1[f] for f in common_features]
        ranks2 = [imp2[f] for f in common_features]
        rho, p = stats.spearmanr(ranks1, ranks2)
        correlations.append({
            "period_1": window_importances[i]["quarter"],
            "period_2": window_importances[i + 1]["quarter"],
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
        })

    mean_rho = float(np.mean([c["spearman_rho"] for c in correlations])) if correlations else None

    result = {
        "experiment": "D: Feature importance stability",
        "hypothesis": "Feature importance rankings change across time",
        "num_windows": len(window_importances),
        "correlations": correlations,
        "mean_spearman_rho": round(mean_rho, 4) if mean_rho is not None else None,
        "window_summaries": [
            {"quarter": w["quarter"], "n": w["n_orders"], "late_rate": w["late_rate"],
             "top5": [x[0] for x in w["top5"]]}
            for w in window_importances
        ],
        "conclusion": "",
    }

    if mean_rho is not None:
        if mean_rho > 0.85:
            result["conclusion"] = f"STABLE: mean Spearman ρ = {mean_rho:.3f} — feature importance is temporally consistent"
        elif mean_rho > 0.70:
            result["conclusion"] = f"MODERATE: mean Spearman ρ = {mean_rho:.3f} — some temporal variation"
        else:
            result["conclusion"] = f"UNSTABLE: mean Spearman ρ = {mean_rho:.3f} — feature rankings change significantly over time"

    save_result("experiment_d_importance_stability", result)
    return result


# =========================================================================
# EXPERIMENT E: Drop purchase_month
# =========================================================================
def experiment_e_drop_purchase_month(train_raw, val_raw, test_raw):
    """Test removing purchase_month as a potential temporal leak."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT E: Drop purchase_month")
    logger.info("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_full(train_raw, val_raw, test_raw)

    baseline = train_and_evaluate_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, "baseline")

    cols = [c for c in X_train.columns if c != "purchase_month"]
    ablated = train_and_evaluate_lgbm(
        X_train[cols], y_train,
        X_val[cols], y_val,
        X_test[cols], y_test,
        "no_purchase_month",
    )

    result = {
        "experiment": "E: Drop purchase_month",
        "hypothesis": "purchase_month creates temporal data leak — model memorises month-specific patterns",
        "baseline": baseline,
        "ablated": ablated,
        "val_prauc_change": round(ablated["val"]["pr_auc"] - baseline["val"]["pr_auc"], 4),
        "test_prauc_change": round(ablated["test"]["pr_auc"] - baseline["test"]["pr_auc"], 4),
        "val_rocauc_change": round(ablated["val"]["roc_auc"] - baseline["val"]["roc_auc"], 4),
        "test_rocauc_change": round(ablated["test"]["roc_auc"] - baseline["test"]["roc_auc"], 4),
        "conclusion": "",
    }

    if result["test_prauc_change"] > -0.005:
        result["conclusion"] = "SAFE TO REMOVE: dropping purchase_month has negligible impact — remove to reduce temporal leak risk"
    else:
        result["conclusion"] = f"KEEP: dropping it hurts test PR-AUC by {result['test_prauc_change']:.4f}"

    save_result("experiment_e_drop_purchase_month", result)
    return result


# =========================================================================
# EXPERIMENT F: Test without estimated_delivery_days
# =========================================================================
def experiment_f_no_estimated_delivery(train_raw, val_raw, test_raw):
    """Test whether the model is just a wrapper around the delivery estimate."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT F: Without estimated_delivery_days")
    logger.info("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_full(train_raw, val_raw, test_raw)

    baseline = train_and_evaluate_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, "baseline")

    cols = [c for c in X_train.columns if c != "estimated_delivery_days"]
    ablated = train_and_evaluate_lgbm(
        X_train[cols], y_train,
        X_val[cols], y_val,
        X_test[cols], y_test,
        "no_est_delivery",
    )

    pct_drop = 0
    if baseline["test"]["pr_auc"] > 0:
        pct_drop = (baseline["test"]["pr_auc"] - ablated["test"]["pr_auc"]) / baseline["test"]["pr_auc"] * 100

    result = {
        "experiment": "F: Without estimated_delivery_days",
        "hypothesis": "If model collapses without estimated_delivery_days, we are wrapping a heuristic",
        "baseline": baseline,
        "ablated": ablated,
        "test_prauc_change": round(ablated["test"]["pr_auc"] - baseline["test"]["pr_auc"], 4),
        "test_prauc_pct_drop": round(pct_drop, 1),
        "test_rocauc_change": round(ablated["test"]["roc_auc"] - baseline["test"]["roc_auc"], 4),
        "conclusion": "",
    }

    if pct_drop > 50:
        result["conclusion"] = f"HEURISTIC WRAPPER: {pct_drop:.0f}% drop — model heavily depends on delivery estimate"
    elif pct_drop > 20:
        result["conclusion"] = f"SIGNIFICANT DEPENDENCY: {pct_drop:.0f}% drop — estimated_delivery_days is critical but other features contribute"
    else:
        result["conclusion"] = f"INDEPENDENT SIGNAL: only {pct_drop:.0f}% drop — model captures signal beyond the delivery estimate"

    save_result("experiment_f_no_estimated_delivery", result)
    return result


# =========================================================================
# MAIN
# =========================================================================
def main():
    logger.info("=" * 70)
    logger.info("STARTING FULL AUDIT EXPERIMENT SUITE")
    logger.info("=" * 70)

    # Load data once
    raw_df, train_raw, val_raw, test_raw = load_and_split()

    # Prepare features once for shared use
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_full(train_raw, val_raw, test_raw)

    results = {}

    # Run all experiments
    results["A"] = experiment_a_seller_history(train_raw, val_raw, test_raw)
    results["B"] = experiment_b_rolling_cv(raw_df)
    results["C"] = experiment_c_feature_drift(X_train, X_val, X_test)
    results["D"] = experiment_d_importance_stability(raw_df)
    results["E"] = experiment_e_drop_purchase_month(train_raw, val_raw, test_raw)
    results["F"] = experiment_f_no_estimated_delivery(train_raw, val_raw, test_raw)

    # Summary
    logger.info("=" * 70)
    logger.info("AUDIT EXPERIMENT SUITE COMPLETE")
    logger.info("=" * 70)
    for key, res in results.items():
        if res and "conclusion" in res:
            logger.info("[%s] %s", key, res["conclusion"])

    # Save combined summary
    summary = {
        exp: {"conclusion": r.get("conclusion", ""), "experiment": r.get("experiment", "")}
        for exp, r in results.items() if r
    }
    save_result("_audit_summary", summary)
    return results


if __name__ == "__main__":
    main()
