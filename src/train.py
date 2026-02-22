"""
train.py — Model training pipeline with temporal split and experiment tracking.

Trains a LightGBM classifier (primary) and a Logistic Regression baseline for
the late‑delivery prediction task.  All experiments are logged to experiments/.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Allow running as ``python src/train.py``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_order_dataset
from src.features import (
    CATEGORICAL_FEATURES,
    compute_seller_history,
    engineer_features,
    get_feature_columns,
    prepare_for_training,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------
def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2018-04-30",
    val_end: str = "2018-06-30",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split orders into train / validation / test by purchase date.

    Data range: Sept 2016 – Aug 2018 (delivered orders).
    - Train: orders up to ``train_end`` (inclusive)  ~80%
    - Val:   orders from ``train_end`` + 1 day to ``val_end`` (inclusive)  ~10%
    - Test:  orders after ``val_end``  ~10%

    Why temporal? In production the model will predict *future* orders using
    past data.  A random split would leak future patterns into training.
    """
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    train_mask = df["order_purchase_timestamp"] <= train_end
    val_mask = (df["order_purchase_timestamp"] > train_end) & (
        df["order_purchase_timestamp"] <= val_end
    )
    test_mask = df["order_purchase_timestamp"] > val_end

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        "Split sizes  — Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Late rate    — Train: %.3f | Val: %.3f | Test: %.3f",
        train_df["is_late"].mean(),
        val_df["is_late"].mean(),
        test_df["is_late"].mean(),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return a dictionary of classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Recall at fixed precision levels
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    recall_at_50_precision = 0.0
    for p, r in zip(precision_arr, recall_arr):
        if p >= 0.50:
            recall_at_50_precision = max(recall_at_50_precision, r)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc),
        "f1": float(f1),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "recall_at_50_precision": float(recall_at_50_precision),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
    }


# ---------------------------------------------------------------------------
# Find optimal threshold (maximize F1 on validation)
# ---------------------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Search for the probability threshold that maximises F1 on the given data."""
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return round(float(best_t), 2)


# ---------------------------------------------------------------------------
# Train LightGBM
# ---------------------------------------------------------------------------
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with early stopping.

    We deliberately do NOT use class weighting (scale_pos_weight /
    is_unbalance).  With only ~8 % positive rate the natural
    distribution is mild enough for gradient boosting to learn
    effectively on its own.  Instead we compensate for the imbalance
    by tuning the classification threshold post-hoc.
    """
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=63,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["valid"],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),
        ],
        categorical_feature=cat_cols,
    )
    logger.info(
        "LightGBM best iteration: %d", model.best_iteration_
    )
    return model


# ---------------------------------------------------------------------------
# Train Logistic Regression (baseline)
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple:
    """Train a logistic regression baseline.  Returns (model, scaler, label_encoders)."""
    X = X_train.copy()
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    model = LogisticRegression(
        max_iter=1000,
        class_weight={0: 1.0, 1: n_neg / n_pos},
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    return model, scaler, label_encoders


def predict_lr(
    model, scaler, label_encoders,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict probabilities with the logistic regression baseline."""
    X = X.copy()
    for col, le in label_encoders.items():
        if col in X.columns:
            # Handle unseen labels gracefully
            X[col] = X[col].astype(str).map(
                lambda v, _le=le: (
                    _le.transform([v])[0] if v in _le.classes_
                    else -1
                )
            )
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------
def log_experiment(
    name: str,
    params: dict,
    metrics_train: dict,
    metrics_val: dict,
    metrics_test: dict,
    feature_importance: dict | None = None,
    notes: str = "",
) -> str:
    """Save experiment results to experiments/<timestamp>_<name>.json."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{name}.json"
    filepath = EXPERIMENTS_DIR / filename

    record = {
        "experiment_name": name,
        "timestamp": ts,
        "parameters": params,
        "metrics": {
            "train": metrics_train,
            "validation": metrics_val,
            "test": metrics_test,
        },
        "feature_importance_top20": feature_importance,
        "notes": notes,
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    logger.info("Experiment logged → %s", filepath)
    return str(filepath)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def run_training_pipeline(data_dir: str | None = None) -> dict:
    """Execute the full training pipeline.  Returns a summary dict."""

    # 1. Load & join data
    logger.info("=" * 60)
    logger.info("Step 1: Loading and joining data")
    raw_df = build_order_dataset(data_dir)

    # 2. Temporal split (BEFORE feature engineering to prevent leakage)
    logger.info("=" * 60)
    logger.info("Step 2: Temporal split")
    train_raw, val_raw, test_raw = temporal_split(raw_df)

    # 3. Feature engineering
    logger.info("=" * 60)
    logger.info("Step 3: Feature engineering")
    # Compute seller history from training data ONLY
    train_feat = engineer_features(train_raw)
    seller_stats = compute_seller_history(train_feat)
    # Apply same seller stats to val and test (look-up, no leakage)
    val_feat = engineer_features(val_raw, seller_stats)
    test_feat = engineer_features(test_raw, seller_stats)

    # 4. Prepare features
    X_train, y_train = prepare_for_training(train_feat)
    X_val, y_val = prepare_for_training(val_feat)
    X_test, y_test = prepare_for_training(test_feat)

    logger.info("Feature matrix shapes — Train: %s | Val: %s | Test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    results = {}

    # ------------------------------------------------------------------
    # 5a. Train LightGBM (primary model)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5a: Training LightGBM")
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)

    # Predictions
    lgbm_prob_train = lgbm_model.predict_proba(X_train)[:, 1]
    lgbm_prob_val = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_prob_test = lgbm_model.predict_proba(X_test)[:, 1]

    # Optimal threshold from validation
    optimal_t = find_optimal_threshold(y_val.values, lgbm_prob_val)
    logger.info("Optimal threshold (F1-maximised on val): %.2f", optimal_t)

    lgbm_metrics_train = compute_metrics(y_train.values, lgbm_prob_train, optimal_t)
    lgbm_metrics_val = compute_metrics(y_val.values, lgbm_prob_val, optimal_t)
    lgbm_metrics_test = compute_metrics(y_test.values, lgbm_prob_test, optimal_t)

    logger.info("LightGBM Val  — PR-AUC: %.4f | F1: %.4f | Recall: %.4f",
                lgbm_metrics_val["pr_auc"], lgbm_metrics_val["f1"],
                lgbm_metrics_val["recall"])
    logger.info("LightGBM Test — PR-AUC: %.4f | F1: %.4f | Recall: %.4f",
                lgbm_metrics_test["pr_auc"], lgbm_metrics_test["f1"],
                lgbm_metrics_test["recall"])

    # Feature importance
    feat_importance = dict(
        sorted(
            zip(X_train.columns, lgbm_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:20]
    )

    # Log experiment
    log_experiment(
        name="lightgbm_v1",
        params={
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "best_iteration": lgbm_model.best_iteration_,
            "optimal_threshold": optimal_t,
        },
        metrics_train=lgbm_metrics_train,
        metrics_val=lgbm_metrics_val,
        metrics_test=lgbm_metrics_test,
        feature_importance={k: int(v) for k, v in feat_importance.items()},
        notes="Primary LightGBM model with temporal split and seller history features.",
    )

    # Save model + artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgbm_late_delivery.joblib"
    joblib.dump(lgbm_model, model_path)

    # Save seller_stats and optimal_threshold for serving
    joblib.dump(seller_stats, MODELS_DIR / "seller_stats.joblib")
    joblib.dump(optimal_t, MODELS_DIR / "optimal_threshold.joblib")
    # Save feature column list for serving
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_columns.joblib")
    logger.info("Model saved → %s", model_path)

    results["lightgbm"] = {
        "val": lgbm_metrics_val,
        "test": lgbm_metrics_test,
        "threshold": optimal_t,
    }

    # ------------------------------------------------------------------
    # 5b. Train Logistic Regression (baseline)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5b: Training Logistic Regression baseline")
    lr_model, lr_scaler, lr_encoders = train_logistic_regression(X_train, y_train)

    lr_prob_val = predict_lr(lr_model, lr_scaler, lr_encoders, X_val)
    lr_prob_test = predict_lr(lr_model, lr_scaler, lr_encoders, X_test)
    lr_prob_train = predict_lr(lr_model, lr_scaler, lr_encoders, X_train)

    lr_t = find_optimal_threshold(y_val.values, lr_prob_val)
    lr_metrics_train = compute_metrics(y_train.values, lr_prob_train, lr_t)
    lr_metrics_val = compute_metrics(y_val.values, lr_prob_val, lr_t)
    lr_metrics_test = compute_metrics(y_test.values, lr_prob_test, lr_t)

    logger.info("LR Val  — PR-AUC: %.4f | F1: %.4f | Recall: %.4f",
                lr_metrics_val["pr_auc"], lr_metrics_val["f1"],
                lr_metrics_val["recall"])
    logger.info("LR Test — PR-AUC: %.4f | F1: %.4f | Recall: %.4f",
                lr_metrics_test["pr_auc"], lr_metrics_test["f1"],
                lr_metrics_test["recall"])

    log_experiment(
        name="logistic_regression_baseline",
        params={"max_iter": 1000, "optimal_threshold": lr_t},
        metrics_train=lr_metrics_train,
        metrics_val=lr_metrics_val,
        metrics_test=lr_metrics_test,
        notes="Baseline logistic regression for comparison.",
    )

    results["logistic_regression"] = {
        "val": lr_metrics_val,
        "test": lr_metrics_test,
        "threshold": lr_t,
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("-" * 60)
    logger.info(
        "LightGBM  — Test PR-AUC: %.4f | Test F1: %.4f",
        lgbm_metrics_test["pr_auc"], lgbm_metrics_test["f1"],
    )
    logger.info(
        "LogRegr   — Test PR-AUC: %.4f | Test F1: %.4f",
        lr_metrics_test["pr_auc"], lr_metrics_test["f1"],
    )
    logger.info("Model artefacts saved to %s", MODELS_DIR)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    results = run_training_pipeline()
    print("\n" + json.dumps(results, indent=2))
