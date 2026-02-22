"""
features.py — Feature engineering for late delivery prediction.

All features are computed using ONLY information available at ORDER TIME
(purchase / approval).  No post-purchase data leaks into the feature set.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seller historical aggregates (computed from training data ONLY)
# ---------------------------------------------------------------------------

def compute_seller_history(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    From training data, compute per-seller historical statistics.

    These statistics summarise past performance and are used as look-up
    features at prediction time.  Computing them from train-only data
    prevents target leakage.
    """
    seller_stats = train_df.groupby("primary_seller_id").agg(
        seller_order_count=("order_id", "nunique"),
        seller_avg_review=("review_score", "mean"),
        seller_historical_late_pct=("is_late", "mean"),
        seller_avg_delivery_days=(
            "actual_delivery_days", "mean"
        ),
        seller_avg_freight=("total_freight", "mean"),
        seller_avg_price=("total_price", "mean"),
    )
    return seller_stats.reset_index()


# ---------------------------------------------------------------------------
# Feature engineering (order-level)
# ---------------------------------------------------------------------------

# Columns we keep as raw features (available at order time)
RAW_FEATURES = [
    # Order metadata
    "num_items",
    "total_price",
    "total_freight",
    "avg_price",
    "max_price",
    "num_sellers",
    # Payment
    "dominant_payment_installments",
    "total_payment",
    "num_payment_methods",
    # Product (primary)
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "product_photos_qty",
    "product_name_lenght",
    "product_description_lenght",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "dominant_payment_type",
    "product_category_name_english",
]

ENGINEERED_NUMERIC = [
    # Geographic
    "is_same_state",
    # Temporal
    "purchase_day_of_week",
    "purchase_month",
    "purchase_hour",
    "is_weekend",
    # Delivery estimate buffer
    "estimated_delivery_days",
    # Freight ratio
    "freight_price_ratio",
    # Product volume
    "product_volume_cm3",
    # Seller history
    "seller_order_count",
    "seller_avg_review",
    "seller_historical_late_pct",
    "seller_avg_delivery_days",
    "seller_avg_freight",
    "seller_avg_price",
]


def engineer_features(
    df: pd.DataFrame,
    seller_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Order-level dataset from ``data.build_order_dataset()``.
    seller_stats : pd.DataFrame, optional
        Pre-computed seller history (from training set).  If ``None``,
        seller history features will be filled with global defaults.

    Returns
    -------
    pd.DataFrame with additional feature columns.
    """
    out = df.copy()

    # --- Actual delivery days (only for training target analysis, NOT a feature) ---
    if "order_delivered_customer_date" in out.columns and "order_purchase_timestamp" in out.columns:
        out["actual_delivery_days"] = (
            out["order_delivered_customer_date"] - out["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    # --- Geographic ---
    out["is_same_state"] = (
        out["customer_state"] == out["seller_state"]
    ).astype(int)

    # --- Temporal (available at purchase time) ---
    out["purchase_day_of_week"] = out["order_purchase_timestamp"].dt.dayofweek
    out["purchase_month"] = out["order_purchase_timestamp"].dt.month
    out["purchase_hour"] = out["order_purchase_timestamp"].dt.hour
    out["is_weekend"] = out["purchase_day_of_week"].isin([5, 6]).astype(int)

    # --- Delivery estimate buffer ---
    out["estimated_delivery_days"] = (
        out["order_estimated_delivery_date"] - out["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # --- Freight ratio ---
    out["freight_price_ratio"] = np.where(
        out["total_price"] > 0,
        out["total_freight"] / out["total_price"],
        0,
    )

    # --- Product volume ---
    out["product_volume_cm3"] = (
        out["product_length_cm"].fillna(0)
        * out["product_height_cm"].fillna(0)
        * out["product_width_cm"].fillna(0)
    )

    # --- Seller history features ---
    if seller_stats is not None:
        out = out.merge(
            seller_stats,
            on="primary_seller_id",
            how="left",
            suffixes=("", "_hist"),
        )
        # Handle duplicate column names from merge
        for col in seller_stats.columns:
            if col == "primary_seller_id":
                continue
            hist_col = f"{col}_hist"
            if hist_col in out.columns:
                out[col] = out[hist_col]
                out.drop(columns=[hist_col], inplace=True)
    else:
        # Fill with defaults when no seller history is available
        for col in [
            "seller_order_count", "seller_avg_review",
            "seller_historical_late_pct", "seller_avg_delivery_days",
            "seller_avg_freight", "seller_avg_price",
        ]:
            if col not in out.columns:
                out[col] = np.nan

    # --- Fill NaN seller stats with global medians ---
    global_defaults = {
        "seller_order_count": 8,       # median from profiling
        "seller_avg_review": 4.13,     # mean from profiling
        "seller_historical_late_pct": 0.081,  # global late rate
        "seller_avg_delivery_days": 12.6,
        "seller_avg_freight": 20.0,
        "seller_avg_price": 120.7,
    }
    for col, default in global_defaults.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)

    return out


def get_feature_columns() -> list[str]:
    """Return the full list of feature column names used by the model."""
    return RAW_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_NUMERIC


def prepare_for_training(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select feature columns and target, encode categoricals.

    Returns (X, y) ready for model fitting.
    """
    feature_cols = get_feature_columns()

    X = df[feature_cols].copy()
    y = df["is_late"].copy()

    # --- Encode categoricals as pandas category codes ---
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    # --- Fill remaining NaN in numeric columns with median ---
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_FEATURES]
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    return X, y
