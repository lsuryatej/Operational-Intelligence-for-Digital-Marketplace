"""
data.py — Data loading, joining, and cleaning for the Olist marketplace dataset.

Produces a single, analysis-ready DataFrame at the ORDER level with all tables
joined and datetime columns parsed.  Only delivered orders are included (the
target requires actual delivery timestamps).
"""

import os
import logging
from pathlib import Path

import pandas as pd
import kagglehub

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: str | None = None  # resolved lazily via download

DATETIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_dataset() -> str:
    """Download the Olist dataset via kagglehub and return the local path."""
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    logger.info("Dataset downloaded to %s", path)
    return path


def _resolve_data_dir(data_dir: str | None = None) -> str:
    """Return an explicit path or download the dataset."""
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    return download_dataset()


# ---------------------------------------------------------------------------
# Loading individual tables
# ---------------------------------------------------------------------------
def load_orders(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_order_items(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))


def load_payments(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))


def load_reviews(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))


def load_customers(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))


def load_sellers(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"))


def load_products(data_dir: str) -> pd.DataFrame:
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    translation = pd.read_csv(
        os.path.join(data_dir, "product_category_name_translation.csv")
    )
    products = products.merge(translation, on="product_category_name", how="left")
    return products


# ---------------------------------------------------------------------------
# Aggregations (order-level)
# ---------------------------------------------------------------------------
def _agg_items(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order_items to the order level."""
    agg = items.groupby("order_id").agg(
        num_items=("order_item_id", "max"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        num_sellers=("seller_id", "nunique"),
        # Keep the first (primary) seller for feature engineering
        primary_seller_id=("seller_id", "first"),
        primary_product_id=("product_id", "first"),
    )
    return agg.reset_index()


def _agg_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payments to the order level."""
    # Dominant payment type = the one with the highest value
    dominant = (
        payments.sort_values("payment_value", ascending=False)
        .groupby("order_id")
        .first()[["payment_type", "payment_installments"]]
        .rename(
            columns={
                "payment_type": "dominant_payment_type",
                "payment_installments": "dominant_payment_installments",
            }
        )
    )
    totals = payments.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        num_payment_methods=("payment_type", "nunique"),
    )
    return dominant.join(totals).reset_index()


# ---------------------------------------------------------------------------
# Master join
# ---------------------------------------------------------------------------
def build_order_dataset(data_dir: str | None = None) -> pd.DataFrame:
    """
    Join all tables into a single order-level DataFrame.

    Returns only *delivered* orders with the target column ``is_late``.
    """
    data_dir = _resolve_data_dir(data_dir)
    logger.info("Loading tables from %s …", data_dir)

    orders = load_orders(data_dir)
    items = load_order_items(data_dir)
    payments = load_payments(data_dir)
    reviews = load_reviews(data_dir)
    customers = load_customers(data_dir)
    sellers = load_sellers(data_dir)
    products = load_products(data_dir)

    # --- Filter to delivered orders only ---
    orders = orders[orders["order_status"] == "delivered"].copy()
    # Drop rows where delivery timestamp is missing (can't compute target)
    orders = orders.dropna(subset=["order_delivered_customer_date"])
    logger.info("Delivered orders with delivery date: %d", len(orders))

    # --- Target variable ---
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"]
        - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    orders["is_late"] = (orders["delivery_delay_days"] > 0).astype(int)

    # --- Aggregate line-item tables ---
    items_agg = _agg_items(items)
    payments_agg = _agg_payments(payments)

    # --- Join everything ---
    df = orders.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(
        reviews[["order_id", "review_score"]],
        on="order_id",
        how="left",
    )
    df = df.merge(
        customers[["customer_id", "customer_state", "customer_city",
                    "customer_zip_code_prefix"]],
        on="customer_id",
        how="left",
    )
    # Seller info via the primary seller
    df = df.merge(
        sellers[["seller_id", "seller_state", "seller_city",
                 "seller_zip_code_prefix"]],
        left_on="primary_seller_id",
        right_on="seller_id",
        how="left",
        suffixes=("", "_seller"),
    )
    # Product info via the primary product
    df = df.merge(
        products[[
            "product_id", "product_category_name_english",
            "product_weight_g", "product_length_cm",
            "product_height_cm", "product_width_cm",
            "product_photos_qty", "product_name_lenght",
            "product_description_lenght",
        ]],
        left_on="primary_product_id",
        right_on="product_id",
        how="left",
        suffixes=("", "_product"),
    )

    logger.info("Final dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_order_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nis_late distribution:\n{df['is_late'].value_counts()}")
    print(f"\nSample row:\n{df.iloc[0]}")
