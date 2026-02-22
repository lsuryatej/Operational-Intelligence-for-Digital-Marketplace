"""
serve.py — FastAPI REST API for late-delivery prediction with SHAP explanations.

Endpoints
---------
GET  /health       → Health check
POST /predict      → Predict late delivery probability + top contributing features
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import CATEGORICAL_FEATURES, get_feature_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input payload — mirrors what the platform knows at order time."""

    customer_state: str = Field(..., example="RJ")
    seller_state: str = Field(..., example="SP")
    product_weight_g: float = Field(..., example=1200)
    product_length_cm: float = Field(..., example=30)
    product_height_cm: float = Field(..., example=15)
    product_width_cm: float = Field(..., example=20)
    product_category: str = Field(..., example="electronics")
    price: float = Field(..., example=199.90)
    freight_value: float = Field(..., example=42.50)
    payment_type: str = Field(..., example="credit_card")
    payment_installments: int = Field(..., example=3)
    num_items: int = Field(1, example=1)

    # Seller history (pre-aggregated by the platform)
    seller_historical_late_pct: float = Field(0.081, example=0.12)
    seller_avg_delivery_days: float = Field(12.6, example=14.3)
    seller_order_count: int = Field(8, example=87)
    seller_avg_review: float = Field(4.13, example=3.8)
    seller_avg_freight: float = Field(20.0, example=22.5)
    seller_avg_price: float = Field(120.7, example=150.0)

    estimated_delivery_days: float = Field(..., example=21)
    purchase_day_of_week: int = Field(..., ge=0, le=6, example=2)
    purchase_month: int = Field(..., ge=1, le=12, example=7)
    purchase_hour: int = Field(12, ge=0, le=23, example=14)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    top_contributing_features: list[FeatureContribution]
    recommended_action: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Late Delivery Prediction API",
    description=(
        "Predicts whether an order on a Brazilian marketplace will be "
        "delivered after the estimated delivery date. Returns the "
        "probability and the top contributing features (via SHAP)."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_threshold: float = 0.5
_feature_columns: list[str] = []
_explainer = None


@app.on_event("startup")
def load_model():
    """Load model artifacts at startup."""
    global _model, _threshold, _feature_columns, _explainer

    model_path = MODELS_DIR / "lgbm_late_delivery.joblib"
    threshold_path = MODELS_DIR / "optimal_threshold.joblib"
    features_path = MODELS_DIR / "feature_columns.joblib"

    if not model_path.exists():
        logger.error("Model file not found at %s. Run train.py first.", model_path)
        return

    _model = joblib.load(model_path)
    _threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    _feature_columns = joblib.load(features_path) if features_path.exists() else []
    _explainer = shap.TreeExplainer(_model)

    logger.info("Model loaded. Threshold=%.2f, Features=%d",
                _threshold, len(_feature_columns))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feature_row(req: PredictionRequest) -> pd.DataFrame:
    """Convert an API request into a single-row DataFrame matching the training schema."""
    row = {
        # Raw numeric features
        "num_items": req.num_items,
        "total_price": req.price * req.num_items,
        "total_freight": req.freight_value * req.num_items,
        "avg_price": req.price,
        "max_price": req.price,
        "num_sellers": 1,
        "dominant_payment_installments": req.payment_installments,
        "total_payment": (req.price + req.freight_value) * req.num_items,
        "num_payment_methods": 1,
        "product_weight_g": req.product_weight_g,
        "product_length_cm": req.product_length_cm,
        "product_height_cm": req.product_height_cm,
        "product_width_cm": req.product_width_cm,
        "product_photos_qty": 1.0,
        "product_name_lenght": 50.0,
        "product_description_lenght": 500.0,
        # Categorical
        "customer_state": req.customer_state,
        "seller_state": req.seller_state,
        "dominant_payment_type": req.payment_type,
        "product_category_name_english": req.product_category,
        # Engineered
        "is_same_state": int(req.customer_state == req.seller_state),
        "purchase_day_of_week": req.purchase_day_of_week,
        "purchase_month": req.purchase_month,
        "purchase_hour": req.purchase_hour,
        "is_weekend": int(req.purchase_day_of_week in (5, 6)),
        "estimated_delivery_days": req.estimated_delivery_days,
        "freight_price_ratio": (
            req.freight_value / req.price if req.price > 0 else 0
        ),
        "product_volume_cm3": (
            req.product_length_cm * req.product_height_cm * req.product_width_cm
        ),
        # Seller history
        "seller_order_count": req.seller_order_count,
        "seller_avg_review": req.seller_avg_review,
        "seller_historical_late_pct": req.seller_historical_late_pct,
        "seller_avg_delivery_days": req.seller_avg_delivery_days,
        "seller_avg_freight": req.seller_avg_freight,
        "seller_avg_price": req.seller_avg_price,
    }

    df = pd.DataFrame([row])

    # Ensure column order matches training
    if _feature_columns:
        for col in _feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[_feature_columns]

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    return "low"


def _recommended_action(risk: str) -> str:
    actions = {
        "high": "Flag for expedited handling or proactive customer notification about potential delay.",
        "medium": "Monitor this order closely. Consider setting customer expectations.",
        "low": "No action needed — delivery expected on time.",
    }
    return actions.get(risk, "Monitor.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict whether an order will be delivered late.

    Returns the probability, risk level, and top contributing features
    with SHAP-based explanations.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first.",
        )

    # Build feature row
    X = _build_feature_row(request)

    # Predict
    prob = float(_model.predict_proba(X)[:, 1][0])
    prediction = "late" if prob >= _threshold else "on_time"
    risk = _risk_level(prob)

    # SHAP explanation
    contributions: list[FeatureContribution] = []
    try:
        shap_values = _explainer.shap_values(X)
        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 (late)
        else:
            sv = shap_values[0]

        feature_names = list(X.columns)
        indexed = sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        for feat, val in indexed:
            contributions.append(
                FeatureContribution(
                    feature=feat,
                    shap_value=round(float(val), 4),
                    direction="increases risk" if val > 0 else "decreases risk",
                )
            )
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)

    return PredictionResponse(
        prediction=prediction,
        probability=round(prob, 4),
        risk_level=risk,
        top_contributing_features=contributions,
        recommended_action=_recommended_action(risk),
    )


# ---------------------------------------------------------------------------
# CLI (run with uvicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
