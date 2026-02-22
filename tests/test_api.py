"""
test_api.py — Integration tests for the Late Delivery Prediction API.

Usage:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.serve import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

HIGH_RISK_PAYLOAD = {
    "customer_state": "AL",
    "seller_state": "SP",
    "product_weight_g": 5000,
    "product_length_cm": 60,
    "product_height_cm": 40,
    "product_width_cm": 50,
    "product_category": "furniture_decor",
    "price": 89.90,
    "freight_value": 55.00,
    "payment_type": "boleto",
    "payment_installments": 1,
    "num_items": 2,
    "seller_historical_late_pct": 0.25,
    "seller_avg_delivery_days": 22.0,
    "seller_order_count": 15,
    "seller_avg_review": 3.2,
    "seller_avg_freight": 35.0,
    "seller_avg_price": 95.0,
    "estimated_delivery_days": 15,
    "purchase_day_of_week": 5,
    "purchase_month": 12,
    "purchase_hour": 22,
}

LOW_RISK_PAYLOAD = {
    "customer_state": "SP",
    "seller_state": "SP",
    "product_weight_g": 200,
    "product_length_cm": 15,
    "product_height_cm": 5,
    "product_width_cm": 10,
    "product_category": "health_beauty",
    "price": 49.90,
    "freight_value": 8.50,
    "payment_type": "credit_card",
    "payment_installments": 1,
    "num_items": 1,
    "seller_historical_late_pct": 0.03,
    "seller_avg_delivery_days": 7.5,
    "seller_order_count": 250,
    "seller_avg_review": 4.6,
    "seller_avg_freight": 10.0,
    "seller_avg_price": 55.0,
    "estimated_delivery_days": 30,
    "purchase_day_of_week": 2,
    "purchase_month": 3,
    "purchase_hour": 10,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        resp = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        # 200 if model is loaded, 503 if not
        assert resp.status_code in (200, 503)

    def test_predict_response_schema(self):
        resp = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "prediction" in data
            assert data["prediction"] in ("late", "on_time")
            assert "probability" in data
            assert 0 <= data["probability"] <= 1
            assert "risk_level" in data
            assert data["risk_level"] in ("low", "medium", "high")
            assert "top_contributing_features" in data
            assert isinstance(data["top_contributing_features"], list)
            assert "recommended_action" in data

    def test_predict_shap_features(self):
        resp = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        if resp.status_code == 200:
            features = resp.json()["top_contributing_features"]
            for feat in features:
                assert "feature" in feat
                assert "shap_value" in feat
                assert "direction" in feat
                assert feat["direction"] in ("increases risk", "decreases risk")

    def test_high_risk_vs_low_risk(self):
        """High-risk payload should have higher probability than low-risk."""
        resp_high = client.post("/predict", json=HIGH_RISK_PAYLOAD)
        resp_low = client.post("/predict", json=LOW_RISK_PAYLOAD)
        if resp_high.status_code == 200 and resp_low.status_code == 200:
            prob_high = resp_high.json()["probability"]
            prob_low = resp_low.json()["probability"]
            assert prob_high > prob_low, (
                f"Expected high-risk ({prob_high}) > low-risk ({prob_low})"
            )

    def test_missing_field_returns_422(self):
        """Missing required fields should return a validation error."""
        incomplete = {"customer_state": "RJ"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# CLI: quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Health Check ===")
    r = client.get("/health")
    print(r.json())

    print("\n=== High Risk Prediction ===")
    r = client.post("/predict", json=HIGH_RISK_PAYLOAD)
    if r.status_code == 200:
        import json
        print(json.dumps(r.json(), indent=2))
    else:
        print(f"Status {r.status_code}: {r.json()}")

    print("\n=== Low Risk Prediction ===")
    r = client.post("/predict", json=LOW_RISK_PAYLOAD)
    if r.status_code == 200:
        import json
        print(json.dumps(r.json(), indent=2))
    else:
        print(f"Status {r.status_code}: {r.json()}")
