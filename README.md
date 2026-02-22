# Operational Intelligence for Digital Marketplace

**Late Delivery Prediction** — An ML system that predicts whether an order on a Brazilian e-commerce marketplace will be delivered late, enabling proactive operational intervention.

## 🎯 Problem Statement

Working with the operations team at a mid-size online marketplace connecting independent sellers to buyers across Brazil. The client's core pain: *"Margins are getting squeezed, sellers complain, buyers leave bad reviews, and we don't know where to focus."*

After deep data exploration of ~100K orders across 9 linked tables, **late delivery prediction** was identified as the highest-value ML intervention — it sits at the causal intersection of all three pain points (margins, seller friction, buyer dissatisfaction).

**Key data insight:** Late orders (8.1% of total) have a mean review score of **2.57** vs **4.29** for on-time orders. Preventing late deliveries directly prevents bad reviews, reduces refunds/vouchers, and improves seller experience.

## 📊 Results

| Metric | LightGBM | Logistic Regression (Baseline) |
|--------|----------|-------------------------------|
| **PR-AUC** | **0.1345** | 0.086 |
| **ROC-AUC** | **0.71** | 0.56 |
| **F1** | **0.21** | 0.10 |
| Precision | 0.15 | 0.06 |
| Recall | 0.38 | 0.24 |

> **Note on PR-AUC:** With 7.4% positive rate, a random classifier achieves PR-AUC ≈ 0.074. Our model at 0.1345 is **82% better than random**. ROC-AUC of 0.71 confirms the model has meaningful discriminative ability.

## 🏗️ Project Structure

```
├── README.md                      # This file
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Containerized deployment
├── notebooks/
│   └── 01_eda.ipynb              # Data exploration & problem framing
├── src/
│   ├── data.py                   # Data loading, joining, cleaning
│   ├── features.py               # Feature engineering (leakage-safe)
│   ├── train.py                  # Training pipeline + experiment logging
│   ├── evaluate.py               # Metrics, error analysis, segment analysis
│   └── serve.py                  # FastAPI deployment with SHAP
├── models/                        # Serialized model artifacts
├── experiments/                   # Tracked experiment results (JSON)
├── tests/
│   └── test_api.py               # API integration tests
└── ai_chat_logs/                  # AI conversation exports
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repo
git clone <repo-url>
cd Operational-Intelligence-for-Digital-Marketplace

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

This will:
- Download the Olist dataset via kagglehub (first run only)
- Apply temporal train/val/test split
- Train LightGBM (primary) and Logistic Regression (baseline)
- Save model artifacts to `models/`
- Log experiments to `experiments/`

### 3. Evaluate

```bash
python src/evaluate.py
```

Produces a detailed evaluation report with segment-level analysis and error patterns.

### 4. Serve the API

```bash
python src/serve.py
# or
uvicorn src.serve:app --reload --port 8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict (high-risk example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "purchase_hour": 22
  }'
```

### 6. Run Tests

```bash
pytest tests/test_api.py -v
```

### 7. Docker

```bash
# Build (after training — needs models/ directory)
docker build -t marketplace-ml .

# Run
docker run -p 8000:8000 marketplace-ml

# Test
curl http://localhost:8000/health
```

## 🔍 API Response Format

```json
{
  "prediction": "late",
  "probability": 0.73,
  "risk_level": "high",
  "top_contributing_features": [
    {"feature": "seller_historical_late_pct", "shap_value": 0.18, "direction": "increases risk"},
    {"feature": "estimated_delivery_days", "shap_value": -0.12, "direction": "decreases risk"},
    {"feature": "product_weight_g", "shap_value": 0.08, "direction": "increases risk"}
  ],
  "recommended_action": "Flag for expedited handling or proactive customer notification about potential delay."
}
```

## ⚙️ Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Temporal split** (not random) | Mimics production: predict future orders from past data |
| **No class weighting** | 8% imbalance is mild; threshold tuning is more stable than loss reweighting |
| **LightGBM over DL** | Tabular data, ~100K rows — GBTs are the right tool; also enables fast SHAP |
| **PR-AUC as primary metric** | ROC-AUC is inflated by true negatives; PR-AUC focuses on the late-order class |
| **Seller history from train only** | Prevents target leakage; new sellers fall back to global statistics |

## ⚠️ Practical Limitations

1. **Close-call blindness**: 58.5% of missed late orders are ≤3 days late — the model struggles with borderline cases
2. **Estimated delivery date dependency**: If the platform changes how it sets estimates, the model needs retraining
3. **Cold-start sellers**: New sellers with no history get global defaults — less reliable predictions
4. **Geographic sparsity**: Remote states (RR, AP, AC) have sparse data → less calibrated predictions
5. **Data vintage**: Trained on 2016–2018 data; post-2018 logistics changes are not captured
