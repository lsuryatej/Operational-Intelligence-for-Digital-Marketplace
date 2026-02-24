# Operational Intelligence for Digital Marketplace

**Late Delivery Prediction** — An ML system that predicts whether an order on a Brazilian e-commerce marketplace will be delivered late, enabling proactive operational intervention.

## 🎯 Problem Statement

Working with the operations team at a mid-size online marketplace connecting independent sellers to buyers across Brazil. The client's core pain: *"Margins are getting squeezed, sellers complain, buyers leave bad reviews, and we don't know where to focus."*

After deep data exploration of ~100K orders across 9 linked tables, **late delivery prediction** was identified as the highest-value ML intervention — it sits at the causal intersection of all three pain points (margins, seller friction, buyer dissatisfaction).

**Key data insight:** Late orders (8.1% of total) have a mean review score of **2.57** vs **4.29** for on-time orders. Preventing late deliveries directly prevents bad reviews, reduces refunds/vouchers, and improves seller experience.

## 📊 Results (Final v2)

| Metric | LightGBM (Post-Audit) | Logistic Regression (Baseline v2) |
|--------|-----------------------|---------------------------------|
| **PR-AUC** | **0.1308** | 0.1178 |
| **ROC-AUC** | **0.7070** | 0.6408 |
| **F1** | **0.1968** | 0.1503 |
| Precision | 0.1391 | 0.1032 |
| Recall | 0.3362 | 0.2764 |

> **Note on PR-AUC:** With 7.4% positive rate in the test set, a random classifier achieves PR-AUC ≈ 0.074. Our model at 0.1308 is **76% better than random**. ROC-AUC of 0.71 confirms the model has meaningful discriminative ability.

## 🏗️ Project Structure

```
├── README.md                      # This file
├── Makefile                       # Automation for common tasks
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Containerized deployment
├── notebooks/                     # Exploratory Data Analysis
├── src/
│   ├── data.py                   # Data loading, joining, cleaning
│   ├── features.py               # Feature engineering (leakage-safe)
│   ├── train.py                  # Training pipeline + dynamic logging
│   ├── evaluate.py               # Metrics, error analysis, segments
│   └── serve.py                  # FastAPI deployment with SHAP explanations
├── models/                        # Serialized model artifacts
├── experiments/                   # Tracked experiments & Audit results
├── tests/                         # API integration tests
└── ai_chat_logs/                  # Development & Audit history
```

## 🛠️ ML Audit & Quality Improvement

This project underwent a rigorous **ML Audit** which identified and resolved critical issues:

1.  **LR Baseline Collapse (Fixed)**: Replaced `LabelEncoder` with `OneHotEncoder` to fix a double-encoding bug and ordinal mis-specification. **Test ROC-AUC improved 0.55 → 0.64**.
2.  **Overfitting (Regularized)**: Reduced LightGBM complexity (`max_depth` 7→5, `num_leaves` 63→31). Reduced overfit ratio from **3.4x to 2.7x** while maintaining validation performance.
3.  **Parameter Transparency**: Fixed experiment logging to dynamically extract model hyperparameters instead of using hardcoded values.
4.  **Leakage/Noise Reduction**: Removed 6 seller history features after diagnostic experiments proved they added zero signal and posed a potential leakage risk.

Full details in [audit_validation_report.md](./audit_validation_report.md).

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
make setup
```

### 2. Run Pipeline

```bash
make train      # Train v2 models
make evaluate   # Generate evaluation report
make test       # Run API tests
make serve      # Start API locally
```

### 3. API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict
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
    "estimated_delivery_days": 15,
    "purchase_day_of_week": 5,
    "purchase_month": 12,
    "purchase_hour": 22
  }'
```

## 🔍 API Response Format

```json
{
  "prediction": "late",
  "probability": 0.73,
  "risk_level": "high",
  "top_contributing_features": [
    {"feature": "estimated_delivery_days", "shap_value": 0.18, "direction": "increases risk"},
    {"feature": "freight_value", "shap_value": 0.12, "direction": "increases risk"},
    {"feature": "customer_state", "shap_value": -0.08, "direction": "decreases risk"}
  ],
  "recommended_action": "Flag for expedited handling or proactive customer notification about potential delay."
}
```

## ⚙️ Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Temporal split** (not random) | Mimics production: predict future orders from past data |
| **No class weighting** | 8% imbalance is mild; threshold tuning (optimal F1) is more stable than loss reweighting |
| **LightGBM over DL** | Tabular data — GBTs are the right tool; also enables fast SHAP explanations |
| **PR-AUC as primary metric** | ROC-AUC is inflated by true negatives; PR-AUC focuses on the late-order class |
| **Removed Seller History** | Auditing proved these features were noise in the current dataset; simplifies cold-start API logic |

## ⚠️ Practical Limitations

1.  **Close-call blindness**: 60.3% of missed late orders are ≤3 days late — the model struggles with borderline cases.
2.  **Estimate dependency**: Model is 43% dependent on platform-set estimated delivery dates.
3.  **Geographic sparsity**: Remote states (RR, AP, AC) have sparse data → less calibrated predictions.
4.  **Temporal Drift**: 3 features (month, delivery estimate, freight) show moderate-to-high drift across splits.
