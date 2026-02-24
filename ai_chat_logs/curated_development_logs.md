# Curated AI Chat Logs: Late Delivery Prediction ML System

This document contains curated highlights from the AI development logs for the Operational Intelligence ML system, focusing on system architecture, preventative methodology, model diagnostics, and ML audit reasoning. These excerpts demonstrate the structured reasoning applied throughout the project.

---

## Log 1: Problem Framing and Business Reasoning

**User:**
I have an e-commerce dataset (Olist) with ~100k orders. The operations team says margins are squeezed, sellers are complaining, and buyers are leaving bad reviews. We need an ML intervention. What's the highest-value problem to solve?

**AI Developer:**
This is a classic operational triage problem. We need an ML intervention that sits at the causal intersection of your three pain points: buyer dissatisfaction, seller friction, and margin compression.

**Hypothesis:** Late deliveries are the central driver of all three.
1.  **Buyer Dissatisfaction:** Buyers receiving items late leave disproportionately bad reviews.
2.  **Seller Friction:** Sellers get blamed for slow carrier performance or platform delays, leading to frustration.
3.  **Margin Squeeze:** Late deliveries cause support ticket spikes and refund/voucher claims.

**Data Profiling Confirmation:**
Let's profile the data to confirm. We define the target variable `is_late` as `actual_delivery_date > estimated_delivery_date`.
*   *Result:* Late orders represent ~8.1% of all delivered orders.
*   *Result:* The average review score for on-time orders is **4.29 / 5.0**. The average for late orders plummets to **2.57 / 5.0**.

**Conclusion:**
Predicting late deliveries *at the time of purchase* is the highest ROI intervention. If the system can flag high-risk orders immediately, operations can proactively communicate with the buyer, switch the carrier, or notify the seller, turning a guaranteed 1-star review into a mitigated operational event.

---

## Log 2: Leakage Prevention and Temporal Validation Design

**User:**
Let's build the feature pipeline and do the train/test split. How should we partition the data, and how do we prevent data leakage?

**AI Developer:**
For operational ML predicting future risk, **random k-fold cross-validation or a random train/test split is a strictly incorrect methodology.** It causes severe temporal leakage. If we randomly select orders, the model might train on an order from December to predict an order from June, "learning" future holiday carrier delays that it couldn't possibly know at inference time.

**Methodology: Strict Temporal Split**
We must split the data chronologically based on `order_purchase_timestamp`:
1.  Sort the entire dataset chronologically.
2.  **Train:** First ~70% of chronological time (e.g., up to April 2018).
3.  **Validation:** Next ~15% (e.g., May - June 2018).
4.  **Test:** Final ~15% (e.g., July - Aug 2018).

**Feature Engineering & Leakage Vectors to Prevent:**
1.  **Target Leakage:** We must absolutely ensure no post-purchase information (like `actual_delivery_date`, actual freight duration, or review scores) enters the feature matrix.
2.  **Seller History Leakage:** If we create a feature like `seller_historical_late_pct`, we cannot compute this over the entire dataset. We must compute the seller's track record *only on the training set fold*, and then left-join those static stats onto the validation and test sets. Otherwise, the validation set uses future knowledge of how the seller performed.

---

## Log 3: Metric Selection & Threshold Tuning (PR-AUC vs ROC-AUC)

**User:**
The LightGBM model got a ROC-AUC of 0.86 on train and 0.74 on validation. Accuracy is 92%. The Logistic Regression baseline is much worse. Should we use SMOTE since the classes are imbalanced (only 8% late)?

**AI Developer:**
First, we must discard Accuracy as an evaluation metric. With an 8.1% positive rate, a dummy model that predicts "on time" for everything achieves 91.9% accuracy, which is operationally useless.

**ROC-AUC vs PR-AUC:**
While ROC-AUC (0.74) indicates decent rank-ordering, it is dangerously optimistic in highly imbalanced datasets because True Negatives (the dominant "on time" class) inflate the score.

The definitive metric for this system is **PR-AUC (Precision-Recall Area Under Curve)**.
*   The baseline random PR-AUC is equal to the positive class rate (**~0.074** for our test set).
*   Our LightGBM model achieves a validation PR-AUC of **0.141**.
*   *Interpretation:* The model is performing **~1.9x better than random guessing** across the recall spectrum. This reflects real signal, but highlights that the problem is difficult (logistic logistics data often is).

**Class Imbalance Strategy:**
I strongly advise against SMOTE or oversampling here. Synthetic data distorts the calibrated probabilities, and for an operational API, we need real probability distributions (e.g., knowing an order has an *exact* 25% chance of being late).
Instead, we:
1.  Train without class weighting (lightGBM handles mild 8% imbalance well natively).
2.  **Tune the decision threshold post-hoc** by maximizing the F1 score specifically on the validation set. We found the optimal threshold drops from the default 0.50 down to **0.10**. This explicitly trades off precision to maximize recall of late orders.

---

## Log 4: Diagnostic Hypothesis Testing (The ML Audit)

**User:**
I ran an ML Audit on the V1 pipeline. The LightGBM model's PR-AUC drops from 0.464 on train to 0.136 on validation (a 3.4x overfit ratio). Also, the Logistic Regression baseline shows a train ROC-AUC of 0.667 but a validation ROC-AUC of 0.740 (train is worse than val?!). What is happening?

**AI Developer:**
These are brilliant diagnostic catches. We have two distinct pathologies. Let's break down the hypotheses and validation logic.

### Issue 1: The LR Baseline Collapse (Train < Val Anomaly)
**Diagnosis:** A Train score significantly *lower* than a Validation score is pathognomonic for a data pipeline bug, not normal generalization behavior.
**Root Cause Verification:** I reviewed the encoding pipeline. The `prepare_for_training()` function was correctly casting high-cardinality strings to `pd.Categorical` integer codes for LightGBM. However, the Logistic Regression pipeline blindly applied `LabelEncoder` *on top of* those integer codes, and then mapped unseen val/test categories to `-1`.
**Action:** The LR model was literally training on scrambled, doubly-encoded category integers.
**The Fix:** I replaced `LabelEncoder` with `OneHotEncoder(handle_unknown="ignore")` specifically for the LR branch.
**Validation:** Post-fix, the LR Test ROC-AUC jumped from a broken **0.558** to **0.641 (+14.9%)**, restoring a normal (Train > Val > Test) hierarchy.

### Issue 2: The LightGBM 3.4x Overfit Ratio
**Hypothesis formulation:**
1.  **Overfitting from boosting capacity:** High tree depth memorizing noise.
2.  **Temporal Distribution Shift:** The base rate of late orders changes wildly over time.
3.  **Target Leakage via Seller Stats:** The `seller_historical_late_pct` feature, computed on train, includes the order's *own* label, acting as a minor self-referencing leak that artificially inflates Train scores.

**Experiment Design:**
*   *Experiment A (Ablation - Leakage check):* Drop all 6 seller history features and retrain.
    *   *Result:* Zero impact. Metrics were identical to the fourth decimal place. Hypothesis 3 refuted. The seller features were pure noise. We permanently deleted them to reduce feature payload and leakage risk.
*   *Experiment B (Regularization check):* Constrain LightGBM. Reduce `max_depth` (7→5) and `num_leaves` (63→31). Increase L1/L2 penalties.
    *   *Result:* Train PR-AUC dropped from 0.464 to 0.380. Val PR-AUC stabilized at 0.141. The overfit ratio dropped to 2.7x. Hypothesis 1 confirmed as the primarily controllable driver.

---

## Log 5: Deployment Reasoning (API Design & SHAP)

**User:**
The models are fixed. Let's design the FastAPI serving layer and containerize it. How should we format the response?

**AI Developer:**
For an operational machine learning API, just returning `{"prediction": "late"}` is a failure of UX. The operations team needs to know *why* to trust the model, and *what* action to take.

**API Architecture & SHAP Integration:**
I designed the `/predict` endpoint to seamlessly load the pickled feature array and the trained LightGBM model. Crucially, I implemented a fast TreeExplainer (SHAP).
The API response schema now includes:
1.  The raw probability and the binary prediction (using our tuned optimal threshold).
2.  A discretized `risk_level` (Low/Medium/High) for easy frontend color-coding.
3.  **Local Interpretability:** A `top_contributing_features` array using SHAP values. The API dynamically calculates which 3 specific features pushed *this specific order* toward (or away from) a late prediction.
4.  An actionable `recommended_action` string.

**Docker Reproducibility:**
We identified that `lightgbm` requires `libgomp1` (OpenMP) at runtime, which isn't present in the base `python:3.11-slim` image. I amended the `Dockerfile` to include an `apt-get install -y libgomp1` layer, ensuring the container will run smoothly in any orchestrator (ECS, Kubernetes, etc.) without mysterious C-library crashes.
