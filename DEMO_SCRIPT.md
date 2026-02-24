# Operational Intelligence ML: End-to-End Walkthrough & Demo Script

*Use this script as a guide for presenting your project. It’s structured to highlight your engineering mentality, technical rigor, and deep understanding of operational ML.*

---

## 🎬 Part 1: The Problem & The Business Case

**(Slide/Visual: The E-commerce Operations problem - Margins, Reviews, Sellers)**

"Hi everyone. Today I'm walking you through an Operational Intelligence ML system I built for a Brazilian e-commerce marketplace.

"When I started, the operations team came to me with three massive headaches: customer support costs were spiking, sellers were furious about logistics blame, and buyers were leaving 1-star reviews. They didn't know where to focus. 

"Instead of just throwing XGBoost at a database, I started with causal data exploration across about 100,000 orders. I found the smoking gun: **Late Deliveries.** 

"Only about 8.1% of orders arrive late. But when they do, the average review score plummets from **4.29 out of 5, down to 2.57**. A late delivery doesn't just mean a delayed package; it means guaranteed support tickets, refunds, and churned users.

"The business thesis became clear: If we can predict which orders *will* be late right at the moment of purchase, the operations team can intervene early. They can swap carriers, alert the seller, or proactively set expectations with the buyer."

---

## 🏗️ Part 2: Engineering Mentality & Avoiding Leakage

**(Slide/Visual: The architecture or the code showing the Temporal Split)**

"A lot of data scientists fail at operational ML because they leak future data into their models. If you use `train_test_split` or random k-fold cross-validation on this dataset, you will build a time machine. The model will look at a Christmas surge in December and use that knowledge to predict an order that happened in June. 

"To prevent this, I explicitly architected a **strict temporal split**.
*   **Train:** Everything up to April 2018.
*   **Validation:** May and June.
*   **Test:** July and August.

"I also had to be ruthless with feature engineering. It is incredibly easy to accidentally include post-purchase information—like how long the freight *actually* took. I built a pipeline that strictly filters only features available at the exact millisecond a customer clicks 'Buy'."

---

## 📈 Part 3: Why Accuracy is a Lie (Metrics Selection)

**(Slide/Visual: The Evaluation Metrics - PR-AUC vs ROC-AUC)**

"Because only ~8% of our dataset is positive, the target is highly imbalanced.

"If I had optimized for Accuracy, I could have handed the business a model that achieved 92% accuracy by simply predicting 'On Time' for every single order. A 92% accurate model that does absolutely nothing.

"ROC-AUC is also misleading here because the massive volume of True Negatives inflates the score. 

"Instead, I anchored the entire project on **PR-AUC (Precision-Recall Area Under the Curve)**. A random guess on this test set would give a PR-AUC of 0.074. My final LightGBM model achieves **0.1308**. 

"That sounds low until you realize it is **76% better than random guessing** in a highly volatile logistics environment. 
"Furthermore, since operations teams need tangible thresholds, I didn't rely on the default 0.50 cutoff. I explicitly tuned the decision threshold on the validation set, anchoring it down to **0.10** to maximize the F1 score and catch as many late orders as operationally viable."

---

## 🕵️ Part 4: The ML Audit (Highlighting Diagnostic Rigor)

**(Slide/Visual: The ML Audit / Bug fixes)**

"One of the things I am most proud of in this project is the rigorous self-audit I conducted on the initial V1 pipeline. I didn't just accept the first set of results; I explicitly hunted for pathologies. I found three major issues and fixed them.

**1. The Baseline Collapse (Double-Encoding Bug)**
"My Logistic Regression baseline was acting strangely. The training ROC-AUC was actually *lower* than the validation ROC-AUC. That is pathognomonic for a data pipeline bug. I dug into the feature transformations and found a double-encoding bug: `LabelEncoder` was being applied on top of Pandas categorical variables, essentially scrambling the linear combinations. I ripped it out, replaced it with a strict `OneHotEncoder`, and the **Test ROC-AUC immediately improved from 0.55 to 0.64**.

**2. The LightGBM Overfit Ratio**
"Initially, my LightGBM model had a train PR-AUC of 0.46 and a validation of 0.13—a massive 3.4x overfit ratio. The tree depth was 7 with 63 leaves. I manually regularized the model capacity down to depth 5 and 31 leaves. The overfit ratio dropped to 2.7x without sacrificing validation performance. 

**3. Ruthless Ablation of 'Noise' Features**
"I had engineered 6 complex historical seller features (like historical late percentage). I suspected they might be causing a subtle self-referential leak. I ran an exclusion experiment—and the model performance didn't drop by a single decimal point. Those features were pure noise. I immediately deleted them from the codebase to simplify the API payload and eliminate the leakage risk."

---

## 🚀 Part 5: Deployment & Interpretability (The Demo portion)

**(Slide/Visual: Run the API live or show the JSON payload)**

"Finally, an ML model is useless if it lives in a Jupyter Notebook. I deployed this using **FastAPI** and packaged it in a **Docker container** to ensure strict environment reproducibility.

"But more importantly, operations teams won't trust a black box. 

*(If doing live demo, send a `curl` request to the API here)*

"Look at the API response. It doesn't just return `prediction: late`. Because I embedded a **SHAP TreeExplainer** directly into the inference layer, the API returns:
1.  The raw probability.
2.  A clear Risk Level (High/Medium/Low).
3.  **The exact top 3 contributing features** that drove *this specific prediction*. (e.g., 'Estimated delivery days decreased risk, but total freight increased risk').
4.  A recommended operational action."

---

## ⚖️ Part 6: Conclusion & Honest Limitations

"To wrap up, I want to be entirely transparent about where this model struggles.

"This system suffers from what I call 'close-call blindness'. Over 60% of the late orders we miss are orders that were late by 3 days or less. The model is amazing at catching systemic failures, but borderline logistical delays remain essentially random noise. 
"Additionally, the model relies heavily on the platform-set 'estimated delivery date'. If the business fundamentally changes how they calculate that estimate, this model will need to be retrained.

"By anticipating these limitations, focusing ruthlessly on causal business metrics, and auditing my own pipeline for data leaks, I built a system that is production-ready, highly interpretable, and engineered for reality."
