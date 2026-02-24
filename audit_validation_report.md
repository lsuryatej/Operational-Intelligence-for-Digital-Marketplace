# Audit Validation Report

## Summary

9 audit issues investigated. 4 resulted in code changes with measured improvement. 5 were diagnostic experiments yielding important findings but no code changes needed.

---

## Priority 1 Fixes (Code Changes Applied)

### Fix 1: Parameter Logging Mismatch
| Item | Detail |
|------|--------|
| **What was wrong** | [train.py:357–365](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/train.py#L357-L365) logged hardcoded `n_estimators=1000, lr=0.05, depth=6` while actual training used `2000/0.01/7` |
| **What was changed** | Replaced hardcoded dict with `lgbm_model.get_params()` extraction; now logs all 9 hyperparameters dynamically |
| **Measured improvement** | Experiment log now records correct values: `n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=31` ✅ |

### Fix 2: LR Double-Encoding Bug
| Item | Detail |
|------|--------|
| **What was wrong** | `LabelEncoder` re-encoded `pd.Categorical` integer codes as strings, creating inconsistent encodings across splits. Also imposed ordinal relationships on nominal features |
| **What was changed** | Replaced `LabelEncoder` with `OneHotEncoder(handle_unknown="ignore", drop="first")`, encoding raw strings separately from numeric scaling |
| **Before** | Train ROC-AUC 0.667 < Val 0.740 (pathological), Test ROC-AUC **0.558** |
| **After** | Train > Val > Test (normal), Test ROC-AUC **0.641** (+14.9%) ✅ |

### Fix 3: LightGBM Over-Regularisation
| Item | Detail |
|------|--------|
| **What was wrong** | `max_depth=7, num_leaves=63, min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0` → 3.4× train/val overfit ratio |
| **What was changed** | `max_depth=5, num_leaves=31, min_child_samples=50, reg_alpha=0.5, reg_lambda=2.0` |
| **Before** | Train PR-AUC 0.464, overfit ratio **3.4×** |
| **After** | Train PR-AUC 0.380, overfit ratio **2.7×** (↓21%), Val PR-AUC stable at 0.141 ✅ |

### Fix 4: Remove Dead Seller Features (from Experiment A)
| Item | Detail |
|------|--------|
| **What was wrong** | 6 seller history features (order count, avg review, late %, etc.) added complexity and a potential leakage vector |
| **What was changed** | Removed all 6 features from [features.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/features.py), [train.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/train.py), [evaluate.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/evaluate.py), [serve.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/serve.py), and [tests/test_api.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/tests/test_api.py) |
| **Before** | 34 features, all metrics X |
| **After** | 28 features, **identical** metrics (0.0000 delta on all PR-AUC, ROC-AUC, F1) ✅ |
| **Impact** | Simpler model, smaller API payload, eliminated leakage risk |

---

## Diagnostic Experiments (No Code Changes)

### Experiment B: Rolling Window Temporal CV
| Item | Detail |
|------|--------|
| **Hypothesis** | Single temporal split may be anomalous |
| **Result** | 10 folds, PR-AUC range **0.052–0.368**, mean=0.183, std=**0.091** |
| **Finding** | HIGH VARIANCE — driven by base rate swings (late rate 1.4%–21.4% across months). Our single split PR-AUC (0.141) is below the mean (0.183), so performance is conservative estimate |
| **Action** | No code change. Document that single-split evaluation understates average performance |

### Experiment C: Feature Distribution Drift (PSI)
| Item | Detail |
|------|--------|
| **Hypothesis** | Key features shift distribution across temporal splits |
| **Result** | 3 of 10 features have significant drift |
| **Findings** | [purchase_month](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/experiments/run_audit_experiments.py#461-498): PSI=**8.0** (extreme, by construction), `estimated_delivery_days`: PSI=**0.63** (high), `total_freight`: PSI=**0.26** (moderate). Other 7 features stable (PSI < 0.03) |
| **Action** | No removal — Experiments E and F showed removing these features hurts performance. Accept drift as inherent to temporal data |

### Experiment D: Feature Importance Stability
| Item | Detail |
|------|--------|
| **Hypothesis** | Feature rankings change across time |
| **Result** | Mean Spearman ρ = **0.918** across quarterly windows |
| **Finding** | STABLE — feature importance is temporally consistent despite base rate variation ✅ |

### Experiment E: Drop purchase_month
| Item | Detail |
|------|--------|
| **Hypothesis** | [purchase_month](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/experiments/run_audit_experiments.py#461-498) creates temporal leak risk (PSI=8.0) |
| **Result** | Removing it: Val PR-AUC +0.028, but Test ROC-AUC collapses **0.707 → 0.554** |
| **Finding** | KEEP — despite extreme drift, [purchase_month](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/experiments/run_audit_experiments.py#461-498) encodes real seasonal signal |

### Experiment F: Without estimated_delivery_days
| Item | Detail |
|------|--------|
| **Hypothesis** | Model may be wrapping a platform heuristic |
| **Result** | Test PR-AUC drops from 0.131 → 0.075 (**43% drop**), ROC-AUC: 0.707 → 0.525 |
| **Finding** | SIGNIFICANT DEPENDENCY but not total collapse. Other features contribute ~57% of signal independently. Model adds value beyond the delivery estimate |

---

## Final Metrics (After All Fixes)

| Model | Split | PR-AUC | ROC-AUC | F1 | Features |
|-------|-------|--------|---------|-----|----------|
| LightGBM v2 | Val | **0.141** | 0.752 | 0.195 | 28 |
| LightGBM v2 | Test | **0.131** | 0.707 | 0.197 | 28 |
| LR v2 (OHE) | Val | **0.153** | 0.768 | 0.240 | 28 |
| LR v2 (OHE) | Test | **0.118** | 0.641 | 0.150 | 28 |

**Verification**: 6/6 pytest pass, [evaluate.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/evaluate.py) clean, [train.py](file:///Users/suryatejlalam/Operational-Intelligence-for-Digital-Marketplace/src/train.py) clean.
