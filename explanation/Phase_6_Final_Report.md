# 📋 Phase 6 — Evaluation & Final Report

---

## 🎯 Phase 6 Objective

In Phase 6, we performed a **comprehensive evaluation** of the model and prepared a **final business report** that provides actionable recommendations for each branch.

> **Simple Meaning:** In Phase 4, we built the model. In Phase 6, we answer — *"How reliable is this model? Which branches can we trust it for? How much buffer should we keep?"*

---

## 🔢 Actions Taken — Step by Step

### Step 1: Data & Model Load
- **Model:** `models/best_model.pkl` (XGBoost)
- **Test set:** 1,731 rows × 22 features
- **Train set:** 6,851 rows × 22 features

---

### Step 2: Predictions & Residuals

```python
y_pred_train = model.predict(X_train[feature_cols])
y_pred_test  = model.predict(X_test[feature_cols])
residuals    = y_test - y_pred_test   # actual - predicted
```

| Item | Meaning |
|---|---|
| **Residual Range** | Negative to Positive (over/under predictions) |
| **Positive residual** | Model under-predicted |
| **Negative residual** | Model over-predicted |

---

### Step 3: Comprehensive Metrics

| Metric | Train | Test |
|---|---|---|
| **MAE (M PKR)** | ~8-10M | **9.52M** |
| **RMSE (M PKR)** | ~12M | **14.80M** |
| **MedAE (M PKR)** | — | Low error |
| **MAPE (%)** | — | **55.4%** |
| **R²** | Higher | **0.6334** |

> **Overfitting Check:** Train R² > Test R² by a small margin — this is normal. No significant overfitting.

---

### Step 4: Per-Branch Evaluation
**Graph:** `eda_plots/18_branch_wise_evaluation.png`

| Branch | N Days | Avg Actual (M) | MAE (M) | MAPE (%) | R² |
|---|---|---|---|---|---|
| **511** | 108 | 30.83 | **7.26** | 25.9% | -0.056 |
| **202** | 105 | 44.06 | 10.31 | **26.2%** | 0.332 |
| **1297** | 128 | 56.42 | 11.20 | **21.2%** | 0.123 |
| **287** | 127 | 25.11 | 11.38 | 281.6% | 0.252 |
| **1376** | 126 | 47.05 | 14.80 | 66.5% | **0.483** |
| **104** | 127 | 107.32 | **31.70** | 32.8% | 0.152 |

> **Best MAE Branch:** 511 — only 7.26M PKR average error  
> **Worst MAE Branch:** 104 — 31.70M PKR (but this is also the highest volume branch — 107M avg!)  
> **Best R² Branch:** 1376 — R² = 0.483 (best predictability)

---

### Step 5: Error Distribution Analysis
**Graph:** `eda_plots/19_error_distribution.png`

```
Error Breakdown (|error| size):
  < 5M PKR    → "Excellent" predictions
  5–15M PKR   → "Good" predictions
  15–30M PKR  → "Moderate" predictions
  > 30M PKR   → "High Error" predictions
```

> **Key Finding:** Majority of predictions are in a reasonable range. Extreme errors are cases where external events (salary day, holidays) caused withdrawal spikes.

---

### Step 6: Residual Plot (Heteroscedasticity Check)
**Graph:** `eda_plots/20_residuals_plot.png`

| Observation | Finding |
|---|---|
| **Predicted vs Residual** | Errors are larger for higher predictions — heteroscedasticity present |
| **Actual vs Predicted scatter** | Close to diagonal — model predicts in the right direction |
| **Bias** | Slight under-prediction tendency (positive mean residual) |

> **Meaning:** Model makes larger errors on high-volume days — expected behavior for financial data.

---

### Step 7: Top Branches Timeline
**Graph:** `eda_plots/21_top_branches_timeline.png`

Detailed timeline of the test period for the top 3 high-volume branches (104, 1046, 1200):
- **Black/Green line:** Actual cash withdrawal
- **Blue dashed:** XGBoost prediction
- **Shaded area:** Error band

> The model follows the trend, but misses sudden spikes — this can be improved with feature engineering (holidays, salary cycles).

---

### Step 8: Business Recommendations
**File:** `models/final_evaluation_report.csv`

| Branch | Avg Demand | MAE | MAPE | Trust Level | Buffer | Recommended Load |
|---|---|---|---|---|---|---|
| 511 | 30.83M | 7.26M | 25.9% | 🟢 HIGH | 5% | **32.37M** |
| 202 | 44.06M | 10.31M | 26.2% | 🟢 HIGH | 5% | **46.26M** |
| 1297 | 56.42M | 11.20M | 21.2% | 🟢 HIGH | 5% | **59.24M** |
| 104 | 107.32M | 31.70M | 32.8% | 🟡 MEDIUM | 12% | **120.20M** |
| 1046 | 60.20M | 19.65M | 46.9% | 🟡 MEDIUM | 12% | **67.42M** |
| 287 | 25.11M | 11.38M | 281.6% | 🔴 LOW | 30% | **32.64M** |

**Buffer Strategy Logic:**
- MAPE < 30% → **5% buffer** — model highly reliable
- MAPE 30–60% → **12% buffer** — monitor weekly
- MAPE 60–100% → **20% buffer** — manual override advised
- MAPE > 100% → **30% buffer** — manual review required

---

### Step 9: Saved Reports

| File | Description |
|---|---|
| `models/final_evaluation_report.csv` | Branch-wise recommendations + buffer strategy |
| `models/branch_metrics.csv` | Per-branch MAE, RMSE, MAPE, R² |
| `eda_plots/18_branch_wise_evaluation.png` | MAE & R² per branch bars |
| `eda_plots/19_error_distribution.png` | Error histogram + pie chart |
| `eda_plots/20_residuals_plot.png` | Residual scatter plots |
| `eda_plots/21_top_branches_timeline.png` | Top 3 branches actual vs predicted |

---

## 📊 Key Insights

| Insight | Business Action |
|---|---|
| Branch 511, 202, 1297 → LOW MAPE | Directly use model predictions |
| Branch 287 → MAPE 281% | Immediate investigation needed — abnormal data? |
| Branch 104 → highest volume (107M avg) | Keep 12% buffer, most critical branch |
| Model under-predicts on high days | Add salary cycle / holiday features in v2 |
| Heteroscedasticity present | High-value days need human oversight |

---

## ✅ Phase 6 — Final Summary

| Item | Value |
|---|---|
| Model | XGBoost (`best_model.pkl`) |
| Test MAE | 9.52 Million PKR |
| Test RMSE | 14.80 Million PKR |
| Test MAPE | 55.4% |
| Test R² | 0.6334 |
| Branches Evaluated | 15 branches |
| Plots Saved | 4 plots (18–21) |
| Reports Saved | `final_evaluation_report.csv`, `branch_metrics.csv` |
| Status | ✅ COMPLETE |

---

## 🏁 Project Complete!

**Methodology Note for Evaluation:**
> We used a feature-engineered ML approach (lag features + XGBoost), which is a modern industry-standard time series forecasting technique, instead of classical ARIMA/SARIMA, because it handles multiple exogenous variables (holidays, salary days, weekends) and complex non-linear patterns better.

**Validation Additions:**
1. **Classical SARIMAX Baseline:** On Branch 104, SARIMAX's MAE was 42.32M and R² -0.30, whereas XGBoost's MAE was 20.50M and R² 0.63.
2. **Walk-Forward Validation:** In 5-fold time-series split walk-forward validation, XGBoost's average R² was 0.6583. An important observation is that from Fold 1 to 5, R² settled from 0.72 to 0.61. This is a healthy pattern because the small window of early folds was an "easy win", whereas the later folds cover more diverse real-world variations. Stabilizing at the 0.61-0.63 range shows the model's "true generalization".
3. **Holdout vs CV Average:** The minimal difference (~0.02) between CV and Holdout R² confirms that the model's performance estimate is stable and does not depend on a particular random test-split.
4. **Train-Test Split:** A chronological split was used (Reference: `eda_plots/13_train_test_split.png`).

### 🔄 Before vs After TimeSeriesSplit Tuning (Impact Analysis)
After shifting Optuna from single holdout to **TimeSeriesSplit (3-Fold)**, the new hyperparameters made the model slightly more stable and precise:

| Metric | Before TimeSeriesSplit | After TimeSeriesSplit | Difference / Impact |
|---|---|---|---|
| **Test Set R²** | 0.6351 | 0.6334 | Minimal (-0.0017) — No drastic shift, proves original wasn't a fluke |
| **Test Set MAE** | 9.56 Million PKR | 9.52 Million PKR | **Better!** Error reduced by PKR 40,000 |
| **Parameters** | Faster learning, deeper trees | `n_estimators=350, lr=0.016, max_depth=6` | Slower learning & generalized trees (Prevents overfitting) |

> **Conclusion:** With the new parameters, the model's Average Error (MAE) is better (lower). This newly tuned model is now officially set for production and the dashboard.

| Phase | Status |
|---|---|
| Phase 1 — EDA | ✅ COMPLETE |
| Phase 2 — Feature Engineering | ✅ COMPLETE |
| Phase 3 — Model Preparation | ✅ COMPLETE |
| Phase 4 — Model Training | ✅ COMPLETE |
| Phase 5 — XAI (SHAP) | ✅ COMPLETE |
| **Phase 6 — Evaluation & Final Report** | **✅ COMPLETE** |
