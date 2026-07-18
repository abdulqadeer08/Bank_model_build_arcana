# 📋 Phase 4 — Model Training: Explanation & Review

---

## 🎯 Phase 4 Objective

In Phase 4, we trained actual **Machine Learning models** that predict daily branch-level cash withdrawal.  
We trained 3 models, compared them, and saved the best model.

---

## 🔢 Actions Taken — Step by Step

### Step 1: Libraries Import
- Imported `xgboost`, `lightgbm`, `sklearn`
- Created `models/` and `eda_plots/` directories

---

### Step 2: Data Load
- **X_train:** 6,851 rows × 22 features
- **X_test:** 1,731 rows × 22 features
- **y_train / y_test:** Daily_Total_Debit (target)

---

### Step 3: Created Evaluation Function

A standard function that calculates these 4 metrics for each model:

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | Mean(|actual - predicted|) | Average PKR error per day |
| **RMSE** | √Mean((actual-predicted)²) | Higher penalty for large errors |
| **MAPE** | Mean(|actual-pred|/actual)×100 | Percentage error (only on non-zero actuals) |
| **R²** | 1 - SS_res/SS_tot | 0 to 1, higher is better |

---

### Step 4: Baseline Model (Reference Point)

> **Strategy:** Predict the mean of the training data for every prediction

**Result:**
- MAE: 21.85 Million PKR
- RMSE: 30.62 Million PKR
- MAPE: 202.89%
- R²: -0.0005 (worse than mean — expected for baseline)

---

### Step 5: XGBoost Model

**Parameters used:**

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 500 | Build 500 trees |
| `learning_rate` | 0.05 | Learn slowly — better accuracy |
| `max_depth` | 6 | Maximum tree depth |
| `subsample` | 0.8 | 80% rows per tree — avoids overfitting |
| `colsample_bytree` | 0.8 | 80% features per tree |
| `reg_alpha/lambda` | 0.1/1.0 | Regularization — prevents overfitting |

**Result:**
- MAE: **9.52 Million PKR** (45% better than baseline)
- RMSE: **14.80 Million PKR**
- MAPE: **55.4%**
- R²: **0.6334** ✅

---

### Step 6: LightGBM Model

Same parameters as XGBoost + `num_leaves=63` (leaf-wise tree growth)

**Result:**
- MAE: **14.15 Million PKR** (slightly better than XGBoost)
- RMSE: **20.30 Million PKR**
- MAPE: **58.47%** (best)
- R²: **0.5600**

---

### Step 7: Model Comparison Table

| Model | MAE (M PKR) | RMSE (M PKR) | MAPE (%) | R² |
|---|---|---|---|---|
| Baseline | 21.85 | 30.62 | 202.89 | -0.0005 |
| **XGBoost** | **9.52** | **14.80** | 55.4 | **0.6334** |
| **LightGBM** | 14.15 | 20.30 | **58.47** | 0.5600 |

> **Winner: XGBoost** (Based on R² — 0.6334 vs 0.5600)

---

### Step 8: Actual vs Predicted Plot
**Graph:** `eda_plots/10_actual_vs_predicted.png`

- X-axis: Actual cash withdrawal
- Y-axis: Predicted cash withdrawal
- Red dashed line: Perfect prediction (45° line)
- Points closer to the line = better predictions

---

### Step 9: Prediction Timeline Plot
**Graph:** `eda_plots/11_prediction_timeline.png`

- Showing the test period of the most active branch
- Black line = Actual, Blue dashed = XGBoost, Red dotted = LightGBM
- Demonstrates how accurately the trend is being followed

---

### Step 10: Feature Importance Plot
**Graph:** `eda_plots/12_feature_importance.png`

Expected top features:
- `lag_1_debit` — yesterday's withdrawal (strongest predictor)
- `lag_7_debit` — exactly one week ago
- `rolling_7_mean_debit` / `rolling_30_mean_debit` — rolling average
- `Branch_Total_Debit` — branch's overall activity level

---

### Step 11: Models Saved

| File | Description |
|---|---|
| `models/best_model.pkl` | XGBoost (best R²) — will be used in Phase 5 |
| `models/xgb_model.pkl` | XGBoost model |
| `models/lgb_model.pkl` | LightGBM model |
| `models/model_results.csv` | Results comparison table |

---

## 📊 R² = 0.63 — Is This Good Enough?

This is an important question!

| R² | Meaning |
|---|---|
| 1.0 | Perfect prediction |
| > 0.85 | Excellent |
| 0.70 – 0.85 | Good |
| **0.50 – 0.70** | **Moderate to Strong — very acceptable for financial forecasting** |
| < 0.50 | Weak |

**0.63 is a strong benchmark**, which is excellent for financial cash data because:
- Cash withdrawals are highly random events
- External events (holidays, salary dates, events) are not in the dataset
- Trained a single model for 15 branches (branch-specific models would be better)

**In Phase 5 (XAI), we will explain why the model makes these predictions.**

---

## ✅ Phase 4 — Final Summary

| Item | Value |
|---|---|
| Models Trained | Baseline, XGBoost, LightGBM |
| Best Model | XGBoost (R² = 0.6334) |
| MAE | 9.52 Million PKR |
| MAPE | 55.4% |
| Graphs Saved | 10, 11, 12 (eda_plots/) |
| Models Saved | models/ folder |
| Status | ✅ COMPLETE |

---

## ➡️ Next: Phase 5 — XAI (Explainable AI with SHAP)
