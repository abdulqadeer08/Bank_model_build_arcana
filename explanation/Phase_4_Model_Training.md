# 📋 Phase 4 — Model Training: Explanation & Review

---

## 🎯 Phase 4 ka Maqsad (Objective)

Phase 4 mein humne actual **Machine Learning models train** kiye jo daily branch-level cash withdrawal predict karte hain.  
Hum ne 3 models train kiye, compare kiye, aur best model save kiya.

---

## 🔢 Kya Kiya — Step by Step

### Step 1: Libraries Import
- `xgboost`, `lightgbm`, `sklearn` import kiye
- `models/` aur `eda_plots/` folders create kiye

---

### Step 2: Data Load
- **X_train:** 6,851 rows × 22 features
- **X_test:** 1,731 rows × 22 features
- **y_train / y_test:** Daily_Total_Debit (target)

---

### Step 3: Evaluation Function Banai

Ek standard function jo har model ke liye yeh 4 metrics calculate karta hai:

| Metric | Formula | Matlab |
|---|---|---|
| **MAE** | Mean(|actual - predicted|) | Average PKR error per day |
| **RMSE** | √Mean((actual-predicted)²) | Bade errors pe zyada penalty |
| **MAPE** | Mean(|actual-pred|/actual)×100 | % mein galti (sirf non-zero actuals pe) |
| **R²** | 1 - SS_res/SS_tot | 0 to 1, jitna zyada utna better |

---

### Step 4: Baseline Model (Reference Point)

> **Strategy:** Har prediction ke liye training data ka mean predict karo

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
| `n_estimators` | 500 | 500 trees banao |
| `learning_rate` | 0.05 | Slowly seekho — better accuracy |
| `max_depth` | 6 | Tree ki maximum depth |
| `subsample` | 0.8 | 80% rows per tree — avoids overfitting |
| `colsample_bytree` | 0.8 | 80% features per tree |
| `reg_alpha/lambda` | 0.1/1.0 | Regularization — prevents overfitting |

**Result:**
- MAE: **14.25 Million PKR** (35% better than baseline)
- RMSE: **20.29 Million PKR**
- MAPE: **62.13%**
- R²: **0.5607** ✅

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
| **XGBoost** | **14.25** | **20.29** | 62.13 | **0.5607** |
| **LightGBM** | 14.15 | 20.30 | **58.47** | 0.5600 |

> **Winner: XGBoost** (R² ke hisaab se — 0.5607 vs 0.5600)

---

### Step 8: Actual vs Predicted Plot
**Graph:** `eda_plots/10_actual_vs_predicted.png`

- X-axis: Actual cash withdrawal
- Y-axis: Predicted cash withdrawal
- Red dashed line: Perfect prediction (45° line)
- Points jo line ke qareeb hain = better predictions

---

### Step 9: Prediction Timeline Plot
**Graph:** `eda_plots/11_prediction_timeline.png`

- Sabse active branch ka test period dikhaya
- Black line = Actual, Blue dashed = XGBoost, Red dotted = LightGBM
- Trend kitna accurately follow ho raha hai yeh dikh raha hai

---

### Step 10: Feature Importance Plot
**Graph:** `eda_plots/12_feature_importance.png`

Expected top features:
- `lag_1_debit` — kal ka withdrawal (strongest predictor)
- `lag_7_debit` — ek hafte pehle same din
- `rolling_7_mean_debit` / `rolling_30_mean_debit` — rolling average
- `Branch_Total_Debit` — branch ki overall activity level

---

### Step 11: Models Saved

| File | Description |
|---|---|
| `models/best_model.pkl` | XGBoost (best R²) — Phase 5 mein use hoga |
| `models/xgb_model.pkl` | XGBoost model |
| `models/lgb_model.pkl` | LightGBM model |
| `models/model_results.csv` | Results comparison table |

---

## 📊 R² = 0.56 — Kya Yeh Theek Hai?

Yeh sawaal zaroori hai!

| R² | Matlab |
|---|---|
| 1.0 | Perfect prediction |
| > 0.85 | Excellent |
| 0.70 – 0.85 | Good |
| **0.50 – 0.70** | **Moderate — acceptable for financial forecasting** |
| < 0.50 | Weak |

**0.56 moderate hai**, lekin financial cash data mein yeh normal hai kyunki:
- Cash withdrawals highly random events hain
- External events (holidays, salary dates, events) dataset mein nahi hain
- 15 branches ke liye ek hi model train kiya (branch-specific models better hote)

**Phase 5 (XAI) mein yeh explain karein ge ki model kis wajah se kya predict karta hai.**

---

## ✅ Phase 4 — Final Summary

| Item | Value |
|---|---|
| Models Trained | Baseline, XGBoost, LightGBM |
| Best Model | XGBoost (R² = 0.5607) |
| MAE | 14.25 Million PKR |
| MAPE | 62.13% |
| Graphs Saved | 10, 11, 12 (eda_plots/) |
| Models Saved | models/ folder |
| Status | ✅ COMPLETE |

---

## ➡️ Next: Phase 5 — XAI (Explainable AI with SHAP)
