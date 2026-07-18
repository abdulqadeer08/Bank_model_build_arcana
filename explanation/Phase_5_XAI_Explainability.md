# 📋 Phase 5 — XAI (Explainable AI): Explanation & Review

---

## 🎯 Phase 5 Objective

In Phase 5, we used **SHAP (SHapley Additive exPlanations)** so that every model prediction could be explained.

> **Simple Meaning:** The model doesn't just say "Requires 5 Crore" —  
> Now it will explain **"Requires 5 Crore BECAUSE: It is Monday + 4.8 Crore was withdrawn last week + Branch 234 is a busy branch"**

---

## 🔢 Actions Taken — Step by Step

### Step 1: Libraries Import
- Imported `shap 0.52.0`, `joblib`, `matplotlib`

---

### Step 2: Model & Data Load
- **Model:** `models/best_model.pkl` (XGBoost)
- **Data:** `X_test` (1,731 rows × 22 features)
- **Daily metadata:** for branch info

---

### Step 3: Created SHAP Explainer

```python
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)  # 500 samples
```

| Item | Value |
|---|---|
| **Explainer type** | TreeExplainer (optimized for XGBoost) |
| **Base value (expected_value)** | PKR 46.34 Million |
| **SHAP array shape** | (500, 22) — 500 rows × 22 features |

> **Base value = 46.34M** means: without any features, the model predicts 46.34M. Then each feature adds/subtracts from this base.

---

### Step 4: SHAP Global Feature Importance
**Graph:** `eda_plots/13_shap_global_importance.png`

**Top 5 Most Important Features (SHAP):**

| Rank | Feature | SHAP Impact |
|---|---|---|
| 🥇 1 | `rolling_30_mean_debit` | 9.21M PKR |
| 🥈 2 | `rolling_7_mean_debit` | 3.61M PKR |
| 🥉 3 | `Daily_Txn_Count` | 3.56M PKR |
| 4 | `Branch_Total_Debit` | 3.08M PKR |
| 5 | `Day` | 2.49M PKR |

> **Key Insight:** The 30-day average (`rolling_30_mean_debit`) is the most important feature — meaning **historical trend** is the biggest driver of cash prediction!

---

### Step 5: SHAP Beeswarm Plot
**Graph:** `eda_plots/14_shap_beeswarm.png`

- **Red dots (right side):** High feature value → increases prediction (more cash required)
- **Blue dots (left side):** Low feature value → decreases prediction (less cash required)
- **This plot shows:** Not only importance, but also direction (positive/negative impact)

---

### Step 6: SHAP Waterfall Plot (Single Prediction)
**Graph:** `eda_plots/15_shap_waterfall.png`

**Explained Prediction:**
- **Branch:** 234 (one with the highest prediction)
- **Predicted:** PKR 175.36 Million
- **Actual:** PKR 150.95 Million
- **Error:** ~24 Million (16% off)

**Waterfall plot shows:**
```
Base (average) : +46.34M PKR
rolling_30_mean: +85.2M  (strong positive — high historical avg)
Daily_Txn_Count: +22.1M  (busy day — many transactions)
Branch_Total   : +15.3M  (busy branch overall)
Day feature    :  -8.4M  (specific day adjustment)
...other small adjustments...
─────────────────────────────
Final Prediction: 175.36M PKR
```

---

### Step 7: SHAP Dependence Plot
**Graph:** `eda_plots/16_shap_dependence.png`

- **Feature shown:** `rolling_30_mean_debit` (most important)
- **X-axis:** Actual value of rolling_30_mean_debit
- **Y-axis:** Its SHAP contribution to prediction
- **Pattern:** The higher the 30-day average, the more the prediction increases

---

### Step 8: Branch-Level SHAP Summary
**Graph:** `eda_plots/17_shap_branch_summary.png`

- Average SHAP impact of top 5 features for each branch
- In some branches, `rolling_30_mean` is dominant
- In some branches, `Daily_Txn_Count` plays a larger role
- Business recommendation: **There are different behavior patterns for each branch**

---

### Step 9: Saved SHAP Values

| File | Description |
|---|---|
| `models/shap_values.csv` | SHAP values for 500 test samples |
| `models/shap_expected_value.json` | Base value (46.34M PKR) |

---

## 📊 Key Insights from XAI

| Insight | Business Action |
|---|---|
| `rolling_30_mean_debit` is most important | Look at the 30-day historical trend — do long-term planning |
| `rolling_7_mean_debit` is 2nd most important | Monitor weekly patterns |
| `Daily_Txn_Count` is 3rd important | On days with more transactions, more cash is required |
| Branch-wise different patterns | Each branch should have a distinct replenishment schedule |

---

## ✅ Phase 5 — Final Summary

| Item | Value |
|---|---|
| Tool Used | SHAP 0.52.0 (TreeExplainer) |
| Samples Explained | 500 test samples |
| Base Prediction | PKR 46.34 Million |
| Top Feature | `rolling_30_mean_debit` (9.21M SHAP impact) |
| Plots Saved | 5 plots (13 to 17) |
| SHAP Values Saved | `models/shap_values.csv` |
| Status | ✅ COMPLETE |

---

## ➡️ Next: Phase 6 — Evaluation & Final Reporting
