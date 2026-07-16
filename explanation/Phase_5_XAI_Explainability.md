# 📋 Phase 5 — XAI (Explainable AI): Explanation & Review

---

## 🎯 Phase 5 ka Maqsad (Objective)

Phase 5 mein humne **SHAP (SHapley Additive exPlanations)** use ki taakay model ki har prediction explain ho sake.

> **Simple Matlab:** Model sirf "5 Crore chahiye" nahi kehta —  
> Ab batayega **"5 Crore chahiye KYUNKI: Monday hai + pichle hafte 4.8 Crore nikla tha + Branch 234 busy branch hai"**

---

## 🔢 Kya Kiya — Step by Step

### Step 1: Libraries Import
- `shap 0.52.0`, `joblib`, `matplotlib` import kiye

---

### Step 2: Model & Data Load
- **Model:** `models/best_model.pkl` (XGBoost)
- **Data:** `X_test` (1,731 rows × 22 features)
- **Daily metadata:** branch info ke liye

---

### Step 3: SHAP Explainer Create kiya

```python
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)  # 500 samples
```

| Item | Value |
|---|---|
| **Explainer type** | TreeExplainer (XGBoost ke liye optimized) |
| **Base value (expected_value)** | PKR 46.34 Million |
| **SHAP array shape** | (500, 22) — 500 rows × 22 features |

> **Base value = 46.34M** matlab: koi bhi feature na ho toh model 46.34M predict karega. Phir har feature is base par plus/minus karta hai.

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

> **Key Insight:** Pichle 30 din ka average (`rolling_30_mean_debit`) sab se zyada important feature hai — matlab **historical trend** cash prediction ka sabse bada driver hai!

---

### Step 5: SHAP Beeswarm Plot
**Graph:** `eda_plots/14_shap_beeswarm.png`

- **Red dots (right side):** High feature value → prediction increase karta hai (zyada cash chahiye)
- **Blue dots (left side):** Low feature value → prediction decrease karta hai (kam cash chahiye)
- **Yeh plot batata hai:** Na sirf importance, balki direction bhi (positive/negative impact)

---

### Step 6: SHAP Waterfall Plot (Single Prediction)
**Graph:** `eda_plots/15_shap_waterfall.png`

**Explained Prediction:**
- **Branch:** 234 (sabse high prediction wali)
- **Predicted:** PKR 175.36 Million
- **Actual:** PKR 150.95 Million
- **Error:** ~24 Million (16% off)

**Waterfall plot batata hai:**
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
- **Pattern:** Jitna zyada 30-day average, utna zyada prediction increase

---

### Step 8: Branch-Level SHAP Summary
**Graph:** `eda_plots/17_shap_branch_summary.png`

- Har branch ke liye top 5 features ka average SHAP impact
- Kuch branches mein `rolling_30_mean` dominant hai
- Kuch branches mein `Daily_Txn_Count` zyada role ada karta hai
- Business recommendation: **Har branch ke liye alag behavior pattern hai**

---

### Step 9: SHAP Values Save kiye

| File | Description |
|---|---|
| `models/shap_values.csv` | 500 test samples ke SHAP values |
| `models/shap_expected_value.json` | Base value (46.34M PKR) |

---

## 📊 Key Insights from XAI

| Insight | Business Action |
|---|---|
| `rolling_30_mean_debit` sab se important | 30-din ka historical trend dekho — long-term planning karo |
| `rolling_7_mean_debit` 2nd most important | Weekly patterns monitor karo |
| `Daily_Txn_Count` 3rd important | Jis din transactions zyada, cash zyada chahiye |
| Branch-wise different patterns | Har branch ka alag replenishment schedule hona chahiye |

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
