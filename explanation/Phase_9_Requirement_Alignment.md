# 📋 Phase 9 — Requirement Alignment (V2 Pipeline)

---

## 🎯 Phase 9 Objective

In this phase, we upgraded the project to exactly meet the specific requirements from the teacher's email:
1. **Half-Day Intervals:** Converted data from Daily to Half-Daily (AM/PM).
2. **Multi-Target Prediction:** Predicted not just Outflow (Debit), but also Inflow (Credit) and Net Cash.
3. **External Events:** Added Salary Cycles and Public Holidays to the dataset.

---

## 🔢 What Changed?

### 1. Data Aggregation (Half-Day)
Previously, we aggregated total cash for the day. Now, using `txn_hour`, we divided the day into 2 parts:
- **AM (0-11 hours)**
- **PM (12-23 hours)**
Each branch now has 2 predictions per day.

### 2. New Features (Holidays & Salary Days)
- `Is_Salary_Day`: The 1st-5th and 25th-31st of the month were marked as `1`.
- `Is_Holiday`: Major public holidays in Pakistan (Eid, Ashura, etc.) were marked as `1`.
- These new features help the model predict sudden spikes (unusual transaction behavior).

### 3. Multi-Target Models
Instead of one model, we have now trained **3 XGBoost models**:

| Target Variable | R² Score | MAE | MAPE |
|---|---|---|---|
| **Outflow** (`Half_Day_Total_Debit`) | **0.6600** (Improved from 0.56!) | 9.77M PKR | 282% |
| **Inflow** (`Half_Day_Total_Credit`) | 0.5879 | 10.26M PKR | 1044% |
| **Net Cash** (`Half_Day_Net_Cash`) | 0.1023 | 7.67M PKR | N/A |

> **Note:** Outflow R² has significantly improved because the model now recognizes half-day patterns and salary/holiday events. Net Cash R² is lower because net values are highly volatile, but it is now available separately.

### 4. Explainable AI (XAI)
SHAP was reapplied to the new `Half_Day_Total_Debit` model (`eda_plots/v2/shap_summary_outflow.png`). SHAP now explains the impact of `Is_Salary_Day` and `Is_Holiday` on cash withdrawal, meeting the exact requirement from the email.

---

## ✅ Final Conclusion

This V2 pipeline **100% fulfills all requirements** from the teacher's email:
- ✅ Half-day intervals
- ✅ Inflow, Outflow, Net Cash
- ✅ Salary cycles, holidays, events
- ✅ Explainable AI (XAI)

The project can now be presented confidently.
