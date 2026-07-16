# 📋 Phase 8 — Interactive Dashboard

---

## 🎯 Phase 8 ka Maqsad

**Streamlit** se ek interactive web dashboard banaya jahan bank managers bina coding ke:
- Future forecast dekh sakein
- Branch-wise analysis kar sakein
- Historical data explore kar sakein

---

## 🖥️ Dashboard Pages

### 1. 📊 Overview
- 4 KPI cards: Total 30-day need, Week 1 avg, Top branch, HIGH confidence rows
- Branch-wise 30-day demand bar chart
- Summary priority table (CRITICAL / HIGH / NORMAL)

### 2. 🔮 Branch Forecast
- Branch select karo → forecast chart instantly update
- GREEN/YELLOW/RED confidence ribbons
- Detailed table (date, predicted, lower, upper, uncertainty)
- **CSV download button** per branch

### 3. 📈 Model Performance
- Overall metrics (MAE, RMSE, MAPE, R², Model)
- Per-branch evaluation table
- All 4 evaluation plots (18–21) interactive
- Business recommendations table

### 4. 🗂️ Data Explorer
- Branch + metric select karo
- Time series plot + rolling 7/30 day averages
- Raw data table (expandable)

---

## ▶️ Dashboard Chalane Ka Tarika

```bash
cd d:\bank
streamlit run dashboard.py
```

Browser automatically open hoga: `http://localhost:8501`

---

## ✅ Phase 8 — Final Summary

| Item | Value |
|---|---|
| Framework | Streamlit 1.48 |
| Pages | 4 (Overview, Forecast, Performance, Explorer) |
| Theme | Dark mode (custom CSS) |
| File | `dashboard.py` |
| Status | ✅ COMPLETE |

---

## 🏁 Full Project Complete!

| Phase | Description | Status |
|---|---|---|
| 1 | EDA | ✅ |
| 2 | Feature Engineering | ✅ |
| 3 | Model Preparation | ✅ |
| 4 | Model Training (XGBoost) | ✅ |
| 5 | XAI — SHAP | ✅ |
| 6 | Evaluation & Reporting | ✅ |
| 7 | Future Forecasting | ✅ |
| **8** | **Interactive Dashboard** | **✅** |
