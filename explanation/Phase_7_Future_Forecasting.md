# 📋 Phase 7 — Future Forecasting: Explanation & Review

---

## 🎯 Phase 7 Objective

In Phase 7, we implemented **Recursive Multi-Step Forecasting** — predicting the branch-wise daily cash withdrawal for the **next 30 days** based on past data (Feb 2024 – Apr 2026).

> **Simple Meaning:** The model does not know in advance what will happen tomorrow. Therefore, we **make tomorrow's prediction the lag feature for the day after** — then predict the day after — and this chain continues for 30 days!

---

## 🔢 Actions Taken — Step by Step

### Step 1: Identified Last Known State

| Item | Value |
|---|---|
| **Last Date in Data** | 2026-04-04 |
| **Forecast Start** | 2026-04-05 |
| **Forecast End** | 2026-05-04 |
| **Total Forecast Days** | 30 per branch |
| **Total Predictions** | 450 (15 branches × 30 days) |

---

### Step 2: Recursive Forecasting Engine

```python
for day in next_30_days:
    lag_1  = yesterday's value   # actual or previously predicted
    lag_7  = 7 days ago value
    lag_14 = 14 days ago value
    lag_30 = 30 days ago value
    rolling_7_mean  = average of last 7 days
    rolling_30_mean = average of last 30 days
    
    prediction = model.predict(features)
    
    # KEY STEP: predicted value → added to history → becomes lag for next day
    history[target_date] = prediction
```

**Handling Unknown Future Features:**

| Feature | Solution |
|---|---|
| `Daily_Txn_Count` | Used branch-wise historical average |
| `Branch_Total_Debit` | Branch-wise historical average |
| `Peak_Hour_Txns` | Branch-wise historical average |
| Calendar features | Computed exactly (weekday, month, etc.) |

---

### Step 3: Uncertainty / Confidence Intervals

The further ahead you predict, the higher the uncertainty increases:

| Horizon | Uncertainty | Confidence |
|---|---|---|
| Day 1–7 | ±5% to ±10% | 🟢 HIGH |
| Day 8–14 | ±10% to ±15% | 🟡 MEDIUM |
| Day 15–30 | ±15% to ±25% | 🔴 LOW |

> **This is an industry standard:** A weather forecast is accurate for tomorrow, but less accurate for next month.

---

### Step 4: Forecast Results

**Grand Total (All 15 Branches, 30 Days):**

> 🏦 **18,049 Million PKR = 18.05 Billion PKR**

**Per-Branch 30-Day Forecast:**

| Branch | Daily Avg (M) | 30-Day Total (M) | Notes |
|---|---|---|---|
| **104** | **93.3** | **2,797.7** | Highest demand branch |
| **1046** | **65.6** | **1,968.5** | 2nd highest |
| **1297** | **58.6** | **1,758.6** | 3rd highest |
| **1200** | **55.2** | **1,657.0** | Large volume |
| **1376** | **43.9** | **1,316.4** | Medium-high |
| **234** | **43.1** | **1,293.9** | Medium |
| 202 | 33.8 | 1,014.9 | Medium |
| 1092 | 32.3 | 969.2 | Medium |
| 394 | 28.1 | 842.8 | Lower-medium |
| 511 | 27.6 | 828.8 | Lower-medium |
| 372 | 26.6 | 796.9 | Lower |
| 593 | 26.4 | 790.7 | Lower |
| 659 | 25.9 | 776.7 | Lower |
| 1739 | 21.1 | 633.7 | Low |
| **287** | **20.1** | **603.2** | Lowest demand |

**Sample — Branch 104 First Week:**

| Date | Predicted (M) | Lower (M) | Upper (M) | Confidence |
|---|---|---|---|---|
| 2026-04-05 | 101.81 | 96.72 | 106.90 | HIGH |
| 2026-04-06 | 137.61 | 129.81 | 145.40 | HIGH |
| 2026-04-07 | 130.85 | 122.56 | 139.14 | HIGH |
| 2026-04-08 | 110.22 | 102.51 | 117.94 | HIGH |
| 2026-04-09 | 121.57 | 112.25 | 130.89 | HIGH |

---

### Step 5: Plots Generated

**Plot 22 — All Branches Forecast Grid**
`eda_plots/22_all_branches_forecast.png`

- All 15 branches together — 5×3 grid
- Green line = past 60 days actual
- Blue dashed = 30-day forecast
- Shaded area = confidence band
- Orange dotted line = forecast start

**Plot 23 — Top 6 Branches Confidence Ribbons**
`eda_plots/23_confidence_ribbons.png`

- Detailed view of the top 6 high-volume branches
- 🟢 GREEN ribbon = Week 1 (HIGH confidence)
- 🟡 YELLOW ribbon = Week 2 (MEDIUM confidence)
- 🔴 RED ribbon = Week 3–4 (LOW confidence)
- Weekly markers show when confidence drops

**Plot 24 — Branch × Date Heatmap**
`eda_plots/24_forecast_heatmap.png`

- Rows = Branches, Columns = Dates (every 3rd day shown)
- Darker color = more cash required
- Shows at a glance where and when more liquidity is needed

---

### Step 6: Excel Export — 17 Sheets

**File:** `models/forecast_next30days.xlsx`

| Sheet | Contents |
|---|---|
| `All_Branches` | 450 rows — all branches all dates |
| `Weekly_Summary` | Branch × Week pivot table |
| `Br_104` | Detailed 30-day forecast for Branch 104 |
| `Br_202` | Branch 202 ka detailed forecast |
| … | (separate sheet for each branch) |

---

## 📊 Key Insights

| Insight | Business Action |
|---|---|
| Branch 104 → 93.3M/day (30 days) | Highest priority — ensure 100M+ daily liquidity |
| Week 1 predictions → HIGH confidence | Use directly for immediate replenishment |
| Week 3–4 → LOW confidence | Buffer 25% extra, review weekly |
| Branch 287 lowest demand (20M/day) | Reduce excess cash — optimize capital allocation |
| Total 18B PKR needed (30 days) | Quarterly forecast input for treasury planning |

---

## ✅ Phase 7 — Final Summary

| Item | Value |
|---|---|
| **Method** | Recursive Multi-Step Forecasting |
| **Model Used** | XGBoost (`best_model.pkl`) |
| **Forecast Period** | 2026-04-05 to 2026-05-04 |
| **Branches Forecasted** | 15 |
| **Total Predictions** | 450 rows |
| **Grand Total Need** | 18,049 M PKR (18.05 B PKR) |
| **Plots Saved** | 3 plots (22, 23, 24) |
| **Excel Saved** | `forecast_next30days.xlsx` (17 sheets) |
| **Status** | ✅ COMPLETE |

---

## ➡️ Next Options

| Option | Description |
|---|---|
| **Phase 8 — Dashboard** | Streamlit dashboard — interactive forecasting UI |
| **Model v2** | Improve accuracy by adding holiday/salary cycle features |
| **Auto-Retraining** | Automatically retrain model when new data arrives |

---

## ⚠️ Pipeline Reusability & Architecture Limitations

This pipeline can generate forecasts on new data *without* retraining the model. However, there are two important constraints to note for production deployment:

1. **Standalone Execution vs. Live Dashboard Integration:** Currently, this pipeline is a standalone reusable function (`forecast_pipeline.py`) that exports results to an Excel file. The dashboard statically reads from this generated file. Full live-integration (e.g., auto-triggering the forecast generation via a button when new data is uploaded) is considered a future enhancement.
2. **Continuous Saving:** The pipeline currently appends new data dynamically during execution to compute lags, but does not permanently save this extended history. In a full production environment, this would be updated to persistently save incoming daily data to the database or historical CSV file.
