# 📋 Phase 7 — Future Forecasting: Explanation & Review

---

## 🎯 Phase 7 ka Maqsad (Objective)

Phase 7 mein humne **Recursive Multi-Step Forecasting** implement ki — yani past data (Feb 2024 – Apr 2026) ke basis par **agle 30 din** ka branch-wise daily cash withdrawal predict kiya.

> **Simple Matlab:** Model ko pehle se pata nahi hota ke kal kya hoga. Isliye hum **kal ki prediction ko parso ka lag feature** ban dete hain — phir parso predict karte hain — aur yeh chain 30 din tak chalti hai!

---

## 🔢 Kya Kiya — Step by Step

### Step 1: Last Known State Identify Kiya

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
    lag_1  = yesterday's value   # actual ya previously predicted
    lag_7  = 7 days ago value
    lag_14 = 14 days ago value
    lag_30 = 30 days ago value
    rolling_7_mean  = average of last 7 days
    rolling_30_mean = average of last 30 days
    
    prediction = model.predict(features)
    
    # KEY STEP: predicted value → history mein add → next day ka lag banega
    history[target_date] = prediction
```

**Unknown Future Features ka Handle:**

| Feature | Solution |
|---|---|
| `Daily_Txn_Count` | Branch-wise historical average use kiya |
| `Branch_Total_Debit` | Branch-wise historical average |
| `Peak_Hour_Txns` | Branch-wise historical average |
| Calendar features | Exactly compute hote hain (weekday, month, etc.) |

---

### Step 3: Uncertainty / Confidence Intervals

Jitna aage predict karo, utna zyada uncertainty badhti hai:

| Horizon | Uncertainty | Confidence |
|---|---|---|
| Day 1–7 | ±5% to ±10% | 🟢 HIGH |
| Day 8–14 | ±10% to ±15% | 🟡 MEDIUM |
| Day 15–30 | ±15% to ±25% | 🔴 LOW |

> **Yeh industry standard hai:** Weather forecast bhi kal accurate hota hai, mahine baad less accurate.

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

- 15 branches ek saath — 5×3 grid
- Green line = past 60 din actual
- Blue dashed = 30-din forecast
- Shaded area = confidence band
- Orange dotted line = forecast start

**Plot 23 — Top 6 Branches Confidence Ribbons**
`eda_plots/23_confidence_ribbons.png`

- Top 6 high-volume branches ka detailed view
- 🟢 GREEN ribbon = Week 1 (HIGH confidence)
- 🟡 YELLOW ribbon = Week 2 (MEDIUM confidence)
- 🔴 RED ribbon = Week 3–4 (LOW confidence)
- Weekly markers dikhate hain kab confidence drop hoti hai

**Plot 24 — Branch × Date Heatmap**
`eda_plots/24_forecast_heatmap.png`

- Rows = Branches, Columns = Dates (every 3rd day shown)
- Darker color = zyada cash chahiye
- Ek nazar mein pata chalta hai kahan aur kab zyada liquidity chahiye

---

### Step 6: Excel Export — 17 Sheets

**File:** `models/forecast_next30days.xlsx`

| Sheet | Contents |
|---|---|
| `All_Branches` | 450 rows — sab branches sab dates |
| `Weekly_Summary` | Branch × Week pivot table |
| `Br_104` | Branch 104 ka detailed 30-day forecast |
| `Br_202` | Branch 202 ka detailed forecast |
| … | (har branch ki alag sheet) |

---

## 📊 Key Insights

| Insight | Business Action |
|---|---|
| Branch 104 → 93.3M/day (30 days) | Highest priority — ensure 100M+ daily liquidity |
| Week 1 predictions → HIGH confidence | Use directly for immediate replenishment |
| Week 3–4 → LOW confidence | Buffer 25% extra, review weekly |
| Branch 287 lowest demand (20M/day) | Reduce excess cash — optimize capital allocation |
| Total 18B PKR needed (30 days) | Treasury planning ke liye quarterly forecast input |

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
| **Model v2** | Holiday/salary cycle features add karke accuracy improve karo |
| **Auto-Retraining** | Naya data aane par model automatically retrain ho |
