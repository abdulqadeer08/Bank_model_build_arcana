# 📋 Phase 3 — Model Preparation: Explanation & Review

---

## 🎯 Phase 3 Objective

In Phase 3, we prepared the raw feature-engineered data for **model training**.
This step is the essential bridge after EDA and Feature Engineering where actual ML begins.

---

## 🔢 Actions Taken — Step by Step

### Step 1: Libraries Import
- Imported `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `os`, `json`
- Created `model_data/` and `eda_plots/` directories

---

### Step 2: Dataset Load
- **File:** `bank_cash_optimization_features.csv`
- **Rows:** 81,709
- **Columns:** 41
- **Date Range:** 2024-01-02 → 2026-04-04
- **Branches:** 15 unique branches
- **Missing Values:** 0 (clean data)

---

### Step 3: Target Variable Definition

> **Target = `Daily_Total_Debit`**

| | |
|---|---|
| **What will it predict?** | How much cash will be withdrawn from a branch in a day (PKR) |
| **Why Debit?** | Debit = Cash outflow = This is what the bank needs to replenish |
| **Business Meaning** | If Branch X requires 5 Crore tomorrow, arrange the cash today |

---

### Step 4: Transaction Level → Daily Branch Level Aggregation

**Problem:** Original data was transaction-level (each row = one transaction)  
**Solution:** Created a daily summary for each branch (each row = one branch + one day)

| Before | After |
|---|---|
| 81,709 rows (transactions) | 9,032 rows (branch × day) |
| Each row = 1 transaction | Each row = 1 day summary for 1 branch |

**Aggregated features:**
- `Daily_Total_Debit` — total cash withdrawal for the day
- `Daily_Total_Credit` — total cash deposit for the day
- `Daily_Txn_Count` — number of transactions
- Time features (Weekday, Month, Year, Day, etc.)
- Branch-level features (Branch_Total_Debit, Branch_Txn_Count, etc.)
- Peak/Business hour transaction counts

---

### Step 5: Added Lag Features (Core of Time Series)

Lag features = **showing the model the past** so it can predict the future.

| Feature | Meaning |
|---|---|
| `lag_1_debit` | Cash withdrawn yesterday |
| `lag_7_debit` | Cash withdrawn exactly one week ago |
| `lag_14_debit` | Cash withdrawn exactly two weeks ago |
| `lag_30_debit` | Cash withdrawn exactly one month ago |
| `rolling_7_mean_debit` | Average of the past 7 days |
| `rolling_30_mean_debit` | Average of the past 30 days |

> ⚠️ **Important:** Lag features were computed per BRANCH — each branch has its own history

**Rows dropped:** 450 (rows where 30-day history was not available)  
**Rows remaining:** 8,582

---

### Step 6: Separated Features and Target

**Dropped columns (leakage risk):**
- `start_date` — not numeric, used for splitting
- `Daily_Total_Debit` — this is the TARGET, not a feature
- `Daily_Total_Credit` — target se highly correlated, leakage
- `Daily_Net_Cash_Flow` — contains target information
- `Net_Outflow_Periods` — derived from target

**Final Features (X): 22 features**
```
tran_br_code, Daily_Txn_Count, Weekday, Is_Weekend, Month, Year, Day,
Weekday_Sin, Weekday_Cos, Month_Sin, Month_Cos,
Branch_Total_Debit, Branch_Txn_Count, Branch_Avg_Net_CF,
Peak_Hour_Txns, Business_Hour_Txns,
lag_1_debit, lag_7_debit, lag_14_debit, lag_30_debit,
rolling_7_mean_debit, rolling_30_mean_debit
```

---

### Step 7: Chronological Train/Test Split (80/20)

> ⚠️ **In Time Series, we DO NOT shuffle — future data should not leak into training**

```
|← TRAIN (79.8%) →|← TEST (20.2%) →|
| Feb 2024 → Oct 2025 | Nov 2025 → Apr 2026 |
|    6,851 rows       |    1,731 rows        |
```

**Why chronological?**  
If we had shuffled randomly, Nov 2025 data would go to train and Sep 2025 data to test — the model would train by "looking into the future", which is cheating in real life (Data Leakage).

---

### Step 8 & 9: Visualizations + Leakage Verification

**Generated Graphs:**
- `08_train_test_split.png` — Train/Test split timeline
- `09_target_analysis.png` — Target variable distribution (by weekday, by branch)

**Leakage Check Results:**
| Check | Result |
|---|---|
| Train-Test date overlap | ✅ 0 overlap |
| Shuffle applied? | ✅ No |
| Future data in training? | ✅ Not present |

---

### Step 11: Saved Files

| File | Description |
|---|---|
| `model_data/X_train.csv` | Training features (6,851 × 22) |
| `model_data/X_test.csv` | Test features (1,731 × 22) |
| `model_data/y_train.csv` | Training target values |
| `model_data/y_test.csv` | Test target values |
| `model_data/daily_full.csv` | Complete dataset (for final model) |
| `model_data/feature_cols.json` | Feature names list |

---

## ✅ Phase 3 — Final Summary

| Item | Value |
|---|---|
| Input Data | 81,709 transactions |
| After Aggregation | 9,032 daily-branch rows |
| After Lag Drop | 8,582 rows |
| Total Features | 22 |
| Target Variable | `Daily_Total_Debit` |
| Train Period | Feb 2024 → Oct 2025 |
| Test Period | Nov 2025 → Apr 2026 |
| Data Leakage | ❌ None |
| Status | ✅ COMPLETE |

---

## ➡️ Next: Phase 4 — Model Training
