# 📋 Phase 3 — Model Preparation: Explanation & Review

---

## 🎯 Phase 3 ka Maqsad (Objective)

Phase 3 mein humne raw feature-engineered data ko **model training ke liye ready** kiya.
Yeh step EDA aur Feature Engineering ke baad ka zaroori bridge hai jahan se actual ML shuru hota hai.

---

## 🔢 Kya Kiya — Step by Step

### Step 1: Libraries Import
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `os`, `json` import kiye
- `model_data/` aur `eda_plots/` folders create kiye

---

### Step 2: Dataset Load
- **File:** `bank_cash_optimization_features.csv`
- **Rows:** 81,709
- **Columns:** 41
- **Date Range:** 2024-01-02 → 2026-04-04
- **Branches:** 15 unique branches
- **Missing Values:** 0 (clean data)

---

### Step 3: Target Variable Define ki

> **Target = `Daily_Total_Debit`**

| | |
|---|---|
| **Kya predict karega?** | Ek branch par ek din mein kitna cash withdraw hoga (PKR) |
| **Kyun Debit?** | Debit = Cash outflow = Yahi bank ko replenish karna hota hai |
| **Business Meaning** | Agar kal Branch X se 5 Crore niklega, toh aaj hi cash arrange karo |

---

### Step 4: Transaction Level → Daily Branch Level Aggregation

**Problem:** Original data transaction-level tha (har row = ek transaction)  
**Solution:** Har branch ka daily summary banaya (har row = ek branch + ek din)

| Pehle | Baad |
|---|---|
| 81,709 rows (transactions) | 9,032 rows (branch × day) |
| Har row = 1 transaction | Har row = 1 branch ka 1 din ka summary |

**Aggregated features:**
- `Daily_Total_Debit` — din bhar ka total cash withdrawal
- `Daily_Total_Credit` — din bhar ka total cash deposit
- `Daily_Txn_Count` — transactions ki taadaad
- Time features (Weekday, Month, Year, Day, etc.)
- Branch-level features (Branch_Total_Debit, Branch_Txn_Count, etc.)
- Peak/Business hour transaction counts

---

### Step 5: Lag Features Add kiye (Time Series ka Core)

Lag features = **model ko past dikhana** taakay woh future predict kar sake.

| Feature | Matlab |
|---|---|
| `lag_1_debit` | Kal kitna cash nikla |
| `lag_7_debit` | Ek hafta pehle same din kitna nikla |
| `lag_14_debit` | 2 hafte pehle same din |
| `lag_30_debit` | 1 mahina pehle same din |
| `rolling_7_mean_debit` | Pichle 7 din ka average |
| `rolling_30_mean_debit` | Pichle 30 din ka average |

> ⚠️ **Important:** Lag features per BRANCH compute kiye — har branch ka apna history hai

**Rows dropped:** 450 (woh rows jahan 30-day history available nahi thi)  
**Rows remaining:** 8,582

---

### Step 6: Feature aur Target Alag kiye

**Drop kiye gaye columns (leakage risk):**
- `start_date` — numeric nahi, split ke liye use kiya
- `Daily_Total_Debit` — yeh TARGET hai, feature nahi
- `Daily_Total_Credit` — target se highly correlated, leakage
- `Daily_Net_Cash_Flow` — target ki info contain karta hai
- `Net_Outflow_Periods` — target se derived

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

> ⚠️ **Time Series mein shuffle NAHI karte — future data train mein nahi jaana chahiye**

```
|← TRAIN (79.8%) →|← TEST (20.2%) →|
| Feb 2024 → Oct 2025 | Nov 2025 → Apr 2026 |
|    6,851 rows       |    1,731 rows        |
```

**Kyun chronological?**  
Agar random shuffle karte toh Nov 2025 ka data train mein aur Sep 2025 ka data test mein jata — model "future dekh ke" train hota, jo real life mein cheating hai (Data Leakage).

---

### Step 8 & 9: Visualizations + Leakage Verification

**Graphs banaye:**
- `08_train_test_split.png` — Train/Test split timeline
- `09_target_analysis.png` — Target variable distribution (by weekday, by branch)

**Leakage Check Results:**
| Check | Result |
|---|---|
| Train-Test date overlap | ✅ 0 overlap |
| Shuffle applied? | ✅ No |
| Future data in training? | ✅ Not present |

---

### Step 11: Files Save kiye

| File | Description |
|---|---|
| `model_data/X_train.csv` | Training features (6,851 × 22) |
| `model_data/X_test.csv` | Test features (1,731 × 22) |
| `model_data/y_train.csv` | Training target values |
| `model_data/y_test.csv` | Test target values |
| `model_data/daily_full.csv` | Complete dataset (final model ke liye) |
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
