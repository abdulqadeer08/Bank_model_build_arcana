# Bank Cash Flow Forecasting System - Complete Project Documentation

This document serves as the single source of truth for the Bank Cash Flow Forecasting Machine Learning project. It details the entire lifecycle of the data, the algorithms used, the business logic implemented, and the final dashboard interface.


 ## 1. Executive Summary
The goal of this project is to predict the daily cash demand (withdrawals/debits) for various bank branches over a **30-day future horizon**. By accurately forecasting cash needs, the bank can optimize vault balances—avoiding critical cash shortages that frustrate customers, while minimizing idle cash that could otherwise be invested. The system relies on an advanced Machine Learning model (**XGBoost**) and provides explainable, transparent insights via an interactive dashboard.

---

## 2. The Dataset
The model learns from historical transaction data. The raw dataset contains the following key columns:
- **`start_date`**: The date of the transaction.
- **`txn_hour`**: The hour the transaction took place (0-23).
- **`tran_br_code`**: The unique identifier code for the bank branch.
- **`TOTAL_DR`**: Total Debit (Cash withdrawn from the branch).
- **`TOTAL_CR`**: Total Credit (Cash deposited into the branch).

*Note: Cash Forecasting primarily focuses on predicting `TOTAL_DR` (withdrawals), as this dictates the cash required in the vault.*


## 3. Data Processing & Aggregation
Machine learning models perform poorly on raw, noisy hourly data. To solve this, the raw dataset goes through a complex preprocessing pipeline:

### The "Half-Daily" Aggregation Strategy
Instead of predicting directly on a daily level, we aggregate the data into **Half-Daily** intervals (AM and PM).
- **AM Interval:** Any transaction where `txn_hour < 12`.
- **PM Interval:** Any transaction where `txn_hour >= 12`.

**Why?** This allows the AI to learn intraday banking patterns. For example, a branch might have high corporate withdrawals in the morning, but heavy retail deposits in the afternoon. 

* **Net Cash Calculation:** `Half_Day_Net_Cash` = `Half_Day_Total_Credit` - `Half_Day_Total_Debit`.
* **Sunday Handling:** Sundays are included in both training and prediction, since historical data confirms real non-zero Sunday cash activity (average ~11.9M PKR across branches, likely from ATM operations). The model learns Sunday's distinct pattern via the Is_Weekend and Weekday features, producing realistically lower — but never zero — predictions for Sundays.


## 4. Feature Engineering
We "engineer" new columns (features) to give the AI context about *why* transactions are happening.

### A. Calendar & Temporal Features
- **`Month`, `Day`, `Weekday`**: Helps the model understand seasonal trends (e.g., end-of-year rush, Monday spikes).
- **`Is_Weekend`**: Flags Saturdays.
- **`AM_PM_Encoded`**: 0 for AM, 1 for PM.

### B. Business Logic Features
- **`Is_Salary_Day`**: Set to `1` if the day is between the 1st-5th or 25th-31st of the month. Salary days consistently drive massive cash withdrawals.
- **`Is_Holiday`**: Cross-references the date with a static list of Pakistan's public holidays (Eid, Independence Day, Kashmir Day, etc.). Holidays usually see high cash demand right before they begin, and zero demand during.

### C. Lag Features (The Model's "Memory")
To predict tomorrow, the model needs to know what happened recently. We shift historical data to create "Lags":
- **`lag_1`**: What happened 1 half-day ago?
- **`lag_2`**: What happened yesterday at this exact time?
- **`lag_14`**: What happened exactly one week ago?
- **`lag_60`**: What happened exactly one month ago?

### D. Rolling Averages
- **14-Day Rolling Mean:** A smoothed average of the last 14 days of withdrawals. This helps the model ignore one-off random spikes and focus on the broader trend.

---

## 5. The Machine Learning Model (XGBoost V3)
During development, multiple algorithms were tested, including Facebook's Prophet (a standard time-series model). The final production model is **XGBoost (eXtreme Gradient Boosting)**.

### Why XGBoost?
XGBoost uses "decision trees" and is incredibly powerful at finding non-linear relationships (e.g., "If it is a Monday AND a Salary Day AND a week before Eid, then increase demand by 300%"). Standard models struggle with these complex overlapping conditions.

### Training Strategy
- **TimeSeriesSplit:** Because time moves in one direction, we cannot randomly split train/test data (that would allow the model to cheat by looking at the future). We used a 3-fold rolling window to simulate real-world forecasting.
- **Hyperparameter Tuning:** Optuna was used to find the perfect settings for XGBoost (learning rate, tree depth, etc.).

### Overall Performance Metrics
- **MAE (Mean Absolute Error):** **9.52 Million PKR**. On average, predictions are within 9.5M PKR of the actual truth.
- **R² Score:** **0.6334**. The model successfully explains 63% of the variance in the highly chaotic cash flow data (a very strong score for financial behavioral data).



## 6. Explainable AI (SHAP)
A major requirement for financial systems is "Explainability." Bank managers will not trust a black-box AI.
We implemented **SHAP (SHapley Additive exPlanations)**. 

When the model predicts 50M PKR for tomorrow, SHAP breaks down exactly how it got that number.
* *Base Average:* 30M
* *+ 15M* because `Is_Salary_Day` is True.
* *+ 8M* because `14-Day Rolling Average` is trending upward.
* *- 3M* because it is the `PM` half of the day.
* *Final = 50M.*

This logic drives the "Why This Amount" waterfall charts in the dashboard.


## 7. Confidence Tiers & Uncertainty Bounds
Every forecast comes with a margin of error. We grade every branch into **HIGH**, **MEDIUM**, or **LOW** confidence tiers.

### The Methodology (Dynamic Cohort Percentiles)
We **do not** use standard absolute thresholds (like "under 10% error is HIGH"). Cash flow data has many "low value" days where a prediction of 2M instead of 1M looks like a 100% error, artificially inflating the MAPE (Mean Absolute Percentage Error).

Instead, we use **Dynamic Relative Percentiles**:
The system looks at the historical MAPE of *all* branches and calculates the 33rd (p33) and 66th (p66) percentiles. Currently:
- **HIGH Confidence:** Bottom third of branches (MAPE ≤ 39.86%). These are the most predictable, stable branches.
- **MEDIUM Confidence:** Middle third of branches (39.86% < MAPE ≤ 47.64%).
- **LOW Confidence:** Top third of branches (MAPE > 47.64%). These branches are highly volatile and their forecasts require manual managerial review.

If the model is retrained and overall accuracy improves, these percentiles dynamically recalculate, ensuring the tiers always accurately represent the "best, average, and worst" branches.


## 8. The Dashboard Application
The system is accessed via an interactive Streamlit Web Dashboard containing 8 main modules:

1. **Overview:** A high-level view showing which branches have the highest priority/volume and the total 30-day network cash requirement.
2. **Cash Need Calendar:** A 30-day grid specifically designed for branch managers. It shows the daily predicted cash need, the date, and a color-coded Confidence badge (with an informational tooltip explaining the relative percentile logic).
3. **Branch Forecast:** A line chart that seamlessly stitches together past historical data (in grey) with the 30-day future prediction (in blue).
4. **Why This Amount (SHAP):** An interactive tool allowing users to click on any specific future day and see the SHAP waterfall chart and a plain-English explanation of exactly what drove the prediction.
5. **Model Comparison:** A technical page showing how XGBoost outperformed the baseline Prophet model.
6. **Model Performance:** Full transparency into the AI's accuracy. It lists the exact MAE, R², and MAPE for every individual branch, alongside its assigned Quality tier.
7. **Import Data:** The operational engine. Users can upload a fresh CSV file containing new transaction data. The pipeline automatically:
   - Preprocesses the data into half-daily intervals.
   - Generates 30 days of future forecasts using the XGBoost model.
   - Calculates uncertainty bounds based on branch MAPE.
   - Re-saves the `forecast_next30days.xlsx` file.
   - Clears the system cache, instantly refreshing every single page of the dashboard with the new data.
8. **What-If Simulator:** Allows managers to test hypothetical scenarios — toggling holiday/salary-day flags or adjusting historical volume multipliers for a specific branch/date — and instantly see how the prediction shifts, along with an explanation of what changed.
