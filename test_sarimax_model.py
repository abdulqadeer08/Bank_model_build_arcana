import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("Loading Data for SARIMAX Baseline...")
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

# Pick one major branch for classical time-series to keep it clean, e.g., branch with max transactions
top_branch = df.groupby('tran_br_code')['Txn_Count'].sum().idxmax()
print(f"Selecting Top Branch: {top_branch} for SARIMAX Evaluation")

branch_df = df[df['tran_br_code'] == top_branch].copy()
branch_df = branch_df.sort_values('start_date').reset_index(drop=True)

# Define Target and Exogenous variables
# We use Is_Holiday and Is_Salary_Day as Exogenous variables
target = 'Half_Day_Total_Debit'
exog_cols = ['Is_Holiday', 'Is_Salary_Day']

# Train-Test Split Chronological (80/20)
cutoff_idx = int(len(branch_df) * 0.8)

train_df = branch_df.iloc[:cutoff_idx]
test_df = branch_df.iloc[cutoff_idx:]

y_train = train_df[target]
X_train_exog = train_df[exog_cols]

y_test = test_df[target]
X_test_exog = test_df[exog_cols]

print(f"Training SARIMAX model on {len(y_train)} samples...")
# SARIMAX order: (p, d, q) x (P, D, Q, s)
# Since data is half-daily, daily seasonality is s=2
# Using a basic SARIMAX(1, 0, 1)x(1, 0, 1, 2)
model = SARIMAX(y_train, exog=X_train_exog, order=(1, 0, 1), seasonal_order=(1, 0, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

print("Forecasting on Test Data...")
forecast = results.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test_exog)

# Evaluate
# Ignore 0 values just in case for MAPE, but we use MAE and R2
mae = mean_absolute_error(y_test, forecast)
r2 = r2_score(y_test, forecast)

print("\n--- SARIMAX BASELINE RESULTS ---")
print(f"Branch: {top_branch}")
print(f"MAE (Average Error): PKR {mae/1e6:.2f} Million")
print(f"R² Score: {r2:.4f}")

# Compare with XGBoost V3 for the SAME branch
import joblib
xgb_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]
X_test_xgb = test_df[feature_cols]
xgb_preds_log = xgb_model.predict(X_test_xgb)
xgb_preds = np.expm1(xgb_preds_log)
# Enforce 0 for Sundays
xgb_preds = np.where(X_test_xgb['Weekday'] == 6, 0, xgb_preds)

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

print("\n--- XGBoost (V3) RESULTS ON SAME BRANCH ---")
print(f"MAE (Average Error): PKR {xgb_mae/1e6:.2f} Million")
print(f"R² Score: {xgb_r2:.4f}")
print("\nDone!")
