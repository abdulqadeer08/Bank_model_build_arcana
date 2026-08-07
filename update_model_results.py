"""
update_model_results.py -- Add SMAPE and WMAPE columns to models/model_results.csv

Since we can't re-run all 4 models here, we add the columns with NaN for
Baseline/Prophet (which will be populated on next notebook Run All), and
compute actual values for XGBoost and LightGBM from their saved models.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from metrics_utils import mape, smape, wmape
import warnings
warnings.filterwarnings('ignore')

# Load existing
mr = pd.read_csv('models/model_results.csv')

# Load test data (same as notebook)
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    df[f'rolling_14_std_{target}'] = df.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)

feature_cols = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]

df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df['tran_br_code'] = df['tran_br_code'].astype('category')

cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
train_df = df[df['start_date'] < cutoff_date]
test_df = df[df['start_date'] >= cutoff_date]

y_test_raw = test_df['Half_Day_Total_Debit'].values

# --- Baseline: 14-day rolling mean ---
y_pred_baseline = test_df['rolling_14_mean_Half_Day_Total_Debit'].values
s_base, _ = smape(y_test_raw, y_pred_baseline)
w_base = wmape(y_test_raw, y_pred_baseline)

# --- XGBoost ---
xgb_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
y_pred_xgb = np.expm1(xgb_model.predict(test_df[feature_cols]))
s_xgb, _ = smape(y_test_raw, y_pred_xgb)
w_xgb = wmape(y_test_raw, y_pred_xgb)

# --- LightGBM ---
try:
    lgb_model = joblib.load('models/lgb_model.pkl')
    y_pred_lgb = np.expm1(lgb_model.predict(test_df[feature_cols]))
    s_lgb, _ = smape(y_test_raw, y_pred_lgb)
    w_lgb = wmape(y_test_raw, y_pred_lgb)
except Exception as e:
    print(f"LightGBM model load issue: {e}")
    s_lgb, w_lgb = np.nan, np.nan

# Prophet: can't easily re-run, but we can compute from the baseline+rolling approach
# We'll mark as NaN — the notebook re-run will populate it
s_prophet, w_prophet = np.nan, np.nan

# Map SMAPE/WMAPE to each model row
smape_map = {
    'Baseline (14-Day Rolling Mean)': round(s_base, 2),
    'Prophet': round(s_prophet, 2) if not np.isnan(s_prophet) else np.nan,
    'LightGBM': round(s_lgb, 2) if not np.isnan(s_lgb) else np.nan,
    'XGBoost V3 (Tuned)': round(s_xgb, 2),
}
wmape_map = {
    'Baseline (14-Day Rolling Mean)': round(w_base, 2),
    'Prophet': round(w_prophet, 2) if not np.isnan(w_prophet) else np.nan,
    'LightGBM': round(w_lgb, 2) if not np.isnan(w_lgb) else np.nan,
    'XGBoost V3 (Tuned)': round(w_xgb, 2),
}

mr['SMAPE'] = mr['Model'].map(smape_map)
mr['WMAPE'] = mr['Model'].map(wmape_map)

mr.to_csv('models/model_results.csv', index=False)
print("Updated models/model_results.csv with SMAPE and WMAPE columns")
print(mr[['Model', 'R2', 'MAE_M', 'MAPE', 'SMAPE', 'WMAPE']].to_string(index=False))
