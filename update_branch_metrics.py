"""
update_branch_metrics.py -- Add SMAPE/WMAPE columns to models/branch_metrics.csv

This is a one-time script to update the existing branch_metrics.csv with SMAPE/WMAPE
columns before the full notebook is re-run (which will generate them natively).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from metrics_utils import mape, smape, wmape
import warnings
warnings.filterwarnings('ignore')

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
test_df = df[df['start_date'] >= cutoff_date]

model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
preds_transformed = model.predict(test_df[feature_cols])
preds = np.expm1(preds_transformed)

test_copy = test_df.copy()
test_copy['y_pred'] = preds

branch_metrics = []
for br, grp in test_copy.groupby('tran_br_code', observed=True):
    ya = grp['Half_Day_Total_Debit'].values
    yp = grp['y_pred'].values
    
    br_mae = mean_absolute_error(ya, yp)
    br_r2 = r2_score(ya, yp) if len(ya) > 1 else 0.0
    br_mape = mape(ya, yp)
    br_smape, _ = smape(ya, yp)
    br_wmape = wmape(ya, yp)
    
    branch_metrics.append({
        'Branch': int(br),
        'Avg_Demand_M': round(ya.mean() / 1e6, 2),
        'MAE_M': round(br_mae / 1e6, 2),
        'MAPE_%': round(br_mape, 1) if not np.isnan(br_mape) else 0.0,
        'SMAPE_%': round(br_smape, 1) if not np.isnan(br_smape) else 0.0,
        'WMAPE_%': round(br_wmape, 1) if not np.isnan(br_wmape) else 0.0,
        'R2': round(br_r2, 4),
    })

bm_df = pd.DataFrame(branch_metrics).sort_values('Avg_Demand_M', ascending=False)
bm_df.to_csv('models/branch_metrics.csv', index=False)
print("Updated models/branch_metrics.csv with SMAPE_% and WMAPE_% columns")
print(bm_df.to_string(index=False))
