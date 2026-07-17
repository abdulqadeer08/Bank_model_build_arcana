import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Generating 30-day forecast for V2 Dashboard...")

# Load historical data
df = pd.read_csv('model_data/half_daily_features.csv', parse_dates=['start_date'])
last_date = df['start_date'].max()
branches = df['tran_br_code'].unique()

# Load models
model_debit = joblib.load('models/v2/model_Half_Day_Total_Debit.pkl')
model_credit = joblib.load('models/v2/model_Half_Day_Total_Credit.pkl')
model_net = joblib.load('models/v2/model_Half_Day_Net_Cash.pkl')

feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

# Averages per branch to fill static/naive fields
branch_avgs = df.groupby('tran_br_code')['Txn_Count'].mean().to_dict()

def is_salary_day(day): return 1 if day <= 5 or day >= 25 else 0

pk_holidays = pd.to_datetime([
    '2026-04-10', '2026-04-11', '2026-04-12', '2026-05-01',
    '2026-05-27', '2026-05-28', '2026-05-29', '2026-06-25', '2026-06-26', '2026-08-14',
    '2026-08-26', '2026-11-09', '2026-12-25'
])

forecast_records = []

# We will maintain a history dataframe to calculate lags iteratively
history = df[['start_date', 'AM_PM', 'AM_PM_Encoded', 'tran_br_code', 'Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']].copy()
history = history.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded'])

for i in range(1, 31):
    target_date = last_date + pd.Timedelta(days=i)
    if target_date.weekday() == 6:  # Skip Sundays
        continue
        
    for branch in branches:
        br_hist = history[history['tran_br_code'] == branch]
        
        for am_pm, am_pm_enc in [('AM', 0), ('PM', 1)]:
            row = {}
            row['start_date'] = target_date
            row['tran_br_code'] = branch
            row['AM_PM'] = am_pm
            row['AM_PM_Encoded'] = am_pm_enc
            row['Txn_Count'] = branch_avgs.get(branch, 0)
            row['Weekday'] = target_date.weekday()
            row['Is_Weekend'] = 1 if target_date.weekday() >= 5 else 0
            row['Month'] = target_date.month
            row['Day'] = target_date.day
            row['Is_Salary_Day'] = is_salary_day(target_date.day)
            row['Is_Holiday'] = 1 if target_date in pk_holidays else 0
            
            # Fetch Lags
            # Lags are in half-days. lag_1 = last half day. lag_2 = 1 full day ago half day.
            for t_col in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
                try:
                    row[f'lag_1_{t_col}'] = br_hist[t_col].iloc[-1]
                    row[f'lag_2_{t_col}'] = br_hist[t_col].iloc[-2]
                    row[f'lag_14_{t_col}'] = br_hist[t_col].iloc[-14]
                    row[f'lag_60_{t_col}'] = br_hist[t_col].iloc[-60]
                    row[f'rolling_14_mean_{t_col}'] = br_hist[t_col].tail(14).mean()
                except IndexError:
                    row[f'lag_1_{t_col}'] = 0
                    row[f'lag_2_{t_col}'] = 0
                    row[f'lag_14_{t_col}'] = 0
                    row[f'lag_60_{t_col}'] = 0
                    row[f'rolling_14_mean_{t_col}'] = 0
            
            # Predict
            x_df = pd.DataFrame([row])[feature_cols]
            pred_dr = max(0, model_debit.predict(x_df)[0])
            pred_cr = max(0, model_credit.predict(x_df)[0])
            pred_net = pred_cr - pred_dr # Or use net model, but CR-DR is physically accurate
            
            row['Half_Day_Total_Debit'] = pred_dr
            row['Half_Day_Total_Credit'] = pred_cr
            row['Half_Day_Net_Cash'] = pred_net
            
            # Add to records
            forecast_records.append(row)
            
            # Update history for next iterations
            hist_row = pd.DataFrame([row])[['start_date', 'AM_PM', 'AM_PM_Encoded', 'tran_br_code', 'Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']]
            br_hist = pd.concat([br_hist, hist_row])
        
        # update the main history
        history = pd.concat([history[history['tran_br_code'] != branch], br_hist])

fc_df = pd.DataFrame(forecast_records)
fc_df.to_csv('models/v2/forecast_next30days.csv', index=False)
print("Forecast generated and saved to models/v2/forecast_next30days.csv")
