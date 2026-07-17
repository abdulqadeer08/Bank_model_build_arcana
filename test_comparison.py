import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import random
import logging

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

print("Loading Data for Side-by-Side Model Comparison...")
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

df['ds'] = df.apply(lambda row: row['start_date'] + pd.Timedelta(hours=8 if row['AM_PM'] == 'AM' else 14), axis=1)
df['y'] = df['Half_Day_Total_Debit']

# Same cutoff
cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
test_df = df[df['start_date'] >= cutoff_date].copy()

feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

# Load V3 Model (XGBoost)
v3_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
v3_preds = np.expm1(v3_model.predict(test_df[feature_cols]))
# zero out sundays
v3_preds = np.where(test_df['Weekday'] == 6, 0, v3_preds)
test_df['V3_ML_Pred'] = v3_preds

# Load Prophet Models
print("Generating Prophet predictions...")
test_df['Prophet_Pred'] = np.nan
branches = test_df['tran_br_code'].unique()
for branch in branches:
    try:
        m = joblib.load(f'models/prophet/model_br_{branch}.pkl')
        b_test = test_df[test_df['tran_br_code'] == branch]
        future = b_test[['ds', 'Is_Holiday', 'Is_Salary_Day']].copy()
        forecast = m.predict(future)
        p_preds = forecast['yhat'].clip(lower=0).values
        p_preds = np.where(b_test['Weekday'] == 6, 0, p_preds)
        test_df.loc[test_df['tran_br_code'] == branch, 'Prophet_Pred'] = p_preds
    except:
        pass

# Drop NaNs if any branch didn't have prophet model
test_df = test_df.dropna(subset=['Prophet_Pred'])

print("\n--- SIDE BY SIDE COMPARISON (5 RANDOM SAMPLES) ---")
random.seed(42)
sample_indices = random.sample(range(len(test_df)), 5)

for idx in sample_indices:
    row = test_df.iloc[idx]
    actual = row['y']
    ml_pred = row['V3_ML_Pred']
    proph_pred = row['Prophet_Pred']
    
    date_str = row['start_date'].strftime('%Y-%m-%d')
    am_pm = row['AM_PM']
    branch = row['tran_br_code']
    
    print(f"\nDate: {date_str} ({am_pm}) | Branch: {branch}")
    print(f"   Actual Withdrawal:      PKR {actual/1e6:7.2f} M")
    print(f"   V3 ML Model Predicted:  PKR {ml_pred/1e6:7.2f} M   --> (Error: {abs(actual-ml_pred)/1e6:5.2f} M)")
    print(f"   Prophet Predicted:      PKR {proph_pred/1e6:7.2f} M   --> (Error: {abs(actual-proph_pred)/1e6:5.2f} M)")
    print("-" * 60)
