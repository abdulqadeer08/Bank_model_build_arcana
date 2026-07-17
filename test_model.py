import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import random

print("Loading Data and Model...")
# Load preprocessed features from v2_pipeline
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

# Same cutoff as pipeline
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

X_test = test_df[feature_cols]
y_test = test_df['Half_Day_Total_Debit']

# Load model
model = joblib.load('models/v2/model_Half_Day_Total_Debit.pkl')

print("\n--- MODEL PERFORMANCE ON TEST DATA ---")
y_pred = model.predict(X_test)
# Post-process predictions: 0 for Sundays (Weekday=6)
y_pred = np.where(X_test['Weekday'] == 6, 0, y_pred)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Overall MAE (Average Error): PKR {mae/1e6:.2f} Million")
print(f"Overall R² Score: {r2:.4f} (1.0 is perfect)\n")

print("--- 5 RANDOM SAMPLES FROM TEST SET ---")
# Pick 5 random indices
random.seed(42)
sample_indices = random.sample(range(len(test_df)), 5)

for idx in sample_indices:
    row = test_df.iloc[idx]
    actual = row['Half_Day_Total_Debit']
    pred = y_pred[idx]
    
    date_str = row['start_date'].strftime('%Y-%m-%d')
    am_pm = row['AM_PM']
    branch = row['tran_br_code']
    
    print(f"Date: {date_str} ({am_pm}) | Branch: {branch}")
    print(f"   Actual Withdrawal:  PKR {actual/1e6:7.2f} M")
    print(f"   Predicted Amount:   PKR {pred/1e6:7.2f} M")
    print(f"   Error Difference:   PKR {abs(actual-pred)/1e6:7.2f} M")
    print("-" * 50)
