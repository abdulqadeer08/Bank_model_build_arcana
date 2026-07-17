import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
import random
import logging

# Suppress prophet logs
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

print("Loading Data for Prophet Backtesting...")
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

df['ds'] = df.apply(lambda row: row['start_date'] + pd.Timedelta(hours=8 if row['AM_PM'] == 'AM' else 14), axis=1)
df['y'] = df['Half_Day_Total_Debit']

# Same cutoff as ML pipelines (80% Train, 20% Test)
cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']

train_df = df[df['start_date'] < cutoff_date]
test_df = df[df['start_date'] >= cutoff_date]

branches = df['tran_br_code'].unique()

print(f"Training Prophet on Train Data (up to {cutoff_date.date()}) and testing on Test Data...")

all_y_actual = []
all_y_pred = []
test_records = []

for branch in branches:
    b_train = train_df[train_df['tran_br_code'] == branch][['ds', 'y', 'Is_Holiday', 'Is_Salary_Day']].sort_values('ds')
    b_test = test_df[test_df['tran_br_code'] == branch][['ds', 'y', 'Is_Holiday', 'Is_Salary_Day']].sort_values('ds')
    
    if len(b_train) < 10 or len(b_test) == 0:
        continue
        
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_regressor('Is_Holiday')
    m.add_regressor('Is_Salary_Day')
    m.fit(b_train)
    
    # Predict on test dates
    future = b_test[['ds', 'Is_Holiday', 'Is_Salary_Day']].copy()
    forecast = m.predict(future)
    
    preds = forecast['yhat'].clip(lower=0).values
    actuals = b_test['y'].values
    
    # Zero out Sundays for post-processing if needed (Prophet usually learns weekly, but let's be consistent)
    # Weekday 6 is Sunday. ds dayofweek: Monday=0, Sunday=6
    sundays = future['ds'].dt.dayofweek == 6
    preds[sundays] = 0
    
    all_y_actual.extend(actuals)
    all_y_pred.extend(preds)
    
    # Store records for random sampling
    for i in range(len(b_test)):
        test_records.append({
            'Branch': branch,
            'Date': future.iloc[i]['ds'].strftime('%Y-%m-%d'),
            'AM_PM': 'AM' if future.iloc[i]['ds'].hour == 8 else 'PM',
            'Actual': actuals[i],
            'Predicted': preds[i]
        })

overall_mae = mean_absolute_error(all_y_actual, all_y_pred)
overall_r2 = r2_score(all_y_actual, all_y_pred)

print("\n--- PROPHET TIME-SERIES PERFORMANCE ON TEST DATA ---")
print(f"Overall MAE (Average Error): PKR {overall_mae/1e6:.2f} Million")
print(f"Overall R² Score: {overall_r2:.4f}\n")

print("--- 5 RANDOM SAMPLES FROM TEST SET (PROPHET MODEL) ---")
random.seed(42)
sample_indices = random.sample(range(len(test_records)), 5)

for idx in sample_indices:
    rec = test_records[idx]
    actual = rec['Actual']
    pred = rec['Predicted']
    
    print(f"Date: {rec['Date']} ({rec['AM_PM']}) | Branch: {rec['Branch']}")
    print(f"   Actual Withdrawal:  PKR {actual/1e6:7.2f} M")
    print(f"   Predicted Amount:   PKR {pred/1e6:7.2f} M")
    print(f"   Error Difference:   PKR {abs(actual-pred)/1e6:7.2f} M")
    print("-" * 50)
