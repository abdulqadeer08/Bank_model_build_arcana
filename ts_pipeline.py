import os
import warnings
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib

warnings.filterwarnings('ignore')

print("Starting Prophet Time-Series Pipeline...")

# 1. Load Data
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

# 2. Format specifically for Prophet
# Prophet needs 'ds' (datetime) and 'y' (target)
# For AM/PM, we set AM = 08:00:00, PM = 14:00:00
df['ds'] = df.apply(lambda row: row['start_date'] + pd.Timedelta(hours=8 if row['AM_PM'] == 'AM' else 14), axis=1)

# We will forecast Outflow (Debit)
df['y'] = df['Half_Day_Total_Debit']

# Get list of branches
branches = df['tran_br_code'].unique()
print(f"Found {len(branches)} branches. Training Prophet models...")

os.makedirs('models/prophet', exist_ok=True)

# 3. Train Branch-Specific Prophet Models
all_forecasts = []

for branch in branches:
    # Get branch data
    branch_df = df[df['tran_br_code'] == branch][['ds', 'y', 'Is_Holiday', 'Is_Salary_Day']].copy()
    branch_df = branch_df.sort_values('ds').reset_index(drop=True)
    
    # Initialize Prophet with holiday/seasonality awareness
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False # we use 12h intervals, but daily seasonality inside 2 points is tricky, we let it be
    )
    
    # Add external regressors (Supervisor's requirement)
    m.add_regressor('Is_Holiday')
    m.add_regressor('Is_Salary_Day')
    
    # Fit model
    m.fit(branch_df)
    
    # Save model
    joblib.dump(m, f'models/prophet/model_br_{branch}.pkl')
    
    # 4. Predict Future (30 Days = 60 Half-Days)
    future = m.make_future_dataframe(periods=60, freq='12H')
    
    # We need to provide the regressors for the future dates
    def get_holiday(dt):
        return 1 if dt.strftime('%Y-%m-%d') in [
            '2026-02-05', '2026-03-20', '2026-03-21', '2026-03-22', '2026-03-23', '2026-05-01',
            '2026-05-27', '2026-05-28', '2026-05-29', '2026-06-25', '2026-06-26', '2026-08-14',
            '2026-08-26', '2026-11-09', '2026-12-25', '2026-04-10', '2026-04-11' # approximated
        ] else 0

    def get_salary_day(dt):
        return 1 if dt.day <= 5 or dt.day >= 25 else 0

    future['Is_Holiday'] = future['ds'].apply(get_holiday)
    future['Is_Salary_Day'] = future['ds'].apply(get_salary_day)
    
    forecast = m.predict(future)
    
    # Extract useful columns
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'weekly', 'yearly']]
    forecast['Branch'] = branch
    
    # Prevent negative cash predictions
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    
    all_forecasts.append(forecast)

# Combine all forecasts
final_forecast = pd.concat(all_forecasts, ignore_index=True)
final_forecast.to_csv('models/prophet_forecast.csv', index=False)

print("\n✅ Prophet Time Series Training Complete!")
print(f"Saved {len(branches)} models to models/prophet/")
print("Generated future forecast saved to models/prophet_forecast.csv")
