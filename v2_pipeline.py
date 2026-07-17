import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import joblib

warnings.filterwarnings('ignore')

print("Starting V2 Pipeline...")
# 1. Load Data
df = pd.read_excel('Bank Cash Optimization.xlsx')
df['start_date'] = pd.to_datetime(df['start_date'])
print(f"Loaded {len(df)} rows.")

# Handle missing
for col in ['TOTAL_DR', 'TOTAL_CR']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
if df['txn_hour'].isnull().sum() > 0:
    df['txn_hour'].fillna(df['txn_hour'].mode()[0], inplace=True)
if df['start_date'].isnull().sum() > 0:
    df['start_date'].fillna(method='ffill', inplace=True)

# Dedup
df['_total_cash'] = df['TOTAL_DR'] + df['TOTAL_CR']
df = df.sort_values('_total_cash', ascending=False)
df = df.drop_duplicates(subset=['start_date', 'txn_hour', 'tran_br_code'], keep='first')
df = df.drop(columns=['_total_cash']).reset_index(drop=True)

# Winsorize
p99_dr = df['TOTAL_DR'].quantile(0.99)
p99_cr = df['TOTAL_CR'].quantile(0.99)
df['TOTAL_DR'] = df['TOTAL_DR'].clip(upper=p99_dr)
df['TOTAL_CR'] = df['TOTAL_CR'].clip(upper=p99_cr)

# Create AM_PM feature (Half-Day)
df['AM_PM'] = np.where(df['txn_hour'] < 12, 'AM', 'PM')

# Base Features
df['Year'] = df['start_date'].dt.year
df['Month'] = df['start_date'].dt.month
df['Day'] = df['start_date'].dt.day
df['Weekday'] = df['start_date'].dt.dayofweek
df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)

# 2. Add New Features: Salary Day & Holiday
def is_salary_day(day):
    return 1 if day <= 5 or day >= 25 else 0

df['Is_Salary_Day'] = df['Day'].apply(is_salary_day)

# Approximated Public Holidays (Pakistan 2024-2026)
pk_holidays = pd.to_datetime([
    # 2024
    '2024-02-05', '2024-03-23', '2024-04-10', '2024-04-11', '2024-04-12', '2024-05-01',
    '2024-06-17', '2024-06-18', '2024-06-19', '2024-07-16', '2024-07-17', '2024-08-14',
    '2024-09-16', '2024-11-09', '2024-12-25',
    # 2025
    '2025-02-05', '2025-03-23', '2025-03-31', '2025-04-01', '2025-04-02', '2025-05-01',
    '2025-06-06', '2025-06-07', '2025-06-08', '2025-07-05', '2025-07-06', '2025-08-14',
    '2025-09-05', '2025-11-09', '2025-12-25',
    # 2026
    '2026-02-05', '2026-03-20', '2026-03-21', '2026-03-22', '2026-03-23', '2026-05-01',
    '2026-05-27', '2026-05-28', '2026-05-29', '2026-06-25', '2026-06-26', '2026-08-14',
    '2026-08-26', '2026-11-09', '2026-12-25'
])
df['Is_Holiday'] = df['start_date'].isin(pk_holidays).astype(int)

# 3. Aggregate by Half-Day
half_daily = df.groupby(['start_date', 'AM_PM', 'tran_br_code']).agg(
    Half_Day_Total_Debit=('TOTAL_DR', 'sum'),
    Half_Day_Total_Credit=('TOTAL_CR', 'sum'),
    Txn_Count=('TOTAL_DR', 'count'),
    Weekday=('Weekday', 'first'),
    Is_Weekend=('Is_Weekend', 'first'),
    Month=('Month', 'first'),
    Day=('Day', 'first'),
    Is_Salary_Day=('Is_Salary_Day', 'first'),
    Is_Holiday=('Is_Holiday', 'first')
).reset_index()

# Filter out Sundays (Weekday == 6) based on data analysis (only 8 rows in 2 years)
half_daily = half_daily[half_daily['Weekday'] != 6].copy()

half_daily['Half_Day_Net_Cash'] = half_daily['Half_Day_Total_Credit'] - half_daily['Half_Day_Total_Debit']

# Sort chronologically
half_daily = half_daily.sort_values(['start_date', 'AM_PM', 'tran_br_code']).reset_index(drop=True)

# Add Lags per branch
half_daily = half_daily.sort_values(['tran_br_code', 'start_date', 'AM_PM']).reset_index(drop=True)

for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    # Lags: 1 half-day, 2 half-days (1 day), 14 half-days (1 week)
    half_daily[f'lag_1_{target}'] = half_daily.groupby('tran_br_code')[target].shift(1)
    half_daily[f'lag_2_{target}'] = half_daily.groupby('tran_br_code')[target].shift(2)
    half_daily[f'lag_14_{target}'] = half_daily.groupby('tran_br_code')[target].shift(14)
    half_daily[f'lag_60_{target}'] = half_daily.groupby('tran_br_code')[target].shift(60) # 30 days
    
    half_daily[f'rolling_14_mean_{target}'] = half_daily.groupby('tran_br_code')[target].shift(1).rolling(14, min_periods=1).mean().values

half_daily.dropna(subset=['lag_60_Half_Day_Total_Debit'], inplace=True)
half_daily = half_daily.sort_values(['start_date', 'AM_PM']).reset_index(drop=True)

half_daily['AM_PM_Encoded'] = np.where(half_daily['AM_PM'] == 'AM', 0, 1)

feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

cutoff_idx = int(len(half_daily) * 0.8)
cutoff_date = half_daily.iloc[cutoff_idx]['start_date']

train_df = half_daily[half_daily['start_date'] < cutoff_date]
test_df = half_daily[half_daily['start_date'] >= cutoff_date]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

os.makedirs('models/v2', exist_ok=True)
os.makedirs('eda_plots/v2', exist_ok=True)
half_daily.to_csv('model_data/half_daily_features.csv', index=False)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

models = {}
for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    print(f"\nTraining model for {target}...")
    y_train = train_df[target]
    y_test = test_df[target]
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    models[target] = model
    
    joblib.dump(model, f'models/v2/model_{target}.pkl')
    
    y_pred = model.predict(X_test)
    
    # Enforce 0 prediction for Sundays (although filtered, good practice for inference)
    y_pred = np.where(X_test['Weekday'] == 6, 0, y_pred)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Metrics for {target}:")
    print(f"  MAE: {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R2: {r2:,.4f}")
    
    if target != 'Half_Day_Net_Cash':
        print(f"  MAPE: {mape(y_test, y_pred):.2f}%")

# 4. SHAP Explanability (using Outflow model as primary example)
print("\nRunning SHAP Explainer...")
outflow_model = models['Half_Day_Total_Debit']
explainer = shap.TreeExplainer(outflow_model)

# Use a sample of test data to generate plots faster
X_sample = shap.utils.sample(X_test, 1000)
shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig('eda_plots/v2/shap_summary_outflow.png', bbox_inches='tight', dpi=150)
plt.close()

# Evaluate new features specific impact
print("V2 Pipeline completed successfully.")
