import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from metrics_utils import mape, smape, wmape
import joblib
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

train_df = df[df['start_date'] < cutoff_date]
test_df = df[df['start_date'] >= cutoff_date]

print("--- 5. LAG BOUNDARY CHECK (First 5 rows of test_df) ---")
print(test_df[['start_date', 'tran_br_code', 'AM_PM_Encoded', 'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit']].head())

y_train_raw = train_df['Half_Day_Total_Debit']
y_test_raw = test_df['Half_Day_Total_Debit']

cap_val = y_train_raw.quantile(0.99)
num_above = (y_train_raw > cap_val).sum()
median_val = y_train_raw.median()

print("\n--- 4. OUTLIER STATS ---")
print(f"99th Percentile (cap_val): {cap_val}")
print(f"Rows strictly above 99th percentile: {num_above}")
print(f"Median value: {median_val}")
print(f"Ratio 99th to Median: {cap_val / median_val if median_val > 0 else 'N/A'}")

model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
preds_transformed = model.predict(test_df[feature_cols])
preds = np.expm1(preds_transformed)

test_df_copy = test_df.copy()
test_df_copy['y_test_capped'] = y_test_raw.clip(upper=cap_val)
test_df_copy['pred'] = preds
test_df_copy['abs_err'] = np.abs(test_df_copy['y_test_capped'] - test_df_copy['pred'])

print("\n--- CAPPED OVERALL MAE ---")
print(test_df_copy['abs_err'].mean())

test_df_copy['y_test_honest'] = y_test_raw
test_df_copy['abs_err_honest'] = np.abs(test_df_copy['y_test_honest'] - test_df_copy['pred'])

print("\n--- HONEST OVERALL MAE ---")
print(test_df_copy['abs_err_honest'].mean())

print("\n--- 1. MAE BY BRANCH (HONEST) ---")
branch_mae = test_df_copy.groupby('tran_br_code')['abs_err_honest'].mean().sort_values(ascending=False)
print(branch_mae.head(5))

print("\n--- 2. MAE BY WEEKDAY (HONEST) ---")
weekday_mae = test_df_copy.groupby('Weekday')['abs_err_honest'].mean().sort_values(ascending=False)
print(weekday_mae)

# --- SMAPE & WMAPE (supervisor-requested) ---
print("\n--- 3. OVERALL SMAPE & WMAPE ---")
overall_mape = mape(test_df_copy['y_test_honest'].values, test_df_copy['pred'].values)
overall_smape, exc = smape(test_df_copy['y_test_honest'].values, test_df_copy['pred'].values)
overall_wmape = wmape(test_df_copy['y_test_honest'].values, test_df_copy['pred'].values)
print(f"MAPE:  {overall_mape:.2f}%")
print(f"SMAPE: {overall_smape:.2f}%  (excluded {exc} both-zero rows)")
print(f"WMAPE: {overall_wmape:.2f}%  << recommended for bank presentations")

print("\n--- 4. SMAPE & WMAPE BY BRANCH (TOP 5) ---")
branch_smape_wmape = []
for br, grp in test_df_copy.groupby('tran_br_code'):
    ya = grp['y_test_honest'].values
    yp = grp['pred'].values
    s_val, _ = smape(ya, yp)
    w_val = wmape(ya, yp)
    m_val = mape(ya, yp)
    branch_smape_wmape.append({'Branch': br, 'MAPE': round(m_val, 1), 'SMAPE': round(s_val, 1), 'WMAPE': round(w_val, 1)})
bsw = pd.DataFrame(branch_smape_wmape).sort_values('MAPE', ascending=False)
print(bsw.head(5).to_string(index=False))
