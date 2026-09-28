"""
generate_shap.py — Regenerates SHAP values using the V3 XGBoost+LightGBM ensemble.
Uses the XGBoost component for SHAP (TreeExplainer works best with XGBoost).
Feature set matches v3_pipeline.py exactly.
"""
import pandas as pd
import numpy as np
import joblib
import json
import shap
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df['Days_Since_Salary'] = df['Day'].apply(lambda d: d - 25 if d >= 25 else d + (31 - 25))
df['Is_Month_Start'] = df['start_date'].dt.is_month_start.astype(int)
df['Is_Month_End'] = df['start_date'].dt.is_month_end.astype(int)

df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)

# Recompute all v3_pipeline features
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
df['ewma_14_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)

for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    df[f'rolling_14_std_{target}'] = df.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)
    df[f'ewma_14_{target}'] = df.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)
    df[f'dow_avg_4_{target}'] = df.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[target].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)

feature_cols = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'ewma_14_Txn_Count',
    'Days_to_Salary', 'Days_Since_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit',
    'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'rolling_14_std_Half_Day_Total_Debit', 'ewma_14_Half_Day_Total_Debit', 'dow_avg_4_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit',
    'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'rolling_14_std_Half_Day_Total_Credit', 'ewma_14_Half_Day_Total_Credit', 'dow_avg_4_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash',
    'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash',
    'rolling_14_std_Half_Day_Net_Cash', 'ewma_14_Half_Day_Net_Cash', 'dow_avg_4_Half_Day_Net_Cash'
]

df['tran_br_code'] = df['tran_br_code'].astype('category')
df['Weekday'] = df['Weekday'].astype('category')
df['Month'] = df['Month'].astype('category')

cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
test_df = df[df['start_date'] >= cutoff_date].copy()
print(f"Test set: {len(test_df)} rows from {cutoff_date.date()}")

# Load the V3 ensemble — extract XGB component for SHAP
ensemble = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')

if isinstance(ensemble, dict):
    # V3 ensemble format: {'xgb': model, 'lgb': model, 'w_xgb': float}
    model_xgb = ensemble['xgb']
    w_xgb = ensemble['w_xgb']
    print(f"Loaded V3 ensemble (XGB weight={w_xgb:.3f}). Using XGB component for SHAP.")
else:
    # Legacy single model
    model_xgb = ensemble
    print("Loaded single model (legacy format) for SHAP.")

X_test = test_df[feature_cols]
print(f"Computing SHAP values for {len(X_test)} test rows, {len(feature_cols)} features...")

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)

# Save expected value
base_value = explainer.expected_value
if isinstance(base_value, np.ndarray):
    base_value = float(base_value[0])
else:
    base_value = float(base_value)

with open('models/shap_expected_value.json', 'w') as f:
    json.dump({'expected_value': base_value}, f)

# Save shap values
sv_df = pd.DataFrame(shap_values, columns=feature_cols)
sv_df.to_csv('models/shap_values.csv', index=False)

print(f"Saved models/shap_values.csv  ({sv_df.shape[0]} rows x {sv_df.shape[1]} features)")
print(f"Saved models/shap_expected_value.json (base value = {base_value:.4f})")

# Print top SHAP features
mean_abs = sv_df.abs().mean().sort_values(ascending=False)
print("\nTop 10 features by mean |SHAP|:")
for feat, val in mean_abs.head(10).items():
    print(f"  {feat}: {val:.4f}")
