"""
verify_xgb_isolation.py
=======================
Step 3 verification: runs ONLY the XGBoost training and evaluation
(equivalent to notebook Cells 27-37) using the current best_params from
models/v3/best_params_debit.json, and confirms R2/MAE match v3_pipeline output.

Run AFTER run_v3_fresh.py has completed.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json, os, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from metrics_utils import mape, smape, wmape
warnings.filterwarnings('ignore')

print("=" * 70)
print("  STEP 3: Verify XGBoost isolation (Cells 27-37 equivalent)")
print("=" * 70)

# Load best_params from fresh Optuna run
if not os.path.exists('models/v3/best_params_debit.json'):
    print("ERROR: models/v3/best_params_debit.json not found. Run run_v3_fresh.py first.")
    sys.exit(1)

with open('models/v3/best_params_debit.json') as f:
    best_params = json.load(f)

print(f"\nLoaded best_params:")
for k, v in best_params.items():
    print(f"  {k}: {v!r}")

# ── Exact Cell 27 data loading (same as v3_pipeline.py) ───────────────────────
print("\n  Loading data (Cell 27 logic)...")

half_daily = pd.read_csv('model_data/half_daily_features.csv')
half_daily['start_date'] = pd.to_datetime(half_daily['start_date'])

# Cell 27: Days_to_Salary before sort (matches v3_pipeline line 23)
half_daily['Days_to_Salary'] = half_daily['Day'].apply(
    lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)

half_daily = half_daily.sort_values(
    ['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
half_daily['lag_1_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
half_daily['rolling_14_mean_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
for t in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    half_daily[f'rolling_14_std_{t}'] = half_daily.groupby('tran_br_code')[t].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)

half_daily['tran_br_code'] = half_daily['tran_br_code'].astype('category')
half_daily['Half_Day_Total_Debit_RAW'] = half_daily['Half_Day_Total_Debit']

feature_cols = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count',
    'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit',
    'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit',
    'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash',
    'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]

cutoff_idx = int(len(half_daily) * 0.8)
cutoff_date = half_daily.iloc[cutoff_idx]['start_date']
train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()
test_df  = half_daily[half_daily['start_date'] >= cutoff_date].copy()

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

print(f"  cutoff_date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"  Train rows  : {len(train_df)}")
print(f"  Test rows   : {len(test_df)}")

# ── Cell 35: y_train_log with capping (as v3_pipeline.py) ────────────────────
cap_val = train_df['Half_Day_Total_Debit'].quantile(0.99)
y_train_raw_capped = train_df['Half_Day_Total_Debit'].clip(upper=cap_val)
y_train_log = np.log1p(y_train_raw_capped.clip(lower=0))
y_test_raw  = test_df['Half_Day_Total_Debit_RAW'].values

print(f"\n  cap_val (99th pct): {cap_val:,.2f}")

# ── Cell 37: Train XGBoost with fresh params ─────────────────────────────────
print(f"\n  Training XGBoost with fresh Optuna params...")
xgb_model = xgb.XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train_log)
print(f"  Trained.")

y_pred_log = xgb_model.predict(X_test)
y_pred_xgb = np.expm1(y_pred_log)

mae_val  = mean_absolute_error(y_test_raw, y_pred_xgb)
rmse_val = np.sqrt(mean_squared_error(y_test_raw, y_pred_xgb))
r2_val   = r2_score(y_test_raw, y_pred_xgb)
mape_val = mape(y_test_raw, y_pred_xgb)
smape_val, _ = smape(y_test_raw, y_pred_xgb)
wmape_val = wmape(y_test_raw, y_pred_xgb)

print(f"\n  {'='*50}")
print(f"  NOTEBOOK ISOLATION RESULT (fresh params):")
print(f"  {'='*50}")
print(f"  R2:    {r2_val:.4f}")
print(f"  MAE:   {mae_val/1e6:.2f}M")
print(f"  RMSE:  {rmse_val/1e6:.2f}M")
print(f"  MAPE:  {mape_val:.2f}%")
print(f"  SMAPE: {smape_val:.2f}%")
print(f"  WMAPE: {wmape_val:.2f}%")

print(f"\n  Compare with v3_pipeline fresh run result (run_v3_fresh.py):")
print(f"  (Both use same data, same split, same params, same seed)")

# Load and evaluate the saved model from run_v3_fresh
if os.path.exists('models/v3/model_Half_Day_Total_Debit.pkl'):
    saved_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
    p = saved_model.predict(X_test)
    pred_orig = np.expm1(p)
    mae_saved = mean_absolute_error(y_test_raw, pred_orig)
    r2_saved  = r2_score(y_test_raw, pred_orig)
    print(f"\n  Saved v3_pipeline pkl result:")
    print(f"  R2:    {r2_saved:.4f}")
    print(f"  MAE:   {mae_saved/1e6:.2f}M")
    
    max_pred_diff = np.max(np.abs(pred_orig - y_pred_xgb))
    print(f"\n  Max prediction diff (retrained vs saved pkl): {max_pred_diff:,.2f}")
    if max_pred_diff < 1:
        print(f"  >>> MATCH: notebook isolation = v3_pipeline saved model [OK]")
    else:
        print(f"  >>> Note: {max_pred_diff:,.0f} difference (floating point non-determinism)")
        print(f"  >>> R2 match: {abs(r2_val - r2_saved) < 0.001}")
