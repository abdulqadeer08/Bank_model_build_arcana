"""
run_v3_fresh.py
===============
Re-runs v3_pipeline.py from scratch (deletes cached pkl first) on current data.
Reports actual R2/MAE and best_params found by Optuna on current data.
Also saves the best_params to a JSON file for the notebook to use.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from metrics_utils import mape, smape, wmape

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("  STEP 4: Re-running v3_pipeline.py FRESH on current data")
print("  (Deleting old .pkl first to ensure clean retraining)")
print("=" * 70)

# Delete old model to confirm clean retraining
old_pkl = 'models/v3/model_Half_Day_Total_Debit.pkl'
if os.path.exists(old_pkl):
    os.remove(old_pkl)
    print(f"  Deleted old: {old_pkl}")

# ── Exact same logic as v3_pipeline.py ───────────────────────────────────────
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
print(f"  Loaded {len(df)} rows (pipeline_run.log had 17109, current has {len(df)})")

# v3_pipeline.py line 23: Days_to_Salary BEFORE sort
df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)

# v3_pipeline.py line 25: sort
df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    df[f'rolling_14_std_{target}'] = df.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)

feature_cols = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]

df['tran_br_code'] = df['tran_br_code'].astype('category')

cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
train_df = df[df['start_date'] < cutoff_date]
test_df  = df[df['start_date'] >= cutoff_date]

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

print(f"  cutoff_date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"  Train size : {len(X_train)}")
print(f"  Test size  : {len(X_test)}")
print(f"  pipeline_run.log had: Train=16975, Test=134  (DIFFERENT DATASET)")

# Only train for Half_Day_Total_Debit (the one we care about)
target = 'Half_Day_Total_Debit'
y_train_raw = train_df[target]
y_test_raw  = test_df[target]

use_log = True
cap_val = y_train_raw.quantile(0.99)
y_train_raw_capped = y_train_raw.clip(upper=cap_val)
y_train = np.log1p(y_train_raw_capped.clip(lower=0))

print(f"\n  cap_val (99th pct): {cap_val:,.2f}")
print(f"  Running Optuna (20 trials, seed=42)...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True,
        'random_state': 42
    }
    tscv = TimeSeriesSplit(n_splits=3)
    errors = []
    for train_index, val_index in tscv.split(X_train):
        X_t, X_v = X_train.iloc[train_index], X_train.iloc[val_index]
        y_t, y_v = y_train.iloc[train_index], y_train.iloc[val_index]
        model = xgb.XGBRegressor(**params, early_stopping_rounds=20, objective='reg:absoluteerror')
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        preds = model.predict(X_v)
        preds_orig = np.expm1(preds)
        y_v_orig = np.expm1(y_v)
        error = mean_absolute_error(y_v_orig, preds_orig)
        errors.append(error)
    return np.mean(errors)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)
best_params = study.best_params

print(f"\n  Best Optuna params (current data, seed=42):")
for k, v in best_params.items():
    print(f"    {k}: {v!r}")

print(f"\n  Training final model with best params...")
os.makedirs('models/v3', exist_ok=True)
final_model = xgb.XGBRegressor(**best_params, enable_categorical=True, random_state=42, objective='reg:absoluteerror')
final_model.fit(X_train, y_train)
joblib.dump(final_model, old_pkl)
print(f"  Saved: {old_pkl}")

y_pred_transformed = final_model.predict(X_test)
y_pred = np.expm1(y_pred_transformed)

mae_val  = mean_absolute_error(y_test_raw, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test_raw, y_pred))
r2_val   = r2_score(y_test_raw, y_pred)
mape_val = mape(y_test_raw.values, y_pred)
smape_val, _ = smape(y_test_raw.values, y_pred)
wmape_val = wmape(y_test_raw.values, y_pred)

print(f"\n  {'='*50}")
print(f"  CURRENT v3_pipeline RESULTS (fresh Optuna run):")
print(f"  {'='*50}")
print(f"  R2:    {r2_val:.4f}   (old claim: 0.6785,  log says: 0.6628)")
print(f"  MAE:   {mae_val/1e6:.2f}M  (old claim: 9.36M,  log says: 9.37M)")
print(f"  RMSE:  {rmse_val/1e6:.2f}M")
print(f"  MAPE:  {mape_val:.2f}%")
print(f"  SMAPE: {smape_val:.2f}%")
print(f"  WMAPE: {wmape_val:.2f}%")

# Save best_params to JSON for notebook to use
params_for_notebook = dict(best_params)
params_for_notebook['enable_categorical'] = True
params_for_notebook['random_state'] = 42
params_for_notebook['objective'] = 'reg:absoluteerror'

with open('models/v3/best_params_debit.json', 'w') as f:
    json.dump(params_for_notebook, f, indent=2)
print(f"\n  Saved best_params to: models/v3/best_params_debit.json")
print(f"  Content: {json.dumps(params_for_notebook, indent=2)}")

print(f"\n  {'='*50}")
print(f"  COMPARISON: old log vs current run")
print(f"  {'='*50}")
print(f"  OLD LOG (17109 rows): R2=0.6628, MAE=9.37M")
print(f"  CURRENT  ({len(df)} rows): R2={r2_val:.4f}, MAE={mae_val/1e6:.2f}M")
print(f"  Dataset has grown by {len(df) - 17109} rows since old run.")
print(f"  Optuna with seed=42 finds different params on different data.")
