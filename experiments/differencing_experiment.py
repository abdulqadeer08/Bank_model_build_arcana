"""
Differencing Experiment — Controlled Comparison vs Production Model
=====================================================================
Fair, isolated test: same features, same train/test split, same Optuna config (seed=42, 
20 trials), same objective='reg:absoluteerror' — ONLY the target changes to a per-branch 
differenced series.

Inverse transform: predicted_level[T] = actual_level[T-1] + predicted_diff[T]
Evaluation: honest MAE, R², MAPE on reconstructed real-scale predictions vs raw uncapped actuals.
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
import optuna
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET = 'Half_Day_Total_Debit'

# 1. Load Data — same as v3_pipeline.py
print("="*70)
print("DIFFERENCING EXPERIMENT — Controlled Comparison")
print("="*70)

df = pd.read_csv('data/processed/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
print(f"Loaded {len(df)} rows of data.")

df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)

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

df['tran_br_code'] = df['tran_br_code'].astype('category')

# 2. Create Per-Branch Differenced Target
print("\n--- Creating per-branch differenced target ---")

# CRITICAL: diff within each branch separately to avoid cross-branch leakage
df['diff_target'] = df.groupby('tran_br_code')[TARGET].diff(1)

# Verify no cross-branch leakage: first row of each branch should be NaN
# Note: groupby().first() skips NaN by default, so use nth(0) to get the literal first row
first_rows = df.groupby('tran_br_code').nth(0)
assert first_rows['diff_target'].isna().all(), "LEAKAGE: First row of some branches has non-NaN diff!"
print("  [PASS] Cross-branch leakage check PASSED (first row of each branch is NaN)")

# Store the previous actual value for inverse transformation
df['prev_actual'] = df.groupby('tran_br_code')[TARGET].shift(1)

# Drop rows where diff is NaN (first row per branch)
df_diff = df.dropna(subset=['diff_target']).copy()
print(f"  Rows after dropping first-per-branch NaN: {len(df_diff)} (dropped {len(df) - len(df_diff)})")

# 3. ADF Test on Differenced Series
print("\n--- ADF Test on Differenced Series ---")

branches = sorted(df_diff['tran_br_code'].unique())
diff_adf_results = []

for br in branches:
    br_series = df_diff[df_diff['tran_br_code'] == br]['diff_target'].dropna()
    if len(br_series) < 20:
        diff_adf_results.append({'Branch': br, 'p_value': np.nan, 'verdict': 'INSUFFICIENT DATA'})
        continue
    result = adfuller(br_series, autolag='AIC')
    p = result[1]
    diff_adf_results.append({'Branch': br, 'p_value': round(p, 6), 'verdict': 'STATIONARY' if p < 0.05 else 'NON-STATIONARY'})

diff_adf_df = pd.DataFrame(diff_adf_results)
n_stat = (diff_adf_df['verdict'] == 'STATIONARY').sum()
print(f"  After differencing: {n_stat}/{len(branches)} branches are stationary")
for _, row in diff_adf_df.iterrows():
    print(f"    Branch {row['Branch']}: p={row['p_value']:.6f} → {row['verdict']}")

# 4. Chronological Split — same as v3_pipeline.py
# Use the original df for cutoff calculation (same as production)
cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
print(f"\nCutoff date: {cutoff_date} (same as production)")

# Now split the differenced df
train_df = df_diff[df_diff['start_date'] < cutoff_date]
test_df = df_diff[df_diff['start_date'] >= cutoff_date]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 5. Target: log1p of differenced values (matching production's log1p)
# NOTE: Differenced values can be negative (value dropped from T-1 to T).
# log1p only works for non-negative values. We must decide:
#   Option A: Use log1p on diff (but diff can be negative → invalid)
#   Option B: Use raw diff without log transform
# Since differencing produces positive AND negative values, we CANNOT apply log1p.
# We use the raw differenced target directly.

y_train_raw_diff = train_df['diff_target']
y_test_raw_diff = test_df['diff_target']

# For evaluation: we need the actual raw (undifferenced) test values and prev_actual
y_test_actual_raw = test_df[TARGET].values
prev_actual_test = test_df['prev_actual'].values

# Outlier capping on differenced target (same 99th percentile approach)
cap_val = y_train_raw_diff.quantile(0.99)
cap_val_low = y_train_raw_diff.quantile(0.01)
y_train_capped = y_train_raw_diff.clip(lower=cap_val_low, upper=cap_val)
y_train = y_train_capped  # No log transform for differenced target

print(f"\nDiff target stats: mean={y_train_raw_diff.mean():.0f}, std={y_train_raw_diff.std():.0f}")
print(f"Cap range: [{cap_val_low:.0f}, {cap_val:.0f}]")

# 6. Optuna Hyperparameter Tuning — same config as v3_pipeline.py
print(f"\nRunning Optuna Optimization (20 trials, seed=42)...")

from sklearn.model_selection import TimeSeriesSplit

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
        
        preds_diff = model.predict(X_v)
        # For tuning, evaluate MAE on the differenced scale (fair within-experiment)
        error = mean_absolute_error(y_v, preds_diff)
        errors.append(error)
        
    return np.mean(errors)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)

best_params = study.best_params
print(f"Best Optuna Parameters: {best_params}")

# 7. Train Final Model
print("Training final differenced model with best parameters...")
final_model = xgb.XGBRegressor(**best_params, enable_categorical=True, random_state=42, objective='reg:absoluteerror')
final_model.fit(X_train, y_train)

# 8. Predict & Inverse Transform (CRITICAL STEP)
print("\n--- Inverse Transformation ---")
print("Formula: predicted_level[T] = actual_level[T-1] + predicted_diff[T]")
print("Using TRUE actual value at T-1 (not prior prediction) for fair one-step-ahead evaluation")

predicted_diff = final_model.predict(X_test)

# Reconstruct actual predicted levels
# predicted_level[T] = actual_level[T-1] + predicted_diff[T]
predicted_level = prev_actual_test + predicted_diff

# Show a few examples of the reconstruction
print("\nSample reconstruction (first 5 test points):")
print(f"  {'prev_actual':>15} + {'pred_diff':>12} = {'pred_level':>15} | {'actual':>15}")
for i in range(min(5, len(predicted_level))):
    print(f"  {prev_actual_test[i]:>15,.0f} + {predicted_diff[i]:>12,.0f} = {predicted_level[i]:>15,.0f} | {y_test_actual_raw[i]:>15,.0f}")

# 9. Honest Evaluation — Same metrics as production
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae_diff = mean_absolute_error(y_test_actual_raw, predicted_level)
rmse_diff = np.sqrt(mean_squared_error(y_test_actual_raw, predicted_level))
r2_diff = r2_score(y_test_actual_raw, predicted_level)
mape_diff = mape(y_test_actual_raw, predicted_level)

print(f"\n{'='*70}")
print("HONEST COMPARISON — Side by Side")
print(f"{'='*70}")
print(f"{'Metric':>10} | {'Production (no diff)':>22} | {'With Differencing':>22}")
print(f"{'-'*10}-+-{'-'*22}-+-{'-'*22}")
print(f"{'R²':>10} | {'0.6785':>22} | {r2_diff:>22.4f}")
print(f"{'MAE':>10} | {'9.36M PKR':>22} | {mae_diff/1e6:>19.2f}M PKR")
print(f"{'RMSE':>10} | {'15.72M PKR':>22} | {rmse_diff/1e6:>19.2f}M PKR")
print(f"{'MAPE':>10} | {'132.7%':>22} | {mape_diff:>21.1f}%")

# 10. Decision
print(f"\n{'='*70}")
print("DECISION")
print(f"{'='*70}")

mae_prod = 9.36e6
r2_prod = 0.6785
mape_prod = 132.7

# Check if differencing is a meaningful improvement
mae_improvement = (mae_prod - mae_diff) / mae_prod * 100
r2_improvement = (r2_diff - r2_prod) / abs(r2_prod) * 100 if r2_prod != 0 else 0
mape_improvement = (mape_prod - mape_diff) / mape_prod * 100

print(f"MAE  change: {mae_improvement:+.1f}% ({'improved' if mae_improvement > 0 else 'worsened'})")
print(f"R²   change: {r2_improvement:+.1f}% ({'improved' if r2_improvement > 0 else 'worsened'})")
print(f"MAPE change: {mape_improvement:+.1f}% ({'improved' if mape_improvement > 0 else 'worsened'})")

# Threshold: at least 5% improvement in MAE to be considered meaningful
THRESHOLD = 5.0
if mae_improvement > THRESHOLD and r2_improvement > 0:
    print(f"\n>> RECOMMENDATION: ADOPT differencing (MAE improved by {mae_improvement:.1f}%)")
    adopted = True
else:
    print(f"\n>> RECOMMENDATION: DO NOT ADOPT differencing")
    if mae_improvement <= THRESHOLD:
        print(f"   Reason: MAE {'improvement' if mae_improvement > 0 else 'degradation'} of {mae_improvement:.1f}% does not exceed the {THRESHOLD}% significance threshold")
    if r2_improvement <= 0:
        print(f"   Reason: R² {'did not improve' if r2_improvement == 0 else 'worsened'} ({r2_improvement:+.1f}%)")
    print("   The existing lag/rolling features (lag_1, lag_2, lag_14, lag_60, rolling_14_mean)")
    print("   already implicitly capture trend/level changes, making explicit differencing redundant.")
    adopted = False

# Save decision for downstream use
decision = {
    'adopted': adopted,
    'production_mae': mae_prod,
    'production_r2': r2_prod,
    'production_mape': mape_prod,
    'diff_mae': float(mae_diff),
    'diff_r2': float(r2_diff),
    'diff_mape': float(mape_diff),
    'diff_rmse': float(rmse_diff),
    'mae_improvement_pct': float(mae_improvement),
    'r2_improvement_pct': float(r2_improvement),
    'mape_improvement_pct': float(mape_improvement)
}

with open('models/differencing_decision.json', 'w') as f:
    json.dump(decision, f, indent=2)
print(f"\nDecision saved to models/differencing_decision.json")

# 11. Determinism Check — Run prediction a second time
print("\n--- Determinism Check ---")
final_model_2 = xgb.XGBRegressor(**best_params, enable_categorical=True, random_state=42, objective='reg:absoluteerror')
final_model_2.fit(X_train, y_train)
predicted_diff_2 = final_model_2.predict(X_test)
predicted_level_2 = prev_actual_test + predicted_diff_2

mae_2 = mean_absolute_error(y_test_actual_raw, predicted_level_2)
if np.isclose(mae_diff, mae_2, rtol=1e-10):
    print(f"  [PASS] DETERMINISTIC: Run 1 MAE = {mae_diff:.6f}, Run 2 MAE = {mae_2:.6f} (identical)")
else:
    print(f"  [FAIL] NON-DETERMINISTIC: Run 1 MAE = {mae_diff:.6f}, Run 2 MAE = {mae_2:.6f}")

print(f"\n{'='*70}")
print("DIFFERENCING EXPERIMENT COMPLETE")
print(f"{'='*70}")
