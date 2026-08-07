"""
precise_diagnosis.py
====================
Step-by-step comparison of v3_pipeline.py vs notebook Cell 37.
Identifies EVERY difference that could explain R²=0.6785 vs R²=0.4838.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  PRECISE DIAGNOSIS: v3_pipeline.py vs Notebook Cell 37")
print("=" * 70)

# ── POINT 1: best_params comparison ──────────────────────────────────────────
print("\n[POINT 1] best_params comparison")
print("-" * 50)

# Notebook Cell 37 hardcoded params (from current source):
nb_params = {
    'n_estimators': 250,
    'learning_rate': 0.019553708662745254,
    'max_depth': 7,
    'subsample': 0.6557975442608167,
    'colsample_bytree': 0.7168578594140873,
    'min_child_weight': 4,
    'enable_categorical': True,
    'random_state': 42,
    'objective': 'reg:absoluteerror'
}

print("  Notebook Cell 37 params (current hardcoded):")
for k, v in nb_params.items():
    print(f"    {k}: {v!r}")

# Load actual saved model from v3_pipeline.py to get its params
try:
    v3_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
    v3_actual_params = v3_model.get_params()
    print("\n  v3_pipeline.py model (from saved .pkl) actual params:")
    for k in sorted(nb_params.keys()):
        v = v3_actual_params.get(k, 'NOT_PRESENT')
        print(f"    {k}: {v!r}")
    
    print("\n  DIFFERENCES:")
    found_diff = False
    for k in nb_params:
        nb_v = nb_params[k]
        v3_v = v3_actual_params.get(k)
        if nb_v != v3_v:
            print(f"    *** {k}: notebook={nb_v!r}  vs  v3_pkl={v3_v!r}")
            found_diff = True
    if not found_diff:
        print("    (none — params match)")
except Exception as e:
    print(f"  Could not load v3 model: {e}")

# ── POINT 2: Training target (y_train) ───────────────────────────────────────
print("\n[POINT 2] Training target (y_train_log) construction")
print("-" * 50)
print("  v3_pipeline.py (lines 64-76):")
print("    y_train_raw = train_df[target]  # raw series")
print("    cap_val = y_train_raw.quantile(0.99)")
print("    y_train_raw_capped = y_train_raw.clip(upper=cap_val)")
print("    y_train = np.log1p(y_train_raw_capped.clip(lower=0))")
print()
print("  Notebook Cell 35 (LightGBM, also used for XGBoost in Cell 37):")
print("    cap_val = train_df['Half_Day_Total_Debit'].quantile(0.99)")
print("    y_train_raw = train_df['Half_Day_Total_Debit'].clip(upper=cap_val).values")
print("    y_train_log = np.log1p(np.clip(y_train_raw, 0, None))")
print()
print("  >>> CRITICAL: Cell 37 uses y_train_log set in Cell 35.")
print("      Cell 35 clips to .values (numpy) and uses np.clip(lower=0).")
print("      v3_pipeline uses pandas Series and .clip(lower=0) on the capped series.")
print("      FUNCTIONALLY IDENTICAL if cap_val is the same.")

# Verify the cap values
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
for t in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    df[f'rolling_14_std_{t}'] = df.groupby('tran_br_code')[t].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)
df['tran_br_code'] = df['tran_br_code'].astype('category')

cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
train_df = df[df['start_date'] < cutoff_date].copy()
test_df  = df[df['start_date'] >= cutoff_date].copy()

print(f"\n  cap_val (99th pct of train Half_Day_Total_Debit): {train_df['Half_Day_Total_Debit'].quantile(0.99):,.2f}")

# Build y_train_log exactly as v3_pipeline:
y_train_v3 = train_df['Half_Day_Total_Debit']
cap_val = y_train_v3.quantile(0.99)
y_train_v3_capped = y_train_v3.clip(upper=cap_val)
y_train_v3_log = np.log1p(y_train_v3_capped.clip(lower=0))

# Build y_train_log exactly as notebook Cell 35:
cap_val_nb = train_df['Half_Day_Total_Debit'].quantile(0.99)
y_train_nb_raw = train_df['Half_Day_Total_Debit'].clip(upper=cap_val_nb).values
y_train_nb_log = np.log1p(np.clip(y_train_nb_raw, 0, None))

diff = np.abs(y_train_v3_log.values - y_train_nb_log)
print(f"  Max diff between v3 and notebook y_train_log: {diff.max():.2e}")
print(f"  >>> {'IDENTICAL' if diff.max() < 1e-10 else 'DIFFERENT!'}")

# ── POINT 3: What y_train does Cell 37 actually use? ────────────────────────
print("\n[POINT 3] Which y_train does notebook Cell 37 actually use?")
print("-" * 50)
print("  Cell 37 calls: xgb_model.fit(X_train, y_train_log)")
print("  y_train_log is defined in Cell 35 (LightGBM cell)")
print("  So XGBoost in notebook is trained on THE SAME y_train_log as LightGBM")
print()
print("  v3_pipeline.py recomputes y_train FRESH for each target in the loop:")
print("    y_train_raw = train_df[target]")
print("    y_train = np.log1p(y_train_raw.clip(upper=cap_val).clip(lower=0))")
print()
print("  THEY SHOULD BE THE SAME for Half_Day_Total_Debit.")

# ── POINT 4: feature_cols comparison ─────────────────────────────────────────
feature_cols_v3 = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]
feature_cols_nb = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]

print("\n[POINT 4] feature_cols comparison")
print("-" * 50)
print(f"  v3_pipeline.py: {len(feature_cols_v3)} features")
print(f"  Notebook Cell 27: {len(feature_cols_nb)} features")
print(f"  SAME? {feature_cols_v3 == feature_cols_nb}")

# ── POINT 5: eval target — v3 uses y_test_raw (raw), notebook uses y_test_raw ──
print("\n[POINT 5] Evaluation target for XGBoost metrics")
print("-" * 50)
print("  v3_pipeline.py (line 139): mae = mean_absolute_error(y_test_raw, y_pred)")
print("    where y_test_raw = test_df[target]  (raw, uncapped)")
print()
print("  Notebook Cell 35 sets:  y_test_raw = test_df['Half_Day_Total_Debit_RAW'].values")
print("  Notebook Cell 37 uses:  evaluate(y_test_raw, y_pred_xgb, ...)")
print("  Half_Day_Total_Debit_RAW is set in Cell 27 as a copy of Half_Day_Total_Debit")
print("  >>> FUNCTIONALLY SAME unless RAW was set before capping.")

# ── POINT 6: MOST CRITICAL — n_estimators in notebook best_params vs Optuna ──
print("\n[POINT 6] n_estimators: notebook hardcoded=250 vs what Optuna actually found")
print("-" * 50)
print("  v3_pipeline.py search space: n_estimators = suggest_int(100, 500, step=50)")
print("  Notebook Cell 37 hardcoded: n_estimators = 250")
print()
print("  The notebook comment says 'exact hyperparams from v3_pipeline.py'")
print("  But v3_pipeline.py runs Optuna to FIND best_params — it doesn't hardcode them.")
print("  The notebook has stale hardcoded params from a PREVIOUS Optuna run.")
print()
print("  QUESTION: Do these match what Optuna CURRENTLY finds with seed=42 on")
print("  the CURRENT data and XGBoost version?")

# ── POINT 7: Check pipeline_run.log for last v3_pipeline run output ───────────
print("\n[POINT 7] pipeline_run.log — what did last v3_pipeline run actually report?")
print("-" * 50)
try:
    with open('pipeline_run.log', 'r', encoding='utf-8', errors='replace') as f:
        log = f.read()
    print(log[-3000:])  # Last 3000 chars
except Exception as e:
    print(f"  Could not read: {e}")

# ── POINT 8: Load the current saved v3 model and evaluate it directly ─────────
print("\n[POINT 8] Direct evaluation of current models/v3/model_Half_Day_Total_Debit.pkl")
print("-" * 50)
feature_cols = feature_cols_v3
X_train_df = train_df[feature_cols]
X_test_df = test_df[feature_cols]
y_test_raw = test_df['Half_Day_Total_Debit'].values

try:
    v3_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
    preds_log = v3_model.predict(X_test_df)
    preds = np.expm1(preds_log)
    mae = mean_absolute_error(y_test_raw, preds)
    r2  = r2_score(y_test_raw, preds)
    print(f"  Current v3 pkl:  R²={r2:.4f}, MAE={mae/1e6:.2f}M")
    print(f"  Expected (claim): R²=0.6785, MAE=9.36M")
    print(f"  Match R²? {abs(r2 - 0.6785) < 0.01}")
    print(f"  Match MAE? {abs(mae/1e6 - 9.36) < 0.1}")
    print(f"  Actual n_estimators: {v3_model.get_params().get('n_estimators')}")
    print(f"  Actual learning_rate: {v3_model.get_params().get('learning_rate')}")
    print(f"  Actual max_depth: {v3_model.get_params().get('max_depth')}")
except Exception as e:
    print(f"  Error: {e}")
