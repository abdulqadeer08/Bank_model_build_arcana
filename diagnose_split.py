"""
diagnose_split.py
Prints the exact cutoff_date and train/test row counts for BOTH orderings:
  A) v3_pipeline.py ordering: Days_to_Salary BEFORE sort
  B) eval_mae.py / verify_new_metrics.py ordering: Sort FIRST, Days_to_Salary AFTER

This definitively proves whether the two scripts use the same or different splits.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

print("=" * 70)
print("  SPLIT MISMATCH DIAGNOSTIC")
print("=" * 70)

# --------------------------------------------------------------------------
# PATH A: v3_pipeline.py order
#   1. Load
#   2. Days_to_Salary (line 23 of v3_pipeline.py)
#   3. Sort (line 25)
#   4. Lag features
#   5. cutoff
# --------------------------------------------------------------------------
print("\n[PATH A] v3_pipeline.py order: Days_to_Salary BEFORE sort")
df_A = pd.read_csv('model_data/half_daily_features.csv')
df_A['start_date'] = pd.to_datetime(df_A['start_date'])

# Step A2: Days_to_Salary BEFORE sort (as in v3_pipeline line 23)
df_A['Days_to_Salary'] = df_A['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)

# Step A3: sort (as in v3_pipeline line 25)
df_A = df_A.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)

# Step A4: lag features
df_A['lag_1_Txn_Count'] = df_A.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df_A['rolling_14_mean_Txn_Count'] = df_A.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    df_A[f'rolling_14_std_{target}'] = df_A.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)

# Step A5: cutoff
cutoff_idx_A = int(len(df_A) * 0.8)
cutoff_date_A = df_A.iloc[cutoff_idx_A]['start_date']
train_A = df_A[df_A['start_date'] < cutoff_date_A]
test_A = df_A[df_A['start_date'] >= cutoff_date_A]

print(f"  Total rows  : {len(df_A)}")
print(f"  cutoff_idx  : {cutoff_idx_A}")
print(f"  cutoff_date : {cutoff_date_A}")
print(f"  Train rows  : {len(train_A)}")
print(f"  Test rows   : {len(test_A)}")

# --------------------------------------------------------------------------
# PATH B: eval_mae.py / verify_new_metrics.py order
#   1. Load
#   2. Sort FIRST (line 13)
#   3. Lag features
#   4. Days_to_Salary AFTER sort (line 27)
#   5. cutoff
# --------------------------------------------------------------------------
print("\n[PATH B] eval_mae.py order: Sort FIRST, Days_to_Salary AFTER")
df_B = pd.read_csv('model_data/half_daily_features.csv')
df_B['start_date'] = pd.to_datetime(df_B['start_date'])

# Step B2: Sort FIRST
df_B = df_B.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)

# Step B3: lag features
df_B['lag_1_Txn_Count'] = df_B.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df_B['rolling_14_mean_Txn_Count'] = df_B.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    df_B[f'rolling_14_std_{target}'] = df_B.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)

# Step B4: Days_to_Salary AFTER sort
df_B['Days_to_Salary'] = df_B['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)

# Step B5: cutoff
cutoff_idx_B = int(len(df_B) * 0.8)
cutoff_date_B = df_B.iloc[cutoff_idx_B]['start_date']
train_B = df_B[df_B['start_date'] < cutoff_date_B]
test_B = df_B[df_B['start_date'] >= cutoff_date_B]

print(f"  Total rows  : {len(df_B)}")
print(f"  cutoff_idx  : {cutoff_idx_B}")
print(f"  cutoff_date : {cutoff_date_B}")
print(f"  Train rows  : {len(train_B)}")
print(f"  Test rows   : {len(test_B)}")

# --------------------------------------------------------------------------
# COMPARISON
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)
same_date = cutoff_date_A == cutoff_date_B
same_train = len(train_A) == len(train_B)
same_test = len(test_A) == len(test_B)

print(f"  cutoff_date SAME? {'YES - identical' if same_date else 'NO - DIFFERENT!'}")
print(f"    Path A: {cutoff_date_A}")
print(f"    Path B: {cutoff_date_B}")
print(f"  train rows SAME? {'YES' if same_train else 'NO'} (A={len(train_A)}, B={len(train_B)})")
print(f"  test rows  SAME? {'YES' if same_test else 'NO'} (A={len(test_A)}, B={len(test_B)})")

if same_date and same_train and same_test:
    print("\n  >>> VERDICT: Both pipelines use IDENTICAL splits. No mismatch.")
    print("  >>> The R2/MAE difference must come from something else.")
else:
    print("\n  >>> VERDICT: SPLIT MISMATCH CONFIRMED.")
    print("  >>> verify_new_metrics.py uses a DIFFERENT split from v3_pipeline.py!")
