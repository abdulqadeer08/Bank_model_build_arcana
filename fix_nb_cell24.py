"""fix_nb_cell24.py — Fix the syntax error in notebook Cell 24 (lag features)."""
import json

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed_source = (
    '# \u2500\u2500 4.3 Lag features \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
    'half_daily = half_daily.sort_values(["tran_br_code", "start_date", "AM_PM"]).reset_index(drop=True)\n'
    '\n'
    'for target in ["Half_Day_Total_Debit", "Half_Day_Total_Credit", "Half_Day_Net_Cash"]:\n'
    '    # Point lags - each explicitly shifted by 1 to avoid leakage\n'
    '    half_daily[f"lag_1_{target}"]  = half_daily.groupby("tran_br_code")[target].shift(1)\n'
    '    half_daily[f"lag_2_{target}"]  = half_daily.groupby("tran_br_code")[target].shift(2)\n'
    '    half_daily[f"lag_14_{target}"] = half_daily.groupby("tran_br_code")[target].shift(14)\n'
    '    half_daily[f"lag_60_{target}"] = half_daily.groupby("tran_br_code")[target].shift(60)\n'
    '    half_daily[f"rolling_14_std_{target}"] = half_daily.groupby("tran_br_code")[target].transform(\n'
    '        lambda x: x.shift(1).rolling(14, min_periods=2).std()\n'
    '    ).fillna(0)\n'
    '    # 14-day rolling mean - shift(1) applied BEFORE rolling to prevent leakage\n'
    '    half_daily[f"rolling_14_mean_{target}"] = half_daily.groupby("tran_br_code")[target].transform(\n'
    '        lambda x: x.shift(1).rolling(14, min_periods=1).mean()\n'
    '    ).fillna(0)\n'
    '    # EWMA - more responsive to recent trends\n'
    '    half_daily[f"ewma_14_{target}"] = half_daily.groupby("tran_br_code")[target].transform(\n'
    '        lambda x: x.shift(1).ewm(span=14, adjust=False).mean()\n'
    '    ).fillna(0)\n'
    '    # Day of week historical average (last 4 same-days)\n'
    '    half_daily[f"dow_avg_4_{target}"] = half_daily.groupby(["tran_br_code", "Weekday", "AM_PM"])[target].transform(\n'
    '        lambda x: x.shift(1).rolling(4, min_periods=1).mean()\n'
    '    ).fillna(0)\n'
    '\n'
    '# Drop early rows where long lags (lag_60) are NaN\n'
    'before = len(half_daily)\n'
    'half_daily = half_daily.dropna(subset=["lag_60_Half_Day_Total_Debit"]).reset_index(drop=True)\n'
    'half_daily["lag_1_Txn_Count"] = half_daily.groupby("tran_br_code")["Txn_Count"].shift(1)\n'
    'half_daily["rolling_14_mean_Txn_Count"] = half_daily.groupby("tran_br_code")["Txn_Count"].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())\n'
    'half_daily["lag_1_Txn_Count"] = half_daily["lag_1_Txn_Count"].fillna(0)\n'
    'half_daily["rolling_14_mean_Txn_Count"] = half_daily["rolling_14_mean_Txn_Count"].fillna(0)\n'
    'half_daily["ewma_14_Txn_Count"] = half_daily.groupby("tran_br_code")["Txn_Count"].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)\n'
    '\n'
    'print(f"Dropped {before - len(half_daily)} warm-up rows. {len(half_daily)} usable records remaining.")\n'
)

fixed = False
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lag_60_Half_Day_Total_Debit' in src and 'rolling_14_mean' in src and 'ewma_14' in src:
        lines = fixed_source.rstrip('\n').split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        print(f'Fixed Cell {i}')
        fixed = True
        break

if not fixed:
    print('ERROR: Target cell not found!')
else:
    with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook saved successfully.')
