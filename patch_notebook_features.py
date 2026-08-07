import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 22 modification: Calendar features
cell22 = nb['cells'][22]['source']
new_cell22 = []
for line in cell22:
    new_cell22.append(line)
    if 'half_daily[\'Days_to_Salary\'] = half_daily[\'Days_to_Salary\'].clip(lower=0, upper=25)' in line:
        new_cell22.append("half_daily['Days_Since_Salary'] = half_daily['Day'].apply(lambda d: d - 25 if d >= 25 else d + (31 - 25))\n")
        new_cell22.append("half_daily['Is_Month_Start'] = half_daily['start_date'].dt.is_month_start.astype(int)\n")
        new_cell22.append("half_daily['Is_Month_End'] = half_daily['start_date'].dt.is_month_end.astype(int)\n")
    if 'Calendar features added: Is_Salary_Day, Is_Holiday' in line:
        new_cell22[-1] = 'print("Calendar features added: Is_Salary_Day, Is_Holiday, Days_Since_Salary, Is_Month_Start, Is_Month_End")\n'
    if 'Days_to_Salary\']].head(10)' in line:
        new_cell22[-1] = line.replace('Days_to_Salary\']].head(10)', 'Days_to_Salary\', \'Days_Since_Salary\', \'Is_Month_Start\', \'Is_Month_End\']].head(10)')

nb['cells'][22]['source'] = new_cell22

# Cell 24 modification: Lag features
cell24 = nb['cells'][24]['source']
new_cell24 = []
for line in cell24:
    new_cell24.append(line)
    if 'half_daily[f\'rolling_14_mean_{target}\'] = ' in line:
        new_cell24.extend([
            "    # EWMA - more responsive to recent trends\n",
            "    half_daily[f'ewma_14_{target}'] = half_daily.groupby('tran_br_code')[target].transform(\n",
            "        lambda x: x.shift(1).ewm(span=14, adjust=False).mean()\n",
            "    ).fillna(0)\n",
            "    # Day of week historical average (last 4 same-days)\n",
            "    half_daily[f'dow_avg_4_{target}'] = half_daily.groupby(['tran_br_code', 'Weekday', 'AM_PM'])[target].transform(\n",
            "        lambda x: x.shift(1).rolling(4, min_periods=1).mean()\n",
            "    ).fillna(0)\n"
        ])
    if 'half_daily[\'rolling_14_mean_Txn_Count\'] = half_daily[\'rolling_14_mean_Txn_Count\'].fillna(0)' in line:
        new_cell24.extend([
            "half_daily['ewma_14_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)\n"
        ])

nb['cells'][24]['source'] = new_cell24

# Cell 27 modification: v3_pipeline parity
cell27 = nb['cells'][27]['source']
new_cell27 = []
for line in cell27:
    new_cell27.append(line)
    if 'half_daily[\'Days_to_Salary\'] = ' in line:
        new_cell27.append("half_daily['Days_Since_Salary'] = half_daily['Day'].apply(lambda d: d - 25 if d >= 25 else d + (31 - 25))\n")
        new_cell27.append("half_daily['Is_Month_Start'] = half_daily['start_date'].dt.is_month_start.astype(int)\n")
        new_cell27.append("half_daily['Is_Month_End'] = half_daily['start_date'].dt.is_month_end.astype(int)\n")
    if 'half_daily[\'rolling_14_mean_Txn_Count\'] = ' in line:
        new_cell27.append("half_daily['ewma_14_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)\n")
    if 'half_daily[f\'rolling_14_std_{target}\'] = ' in line:
        new_cell27.extend([
            "    half_daily[f'ewma_14_{target}'] = half_daily.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)\n",
            "    half_daily[f'dow_avg_4_{target}'] = half_daily.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[target].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)\n"
        ])

# update feature cols
new_feat_cols = []
for line in new_cell27:
    if 'feature_cols = [' in line:
        new_feat_cols.append(line)
    elif '    \'Is_Salary_Day\', \'Is_Holiday\',' in line:
        new_feat_cols.append("    'Is_Salary_Day', 'Is_Holiday', 'Days_Since_Salary', 'Is_Month_Start', 'Is_Month_End',\n")
    elif '    \'lag_1_Txn_Count\', \'rolling_14_mean_Txn_Count\',' in line:
        new_feat_cols.append("    'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'ewma_14_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',\n")
    elif 'rolling_14_std_' in line and 'Half_Day_' in line:
        new_feat_cols.append(line.replace("',", "', 'ewma_14_" + line.split("rolling_14_std_")[1].split("',")[0] + "', 'dow_avg_4_" + line.split("rolling_14_std_")[1].split("',")[0] + "',"))
    else:
        new_feat_cols.append(line)

nb['cells'][27]['source'] = new_feat_cols

# Also update the ADF test conclusion explicitly
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell.get('source', []))
        if 'differencing' in src.lower() and 'r2 dropped' not in src.lower():
            # Let's append the supervisor feedback
            nb['cells'][i]['source'].append("\n\n> **Supervisor Note:** Since explicitly differencing the data caused R² to drop from 0.68 to 0.38 in earlier tests, this indicates that the lag features are already successfully capturing the non-stationary information. Thus, we do not need to use explicit differencing for the final models.\n")

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched with new features!")
