"""fix_nb_feature_cols.py — Fix the broken feature_cols cell in the notebook."""
import json

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The correct feature_cols that match v3_pipeline.py exactly
CORRECT_FEATURE_COLS = [
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

fixed_source = (
    "# Load exact same data and engineer identical features as v3_pipeline.py\n"
    "import os\n"
    "if os.path.exists('model_data/half_daily_features.csv'):\n"
    "    half_daily = pd.read_csv('model_data/half_daily_features.csv')\n"
    "elif 'half_daily' in locals():\n"
    "    pass\n"
    "else:\n"
    "    raise FileNotFoundError('model_data/half_daily_features.csv not found. Please run Section 4 cells first.')\n"
    "half_daily['start_date'] = pd.to_datetime(half_daily['start_date'])\n"
    "half_daily['Days_to_Salary'] = half_daily['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)\n"
    "half_daily['Days_Since_Salary'] = half_daily['Day'].apply(lambda d: d - 25 if d >= 25 else d + (31 - 25))\n"
    "half_daily['Is_Month_Start'] = half_daily['start_date'].dt.is_month_start.astype(int)\n"
    "half_daily['Is_Month_End'] = half_daily['start_date'].dt.is_month_end.astype(int)\n"
    "half_daily = half_daily.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)\n"
    "half_daily['lag_1_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)\n"
    "half_daily['rolling_14_mean_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)\n"
    "half_daily['ewma_14_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)\n"
    "for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:\n"
    "    half_daily[f'rolling_14_std_{target}'] = half_daily.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)\n"
    "    half_daily[f'ewma_14_{target}'] = half_daily.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)\n"
    "    half_daily[f'dow_avg_4_{target}'] = half_daily.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[target].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)\n"
    "half_daily['tran_br_code'] = half_daily['tran_br_code'].astype('category')\n"
    "half_daily['Weekday'] = half_daily['Weekday'].astype('category')\n"
    "half_daily['Month'] = half_daily['Month'].astype('category')\n"
    "\n"
    "# Exact feature_cols matching v3_pipeline.py\n"
    "feature_cols = [\n"
    "    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'ewma_14_Txn_Count',\n"
    "    'Days_to_Salary', 'Days_Since_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',\n"
    "    'Is_Salary_Day', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End',\n"
    "    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit',\n"
    "    'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',\n"
    "    'rolling_14_std_Half_Day_Total_Debit', 'ewma_14_Half_Day_Total_Debit', 'dow_avg_4_Half_Day_Total_Debit',\n"
    "    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit',\n"
    "    'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',\n"
    "    'rolling_14_std_Half_Day_Total_Credit', 'ewma_14_Half_Day_Total_Credit', 'dow_avg_4_Half_Day_Total_Credit',\n"
    "    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash',\n"
    "    'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash',\n"
    "    'rolling_14_std_Half_Day_Net_Cash', 'ewma_14_Half_Day_Net_Cash', 'dow_avg_4_Half_Day_Net_Cash'\n"
    "]\n"
    "\n"
    "# Match v3_pipeline.py exact split logic\n"
    "cutoff_idx = int(len(half_daily) * 0.8)\n"
    "cutoff_date = half_daily.iloc[cutoff_idx]['start_date']\n"
    "train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()\n"
    "test_df  = half_daily[half_daily['start_date'] >= cutoff_date].copy()\n"
    "X_train = train_df[feature_cols]\n"
    "X_test  = test_df[feature_cols]\n"
    "print(f\"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}\")\n"
    "print(f\"Train shape : {X_train.shape}\")\n"
    "print(f\"Test  shape : {X_test.shape}\")\n"
)

fixed = False
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    # Find the broken cell: it has the malformed feature_cols with Net_Cash items split across lines
    if ("dow_avg_4_Half_Day_Net_Cash" in src and 
        "ewma_14_Half_Day_Net_Cash" in src and 
        "cutoff_idx = int(len(half_daily) * 0.8)" in src and
        ("Half_Day_Total_Debit_RAW" in src or "unterminated" not in src)):
        lines = fixed_source.rstrip('\n').split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        print(f'Fixed Cell {i}: feature_cols data prep cell')
        fixed = True
        break

if not fixed:
    print('Searching more broadly...')
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        if "lag_60_Half_Day_Net_Cash" in src and "ewma_14_Half_Day_Net_Cash" in src and "cutoff_idx" in src:
            lines = fixed_source.rstrip('\n').split('\n')
            cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
            print(f'Fixed Cell {i} (broad match)')
            fixed = True
            break

if not fixed:
    print('ERROR: Target cell not found!')
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        if 'ewma_14_Half_Day_Net_Cash' in src:
            print(f'  Cell {i} has ewma_14_Half_Day_Net_Cash')
else:
    with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook saved successfully.')
