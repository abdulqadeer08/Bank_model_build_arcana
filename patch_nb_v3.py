import json
with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'X_train = train_df[feature_cols]' in source and 'Match v3_pipeline' in source:
            new_split = '''# Load exact same data and engineer identical features as v3_pipeline.py
half_daily = pd.read_csv('model_data/half_daily_features.csv')
half_daily['start_date'] = pd.to_datetime(half_daily['start_date'])

half_daily['Days_to_Salary'] = half_daily['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)

half_daily = half_daily.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
half_daily['lag_1_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
half_daily['rolling_14_mean_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)

for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    half_daily[f'rolling_14_std_{target}'] = half_daily.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)

half_daily['tran_br_code'] = half_daily['tran_br_code'].astype('category')
half_daily['Half_Day_Total_Debit_RAW'] = half_daily['Half_Day_Total_Debit']

feature_cols = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]

# Match v3_pipeline.py exact split logic
cutoff_idx = int(len(half_daily) * 0.8)
cutoff_date = half_daily.iloc[cutoff_idx]['start_date']

train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()
test_df = half_daily[half_daily['start_date'] >= cutoff_date].copy()

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

print(f"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Train shape : {X_train.shape}")
print(f"Test  shape : {X_test.shape}")
'''
            source = new_split
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
            if cell['source'] and cell['source'][-1].endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Replaced data loading logic!')
