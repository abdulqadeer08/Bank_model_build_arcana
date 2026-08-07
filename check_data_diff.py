import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
print('Current rows:', len(df))
print('Date range:', df['start_date'].min(), 'to', df['start_date'].max())

df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)

old_rows = 17109
old_cutoff_idx = int(old_rows * 0.8)
old_date = df.iloc[old_cutoff_idx]['start_date'] if old_cutoff_idx < len(df) else 'beyond current'
print(f'Old cutoff_idx ({old_rows}*0.8={old_cutoff_idx}): date = {old_date}')

new_cutoff_idx = int(len(df) * 0.8)
new_date = df.iloc[new_cutoff_idx]['start_date']
print(f'New cutoff_idx ({len(df)}*0.8={new_cutoff_idx}): date = {new_date}')

# Old train/test
old_train = df[df['start_date'] < old_date]
old_test = df[df['start_date'] >= old_date]
print(f'\nWith OLD cutoff ({old_date}): train={len(old_train)}, test={len(old_test)}')
print(f'With NEW cutoff ({new_date}): train={df[df["start_date"] < new_date].shape[0]}, test={df[df["start_date"] >= new_date].shape[0]}')

# How many more rows beyond 2026-04-14?
print(f'\nRows after 2026-04-14: {(df["start_date"] > pd.Timestamp("2026-04-14")).sum()}')
print(f'Latest 10 dates: {sorted(df["start_date"].unique())[-10:]}')
