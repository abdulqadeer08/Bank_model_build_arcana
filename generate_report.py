import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from metrics_utils import smape as calc_smape, wmape as calc_wmape
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

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

df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df['tran_br_code'] = df['tran_br_code'].astype('category')

cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']
test_df = df[df['start_date'] >= cutoff_date]

model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
preds_transformed = model.predict(test_df[feature_cols])
preds = np.expm1(preds_transformed)

test_df_copy = test_df.copy()
test_df_copy['y_pred'] = preds

branch_stats = []
for br, grp in test_df_copy.groupby('tran_br_code'):
    y_true = grp['Half_Day_Total_Debit']
    y_pred = grp['y_pred']
    avg_demand = y_true.mean() / 1e6
    mae = mean_absolute_error(y_true, y_pred) / 1e6
    
    # safe mape
    # We add a small epsilon to avoid div by zero if there's any absolute zero
    y_true_safe = y_true.replace(0, 1)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # SMAPE and WMAPE (supervisor-requested)
    smape_val, _ = calc_smape(y_true.values, y_pred.values)
    wmape_val = calc_wmape(y_true.values, y_pred.values)
    
    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = 0.0
        
    branch_stats.append({
        'Branch': br,
        'Avg_Demand_M': round(avg_demand, 2),
        'MAE_M': round(mae, 2),
        'MAPE_%': round(mape, 1),
        'SMAPE_%': round(smape_val, 1) if not np.isnan(smape_val) else 0.0,
        'WMAPE_%': round(wmape_val, 1) if not np.isnan(wmape_val) else 0.0,
        'R2': round(r2, 4)
    })

report = pd.DataFrame(branch_stats)
# Re-calculate quantiles across branches using MAE to avoid distortion from near-zero-demand outliers
p33 = report['MAE_M'].quantile(0.33)
p66 = report['MAE_M'].quantile(0.66)

def get_trust(mae):
    if mae <= p33: return 'HIGH'
    elif mae <= p66: return 'MEDIUM'
    else: return 'LOW'

def get_buffer(mae):
    if mae <= p33: return 5
    elif mae <= p66: return 12
    else: return 20

def get_action(mae):
    if mae <= p33: return 'Use model directly for replenishment'
    elif mae <= p66: return 'Add safety buffer, monitor weekly'
    else: return 'Use with caution, manual override advised'

report['Trust_Level'] = report['MAE_M'].apply(get_trust)
report['Buffer_%'] = report['MAE_M'].apply(get_buffer)
report['Recommended_M'] = (report['Avg_Demand_M'] + report['MAE_M']) * (1 + report['Buffer_%']/100)
report['Recommended_M'] = report['Recommended_M'].round(2)
report['Action'] = report['MAE_M'].apply(get_action)

report = report.sort_values('MAE_M')
report.to_csv('models/final_evaluation_report.csv', index=False)
print("Updated final_evaluation_report.csv")
print(f"New p33 cutoff (MAE): {p33:.2f}")
print(f"New p66 cutoff (MAE): {p66:.2f}")
