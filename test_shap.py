import pandas as pd
import numpy as np
import shap
import joblib

# Mock the dashboard's environment
def load_model():
    return joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')

model = load_model()

def load_forecast_features():
    ff = pd.read_csv('models/forecast_features.csv')
    ff['start_date'] = pd.to_datetime(ff['start_date'])
    return ff

forecast_feats = load_forecast_features()
feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

def get_shap_for_row(X_row_am, X_row_pm):
    explainer = shap.TreeExplainer(model)
    
    # AM
    sv_am = explainer.shap_values(X_row_am)[0]
    pred_log_am = model.predict(X_row_am)[0]
    pred_am = np.expm1(pred_log_am)
    base_log = explainer.expected_value
    base_val = np.expm1(base_log)
    
    diff_am = pred_am - base_val
    sum_abs_am = sum(abs(sv_am))
    sv_linear_am = (sv_am / sum_abs_am) * diff_am if sum_abs_am != 0 else sv_am * 0
    
    # PM
    sv_pm = explainer.shap_values(X_row_pm)[0]
    pred_log_pm = model.predict(X_row_pm)[0]
    pred_pm = np.expm1(pred_log_pm)
    
    diff_pm = pred_pm - base_val
    sum_abs_pm = sum(abs(sv_pm))
    sv_linear_pm = (sv_pm / sum_abs_pm) * diff_pm if sum_abs_pm != 0 else sv_pm * 0
    
    # Aggregate
    total_shap = sv_linear_am + sv_linear_pm
    total_base = base_val * 2
    
    return total_shap, total_base, pred_am + pred_pm

# Test Branch 104 on 2026-04-07
branch = 104
target_date = pd.to_datetime('2026-04-07')

rows = forecast_feats[(forecast_feats['tran_br_code'] == branch) & (forecast_feats['start_date'] == target_date)]
am_row = rows[rows['AM_PM'] == 'AM'].iloc[0].to_dict()
pm_row = rows[rows['AM_PM'] == 'PM'].iloc[0].to_dict()

X_am = pd.DataFrame([am_row])[feature_cols]
X_pm = pd.DataFrame([pm_row])[feature_cols]

sv, bv, pred = get_shap_for_row(X_am, X_pm)

print(f"Base Value (Daily): {bv/1e6:.2f} M")
print(f"Predicted (Daily): {pred/1e6:.2f} M")
print("\nTop 5 Drivers:")
pairs = sorted(zip(sv, feature_cols), key=lambda x: abs(x[0]), reverse=True)[:5]
for s, f in pairs:
    print(f"{f}: {s/1e6:+.2f} M")
