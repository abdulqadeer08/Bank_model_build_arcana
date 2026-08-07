import pandas as pd
import numpy as np
import joblib
import shap

print("Loading model and data...")
model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
ff = pd.read_csv('models/forecast_features.csv')
ff['start_date'] = pd.to_datetime(ff['start_date'])

feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

xgb_m = model['xgb']
lgb_m = model['lgb']
w = model['w_xgb']

print("Creating explainers...")
explainer_xgb = shap.TreeExplainer(xgb_m)
explainer_lgb = shap.TreeExplainer(lgb_m)

# Pick 3 test cases (branch/date)
test_cases = [
    (104, '2026-04-05'),
    (115, '2026-04-10'),
    (129, '2026-04-15')
]

for branch, dt in test_cases:
    print(f"\\n--- Test Case: Branch {branch} on {dt} ---")
    dt = pd.to_datetime(dt)
    rows = ff[(ff['tran_br_code'] == branch) & (ff['start_date'] == dt)]
    am_row = rows[rows['AM_PM'] == 'AM'].iloc[0].to_dict()
    pm_row = rows[rows['AM_PM'] == 'PM'].iloc[0].to_dict()
    
    X_am = pd.DataFrame([am_row])[feature_cols]
    X_pm = pd.DataFrame([pm_row])[feature_cols]
    
    # AM
    sv_xgb_am = explainer_xgb.shap_values(X_am)[0]
    sv_lgb_am = explainer_lgb.shap_values(X_am)[0]
    sv_am = w * sv_xgb_am + (1 - w) * sv_lgb_am
    
    pred_log_am = w * xgb_m.predict(X_am)[0] + (1 - w) * lgb_m.predict(X_am)[0]
    pred_am = np.expm1(pred_log_am)
    
    base_log = w * explainer_xgb.expected_value + (1 - w) * explainer_lgb.expected_value
    base_val = np.expm1(base_log)
    
    diff_am = pred_am - base_val
    sum_abs_am = sum(abs(sv_am))
    sv_linear_am = (sv_am / sum_abs_am) * diff_am if sum_abs_am != 0 else sv_am * 0
    
    # PM
    sv_xgb_pm = explainer_xgb.shap_values(X_pm)[0]
    sv_lgb_pm = explainer_lgb.shap_values(X_pm)[0]
    sv_pm = w * sv_xgb_pm + (1 - w) * sv_lgb_pm
    
    pred_log_pm = w * xgb_m.predict(X_pm)[0] + (1 - w) * lgb_m.predict(X_pm)[0]
    pred_pm = np.expm1(pred_log_pm)
    
    diff_pm = pred_pm - base_val
    sum_abs_pm = sum(abs(sv_pm))
    sv_linear_pm = (sv_pm / sum_abs_pm) * diff_pm if sum_abs_pm != 0 else sv_pm * 0
    
    total_shap = sv_linear_am + sv_linear_pm
    total_base = base_val * 2
    actual_pred = pred_am + pred_pm
    
    summed_shap = sum(total_shap)
    reconstructed_pred = total_base + summed_shap
    
    print(f"Total Base Value: {total_base:,.2f}")
    print(f"Sum of Blended SHAP: {summed_shap:,.2f}")
    print(f"Reconstructed Prediction (Base + SHAP): {reconstructed_pred:,.2f}")
    print(f"Actual Blended Prediction: {actual_pred:,.2f}")
    print(f"Difference: {abs(reconstructed_pred - actual_pred):,.6f}")
