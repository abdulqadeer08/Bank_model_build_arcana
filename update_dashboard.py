import re

with open('dashboard.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update load_model
code = code.replace("def load_model():\n    return joblib.load('models/best_model.pkl')",
                    "def load_model():\n    return joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')")

# 2. Add load_forecast_features
code = code.replace("def load_features():\n    with open('model_data/feature_cols.json') as f:\n        return json.load(f)",
                    "def load_features():\n    # Using V3 feature columns\n    return [\n        'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',\n        'Is_Salary_Day', 'Is_Holiday',\n        'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',\n        'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',\n        'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'\n    ]\n\n@st.cache_data\ndef load_forecast_features():\n    ff = pd.read_csv('models/forecast_features.csv')\n    ff['start_date'] = pd.to_datetime(ff['start_date'])\n    return ff")

# 3. Add to globals
code = code.replace("feature_cols   = load_features()", "feature_cols   = load_features()\nforecast_feats = load_forecast_features()")

# 4. Update FEATURE_LABELS
new_labels = """FEATURE_LABELS = {
    'rolling_14_mean_Half_Day_Total_Debit' : ('📅 14-Day Average',    'Average withdrawal over the past 14 days'),
    'lag_1_Half_Day_Total_Debit'           : ('⏮️ Last Half-Day Withdrawal', 'Cash withdrawal amount from the previous half-day'),
    'lag_2_Half_Day_Total_Debit'           : ('⏮️ Yesterday Withdrawal', 'Cash withdrawal amount from yesterday same time'),
    'lag_14_Half_Day_Total_Debit'          : ('📅 14 Days Ago',        'Cash withdrawal amount from exactly two weeks ago'),
    'lag_60_Half_Day_Total_Debit'          : ('📅 60 Days Ago',        'Cash withdrawal amount from exactly two months ago'),
    'rolling_14_mean_Half_Day_Total_Credit': ('📅 14-Day Avg Deposit', 'Average deposit over the past 14 days'),
    'lag_1_Half_Day_Total_Credit'          : ('⏮️ Last Half-Day Deposit', 'Deposit amount from the previous half-day'),
    'lag_2_Half_Day_Total_Credit'          : ('⏮️ Yesterday Deposit', 'Deposit amount from yesterday same time'),
    'lag_14_Half_Day_Total_Credit'         : ('📅 14 Days Ago Deposit','Deposit amount from exactly two weeks ago'),
    'lag_60_Half_Day_Total_Credit'         : ('📅 60 Days Ago Deposit','Deposit amount from exactly two months ago'),
    'rolling_14_mean_Half_Day_Net_Cash'    : ('📅 14-Day Avg Net Cash','Average net cash over the past 14 days'),
    'lag_1_Half_Day_Net_Cash'              : ('⏮️ Last Half-Day Net Cash','Net cash amount from the previous half-day'),
    'lag_2_Half_Day_Net_Cash'              : ('⏮️ Yesterday Net Cash', 'Net cash amount from yesterday same time'),
    'lag_14_Half_Day_Net_Cash'             : ('📅 14 Days Ago Net Cash','Net cash amount from exactly two weeks ago'),
    'lag_60_Half_Day_Net_Cash'             : ('📅 60 Days Ago Net Cash','Net cash amount from exactly two months ago'),
    'Txn_Count'             : ('🔢 Transactions',      'Total number of transactions'),
    'Is_Salary_Day'         : ('💰 Salary Day',        'Whether the day is near salary day'),
    'Is_Holiday'            : ('🎉 Holiday',           'Whether the day is a public holiday'),
    'Weekday'               : ('📆 Day (Weekday)',     'Day of the week'),
    'Is_Weekend'            : ('🏖️ Weekend',           'Whether the day is a weekend'),
    'Month'                 : ('🗓️ Month',             'Month of the year'),
    'Day'                   : ('🔢 Date',              'Date of the month'),
    'AM_PM_Encoded'         : ('☀️/🌙 Time of Day',    'Morning (AM) or Afternoon (PM)'),
}"""
# Using regex to replace the entire FEATURE_LABELS dict
code = re.sub(r'FEATURE_LABELS = \{.*?\n\}', new_labels, code, flags=re.DOTALL)


# 5. Modify get_shap_for_row
new_get_shap = """def get_shap_for_row(X_row_am, X_row_pm):
    \"\"\"Compute SHAP values for AM and PM, sum linear impacts, and return total Daily SHAP.\"\"\"
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
    
    return total_shap, total_base, pred_am + pred_pm"""

code = re.sub(r'def get_shap_for_row\(X_row\):.*?(?=def build_explanation)', new_get_shap + "\n\n", code, flags=re.DOTALL)

# 6. Replace get_forecast_features completely
new_get_forecast = """def get_forecast_features(branch, target_date):
    \"\"\"Fetch AM and PM feature rows for a future date from the pre-generated forecast features.\"\"\"
    target_date = pd.to_datetime(target_date)
    rows = forecast_feats[(forecast_feats['tran_br_code'] == branch) & (forecast_feats['start_date'] == target_date)]
    if len(rows) == 0:
        return None, None
    
    am_row = rows[rows['AM_PM'] == 'AM'].iloc[0].to_dict()
    pm_row = rows[rows['AM_PM'] == 'PM'].iloc[0].to_dict()
    return am_row, pm_row"""
code = re.sub(r'def get_forecast_features\(branch, target_date\):.*?(?=\n# ─── SIDEBAR)', new_get_forecast + "\n", code, flags=re.DOTALL)

# 7. Update Page 1 (Calendar SHAP call)
# The old one iterated `get_forecast_features` and built `X_30`.
# We need to rewrite `with st.spinner("Analyzing SHAP reasons for the calendar..."):` block.
old_calendar_shap = """        with st.spinner("Analyzing SHAP reasons for the calendar..."):
            X_30_list = [get_forecast_features(sel_br, dt) for dt in br_fc['Date']]
            X_30 = pd.DataFrame(X_30_list)[feature_cols]
            
            explainer = shap.TreeExplainer(model)
            shap_vals_30 = explainer.shap_values(X_30)
            
            top_drivers = []
            for i in range(len(X_30)):
                top_drivers.append(get_top_shap_reasons_str(X_30.iloc[[i]], shap_vals_30[i]))
            
            br_fc['Top_Drivers'] = top_drivers"""

new_calendar_shap = """        with st.spinner("Analyzing SHAP reasons for the calendar..."):
            top_drivers = []
            for dt in br_fc['Date']:
                am_row, pm_row = get_forecast_features(sel_br, dt)
                if am_row is None:
                    top_drivers.append("Normal Pattern")
                    continue
                X_am = pd.DataFrame([am_row])[feature_cols]
                X_pm = pd.DataFrame([pm_row])[feature_cols]
                
                sv_total, _, _ = get_shap_for_row(X_am, X_pm)
                
                # Use AM row features as proxy for feature values display
                top_drivers.append(get_top_shap_reasons_str(X_am, sv_total))
                
            br_fc['Top_Drivers'] = top_drivers"""
code = code.replace(old_calendar_shap, new_calendar_shap)

# 8. Update Page 2 (Why This Amount? SHAP Call)
old_page2_shap = """        with st.spinner("Running SHAP analysis..."):
            target_dt  = pd.Timestamp(sel_date)
            feat_row   = get_forecast_features(sel_br, target_dt)
            X_row      = pd.DataFrame([feat_row])[feature_cols]
            sv, bv     = get_shap_for_row(X_row)
            pred_val   = float(model.predict(X_row)[0])"""

new_page2_shap = """        with st.spinner("Running SHAP analysis..."):
            target_dt  = pd.Timestamp(sel_date)
            feat_row_am, feat_row_pm = get_forecast_features(sel_br, target_dt)
            
            if feat_row_am is None:
                st.error("Feature data not found for this date. Run forecast pipeline to generate features.")
                st.stop()
                
            X_row_am = pd.DataFrame([feat_row_am])[feature_cols]
            X_row_pm = pd.DataFrame([feat_row_pm])[feature_cols]
            
            sv, bv, pred_val = get_shap_for_row(X_row_am, X_row_pm)
            feat_row = feat_row_am  # for display values"""
code = code.replace(old_page2_shap, new_page2_shap)

# 9. Update Sidebar V3 Info
# It already says V3, but wait, the MAE should be 9.52M (which it is)
# We don't need to change sidebar text, but we should make sure we don't display R2 0.56

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("dashboard.py updated!")
