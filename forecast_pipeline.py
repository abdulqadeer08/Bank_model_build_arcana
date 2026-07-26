import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

pk_holidays = pd.to_datetime([
    '2024-02-05', '2024-03-23', '2024-04-10', '2024-04-11', '2024-04-12', '2024-05-01',
    '2024-06-17', '2024-06-18', '2024-06-19', '2024-07-16', '2024-07-17', '2024-08-14',
    '2024-09-16', '2024-11-09', '2024-12-25',
    '2025-02-05', '2025-03-23', '2025-03-31', '2025-04-01', '2025-04-02', '2025-05-01',
    '2025-06-06', '2025-06-07', '2025-06-08', '2025-07-05', '2025-07-06', '2025-08-14',
    '2025-09-05', '2025-11-09', '2025-12-25',
    '2026-02-05', '2026-03-20', '2026-03-21', '2026-03-22', '2026-03-23', '2026-05-01',
    '2026-05-27', '2026-05-28', '2026-05-29', '2026-06-25', '2026-06-26', '2026-08-14',
    '2026-08-26', '2026-11-09', '2026-12-25'
])

def is_salary_day(day):
    return 1 if day <= 5 or day >= 25 else 0

def preprocess_raw_to_halfdaily(df):
    if len(df) == 0:
        return pd.DataFrame()
        
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['AM_PM'] = np.where(df['txn_hour'] < 12, 'AM', 'PM')
    df['Weekday'] = df['start_date'].dt.dayofweek
    df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
    df['Month'] = df['start_date'].dt.month
    df['Day'] = df['start_date'].dt.day
    df['Is_Salary_Day'] = df['Day'].apply(is_salary_day)
    df['Is_Holiday'] = df['start_date'].isin(pk_holidays).astype(int)

    half_daily = df.groupby(['start_date', 'AM_PM', 'tran_br_code']).agg(
        Half_Day_Total_Debit=('TOTAL_DR', 'sum'),
        Half_Day_Total_Credit=('TOTAL_CR', 'sum'),
        Txn_Count=('TOTAL_DR', 'count'),
        Weekday=('Weekday', 'first'),
        Is_Weekend=('Is_Weekend', 'first'),
        Month=('Month', 'first'),
        Day=('Day', 'first'),
        Is_Salary_Day=('Is_Salary_Day', 'first'),
        Is_Holiday=('Is_Holiday', 'first')
    ).reset_index()

    # Filter out Sundays as they generally don't have banking hours (Removed per QA audit)
    # half_daily = half_daily[half_daily['Weekday'] != 6].copy()
    half_daily['Half_Day_Net_Cash'] = half_daily['Half_Day_Total_Credit'] - half_daily['Half_Day_Total_Debit']
    half_daily['AM_PM_Encoded'] = np.where(half_daily['AM_PM'] == 'AM', 0, 1)
    
    return half_daily

def recalculate_lags(half_daily):
    half_daily = half_daily.sort_values(['tran_br_code', 'start_date', 'AM_PM']).reset_index(drop=True)
    
    half_daily['lag_1_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
    half_daily['rolling_14_mean_Txn_Count'] = half_daily.groupby('tran_br_code')['Txn_Count'].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean()
    ).fillna(0)
    
    for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
        half_daily[f'lag_1_{target}'] = half_daily.groupby('tran_br_code')[target].shift(1)
        half_daily[f'lag_2_{target}'] = half_daily.groupby('tran_br_code')[target].shift(2)
        half_daily[f'lag_14_{target}'] = half_daily.groupby('tran_br_code')[target].shift(14)
        half_daily[f'lag_60_{target}'] = half_daily.groupby('tran_br_code')[target].shift(60)
        half_daily[f'rolling_14_mean_{target}'] = half_daily.groupby('tran_br_code')[target].transform(
            lambda x: x.shift(1).rolling(14, min_periods=1).mean()
        )
        half_daily[f'rolling_14_std_{target}'] = half_daily.groupby('tran_br_code')[target].transform(
            lambda x: x.shift(1).rolling(14, min_periods=2).std()
        ).fillna(0)

    # Drop early rows where rolling features aren't fully stable
    half_daily = half_daily.dropna(subset=['lag_60_Half_Day_Total_Debit']).reset_index(drop=True)
    return half_daily

def generate_forecast(new_raw_df=None, forecast_days=30, history_path='model_data/half_daily_features.csv'):
    """
    Generates a branch-wise daily cash forecast for the next `forecast_days`.
    If `new_raw_df` is provided, it applies the full preprocessing pipeline, appends it 
    to the historical features, updates the lags, and predicts forward from the new max date.
    """
    # 1. Load historical context
    history = pd.read_csv(history_path)
    history['start_date'] = pd.to_datetime(history['start_date'])
    
    # 2. Append new data and recalculate lags if provided
    if new_raw_df is not None and not new_raw_df.empty:
        new_hd = preprocess_raw_to_halfdaily(new_raw_df)
        if not new_hd.empty:
            history = pd.concat([history, new_hd], ignore_index=True)
            history = history.drop_duplicates(subset=['start_date', 'AM_PM', 'tran_br_code'], keep='last')
            history = recalculate_lags(history)
            
    branches = history['tran_br_code'].unique()
    branch_avgs = history.groupby('tran_br_code')['Txn_Count'].mean().to_dict()
    last_date = history['start_date'].max()
    
    # 3. Load V3 Models
    model_debit = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
    model_credit = joblib.load('models/v3/model_Half_Day_Total_Credit.pkl')
    model_net = joblib.load('models/v3/model_Half_Day_Net_Cash.pkl')
    
    feature_cols = [
        'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
        'Is_Salary_Day', 'Is_Holiday',
        'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
        'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
        'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
    ]
    
    working_history = history.sort_values(['tran_br_code', 'start_date', 'AM_PM']).groupby('tran_br_code').tail(65).copy()
    
    forecast_records = []
    
    # 4. Generate Predictions Recursively
    for step in range(1, forecast_days + 1):
        target_date = last_date + pd.Timedelta(days=step)
            
        for branch in branches:
            br_hist = working_history[working_history['tran_br_code'] == branch]
            
            for am_pm, am_pm_enc in [('AM', 0), ('PM', 1)]:
                row = {
                    'start_date': target_date,
                    'tran_br_code': branch,
                    'AM_PM': am_pm,
                    'AM_PM_Encoded': am_pm_enc,
                    'Txn_Count': branch_avgs.get(branch, 0),
                    'Weekday': target_date.weekday(),
                    'Is_Weekend': 1 if target_date.weekday() >= 5 else 0,
                    'Month': target_date.month,
                    'Day': target_date.day,
                    'Is_Salary_Day': is_salary_day(target_date.day),
                    'Is_Holiday': 1 if target_date in pk_holidays else 0,
                    'Days_to_Salary': max(0, min(25, 25 - target_date.day if target_date.day < 25 else (31 - target_date.day + 5)))
                }
                
                try:
                    row['lag_1_Txn_Count'] = br_hist['Txn_Count'].iloc[-1]
                    row['rolling_14_mean_Txn_Count'] = br_hist['Txn_Count'].tail(14).mean()
                except IndexError:
                    row['lag_1_Txn_Count'] = branch_avgs.get(branch, 0)
                    row['rolling_14_mean_Txn_Count'] = branch_avgs.get(branch, 0)
                
                # Fetch Lags from historical queue
                for t_col in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
                    try:
                        row[f'lag_1_{t_col}'] = br_hist[t_col].iloc[-1]
                        row[f'lag_2_{t_col}'] = br_hist[t_col].iloc[-2]
                        row[f'lag_14_{t_col}'] = br_hist[t_col].iloc[-14]
                        row[f'lag_60_{t_col}'] = br_hist[t_col].iloc[-60]
                        row[f'rolling_14_mean_{t_col}'] = br_hist[t_col].tail(14).mean()
                        row[f'rolling_14_std_{t_col}'] = br_hist[t_col].tail(14).std() if len(br_hist[t_col]) > 1 else 0
                    except IndexError:
                        row[f'lag_1_{t_col}'] = 0
                        row[f'lag_2_{t_col}'] = 0
                        row[f'lag_14_{t_col}'] = 0
                        row[f'lag_60_{t_col}'] = 0
                        row[f'rolling_14_mean_{t_col}'] = 0
                        row[f'rolling_14_std_{t_col}'] = 0
                
                x_df = pd.DataFrame([row])[feature_cols]
                
                # Inverse Transform V3 Models (log1p applied during training)
                pred_dr_log = model_debit.predict(x_df)[0]
                pred_cr_log = model_credit.predict(x_df)[0]
                
                pred_dr = np.expm1(pred_dr_log)
                pred_cr = np.expm1(pred_cr_log)
                pred_net = model_net.predict(x_df)[0]  # Net cash is raw
                
                pred_dr = max(0, pred_dr)
                pred_cr = max(0, pred_cr)
                
                row['Half_Day_Total_Debit'] = pred_dr
                row['Half_Day_Total_Credit'] = pred_cr
                row['Half_Day_Net_Cash'] = pred_net
                
                forecast_records.append(row)
                
                hist_row = pd.DataFrame([row])
                working_history = pd.concat([working_history, hist_row], ignore_index=True)
                working_history = working_history.groupby('tran_br_code').tail(65)
                br_hist = working_history[working_history['tran_br_code'] == branch]

    fc_half_daily = pd.DataFrame(forecast_records)
    
    # 5. Format Output
    fc_daily = fc_half_daily.groupby(['start_date', 'tran_br_code']).agg({
        'Half_Day_Total_Debit': 'sum'
    }).reset_index()
    
    fc_daily = fc_daily.rename(columns={
        'start_date': 'Date',
        'tran_br_code': 'Branch',
        'Half_Day_Total_Debit': 'Predicted_PKR'
    })
    
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    all_combinations = pd.MultiIndex.from_product([date_range, branches], names=['Date', 'Branch']).to_frame(index=False)
    
    fc_final = pd.merge(all_combinations, fc_daily, on=['Date', 'Branch'], how='left')
    fc_final['Predicted_PKR'] = fc_final['Predicted_PKR'].fillna(0)
    fc_final['Step'] = (fc_final['Date'] - last_date).dt.days
    
    # Load evaluation report for V3 MAE/MAPE
    try:
        eval_report = pd.read_csv('models/final_evaluation_report.csv')
        eval_report['Branch'] = eval_report['Branch'].astype(str)
        branch_mape = eval_report.set_index('Branch')['MAPE_%'].to_dict()
        p33 = eval_report['MAPE_%'].quantile(0.33)
        p66 = eval_report['MAPE_%'].quantile(0.66)
    except Exception:
        branch_mape = {}
        p33 = 10.0
        p66 = 20.0

    def compute_uncertainty(row):
        step = row['Step']
        br = str(row['Branch'])
        # Use exact branch MAPE without any generic cap or step-based additions
        mape_val = branch_mape.get(br, 10.0)
        uncertainty_pct = mape_val / 100.0
        
        pred = row['Predicted_PKR']
        lower = pred * (1 - uncertainty_pct)
        upper = pred * (1 + uncertainty_pct)
        
        # Derive Confidence from dynamically calculated percentiles (relative to cohort)
        if mape_val <= p33:
            conf = 'HIGH'
        elif mape_val <= p66:
            conf = 'MEDIUM'
        else:
            conf = 'LOW'
            
        return pd.Series({
            'Predicted_M': round(pred / 1e6, 2),
            'Lower_M': round(lower / 1e6, 2),
            'Upper_M': round(upper / 1e6, 2),
            'Uncertainty_Pct': round(uncertainty_pct * 100, 1),
            'Confidence': conf
        })
        
    fc_final[['Predicted_M', 'Lower_M', 'Upper_M', 'Uncertainty_Pct', 'Confidence']] = fc_final.apply(compute_uncertainty, axis=1)
    
    out_cols = ['Branch', 'Date', 'Predicted_M', 'Lower_M', 'Upper_M', 'Uncertainty_Pct', 'Confidence', 'Step']
    fc_final = fc_final[out_cols].sort_values(['Branch', 'Date']).reset_index(drop=True)
    
    return fc_final, fc_half_daily

def save_to_excel(fc_final, fc_half_daily, output_path='models/forecast_next30days.xlsx'):
    print(f"Saving forecast to {output_path}...")
    
    def week_label(step):
        if step <= 7:  return 'Week1 (HIGH)'
        if step <= 14: return 'Week2 (MEDIUM)'
        if step <= 21: return 'Week3 (LOW)'
        return 'Week4 (LOW)'

    fc_final['Week'] = fc_final['Step'].apply(week_label)
    
    weekly = (fc_final
              .groupby(['Branch', 'Week'])['Predicted_M']
              .agg(['mean'])
              .round(1)
              .rename(columns={'mean': 'Daily_Avg_M'})
              .reset_index())
    
    weekly_pivot = weekly.pivot_table(
        index='Branch', columns='Week', values='Daily_Avg_M'
    ).reset_index()
    
    branches = sorted(fc_final['Branch'].unique())
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        fc_final.drop(columns=['Week']).to_excel(writer, sheet_name='All_Branches', index=False)
        weekly_pivot.to_excel(writer, sheet_name='Weekly_Summary', index=False)
        
        for branch in branches:
            br_fc = fc_final[fc_final['Branch'] == branch].drop(columns=['Week']).copy()
            br_fc['Date'] = br_fc['Date'].dt.strftime('%Y-%m-%d')
            br_fc.to_excel(writer, sheet_name=f'Br_{branch}', index=False)
            
    print("Saving forecast features to models/forecast_features.csv...")
    fc_half_daily.to_csv('models/forecast_features.csv', index=False)
    print("Save complete!")

if __name__ == '__main__':
    print("Testing generate_forecast()...")
    fc, fc_hd = generate_forecast(forecast_days=30)
    print("Forecast generated successfully:")
    print(fc.head(10))
    save_to_excel(fc, fc_hd)
