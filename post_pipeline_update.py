"""
post_pipeline_update.py
Run this immediately after v3_pipeline.py completes.
Reads the fresh cv_results.json and propagates the stable CV-averaged metrics
to all downstream files: model_results.csv, final_evaluation_report.csv,
Complete_Project_Documentation.md, forecast_next30days.xlsx, shap_values.csv.
"""

import json
import os
import sys
import subprocess
import re
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load cv_results.json ──────────────────────────────────────────────────
cv_path = 'models/v3/cv_results.json'
if not os.path.exists(cv_path):
    print(f"ERROR: {cv_path} not found. Run v3_pipeline.py first.")
    sys.exit(1)

with open(cv_path) as f:
    cv = json.load(f)

print("=" * 65)
print("  CV Results loaded successfully")
print("=" * 65)

for target, vals in cv.items():
    print(f"\n{target}:")
    print(f"  CV R2   = {vals['cv_r2_mean']:.4f}  +/- {vals['cv_r2_std']:.4f}")
    print(f"  CV MAE  = {vals['cv_mae_mean_M']:.3f}M  +/- {vals['cv_mae_std_M']:.3f}M")
    print(f"  CV RMSE = {vals['cv_rmse_mean_M']:.3f}M  +/- {vals['cv_rmse_std_M']:.3f}M")
    print(f"  CV MAPE = {vals['cv_mape_mean']:.1f}%  +/- {vals['cv_mape_std']:.1f}%")
    print(f"  CV SMAPE= {vals['cv_smape_mean']:.1f}%  +/- {vals['cv_smape_std']:.1f}%")
    print(f"  CV WMAPE= {vals['cv_wmape_mean']:.1f}%  +/- {vals['cv_wmape_std']:.1f}%")

# Use Debit as the headline (primary target for bank cash management)
d = cv.get('Half_Day_Total_Debit', {})
cv_r2     = d.get('cv_r2_mean', 0)
cv_r2_sd  = d.get('cv_r2_std', 0)
cv_mae    = d.get('cv_mae_mean_M', 0)
cv_mae_sd = d.get('cv_mae_std_M', 0)
cv_rmse   = d.get('cv_rmse_mean_M', 0)
cv_mape   = d.get('cv_mape_mean', 0)
cv_mape_sd= d.get('cv_mape_std', 0)
cv_smape  = d.get('cv_smape_mean', 0)
cv_wmape  = d.get('cv_wmape_mean', 0)
n_trials  = d.get('n_optuna_trials', 50)
n_folds   = d.get('n_cv_folds', 3)

print(f"\n{'='*65}")
print(f"  HEADLINE METRIC (Debit target, {n_folds}-fold CV, {n_trials} Optuna trials):")
print(f"  R2    = {cv_r2:.4f}  +/- {cv_r2_sd:.4f}")
print(f"  MAE   = {cv_mae:.3f}M  +/- {cv_mae_sd:.3f}M PKR")
print(f"  RMSE  = {cv_rmse:.3f}M PKR")
print(f"  MAPE  = {cv_mape:.1f}%  +/- {cv_mape_sd:.1f}%")
print(f"  SMAPE = {cv_smape:.1f}%")
print(f"  WMAPE = {cv_wmape:.1f}%")
print(f"{'='*65}\n")

# ── 2. Update models/model_results.csv ──────────────────────────────────────
print("Step 2: Updating models/model_results.csv ...")

ho_r2    = d.get('ho_r2', cv_r2)
ho_mae   = d.get('ho_mae_M', cv_mae)
ho_rmse  = d.get('ho_rmse_M', cv_rmse)
ho_mape  = d.get('ho_mape', cv_mape)
ho_smape = d.get('ho_smape', cv_smape)
ho_wmape = d.get('ho_wmape', cv_wmape)

model_results = pd.DataFrame([
    {
        'Model': 'Baseline (14-Day Rolling Mean)',
        'R2': 0.2964, 'MAE': 22790000.0, 'RMSE': float('nan'),
        'MAE_M': 22.79, 'RMSE_M': float('nan'),
        'MAPE': 571.84, 'SMAPE': 88.79, 'WMAPE': 82.25,
        'CV_R2_Mean': float('nan'), 'CV_R2_Std': float('nan'),
        'CV_MAE_Mean_M': float('nan'), 'CV_MAE_Std_M': float('nan'),
        'Metric_Type': 'Hold-out'
    },
    {
        'Model': 'Prophet',
        'R2': -0.0169, 'MAE': 24830000.0, 'RMSE': float('nan'),
        'MAE_M': 24.83, 'RMSE_M': float('nan'),
        'MAPE': 707.53, 'SMAPE': 93.50, 'WMAPE': 89.61,
        'CV_R2_Mean': float('nan'), 'CV_R2_Std': float('nan'),
        'CV_MAE_Mean_M': float('nan'), 'CV_MAE_Std_M': float('nan'),
        'Metric_Type': 'Hold-out'
    },
    {
        'Model': 'LightGBM',
        'R2': 0.5411, 'MAE': 12520000.0, 'RMSE': float('nan'),
        'MAE_M': 12.52, 'RMSE_M': float('nan'),
        'MAPE': 125.95, 'SMAPE': 57.37, 'WMAPE': 45.18,
        'CV_R2_Mean': float('nan'), 'CV_R2_Std': float('nan'),
        'CV_MAE_Mean_M': float('nan'), 'CV_MAE_Std_M': float('nan'),
        'Metric_Type': 'Hold-out'
    },
    {
        'Model': f'XGBoost+LightGBM Ensemble V3 (CV-Avg, {n_trials}-trial Optuna)',
        'R2': ho_r2, 'MAE': round(ho_mae * 1e6), 'RMSE': float('nan'),
        'MAE_M': ho_mae, 'RMSE_M': ho_rmse,
        'MAPE': ho_mape, 'SMAPE': ho_smape, 'WMAPE': ho_wmape,
        'CV_R2_Mean': cv_r2, 'CV_R2_Std': cv_r2_sd,
        'CV_MAE_Mean_M': cv_mae, 'CV_MAE_Std_M': cv_mae_sd,
        'Metric_Type': f'{n_folds}-fold CV (primary) + Hold-out (supplemental)'
    },
])
model_results.to_csv('models/model_results.csv', index=False)
print("  -> models/model_results.csv updated.")

# ── 3. Regenerate per-branch final_evaluation_report.csv ────────────────────
print("\nStep 3: Regenerating models/final_evaluation_report.csv ...")

try:
    from metrics_utils import mape as _mape, smape as _smape, wmape as _wmape
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = pd.read_csv('model_data/half_daily_features.csv')
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['Days_to_Salary'] = df['Day'].apply(lambda d2: 25 - d2 if d2 < 25 else (31 - d2 + 5)).clip(lower=0, upper=25)
    df['Days_Since_Salary'] = df['Day'].apply(lambda d2: d2 - 25 if d2 >= 25 else d2 + (31 - 25))
    df['Is_Month_Start'] = df['start_date'].dt.is_month_start.astype(int)
    df['Is_Month_End'] = df['start_date'].dt.is_month_end.astype(int)
    df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)

    # Recompute derived features that v3_pipeline adds at runtime
    df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
    df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
    df['ewma_14_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
        lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)

    for tgt in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
        df[f'rolling_14_std_{tgt}'] = df.groupby('tran_br_code')[tgt].transform(
            lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)
        df[f'ewma_14_{tgt}'] = df.groupby('tran_br_code')[tgt].transform(
            lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)
        df[f'dow_avg_4_{tgt}'] = df.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[tgt].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)

    feature_cols = [
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
    df['tran_br_code'] = df['tran_br_code'].astype('category')
    df['Weekday'] = df['Weekday'].astype('category')
    df['Month'] = df['Month'].astype('category')

    cutoff_idx  = int(len(df) * 0.8)
    cutoff_date = df.iloc[cutoff_idx]['start_date']
    test_df = df[df['start_date'] >= cutoff_date].copy()
    print(f"  Test set: {len(test_df)} rows from {cutoff_date.date()}")

    ensemble = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
    X_test = test_df[feature_cols]
    y_test = test_df['Half_Day_Total_Debit']

    p_xgb = ensemble['xgb'].predict(X_test)
    p_lgb = ensemble['lgb'].predict(X_test)
    w = ensemble['w_xgb']
    y_pred_log = w * p_xgb + (1 - w) * p_lgb
    y_pred = np.expm1(y_pred_log)

    branches = sorted(test_df['tran_br_code'].cat.categories.tolist())
    eval_rows = []
    all_maes = []

    for br in branches:
        mask = test_df['tran_br_code'] == br
        if mask.sum() < 2:
            continue
        yt = y_test[mask].values
        yp = y_pred[mask.values]

        br_mae   = mean_absolute_error(yt, yp)
        br_r2    = r2_score(yt, yp)
        br_mape  = _mape(yt, yp)
        br_smape_val, _ = _smape(yt, yp)
        br_wmape = _wmape(yt, yp)
        avg_d    = yt.mean()
        all_maes.append(br_mae / 1e6)

        eval_rows.append({
            'Branch': br,
            'Avg_Demand_M': round(avg_d / 1e6, 2),
            'MAE_M': round(br_mae / 1e6, 2),
            'MAPE_%': round(br_mape, 1),
            'SMAPE_%': round(br_smape_val, 1),
            'WMAPE_%': round(br_wmape, 1),
            'R2': round(br_r2, 4),
        })

    # Compute p33/p66 on MAE for confidence tiers
    all_maes_arr = np.array(all_maes)
    p33 = float(np.percentile(all_maes_arr, 33))
    p66 = float(np.percentile(all_maes_arr, 66))
    print(f"  Confidence tier thresholds: p33={p33:.2f}M, p66={p66:.2f}M")

    for row in eval_rows:
        mae_m = row['MAE_M']
        if mae_m <= p33:
            trust = 'HIGH'; buffer = 5
            action = 'Use model directly for replenishment'
        elif mae_m <= p66:
            trust = 'MEDIUM'; buffer = 12
            action = 'Add safety buffer, monitor weekly'
        else:
            trust = 'LOW'; buffer = 20
            action = 'Use with caution, manual override advised'
        row['Trust_Level'] = trust
        row['Buffer_%'] = buffer
        row['Recommended_M'] = round(row['Avg_Demand_M'] * (1 + buffer / 100), 2)
        row['Action'] = action

    eval_df = pd.DataFrame(eval_rows).sort_values('MAE_M').reset_index(drop=True)
    eval_df.to_csv('models/final_evaluation_report.csv', index=False)
    print("  -> models/final_evaluation_report.csv regenerated.")
    print(eval_df[['Branch', 'MAE_M', 'MAPE_%', 'WMAPE_%', 'R2', 'Trust_Level']].to_string())

except Exception as e:
    print(f"  WARNING: Could not regenerate final_evaluation_report.csv: {e}")
    import traceback; traceback.print_exc()

# ── 4. Update Complete_Project_Documentation.md ──────────────────────────────
print("\nStep 4: Updating Complete_Project_Documentation.md ...")

try:
    with open('Complete_Project_Documentation.md', 'r', encoding='utf-8') as f:
        doc = f.read()

    methodology_note = (
        "\n\n---\n\n"
        "### Why Metrics Changed: Single-Split to Cross-Validated Average\n\n"
        "A single train/test split with ~150 test rows proved sensitive to minor dataset changes, "
        "producing inconsistent point estimates across runs (e.g., R\u00b2 fluctuating 0.6785 \u2192 0.6628 "
        "\u2192 0.4838 \u2192 0.4770). We now report the **3-fold TimeSeriesSplit cross-validated average "
        "(with standard deviation)** as a more stable and defensible metric.\n\n"
        f"**Final stable metric (Debit target, {n_folds}-fold TimeSeriesSplit CV, {n_trials} Optuna trials, "
        f"objective=reg:absoluteerror, tran_br_code as categorical, seed=42):**\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| R\u00b2 | **{cv_r2:.4f} \u00b1 {cv_r2_sd:.4f}** |\n"
        f"| MAE | **{cv_mae:.3f}M \u00b1 {cv_mae_sd:.3f}M PKR** |\n"
        f"| RMSE | **{cv_rmse:.3f}M PKR** |\n"
        f"| MAPE | **{cv_mape:.1f}%** |\n"
        f"| SMAPE | **{cv_smape:.1f}%** |\n"
        f"| WMAPE | **{cv_wmape:.1f}%** |\n\n"
        "_These numbers will not swing wildly if a few more rows are added to the dataset._\n"
    )

    # Remove existing methodology note if present
    doc = re.sub(
        r'\n\n---\n\n### Why Metrics Changed.*',
        '',
        doc,
        flags=re.DOTALL
    )
    doc = doc.rstrip() + methodology_note

    with open('Complete_Project_Documentation.md', 'w', encoding='utf-8') as f:
        f.write(doc)
    print("  -> Complete_Project_Documentation.md updated.")

except Exception as e:
    print(f"  WARNING: Could not update docs: {e}")
    import traceback; traceback.print_exc()

# ── 5. Regenerate forecast_next30days.xlsx ───────────────────────────────────
print("\nStep 5: Regenerating forecast_next30days.xlsx ...")
try:
    from forecast_pipeline import generate_forecast, save_to_excel
    fc, fc_hd = generate_forecast(forecast_days=30)
    save_to_excel(fc, fc_hd)
    print("  -> models/forecast_next30days.xlsx regenerated.")
except Exception as e:
    print(f"  WARNING: Could not regenerate forecast: {e}")
    import traceback; traceback.print_exc()

# ── 6. Regenerate shap_values.csv ────────────────────────────────────────────
print("\nStep 6: Regenerating SHAP values ...")
try:
    result = subprocess.run(['python', 'generate_shap.py'], capture_output=True, text=True, cwd='.')
    if result.returncode == 0:
        print("  -> models/shap_values.csv regenerated.")
        print(result.stdout[-500:] if result.stdout else '')
    else:
        print(f"  WARNING: generate_shap.py returned code {result.returncode}")
        print(result.stderr[:500])
except Exception as e:
    print(f"  WARNING: Could not run generate_shap.py: {e}")

print("\n" + "=" * 65)
print("  post_pipeline_update.py COMPLETE")
print("=" * 65)
print(f"\nFINAL STABLE METRICS (Debit, {n_folds}-fold CV, {n_trials} Optuna trials):")
print(f"  R2    = {cv_r2:.4f}  +/- {cv_r2_sd:.4f}")
print(f"  MAE   = {cv_mae:.3f}M  +/- {cv_mae_sd:.3f}M PKR")
print(f"  RMSE  = {cv_rmse:.3f}M PKR")
print(f"  MAPE  = {cv_mape:.1f}%  +/- {cv_mape_sd:.1f}%")
print(f"  SMAPE = {cv_smape:.1f}%")
print(f"  WMAPE = {cv_wmape:.1f}%")
