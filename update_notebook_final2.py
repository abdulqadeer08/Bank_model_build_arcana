import json

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_37_code = """# ── 6.4 XGBoost + LightGBM Ensemble V3 (Tuned matching production) ───────────────────────────
# Use exact hyperparams from v3_pipeline.py to match production
# We evaluate all 3 targets using 3-fold TimeSeriesSplit to mirror the production CV logic

import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from metrics_utils import mape, smape, wmape

with open('models/v3/cv_results.json', 'r') as f:
    cv_params = json.load(f)

for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    print(f"\\n{'='*55}\\nEvaluating Ensemble Model for: {target}\\n{'='*55}")
    use_log = (target != 'Half_Day_Net_Cash')
    y_train_raw = train_df[target]
    
    cap_val = y_train_raw.quantile(0.99)
    y_train_raw_capped = y_train_raw.clip(upper=cap_val)
    y_train_tgt = np.log1p(y_train_raw_capped.clip(lower=0)) if use_log else y_train_raw_capped

    t_params = cv_params[target]
    best_p = t_params['best_params']
    w_xgb = t_params['w_xgb']
    
    xgb_p = {k.replace('xgb_', ''): v for k, v in best_p.items() if k.startswith('xgb_')}
    xgb_p['enable_categorical'] = True
    xgb_p['random_state'] = 42
    
    lgb_p = {k.replace('lgb_', ''): v for k, v in best_p.items() if k.startswith('lgb_')}
    lgb_p['random_state'] = 42
    lgb_p['verbose'] = -1
    
    tscv = TimeSeriesSplit(n_splits=3)
    fold_mae, fold_r2, fold_rmse_vals = [], [], []
    fold_mape_vals, fold_smape_vals, fold_wmape_vals = [], [], []

    for fold_num, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train_tgt.iloc[tr_idx],  y_train_tgt.iloc[val_idx]
        y_v_raw  = y_train_raw.iloc[val_idx]

        m_xgb = xgb.XGBRegressor(**xgb_p, objective='reg:absoluteerror')
        m_xgb.fit(X_t, y_t)
        p_xgb = m_xgb.predict(X_v)

        m_lgb = lgb.LGBMRegressor(**lgb_p, objective='mae')
        m_lgb.fit(X_t, y_t)
        p_lgb = m_lgb.predict(X_v)

        preds_transformed = w_xgb * p_xgb + (1 - w_xgb) * p_lgb
        preds_orig = np.expm1(preds_transformed) if use_log else preds_transformed

        fold_mae.append(mean_absolute_error(y_v_raw, preds_orig))
        fold_rmse_vals.append(np.sqrt(mean_squared_error(y_v_raw, preds_orig)))
        fold_r2.append(r2_score(y_v_raw, preds_orig))
        fold_mape_vals.append(mape(y_v_raw.values, preds_orig))
        _smape_val, _ = smape(y_v_raw.values, preds_orig)
        fold_smape_vals.append(_smape_val)
        fold_wmape_vals.append(wmape(y_v_raw.values, preds_orig))
    
    mae_mean = np.mean(fold_mae)/1e6
    mae_std = np.std(fold_mae, ddof=1)/1e6
    print(f"  CV R2    : {np.mean(fold_r2):.4f} ± {np.std(fold_r2, ddof=1):.4f}")
    print(f"  CV MAE   : {mae_mean:.2f}M ± {mae_std:.2f}M")
    print(f"  CV RMSE  : {np.mean(fold_rmse_vals)/1e6:.2f}M")
    print(f"  CV MAPE  : {np.mean(fold_mape_vals):.1f}%")
    print(f"  CV SMAPE : {np.mean(fold_smape_vals):.1f}%")
    print(f"  CV WMAPE : {np.mean(fold_wmape_vals):.1f}% ± {np.std(fold_wmape_vals, ddof=1):.1f}%\\n")

    # If it's the primary target (Debit), we train the final model on full train set for subsequent cells (SHAP, plots, etc)
    if target == 'Half_Day_Total_Debit':
        print("Training final ensemble on full training set for Debit target...")
        xgb_model = xgb.XGBRegressor(**xgb_p, objective='reg:absoluteerror')
        xgb_model.fit(X_train, y_train_tgt)
        
        lgb_model = lgb.LGBMRegressor(**lgb_p, objective='mae')
        lgb_model.fit(X_train, y_train_tgt)
        
        y_test_raw = test_df['Half_Day_Total_Debit_RAW'].values
        p_xgb = xgb_model.predict(X_test)
        p_lgb = lgb_model.predict(X_test)
        preds_transformed = w_xgb * p_xgb + (1 - w_xgb) * p_lgb
        y_pred_xgb = np.expm1(preds_transformed)

# For the comparison table, we will use the CV results for the primary target (Debit)
# We append a dummy result here because the comparison table code (Cell 40) explicitly loads the CV results 
# from the JSON file to populate the final comparison table.
results.append({
    'Model': 'XGBoost + LightGBM Ensemble V3',
    'R2': cv_params['Half_Day_Total_Debit']['cv_r2_mean'],
    'MAE_M': cv_params['Half_Day_Total_Debit']['cv_mae_mean_M'],
    'RMSE_M': cv_params['Half_Day_Total_Debit']['cv_rmse_mean_M'],
    'MAPE_%': cv_params['Half_Day_Total_Debit']['cv_mape_mean'],
    'SMAPE_%': cv_params['Half_Day_Total_Debit']['cv_smape_mean'],
    'WMAPE_%': cv_params['Half_Day_Total_Debit']['cv_wmape_mean']
})
"""

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if '6.4 XGBoost + LightGBM Ensemble V3' in src and 'evaluate all 3 targets' in src:
        cell['source'] = [line + '\\n' for line in cell_37_code.split('\\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\\n')
        print(f"Replaced Cell {i} (6.4 Model Training) with updated one")
        
    if '6.5 Plot:' in src:
        # Also fix the title of the plot to say "Ensemble" instead of "XGBoost V3"
        src = src.replace('XGBoost V3', 'Ensemble V3')
        cell['source'] = [line + '\\n' for line in src.split('\\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\\n')
        print(f"Updated Cell {i} plot title")
        

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully.")
