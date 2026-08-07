import os
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
from metrics_utils import mape, smape, wmape

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
N_OPTUNA_TRIALS = 50        # 50 trials for stable hyperparameter search
N_CV_FOLDS      = 3
OPTUNA_SEED     = 42

print("Starting V3 Ensemble Pipeline (XGBoost + LightGBM + CV Evaluation)...")
print(f"  Optuna trials  : {N_OPTUNA_TRIALS}")
print(f"  CV folds       : {N_CV_FOLDS}")

# ── 1. Load Preprocessed Data ─────────────────────────────────────────────────
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df[] = df['Day'].apply(lambda d: d - 25 if d >= 25 else d + (31 - 25))
df[] = df['start_date'].dt.is_month_start.astype(int)
df[] = df['start_date'].dt.is_month_end.astype(int)

df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
df[] = df.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)

for target in ['Half_Day_Total_Debit']:
    df[f'rolling_14_std_{target}'] = df.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)
    df[f'ewma_14_{target}'] = df.groupby('tran_br_code')[target].transform(lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)
    df[f'dow_avg_4_{target}'] = df.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[target].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)

feature_cols = [
    'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 
    'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash'
]

df['tran_br_code'] = df['tran_br_code'].astype('category')
df['Weekday'] = df['Weekday'].astype('category')
df['Month'] = df['Month'].astype('category')

# ── 2. Split ──────────────────────────────────────────────────────────────────
cutoff_idx  = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']

train_df = df[df['start_date'] < cutoff_date]
test_df  = df[df['start_date'] >= cutoff_date]

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

print(f"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Train size  : {len(X_train)}")
print(f"Test  size  : {len(X_test)}")

os.makedirs('models/v3', exist_ok=True)
cv_summary = {}

# ── 3. Modeling ───────────────────────────────────────────────────────────────
for target in ['Half_Day_Total_Debit']:
    print(f"\n{'='*55}\nTraining Advanced Ensemble Model for: {target}\n{'='*55}")

    y_train_raw = train_df[target]
    y_test_raw  = test_df[target]
    use_log = (target != 'Half_Day_Net_Cash')

    cap_val = y_train_raw.quantile(0.99)
    y_train_raw_capped = y_train_raw.clip(upper=cap_val)
    y_train = np.log1p(y_train_raw_capped.clip(lower=0)) if use_log else y_train_raw_capped

    def objective(trial):
        w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
        
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'enable_categorical': True,
            'random_state': OPTUNA_SEED
        }
        
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 7),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.6, 1.0),
            'random_state': OPTUNA_SEED,
            'verbose': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
        errors = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            
            # XGB
            m_xgb = xgb.XGBRegressor(**xgb_params, objective='reg:absoluteerror')
            m_xgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            p_xgb = m_xgb.predict(X_v)
            
            # LGBM
            m_lgb = lgb.LGBMRegressor(**lgb_params, objective='mae')
            m_lgb.fit(X_t, y_t, eval_set=[(X_v, y_v)])
            p_lgb = m_lgb.predict(X_v)
            
            # Blend
            preds = w_xgb * p_xgb + (1 - w_xgb) * p_lgb
            
            if use_log:
                errors.append(mean_absolute_error(np.expm1(y_v), np.expm1(preds)))
            else:
                errors.append(mean_absolute_error(y_v, preds))
                
        return float(np.mean(errors))

    print(f"Running Optuna ({N_OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_p = study.best_params
    w_xgb = best_p['w_xgb']
    
    xgb_params = {k.replace('xgb_', ''): v for k, v in best_p.items() if k.startswith('xgb_')}
    xgb_params['enable_categorical'] = True
    xgb_params['random_state'] = OPTUNA_SEED
    
    lgb_params = {k.replace('lgb_', ''): v for k, v in best_p.items() if k.startswith('lgb_')}
    lgb_params['random_state'] = OPTUNA_SEED
    lgb_params['verbose'] = -1

    # ── 4. CV Evaluation with best params ────────────────────────────────────
    print(f"\nEvaluating Ensemble (w_xgb={w_xgb:.2f}, w_lgb={1-w_xgb:.2f}) on {N_CV_FOLDS}-fold CV...")
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    fold_mae, fold_r2, fold_rmse_vals = [], [], []
    fold_mape_vals, fold_smape_vals, fold_wmape_vals = [], [], []

    for fold_num, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[tr_idx],  y_train.iloc[val_idx]
        y_v_raw  = y_train_raw.iloc[val_idx]

        m_xgb = xgb.XGBRegressor(**xgb_params, objective='reg:absoluteerror')
        m_xgb.fit(X_t, y_t)

        m_lgb = lgb.LGBMRegressor(**lgb_params, objective='mae')
        m_lgb.fit(X_t, y_t)

        p_xgb = m_xgb.predict(X_v)
        p_lgb = m_lgb.predict(X_v)

        preds_transformed = w_xgb * p_xgb + (1 - w_xgb) * p_lgb
        preds_orig = np.expm1(preds_transformed) if use_log else preds_transformed

        # Evaluate against raw UNCAPPED test actuals
        fold_mae.append(mean_absolute_error(y_v_raw, preds_orig))
        fold_rmse_vals.append(np.sqrt(mean_squared_error(y_v_raw, preds_orig)))
        fold_r2.append(r2_score(y_v_raw, preds_orig))
        fold_mape_vals.append(mape(y_v_raw.values, preds_orig))
        _smape_val, _ = smape(y_v_raw.values, preds_orig)
        fold_smape_vals.append(_smape_val)
        fold_wmape_vals.append(wmape(y_v_raw.values, preds_orig))

        print(f"  Fold {fold_num}: MAE={fold_mae[-1]/1e6:.2f}M  RMSE={fold_rmse_vals[-1]/1e6:.2f}M  R²={fold_r2[-1]:.4f}  MAPE={fold_mape_vals[-1]:.1f}%  SMAPE={fold_smape_vals[-1]:.1f}%  WMAPE={fold_wmape_vals[-1]:.1f}%")

    cv_mae_mean   = float(np.mean(fold_mae))
    cv_rmse_mean  = float(np.mean(fold_rmse_vals))
    cv_r2_mean    = float(np.mean(fold_r2))
    cv_mape_mean  = float(np.mean(fold_mape_vals))
    cv_smape_mean = float(np.mean(fold_smape_vals))
    cv_wmape_mean = float(np.mean(fold_wmape_vals))

    print(f"\n  CV Summary ({N_CV_FOLDS}-fold TimeSeriesSplit):")
    print(f"    MAE   : {cv_mae_mean/1e6:.2f}M  ± {float(np.std(fold_mae, ddof=1))/1e6:.2f}M")
    print(f"    RMSE  : {cv_rmse_mean/1e6:.2f}M  ± {float(np.std(fold_rmse_vals, ddof=1))/1e6:.2f}M")
    print(f"    R²    : {cv_r2_mean:.4f}  ± {float(np.std(fold_r2, ddof=1)):.4f}")
    print(f"    MAPE  : {cv_mape_mean:.1f}%  ± {float(np.std(fold_mape_vals, ddof=1)):.1f}%")
    print(f"    SMAPE : {cv_smape_mean:.1f}%  ± {float(np.std(fold_smape_vals, ddof=1)):.1f}%")
    print(f"    WMAPE : {cv_wmape_mean:.1f}%  ± {float(np.std(fold_wmape_vals, ddof=1)):.1f}%")

    # ── 5. Train Final Models ─────────────────────────────────────────────────
    print(f"\nTraining final ensemble models on full train set...")
    final_xgb = xgb.XGBRegressor(**xgb_params, objective='reg:absoluteerror')
    final_xgb.fit(X_train, y_train)
    
    final_lgb = lgb.LGBMRegressor(**lgb_params, objective='mae')
    final_lgb.fit(X_train, y_train)

    joblib.dump({'xgb': final_xgb, 'lgb': final_lgb, 'w_xgb': w_xgb}, f'models/ablation/model_{target}.pkl')
    
    # ── 6. Hold-out test set evaluation ───────────────────────────────────────
    p_xgb = final_xgb.predict(X_test)
    p_lgb = final_lgb.predict(X_test)
    y_pred_transformed = w_xgb * p_xgb + (1 - w_xgb) * p_lgb
    y_pred = np.expm1(y_pred_transformed) if use_log else y_pred_transformed

    ho_mae  = mean_absolute_error(y_test_raw, y_pred)
    ho_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    ho_r2   = r2_score(y_test_raw, y_pred)

    if use_log:
        ho_mape_val  = mape(y_test_raw.values, y_pred)
        ho_smape_val, _ = smape(y_test_raw.values, y_pred)
        ho_wmape_val = wmape(y_test_raw.values, y_pred)
    else:
        ho_mape_val  = mape(y_test_raw.values, y_pred)
        ho_smape_val, _ = smape(y_test_raw.values, y_pred)
        ho_wmape_val = wmape(y_test_raw.values, y_pred)
        
    print(f"\n  Hold-out test set:")
    print(f"    R²   : {ho_r2:.4f}")
    print(f"    MAE  : {ho_mae/1e6:.2f}M")
    print(f"    WMAPE: {ho_wmape_val:.2f}%")

    cv_summary[target] = {
        'n_optuna_trials'   : N_OPTUNA_TRIALS,
        'n_cv_folds'        : N_CV_FOLDS,
        'best_params'       : best_p,
        'w_xgb'             : w_xgb,
        # ── CV-averaged metrics (primary, stable metric) ──
        'cv_mae_mean_M'     : round(cv_mae_mean / 1e6, 3),
        'cv_mae_std_M'      : round(float(np.std(fold_mae, ddof=1)) / 1e6, 3),
        'cv_rmse_mean_M'    : round(cv_rmse_mean / 1e6, 3),
        'cv_rmse_std_M'     : round(float(np.std(fold_rmse_vals, ddof=1)) / 1e6, 3),
        'cv_r2_mean'        : round(cv_r2_mean, 4),
        'cv_r2_std'         : round(float(np.std(fold_r2, ddof=1)), 4),
        'cv_mape_mean'      : round(cv_mape_mean, 2),
        'cv_mape_std'       : round(float(np.std(fold_mape_vals, ddof=1)), 2),
        'cv_smape_mean'     : round(cv_smape_mean, 2),
        'cv_smape_std'      : round(float(np.std(fold_smape_vals, ddof=1)), 2),
        'cv_wmape_mean'     : round(cv_wmape_mean, 2),
        'cv_wmape_std'      : round(float(np.std(fold_wmape_vals, ddof=1)), 2),
        # ── Per-fold breakdown ──
        'cv_fold_mae_M'     : [round(v / 1e6, 3) for v in fold_mae],
        'cv_fold_rmse_M'    : [round(v / 1e6, 3) for v in fold_rmse_vals],
        'cv_fold_r2'        : [round(v, 4) for v in fold_r2],
        'cv_fold_mape'      : [round(v, 2) for v in fold_mape_vals],
        'cv_fold_smape'     : [round(v, 2) for v in fold_smape_vals],
        'cv_fold_wmape'     : [round(v, 2) for v in fold_wmape_vals],
        # ── Hold-out single-split (supplemental only) ──
        'ho_test_rows'      : len(X_test),
        'ho_mae_M'          : round(ho_mae / 1e6, 3),
        'ho_rmse_M'         : round(ho_rmse / 1e6, 3),
        'ho_r2'             : round(ho_r2, 4),
        'ho_mape'           : round(ho_mape_val, 2),
        'ho_smape'          : round(ho_smape_val, 2),
        'ho_wmape'          : round(ho_wmape_val, 2),
    }

with open('models/ablation/v3_config.json', 'w') as f:
    json.dump({'targets_using_log1p': ['Half_Day_Total_Debit', 'Half_Day_Total_Credit']}, f)
with open('models/ablation/cv_results.json', 'w') as f:
    json.dump(cv_summary, f, indent=2)

print("\nV3 Ensemble Pipeline completed successfully.")
