import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from metrics_utils import mape, smape, wmape
import warnings
warnings.filterwarnings('ignore')

print('Loading data...')
df = pd.read_csv('model_data/half_daily_features.csv', parse_dates=['start_date'])
df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM']).reset_index(drop=True)

# Generate features EXCLUDING the new ones (ewma, dow_avg, Days_Since_Salary, Month_Start/End)
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

df['Weekday'] = df['start_date'].dt.dayofweek
df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
df['Month'] = df['start_date'].dt.month
df['Day'] = df['start_date'].dt.day

salary_days = [1, 2, 3, 4, 5, 27, 28, 29, 30, 31]
df['Is_Salary_Day'] = df['Day'].isin(salary_days).astype(int)

pk_holidays = pd.to_datetime([
    '2022-02-05', '2022-03-23', '2022-05-01', '2022-05-03', '2022-05-04', '2022-05-05', '2022-07-10', '2022-07-11', '2022-07-12', '2022-08-08', '2022-08-09', '2022-08-14',
    '2023-02-05', '2023-03-23', '2023-04-21', '2023-04-22', '2023-04-23', '2023-04-24', '2023-05-01', '2023-06-29', '2023-06-30', '2023-07-01', '2023-07-28', '2023-08-14',
    '2024-02-05', '2024-03-23', '2024-04-10', '2024-04-11', '2024-04-12', '2024-05-01', '2024-06-17', '2024-06-18', '2024-06-19', '2024-07-17', '2024-08-14'
])
df['Is_Holiday'] = df['start_date'].isin(pk_holidays).astype(int)
df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d) + 25)

targets = ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']
for t in targets:
    df[f'lag_1_{t}'] = df.groupby('tran_br_code')[t].shift(1)
    df[f'lag_2_{t}'] = df.groupby('tran_br_code')[t].shift(2)
    df[f'lag_14_{t}'] = df.groupby('tran_br_code')[t].shift(14)
    df[f'lag_60_{t}'] = df.groupby('tran_br_code')[t].shift(60)
    df[f'rolling_14_mean_{t}'] = df.groupby('tran_br_code')[t].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df[f'rolling_14_std_{t}']  = df.groupby('tran_br_code')[t].transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())

df = df.dropna().reset_index(drop=True)

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

cutoff_idx = int(len(df) * 0.8)
train_df = df.iloc[:cutoff_idx]
X_train = train_df[feature_cols]

y_train_raw = train_df['Half_Day_Total_Debit']
cap_val = y_train_raw.quantile(0.99)
y_train_raw_capped = y_train_raw.clip(upper=cap_val)
y_train_tgt = np.log1p(y_train_raw_capped.clip(lower=0))

tscv = TimeSeriesSplit(n_splits=3)

def objective(trial):
    w_xgb = trial.suggest_float('w_xgb', 0.1, 0.9)
    xgb_p = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'enable_categorical': True,
        'random_state': 42,
        'objective': 'reg:absoluteerror',
        'n_jobs': 4
    }
    lgb_p = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 7),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'objective': 'mae',
        'verbose': -1,
        'n_jobs': 4
    }
    
    maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train_tgt.iloc[tr_idx], y_train_tgt.iloc[val_idx]
        y_v_raw = y_train_raw.iloc[val_idx]
        
        m_xgb = xgb.XGBRegressor(**xgb_p).fit(X_t, y_t)
        p_xgb = m_xgb.predict(X_v)
        
        m_lgb = lgb.LGBMRegressor(**lgb_p).fit(X_t, y_t)
        p_lgb = m_lgb.predict(X_v)
        
        preds_orig = np.expm1(w_xgb * p_xgb + (1 - w_xgb) * p_lgb)
        maes.append(mean_absolute_error(y_v_raw, preds_orig))
    
    return np.mean(maes)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
print('Starting 50-trial Ablation Study for Half_Day_Total_Debit...')
study.optimize(objective, n_trials=50)

best_p = study.best_params
w_xgb = best_p.pop('w_xgb')
xgb_p = {k.replace('xgb_', ''): v for k, v in best_p.items() if k.startswith('xgb_')}
xgb_p['enable_categorical'] = True; xgb_p['random_state'] = 42; xgb_p['objective'] = 'reg:absoluteerror'; xgb_p['n_jobs'] = 4
lgb_p = {k.replace('lgb_', ''): v for k, v in best_p.items() if k.startswith('lgb_')}
lgb_p['random_state'] = 42; lgb_p['verbose'] = -1; lgb_p['objective'] = 'mae'; lgb_p['n_jobs'] = 4

fold_r2, fold_mae, fold_wmape = [], [], []
for tr_idx, val_idx in tscv.split(X_train):
    X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_t, y_v = y_train_tgt.iloc[tr_idx], y_train_tgt.iloc[val_idx]
    y_v_raw = y_train_raw.iloc[val_idx]
    
    m_xgb = xgb.XGBRegressor(**xgb_p).fit(X_t, y_t)
    p_xgb = m_xgb.predict(X_v)
    
    m_lgb = lgb.LGBMRegressor(**lgb_p).fit(X_t, y_t)
    p_lgb = m_lgb.predict(X_v)
    
    preds = np.expm1(w_xgb * p_xgb + (1 - w_xgb) * p_lgb)
    fold_r2.append(r2_score(y_v_raw, preds))
    fold_mae.append(mean_absolute_error(y_v_raw, preds))
    fold_wmape.append(wmape(y_v_raw.values, preds))

print('\\n================================================')
print('ABLATION STUDY RESULTS (WITHOUT NEW FEATURES):')
print('================================================')
print(f'CV R2: {np.mean(fold_r2):.4f} +/- {np.std(fold_r2, ddof=1):.4f}')
print(f'CV MAE: {np.mean(fold_mae)/1e6:.3f}M PKR +/- {np.std(fold_mae)/1e6:.3f}M')
print(f'CV WMAPE: {np.mean(fold_wmape):.1f}% +/- {np.std(fold_wmape, ddof=1):.1f}%')
