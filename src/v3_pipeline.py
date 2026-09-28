"""
Train the V3 XGBoost + LightGBM ensemble for half-daily branch cash demand.

For each target (debit, credit, net cash):
  1. split the data by date: first 80% of dates for training, last 20% held out;
  2. build 5 rolling-origin CV folds inside the training window;
  3. tune the ensemble with Optuna on those folds;
  4. refit on the full training window, score the hold-out and save the model.

Notes on the evaluation (Section 7.0 of the notebook explains each one):
  - the split and folds use calendar dates, not row positions; the frame is sorted by
    branch, so row positions correspond to blocks of branches rather than periods;
  - the xgb_/lgb_ prefixes are stripped exactly, so colsample_bytree reaches the model;
  - the 99th-percentile cap is fitted inside each fold on that fold's training rows;
  - CV and hold-out are both scored against the uncapped actuals (the *_RAW columns).

Run from the repository root:  python src/v3_pipeline.py
"""
import os
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
from metrics_utils import mape, smape, wmape

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Config
N_OPTUNA_TRIALS = 50
N_CV_FOLDS      = 5      # 5 rolling-origin folds; every branch appears in each
CV_INIT_FRAC    = 0.50   # first 50% of the training timeline seeds the expanding window
HOLDOUT_FRAC    = 0.80   # 80th percentile of unique DATES
OPTUNA_SEED     = 42

TARGETS = ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']

print("Training V3 ensemble...")
print(f"  Optuna trials  : {N_OPTUNA_TRIALS}")
print(f"  CV folds       : {N_CV_FOLDS} (rolling origin, split on calendar date)")

# 1. Load preprocessed data
df = pd.read_csv('data/processed/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

df['Days_to_Salary'] = df['Day'].apply(lambda d: 25 - d if d < 25 else (31 - d + 5)).clip(lower=0, upper=25)
df['Days_Since_Salary'] = df['Day'].apply(lambda d: d - 25 if d >= 25 else d + (31 - 25))
df['Is_Month_Start'] = df['start_date'].dt.is_month_start.astype(int)
df['Is_Month_End'] = df['start_date'].dt.is_month_end.astype(int)

df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
df['lag_1_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].shift(1).fillna(0)
df['rolling_14_mean_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
df['ewma_14_Txn_Count'] = df.groupby('tran_br_code')['Txn_Count'].transform(
    lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)

for target in TARGETS:
    df[f'rolling_14_std_{target}'] = df.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)
    df[f'ewma_14_{target}'] = df.groupby('tran_br_code')[target].transform(
        lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)
    df[f'dow_avg_4_{target}'] = df.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[target].transform(
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


# 2. Splitting helpers
def date_cutoff(dates, frac):
    """Cutoff taken from the unique-date timeline, never from row position."""
    d = np.sort(pd.Series(dates).dropna().unique())
    return d[int(len(d) * frac)]


def rolling_origin_folds(dates, n_folds=N_CV_FOLDS, init_frac=CV_INIT_FRAC):
    """Expanding-window rolling-origin folds, split on the calendar.

    Fold k trains on every row dated before its validation window and validates on the
    next contiguous block of dates. Because all branches report across the whole
    timeline, every branch lands in every validation fold. Returns positional indices.
    """
    dates = pd.Series(np.asarray(dates))
    uniq = np.sort(dates.unique())
    init = int(len(uniq) * init_frac)
    block = (len(uniq) - init) // n_folds
    folds = []
    for k in range(n_folds):
        s = init + k * block
        e = init + (k + 1) * block if k < n_folds - 1 else len(uniq)
        lo, hi = uniq[s], uniq[e - 1]
        tr = np.where((dates < lo).to_numpy())[0]
        va = np.where(((dates >= lo) & (dates <= hi)).to_numpy())[0]
        folds.append((tr, va))
    return folds


def strip_prefix(params, prefix):
    """Exact prefix strip; `.replace()` could corrupt names elsewhere in the key."""
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def eval_column(frame, target):
    """Column to SCORE against: the untouched RAW series where one exists."""
    raw = f'{target}_RAW'
    return raw if raw in frame.columns else target


cutoff_date = date_cutoff(df['start_date'], HOLDOUT_FRAC)
train_df = df[df['start_date'] < cutoff_date].reset_index(drop=True)
test_df = df[df['start_date'] >= cutoff_date].reset_index(drop=True)
X_train, X_test = train_df[feature_cols], test_df[feature_cols]

print(f"\nCutoff date : {pd.Timestamp(cutoff_date).strftime('%Y-%m-%d')}")
print(f"Train size  : {len(X_train):,} rows  ({len(X_train)/len(df)*100:.1f}%)  "
      f"{train_df['start_date'].min().date()} -> {train_df['start_date'].max().date()}")
print(f"Test  size  : {len(X_test):,} rows  ({len(X_test)/len(df)*100:.1f}%)  "
      f"{test_df['start_date'].min().date()} -> {test_df['start_date'].max().date()}")
print(f"Branches in hold-out : {test_df['tran_br_code'].nunique()} of {df['tran_br_code'].nunique()}")

CV_FOLDS = rolling_origin_folds(train_df['start_date'])
print(f"\nCV folds ({N_CV_FOLDS}, rolling origin):")
for k, (tr, va) in enumerate(CV_FOLDS, 1):
    vd = train_df.iloc[va]['start_date']
    print(f"  Fold {k}: train {len(tr):>6,} | val {len(va):>5,} | "
          f"{vd.min().date()} -> {vd.max().date()} | "
          f"branches in val: {train_df.iloc[va]['tran_br_code'].nunique()}")

os.makedirs('models/v3', exist_ok=True)
cv_summary = {}


def build_models(p):
    """Instantiate the pair from a flat best-params dict."""
    xp = strip_prefix(p, 'xgb_')
    xp.update(enable_categorical=True, random_state=OPTUNA_SEED, n_jobs=-1, verbosity=0)
    lp = strip_prefix(p, 'lgb_')
    lp.update(random_state=OPTUNA_SEED, n_jobs=-1, verbose=-1)
    return (xgb.XGBRegressor(**xp, objective='reg:absoluteerror'),
            lgb.LGBMRegressor(**lp, objective='mae'))


# 3. Modeling
for target in TARGETS:
    print(f"\n{'='*60}\nTraining Advanced Ensemble Model for: {target}\n{'='*60}")

    use_log = (target != 'Half_Day_Net_Cash')
    _eval_col = eval_column(df, target)

    y_fit_src = train_df[target]          # winsorized series — what the models are FITTED on
    y_train_raw = train_df[_eval_col]     # untouched series — what we SCORE against
    y_test_raw = test_df[_eval_col]
    if _eval_col != target:
        print(f"  scoring against '{_eval_col}' (uncapped); fitting on winsorized '{target}'")

    def prep_y(raw_slice):
        """Cap fitted on the rows passed in only - never on validation rows."""
        cap = raw_slice.quantile(0.99)
        capped = raw_slice.clip(upper=cap)
        return np.log1p(capped.clip(lower=0)) if use_log else capped

    def objective(trial):
        p = {
            'w_xgb': trial.suggest_float('w_xgb', 0.0, 1.0),
            'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400, step=50),
            'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
            'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
            'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            # named so the stripped key is the real argument `colsample_bytree`
            'xgb_colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 100, 400, step=50),
            'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
            'lgb_max_depth': trial.suggest_int('lgb_max_depth', 3, 7),
            'lgb_subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'lgb_colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        }
        w = p['w_xgb']
        errs = []
        for tr_idx, va_idx in CV_FOLDS:
            X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_t = prep_y(y_fit_src.iloc[tr_idx])
            m_x, m_l = build_models(p)
            m_x.fit(X_t, y_t)
            m_l.fit(X_t, y_t)
            blend = w * m_x.predict(X_v) + (1 - w) * m_l.predict(X_v)
            pred = np.expm1(blend) if use_log else blend
            # scored against RAW uncapped actuals, the same basis as the reported metrics
            errs.append(mean_absolute_error(y_train_raw.iloc[va_idx], pred))
        return float(np.mean(errs))

    print(f"Running Optuna ({N_OPTUNA_TRIALS} trials x {N_CV_FOLDS} folds)...")
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_p = study.best_params
    w_xgb = best_p['w_xgb']
    print(f"Best trial MAE: {study.best_value/1e6:.3f}M | w_xgb={w_xgb:.4f}")

    # 4. CV evaluation with the best params
    print(f"\nEvaluating on {N_CV_FOLDS}-fold rolling-origin CV...")
    fold_mae, fold_r2, fold_rmse_vals = [], [], []
    fold_mape_vals, fold_smape_vals, fold_wmape_vals = [], [], []

    for fold_num, (tr_idx, va_idx) in enumerate(CV_FOLDS, 1):
        X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_t = prep_y(y_fit_src.iloc[tr_idx])
        y_v_raw = y_train_raw.iloc[va_idx]

        m_x, m_l = build_models(best_p)
        m_x.fit(X_t, y_t)
        m_l.fit(X_t, y_t)
        blend = w_xgb * m_x.predict(X_v) + (1 - w_xgb) * m_l.predict(X_v)
        preds_orig = np.expm1(blend) if use_log else blend

        fold_mae.append(mean_absolute_error(y_v_raw, preds_orig))
        fold_rmse_vals.append(np.sqrt(mean_squared_error(y_v_raw, preds_orig)))
        fold_r2.append(r2_score(y_v_raw, preds_orig))
        fold_mape_vals.append(mape(y_v_raw.values, preds_orig))
        _s, _ = smape(y_v_raw.values, preds_orig)
        fold_smape_vals.append(_s)
        fold_wmape_vals.append(wmape(y_v_raw.values, preds_orig))
        print(f"  Fold {fold_num}: MAE={fold_mae[-1]/1e6:6.2f}M  R2={fold_r2[-1]:7.4f}  "
              f"WMAPE={fold_wmape_vals[-1]:6.2f}%  (n_val={len(va_idx)})")

    def ms(v):
        return float(np.mean(v)), float(np.std(v, ddof=1))

    cv_mae_mean, cv_mae_std = ms(fold_mae)
    cv_rmse_mean, cv_rmse_std = ms(fold_rmse_vals)
    cv_r2_mean, cv_r2_std = ms(fold_r2)
    cv_mape_mean, cv_mape_std = ms(fold_mape_vals)
    cv_smape_mean, cv_smape_std = ms(fold_smape_vals)
    cv_wmape_mean, cv_wmape_std = ms(fold_wmape_vals)

    print(f"\n  CV Summary ({N_CV_FOLDS}-fold rolling origin):")
    print(f"    R2    : {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}")
    print(f"    MAE   : {cv_mae_mean/1e6:.2f}M +/- {cv_mae_std/1e6:.2f}M")
    print(f"    WMAPE : {cv_wmape_mean:.2f}% +/- {cv_wmape_std:.2f}%")

    # 5. Final models on the full training window
    print("Training final ensemble on the full training window...")
    y_train_final = prep_y(y_fit_src)
    final_xgb, final_lgb = build_models(best_p)
    final_xgb.fit(X_train, y_train_final)
    final_lgb.fit(X_train, y_train_final)
    joblib.dump({'xgb': final_xgb, 'lgb': final_lgb, 'w_xgb': w_xgb},
                f'models/v3/model_{target}.pkl')

    # 6. Hold-out evaluation
    blend = w_xgb * final_xgb.predict(X_test) + (1 - w_xgb) * final_lgb.predict(X_test)
    y_pred = np.expm1(blend) if use_log else blend

    ho_mae = mean_absolute_error(y_test_raw, y_pred)
    ho_rmse = float(np.sqrt(mean_squared_error(y_test_raw, y_pred)))
    ho_r2 = r2_score(y_test_raw, y_pred)
    ho_mape_val = mape(y_test_raw.values, y_pred)
    ho_smape_val, _ = smape(y_test_raw.values, y_pred)
    ho_wmape_val = wmape(y_test_raw.values, y_pred)
    print(f"  Hold-out ({len(X_test):,} rows): R2={ho_r2:.4f}  MAE={ho_mae/1e6:.2f}M  "
          f"WMAPE={ho_wmape_val:.2f}%")

    # per-branch hold-out breakdown (now meaningful: ~230 rows per branch, not ~11)
    ho_branch = []
    _t = test_df[['tran_br_code']].copy()
    _t['actual'] = y_test_raw.values
    _t['pred'] = y_pred
    for b, g in _t.groupby('tran_br_code', observed=True):
        ho_branch.append({
            'branch': int(b), 'n': int(len(g)),
            'r2': round(float(r2_score(g['actual'], g['pred'])), 4),
            'mae_M': round(float(mean_absolute_error(g['actual'], g['pred'])) / 1e6, 3),
            'wmape': round(float(wmape(g['actual'].values, g['pred'].values)), 2),
            'mape': round(float(mape(g['actual'].values, g['pred'].values)), 2),
        })

    cv_summary[target] = {
        'n_optuna_trials': N_OPTUNA_TRIALS,
        'n_cv_folds': N_CV_FOLDS,
        'cv_scheme': 'rolling-origin expanding window, split on calendar date',
        'best_params': best_p,
        'w_xgb': w_xgb,
        'cv_mae_mean_M': round(cv_mae_mean / 1e6, 3), 'cv_mae_std_M': round(cv_mae_std / 1e6, 3),
        'cv_rmse_mean_M': round(cv_rmse_mean / 1e6, 3), 'cv_rmse_std_M': round(cv_rmse_std / 1e6, 3),
        'cv_r2_mean': round(cv_r2_mean, 4), 'cv_r2_std': round(cv_r2_std, 4),
        'cv_mape_mean': round(cv_mape_mean, 2), 'cv_mape_std': round(cv_mape_std, 2),
        'cv_smape_mean': round(cv_smape_mean, 2), 'cv_smape_std': round(cv_smape_std, 2),
        'cv_wmape_mean': round(cv_wmape_mean, 2), 'cv_wmape_std': round(cv_wmape_std, 2),
        'cv_fold_mae_M': [round(v / 1e6, 3) for v in fold_mae],
        'cv_fold_rmse_M': [round(v / 1e6, 3) for v in fold_rmse_vals],
        'cv_fold_r2': [round(v, 4) for v in fold_r2],
        'cv_fold_mape': [round(v, 2) for v in fold_mape_vals],
        'cv_fold_smape': [round(v, 2) for v in fold_smape_vals],
        'cv_fold_wmape': [round(v, 2) for v in fold_wmape_vals],
        'ho_test_rows': int(len(X_test)),
        'ho_mae_M': round(ho_mae / 1e6, 3), 'ho_rmse_M': round(ho_rmse / 1e6, 3),
        'ho_r2': round(ho_r2, 4), 'ho_mape': round(ho_mape_val, 2),
        'ho_smape': round(ho_smape_val, 2), 'ho_wmape': round(ho_wmape_val, 2),
        'ho_per_branch': ho_branch,
    }

split_cfg = {
    'holdout_frac_of_dates': HOLDOUT_FRAC,
    'cutoff_date': str(pd.Timestamp(cutoff_date).date()),
    'n_train_rows': int(len(X_train)), 'n_test_rows': int(len(X_test)),
    'n_cv_folds': N_CV_FOLDS, 'cv_init_frac': CV_INIT_FRAC,
    'cv_fold_windows': [
        {'fold': k,
         'val_start': str(train_df.iloc[va]['start_date'].min().date()),
         'val_end': str(train_df.iloc[va]['start_date'].max().date()),
         'n_train': int(len(tr)), 'n_val': int(len(va)),
         'branches_in_val': int(train_df.iloc[va]['tran_br_code'].nunique())}
        for k, (tr, va) in enumerate(CV_FOLDS, 1)],
}
with open('models/v3/split_config.json', 'w') as f:
    json.dump(split_cfg, f, indent=2)
with open('models/v3/v3_config.json', 'w') as f:
    json.dump({'targets_using_log1p': ['Half_Day_Total_Debit', 'Half_Day_Total_Credit']}, f)
with open('models/v3/cv_results.json', 'w') as f:
    json.dump(cv_summary, f, indent=2)

print("\nV3 Ensemble Pipeline completed successfully.")
