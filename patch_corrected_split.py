"""
Bring the notebook onto the corrected evaluation harness, matching v3_pipeline.py.

STRUCTURAL CHANGES (numbers are refreshed afterwards by re-executing the notebook):

Cell 28  Train/hold-out split becomes date-based, and two shared helpers are defined:
         `date_cutoff()` and `rolling_origin_folds()`. Everything downstream — Section 6.4,
         Section 7.0 and the hold-out checks — now uses these, so there is one definition
         of the split in the whole notebook.

Cell 38  Section 6.4 stops carrying a large hardcoded parameter block and instead loads
         models/v3/cv_results.json when present, falling back to the production values for
         Colab. Its cross-validation switches from row-index TimeSeriesSplit to the shared
         rolling-origin folds, and the 99th-percentile cap is fitted per fold.

Run:  python patch_corrected_split.py
"""
import json
import shutil
import sys

NB = 'Bank_Cash_Optimization_Workflow.ipynb'

# ── Cell 28: split logic ─────────────────────────────────────────────────────
OLD_SPLIT = """# Match v3_pipeline.py exact split logic
cutoff_idx = int(len(half_daily) * 0.8)
cutoff_date = half_daily.iloc[cutoff_idx]['start_date']
train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()
test_df  = half_daily[half_daily['start_date'] >= cutoff_date].copy()
X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]
print(f"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Train shape : {X_train.shape}")
print(f"Test  shape : {X_test.shape}")"""

NEW_SPLIT = '''# ── Train / hold-out split and CV folds (matches v3_pipeline.py) ─────────────
# CORRECTED — see Section 7.0 for the full diagnosis. The previous version used
#     cutoff_idx  = int(len(half_daily) * 0.8)
#     cutoff_date = half_daily.iloc[cutoff_idx]['start_date']
# which indexes a frame sorted by BRANCH first, so row 80% lands inside one branch's
# block and *that row's date* became the global cutoff. It produced a 164-row hold-out
# (1.0% of the data, 5 days) instead of 20%. Both helpers below work on the calendar.

def date_cutoff(dates, frac=0.8):
    """Date at the given quantile of the UNIQUE-DATE timeline (never a row position)."""
    d = np.sort(pd.Series(dates).dropna().unique())
    return pd.Timestamp(d[int(len(d) * frac)])


def rolling_origin_folds(dates, n_folds=5, init_frac=0.5):
    """Expanding-window rolling-origin CV folds, split on the calendar.

    Fold k trains on every row dated before its validation window and validates on the
    next contiguous block of dates, so no training row ever post-dates its validation
    window. Because all 15 branches report across the whole timeline, every branch
    appears in every validation fold — which is what makes per-branch variance
    measurable at all. Returns (train_pos, val_pos) positional index arrays.
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
        folds.append((np.where((dates < lo).to_numpy())[0],
                      np.where(((dates >= lo) & (dates <= hi)).to_numpy())[0]))
    return folds


cutoff_date = date_cutoff(half_daily['start_date'], 0.8)
train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()
test_df  = half_daily[half_daily['start_date'] >= cutoff_date].copy()
X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

# Shared CV folds, carved out of the TRAINING window only so the hold-out stays untouched
CV_FOLDS = rolling_origin_folds(train_df['start_date'], n_folds=5, init_frac=0.5)

print(f"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Train shape : {X_train.shape}  ({len(train_df)/len(half_daily)*100:.1f}%)  "
      f"{train_df['start_date'].min().date()} -> {train_df['start_date'].max().date()}")
print(f"Test  shape : {X_test.shape}  ({len(test_df)/len(half_daily)*100:.1f}%)  "
      f"{test_df['start_date'].min().date()} -> {test_df['start_date'].max().date()}")
print(f"Hold-out covers {test_df['tran_br_code'].nunique()} of "
      f"{half_daily['tran_br_code'].nunique()} branches\\n")
print(f"{len(CV_FOLDS)} rolling-origin CV folds (all inside the training window):")
for _k, (_tr, _va) in enumerate(CV_FOLDS, 1):
    _vd = train_df.iloc[_va]['start_date']
    print(f"  Fold {_k}: train {len(_tr):>6,} | val {len(_va):>5,} | "
          f"{_vd.min().date()} -> {_vd.max().date()} | "
          f"branches in val: {train_df.iloc[_va]['tran_br_code'].nunique()}")'''

# ── Cell 38: Section 6.4 ─────────────────────────────────────────────────────
NEW_64 = '''# ── 6.4 XGBoost + LightGBM Ensemble V3 (tuned, matching production) ──────────
# Evaluated with the shared rolling-origin CV folds defined in the data-prep cell, which
# is the same scheme v3_pipeline.py uses. Hyperparameters come from the production Optuna
# run; the hardcoded block is the Colab fallback for when models/ is not on disk.

import json, os
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CV_RESULTS_PATH = 'models/v3/cv_results.json'
if os.path.exists(CV_RESULTS_PATH):
    with open(CV_RESULTS_PATH) as _f:
        cv_params = json.load(_f)
    print(f"Loaded tuned hyperparameters from {CV_RESULTS_PATH}")
else:
    cv_params = json.loads(r"""__CV_PARAMS_JSON__""")
    print("models/ not found (Colab) — using the hardcoded production hyperparameters")

print(f"Optuna trials: {cv_params['Half_Day_Total_Debit']['n_optuna_trials']} | "
      f"CV: {cv_params['Half_Day_Total_Debit']['n_cv_folds']}-fold "
      f"{cv_params['Half_Day_Total_Debit'].get('cv_scheme', 'rolling origin')}")


def _strip(params, prefix):
    """Exact prefix strip. The previous code used k.replace('xgb_', ''), which turned
    `xgb_colsample` into `colsample` — not a valid XGBoost/LightGBM argument, so the
    tuned column-subsampling value was silently ignored and defaulted to 1.0."""
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def build_ensemble(p):
    xp = _strip(p, 'xgb_')
    xp.update(enable_categorical=True, random_state=42, n_jobs=-1, verbosity=0)
    lp = _strip(p, 'lgb_')
    lp.update(random_state=42, n_jobs=-1, verbose=-1)
    return (xgb.XGBRegressor(**xp, objective='reg:absoluteerror'),
            lgb.LGBMRegressor(**lp, objective='mae'))


for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    print(f"\\n{'='*58}\\nEvaluating Ensemble Model for: {target}\\n{'='*58}")
    use_log = (target != 'Half_Day_Net_Cash')
    # Score against the untouched RAW series where one exists; fit on the winsorized one.
    # Previously the CV scored against the capped column while the hold-out used RAW, so the
    # two numbers were not comparable and the CV understated the error.
    _eval_col = f'{target}_RAW' if f'{target}_RAW' in train_df.columns else target
    y_fit_src   = train_df[target]        # winsorized — what the models are FITTED on
    y_train_raw = train_df[_eval_col]     # uncapped    — what we SCORE against
    t_params = cv_params[target]
    best_p = t_params['best_params']
    w_xgb = t_params['w_xgb']
    if _eval_col != target:
        print(f"  scoring against '{_eval_col}' (uncapped); fitting on winsorized '{target}'")

    def prep_y(raw_slice):
        """99th-percentile cap fitted on the rows passed in only — never on the
        validation rows, which is what the previous single global cap did."""
        capped = raw_slice.clip(upper=raw_slice.quantile(0.99))
        return np.log1p(capped.clip(lower=0)) if use_log else capped

    fold_mae, fold_r2, fold_rmse_vals = [], [], []
    fold_mape_vals, fold_smape_vals, fold_wmape_vals = [], [], []

    for fold_num, (tr_idx, val_idx) in enumerate(CV_FOLDS, 1):
        X_t, X_v = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_t = prep_y(y_fit_src.iloc[tr_idx])
        y_v_raw = y_train_raw.iloc[val_idx]

        m_xgb, m_lgb = build_ensemble(best_p)
        m_xgb.fit(X_t, y_t)
        m_lgb.fit(X_t, y_t)
        blend = w_xgb * m_xgb.predict(X_v) + (1 - w_xgb) * m_lgb.predict(X_v)
        preds_orig = np.expm1(blend) if use_log else blend

        fold_mae.append(mean_absolute_error(y_v_raw, preds_orig))
        fold_rmse_vals.append(np.sqrt(mean_squared_error(y_v_raw, preds_orig)))
        fold_r2.append(r2_score(y_v_raw, preds_orig))
        fold_mape_vals.append(calc_mape(y_v_raw.values, preds_orig))
        _sm, _ = calc_smape(y_v_raw.values, preds_orig)
        fold_smape_vals.append(_sm)
        fold_wmape_vals.append(calc_wmape(y_v_raw.values, preds_orig))
        print(f"  Fold {fold_num}: MAE={fold_mae[-1]/1e6:6.2f}M  R2={fold_r2[-1]:7.4f}  "
              f"WMAPE={fold_wmape_vals[-1]:6.2f}%  (n_val={len(val_idx)})")

    print(f"\\n  CV R2    : {np.mean(fold_r2):.4f} +/- {np.std(fold_r2, ddof=1):.4f}")
    print(f"  CV MAE   : {np.mean(fold_mae)/1e6:.2f}M +/- {np.std(fold_mae, ddof=1)/1e6:.2f}M")
    print(f"  CV RMSE  : {np.mean(fold_rmse_vals)/1e6:.2f}M")
    print(f"  CV MAPE  : {np.mean(fold_mape_vals):.1f}%   (metric artefact — see the MAPE/WMAPE section)")
    print(f"  CV SMAPE : {np.mean(fold_smape_vals):.1f}%")
    print(f"  CV WMAPE : {np.mean(fold_wmape_vals):.2f}% +/- {np.std(fold_wmape_vals, ddof=1):.2f}%")

    # Final model for the primary target, used by SHAP, the plots and the forecast section
    if target == 'Half_Day_Total_Debit':
        print("\\n  Training the final Debit ensemble on the full training window...")
        y_train_tgt = prep_y(y_fit_src)
        xgb_model, lgb_model = build_ensemble(best_p)
        xgb_model.fit(X_train, y_train_tgt)
        lgb_model.fit(X_train, y_train_tgt)
        W_DEBIT = w_xgb                       # explicit: `w_xgb` is reassigned each loop
        y_test_raw = test_df['Half_Day_Total_Debit_RAW'].values
        y_pred_xgb = np.expm1(W_DEBIT * xgb_model.predict(X_test)
                              + (1 - W_DEBIT) * lgb_model.predict(X_test))
        print(f"  Hold-out ({len(X_test):,} rows): R2={r2_score(y_test_raw, y_pred_xgb):.4f}  "
              f"MAE={mean_absolute_error(y_test_raw, y_pred_xgb)/1e6:.2f}M  "
              f"WMAPE={calc_wmape(y_test_raw, y_pred_xgb):.2f}%")

_d = cv_params['Half_Day_Total_Debit']
results.append({
    'Model': 'XGBoost + LightGBM Ensemble V3',
    'R2': _d['cv_r2_mean'], 'MAE_M': _d['cv_mae_mean_M'], 'RMSE_M': _d['cv_rmse_mean_M'],
    'MAPE_%': _d['cv_mape_mean'], 'SMAPE_%': _d['cv_smape_mean'], 'WMAPE_%': _d['cv_wmape_mean'],
})'''


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: patch_corrected_split.py <path to new cv_results.json>")
    cv_json = json.load(open(sys.argv[1], encoding='utf-8'))
    # keep the embedded fallback compact: params + headline metrics only
    slim = {}
    for t, d in cv_json.items():
        slim[t] = {k: v for k, v in d.items() if k != 'ho_per_branch'}
    payload = json.dumps(slim, indent=2)
    if '"""' in payload:
        sys.exit("ERROR: JSON payload contains a triple quote")

    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    shutil.copy(NB, NB + '.pre_corrected_split.bak')

    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and OLD_SPLIT in ''.join(c['source'])]
    if len(hits) != 1:
        sys.exit(f"ERROR: split cell not found uniquely: {hits}")
    i = hits[0]
    cells[i]['source'] = ''.join(cells[i]['source']).replace(OLD_SPLIT, NEW_SPLIT, 1).splitlines(keepends=True)
    cells[i]['outputs'], cells[i]['execution_count'] = [], None
    print(f"  cell {i}: split logic replaced")

    hits = [j for j, c in enumerate(cells) if c['cell_type'] == 'code'
            and '6.4 XGBoost + LightGBM Ensemble V3' in ''.join(c['source'])]
    if len(hits) != 1:
        sys.exit(f"ERROR: 6.4 cell not found uniquely: {hits}")
    j = hits[0]
    cells[j]['source'] = NEW_64.replace('__CV_PARAMS_JSON__', payload).splitlines(keepends=True)
    cells[j]['outputs'], cells[j]['execution_count'] = [], None
    print(f"  cell {j}: Section 6.4 rewritten ({len(payload):,} char fallback embedded)")

    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"Done. Backup: {NB}.pre_corrected_split.bak")


if __name__ == '__main__':
    main()
