"""
Rewire Section 7.0 onto the notebook's shared evaluation scheme.

Before this, Section 7.0 defined its own folds, its own copy of the hyperparameters and its
own model factory, because at the time it was the only correct evaluation in the notebook.
Now that Cell 28 defines `CV_FOLDS` and Section 6.4 defines `cv_params` / `build_ensemble`
on the same basis as v3_pipeline.py, Section 7.0 consumes those instead - so the three can
no longer drift apart. Section 7.0 also now scores against the raw uncapped target and uses
the real 20% hold-out rather than computing a private one.

Run:  python patch_section70_shared.py
"""
import json
import shutil
import sys

NB = 'Bank_Cash_Optimization_Workflow.ipynb'

STEP2 = r'''# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Per-branch results from the corrected cross-validation
# ═══════════════════════════════════════════════════════════════════════════
# This step deliberately defines nothing of its own. It consumes the notebook's shared
# evaluation scheme so Section 7.0, Section 6.4 and v3_pipeline.py cannot drift apart:
#   CV_FOLDS        rolling-origin folds, built on the calendar in the data-prep cell
#   cv_params       tuned hyperparameters, loaded in Section 6.4
#   build_ensemble  the model factory, defined in Section 6.4
# Models are FITTED on the winsorized target and SCORED against the raw uncapped one,
# which is what the notebook claims throughout and what the pipeline now does.

import numpy as np, pandas as pd, time
from sklearn.metrics import mean_absolute_error, r2_score

CV_TARGET   = 'Half_Day_Total_Debit'
CV_EVAL_COL = 'Half_Day_Total_Debit_RAW' if 'Half_Day_Total_Debit_RAW' in half_daily.columns else CV_TARGET
_BP = cv_params[CV_TARGET]['best_params']
_W  = cv_params[CV_TARGET]['w_xgb']


def cv_wmape(a, p):
    """Weighted MAPE: total absolute error / total absolute actual."""
    a, p = np.asarray(a, float), np.asarray(p, float)
    d = np.abs(a).sum()
    return float(np.abs(a - p).sum() / d * 100) if d > 0 else np.nan


def cv_mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = a != 0
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.sum() else np.nan


def cv_fit_predict(tr, va, cols=None):
    """Fit the production ensemble on `tr`, predict `va`."""
    cols = feature_cols if cols is None else cols
    y = tr[CV_TARGET]                                     # winsorized series for fitting
    y = np.log1p(y.clip(upper=y.quantile(0.99)).clip(lower=0))
    mx, ml = build_ensemble(_BP)
    mx.fit(tr[cols], y)
    ml.fit(tr[cols], y)
    return np.expm1(_W * mx.predict(va[cols]) + (1 - _W) * ml.predict(va[cols]))


print("=" * 84)
print("CORRECTED CV — rolling origin on the calendar, scored on raw uncapped actuals")
print("=" * 84)
print(f"Target scored: {CV_EVAL_COL}   |   folds: {len(CV_FOLDS)}   |   "
      f"blend: XGB {_W:.4f} / LGB {1 - _W:.4f}\n")

_t0, _rows, _agg = time.time(), [], []
for _k, (_tr_i, _va_i) in enumerate(CV_FOLDS, 1):
    _tr, _va = train_df.iloc[_tr_i], train_df.iloc[_va_i]
    _p = cv_fit_predict(_tr, _va)
    _y = _va[CV_EVAL_COL].values
    _agg.append({'Fold': _k,
                 'Val window': f"{_va['start_date'].min().date()} -> {_va['start_date'].max().date()}",
                 'n_train': len(_tr), 'n_val': len(_va),
                 'Branches': int(_va['tran_br_code'].nunique()),
                 'R2': r2_score(_y, _p), 'MAE_M': mean_absolute_error(_y, _p) / 1e6,
                 'WMAPE_%': cv_wmape(_y, _p), 'MAPE_%': cv_mape(_y, _p)})
    _rows.append(pd.DataFrame({'fold': _k, 'branch': _va['tran_br_code'].astype(int).values,
                               'date': _va['start_date'].values, 'actual': _y, 'pred': _p}))
    print(f"  Fold {_k}: {_agg[-1]['Val window']} | train {len(_tr):>6,} | val {len(_va):>5,} | "
          f"branches {_agg[-1]['Branches']:>2} | R2 {_agg[-1]['R2']:.4f} | WMAPE {_agg[-1]['WMAPE_%']:.2f}%")

CV_PREDS    = pd.concat(_rows, ignore_index=True)
CV_FOLD_TBL = pd.DataFrame(_agg)
print(f"\n  ({time.time() - _t0:.0f}s)")

print("\n" + "-" * 84 + "\nFOLD SUMMARY\n" + "-" * 84)
print(CV_FOLD_TBL.round(4).to_string(index=False))

CV_R2_MEAN, CV_R2_STD = CV_FOLD_TBL['R2'].mean(), CV_FOLD_TBL['R2'].std(ddof=1)
CV_WM_MEAN, CV_WM_STD = CV_FOLD_TBL['WMAPE_%'].mean(), CV_FOLD_TBL['WMAPE_%'].std(ddof=1)
print(f"\n  CORRECTED : R2 = {CV_R2_MEAN:.4f} +/- {CV_R2_STD:.4f}   "
      f"WMAPE = {CV_WM_MEAN:.2f}% +/- {CV_WM_STD:.2f}%")
print(f"  Every one of the {int(CV_FOLD_TBL['Branches'].min())} branches appears in all "
      f"{len(CV_FOLDS)} validation folds, which is what makes the rest of this section possible.")

# ── Per branch x fold, then per branch ────────────────────────────────────
_bf = []
for (_b, _k), _g in CV_PREDS.groupby(['branch', 'fold']):
    _bf.append({'branch': _b, 'fold': _k, 'n': len(_g),
                'r2': r2_score(_g['actual'], _g['pred']),
                'mae_M': mean_absolute_error(_g['actual'], _g['pred']) / 1e6,
                'wmape': cv_wmape(_g['actual'].values, _g['pred'].values),
                'mape': cv_mape(_g['actual'].values, _g['pred'].values),
                'abs_err': float(np.abs(_g['actual'] - _g['pred']).sum()),
                'abs_act': float(np.abs(_g['actual']).sum())})
CV_BRANCH_FOLD = pd.DataFrame(_bf)

CV_BRANCH = CV_BRANCH_FOLD.groupby('branch').agg(
    Folds=('fold', 'nunique'), Rows=('n', 'sum'),
    R2_mean=('r2', 'mean'),       R2_std=('r2', lambda s: s.std(ddof=1)),
    WMAPE_mean=('wmape', 'mean'), WMAPE_std=('wmape', lambda s: s.std(ddof=1)),
    MAE_mean=('mae_M', 'mean'),   MAE_std=('mae_M', lambda s: s.std(ddof=1)),
    MAPE_mean=('mape', 'mean')).reset_index()

_tot_e, _tot_a = CV_BRANCH_FOLD['abs_err'].sum(), CV_BRANCH_FOLD['abs_act'].sum()
_sh = CV_BRANCH_FOLD.groupby('branch')[['abs_err', 'abs_act']].sum()
CV_BRANCH = CV_BRANCH.merge(pd.DataFrame({
    'branch': _sh.index.to_numpy(),
    'ErrShare_%': (_sh['abs_err'] / _tot_e * 100).to_numpy(),
    'DemandShare_%': (_sh['abs_act'] / _tot_a * 100).to_numpy()}), on='branch')
CV_BRANCH['Err_vs_Demand'] = CV_BRANCH['ErrShare_%'] / CV_BRANCH['DemandShare_%']

print("\n" + "=" * 84)
print("PER-BRANCH PERFORMANCE — every branch, every fold (sorted worst WMAPE first)")
print("=" * 84)
print("  R2_std / WMAPE_std = that branch's OWN fold-to-fold variability")
print("  Err_vs_Demand      = share of portfolio error / share of portfolio demand; >1 = drags\n")
print(CV_BRANCH.sort_values('WMAPE_mean', ascending=False).round(3).to_string(index=False))


def cv_tier(r):
    if r['R2_mean'] >= 0.60 and r['WMAPE_mean'] <= 37:
        return 'Production ready'
    if r['R2_mean'] >= 0.50 and r['WMAPE_mean'] <= 45:
        return 'Acceptable — monitor'
    return 'Needs attention'


CV_BRANCH['Tier'] = CV_BRANCH.apply(cv_tier, axis=1)
print("\n  Tier counts: " + ", ".join(f"{k} = {v}" for k, v in CV_BRANCH['Tier'].value_counts().items()))
'''

STEP5_OLD_HEAD = """print("Training one dedicated model per branch per fold (this takes a minute)...")
_t0, _recs = time.time(), []
for _k, (_a, _b) in enumerate(CV_FOLDS, 1):
    _tr_all = half_daily[half_daily['start_date'] < _a]
    _glob = CV_PREDS[CV_PREDS['fold'] == _k]                    # reuse Step 2 predictions
    for _br in sorted(CV_PREDS['branch'].unique()):
        _tr_b = _tr_all[_tr_all['tran_br_code'] == _br]
        _va_b = half_daily[(half_daily['start_date'] >= _a) & (half_daily['start_date'] <= _b) &
                           (half_daily['tran_br_code'] == _br)]
        if len(_va_b) < 5 or len(_tr_b) < 50:
            continue
        _p_loc = cv_fit_predict(_tr_b, _va_b, _local_cols)
        _g = _glob[_glob['branch'] == _br]
        _y = _va_b[CV_TARGET].values
        _recs.append({'branch': _br, 'fold': _k, 'branch_train_rows': len(_tr_b),
                      'wmape_pooled': cv_wmape(_g['actual'].values, _g['pred'].values),
                      'wmape_dedicated': cv_wmape(_y, _p_loc),
                      'r2_pooled': r2_score(_g['actual'], _g['pred']),
                      'r2_dedicated': r2_score(_y, _p_loc)})
    print(f"  fold {_k} done")"""

STEP5_NEW_HEAD = """print("Training one dedicated model per branch per fold (this takes a minute)...")
_t0, _recs = time.time(), []
for _k, (_tr_i, _va_i) in enumerate(CV_FOLDS, 1):
    _tr_all, _va_all = train_df.iloc[_tr_i], train_df.iloc[_va_i]
    _glob = CV_PREDS[CV_PREDS['fold'] == _k]                    # reuse Step 2 predictions
    for _br in sorted(CV_PREDS['branch'].unique()):
        _tr_b = _tr_all[_tr_all['tran_br_code'] == _br]
        _va_b = _va_all[_va_all['tran_br_code'] == _br]
        if len(_va_b) < 5 or len(_tr_b) < 50:
            continue
        _p_loc = cv_fit_predict(_tr_b, _va_b, _local_cols)
        _g = _glob[_glob['branch'] == _br]
        _y = _va_b[CV_EVAL_COL].values
        _recs.append({'branch': _br, 'fold': _k, 'branch_train_rows': len(_tr_b),
                      'wmape_pooled': cv_wmape(_g['actual'].values, _g['pred'].values),
                      'wmape_dedicated': cv_wmape(_y, _p_loc),
                      'r2_pooled': r2_score(_g['actual'], _g['pred']),
                      'r2_dedicated': r2_score(_y, _p_loc)})
    print(f"  fold {_k} done")"""

STEP5_OLD_DRV = """_raw = half_daily.groupby('tran_br_code', observed=True)[CV_TARGET].agg(['mean', 'std', 'size'])"""
STEP5_NEW_DRV = """_raw = half_daily.groupby('tran_br_code', observed=True)[CV_EVAL_COL].agg(['mean', 'std', 'size'])"""

STEP5_OLD_LOCAL = "_local_cols = [c for c in CV_FEATURES if c != 'tran_br_code']   # constant within a branch"
STEP5_NEW_LOCAL = "_local_cols = [c for c in feature_cols if c != 'tran_br_code']  # constant within a branch"

# ── Step 6: use the real hold-out instead of computing a private one ─────────
STEP6_OLD = """_dates = np.array(sorted(half_daily['start_date'].unique()))
_ho_start = _dates[int(len(_dates) * 0.8)]
_tr_h = half_daily[half_daily['start_date'] < _ho_start]
_te_h = half_daily[half_daily['start_date'] >= _ho_start]
_p_h = cv_fit_predict(_tr_h, _te_h)
_y_h = _te_h[CV_TARGET].values

_cut_idx = int(len(half_daily) * 0.8)
_cut_date = half_daily.iloc[_cut_idx]['start_date']
_n_old = int((half_daily['start_date'] >= _cut_date).sum())

print(f"  Current hold-out (Cell 27)  : from {_cut_date.date()}  ->  {_n_old:,} rows "
      f"({_n_old / len(half_daily) * 100:.1f}% of data, {(half_daily['start_date'].max() - _cut_date).days} days)")
print(f"  Proper 20% date hold-out    : from {pd.Timestamp(_ho_start).date()}  ->  {len(_te_h):,} rows "
      f"({len(_te_h) / len(half_daily) * 100:.1f}% of data, "
      f"{(half_daily['start_date'].max() - pd.Timestamp(_ho_start)).days} days), all "
      f"{_te_h['tran_br_code'].nunique()} branches")
print(f"\\n  Performance on the proper hold-out:")
print(f"     R2 = {r2_score(_y_h, _p_h):.4f}   MAE = {mean_absolute_error(_y_h, _p_h) / 1e6:.2f}M PKR   "
      f"WMAPE = {cv_wmape(_y_h, _p_h):.2f}%")
print(f"     against cross-validation: R2 = {CV_R2_MEAN:.4f}, WMAPE = {CV_WM_MEAN:.2f}%")
print("\\n  The model holds up — but the evidence base behind the number is 21x larger.")

CV_HOLDOUT = _te_h.assign(pred=_p_h)"""

STEP6_NEW = """# The hold-out is the one defined in the data-prep cell and scored in Section 6.4 —
# no private re-split here, so this table cannot disagree with the rest of the notebook.
_te_h = test_df
_y_h  = y_test_raw                       # Half_Day_Total_Debit_RAW, uncapped
_p_h  = y_pred_xgb                       # final ensemble predictions from Section 6.4

# What the OLD positional split would have produced, for comparison
_old_cut = half_daily.iloc[int(len(half_daily) * 0.8)]['start_date']
_n_old = int((half_daily['start_date'] >= _old_cut).sum())

print(f"  Old positional split would give : from {_old_cut.date()}  ->  {_n_old:,} rows "
      f"({_n_old / len(half_daily) * 100:.1f}% of data, "
      f"{(half_daily['start_date'].max() - _old_cut).days} days)")
print(f"  Corrected date-based hold-out   : from {_te_h['start_date'].min().date()}  ->  "
      f"{len(_te_h):,} rows ({len(_te_h) / len(half_daily) * 100:.1f}% of data, "
      f"{(_te_h['start_date'].max() - _te_h['start_date'].min()).days} days), all "
      f"{_te_h['tran_br_code'].nunique()} branches")
print(f"\\n  Performance on the corrected hold-out:")
print(f"     R2 = {r2_score(_y_h, _p_h):.4f}   MAE = {mean_absolute_error(_y_h, _p_h) / 1e6:.2f}M PKR   "
      f"WMAPE = {cv_wmape(_y_h, _p_h):.2f}%")
print(f"     against cross-validation: R2 = {CV_R2_MEAN:.4f}, WMAPE = {CV_WM_MEAN:.2f}%")
print(f"\\n  The evidence base behind the hold-out number is {len(_te_h) / max(_n_old, 1):.0f}x larger "
      f"than the old split allowed.")

CV_HOLDOUT = _te_h[['tran_br_code', 'start_date']].copy()
CV_HOLDOUT[CV_TARGET] = _y_h
CV_HOLDOUT['pred'] = _p_h"""

STEP6_OLD_LOOP = """for _b, _g in CV_HOLDOUT.groupby('tran_br_code', observed=True):
    _ho_rows.append({'branch': int(_b), 'n': len(_g), 'HO_R2': r2_score(_g[CV_TARGET], _g['pred']),
                     'HO_WMAPE': cv_wmape(_g[CV_TARGET].values, _g['pred'].values),
                     'HO_MAE_M': mean_absolute_error(_g[CV_TARGET], _g['pred']) / 1e6})"""
STEP6_NEW_LOOP = """for _b, _g in CV_HOLDOUT.groupby('tran_br_code', observed=True):
    if len(_g) < 2:
        continue
    _ho_rows.append({'branch': int(_b), 'n': len(_g), 'HO_R2': r2_score(_g[CV_TARGET], _g['pred']),
                     'HO_WMAPE': cv_wmape(_g[CV_TARGET].values, _g['pred'].values),
                     'HO_MAE_M': mean_absolute_error(_g[CV_TARGET], _g['pred']) / 1e6})"""


def sub(cells, i, old, new, label):
    s = ''.join(cells[i]['source'])
    if old not in s:
        sys.exit(f"ERROR [{label}]: pattern not found in cell {i}")
    cells[i]['source'] = s.replace(old, new, 1).splitlines(keepends=True)
    cells[i]['outputs'], cells[i]['execution_count'] = [], None


def find(cells, needle):
    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and needle in ''.join(c['source'])]
    if len(hits) != 1:
        sys.exit(f"ERROR: '{needle}' matched {hits}")
    return hits[0]


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    shutil.copy(NB, NB + '.pre_s70shared.bak')

    i2 = find(cells, 'STEP 2 — Corrected CV')
    cells[i2]['source'] = STEP2.strip('\n').splitlines(keepends=True)
    cells[i2]['outputs'], cells[i2]['execution_count'] = [], None
    print(f"  cell {i2}: STEP 2 rewritten to use shared CV_FOLDS / cv_params / build_ensemble")

    i5 = find(cells, "STEP 5 — The supervisor's decision")
    sub(cells, i5, STEP5_OLD_LOCAL, STEP5_NEW_LOCAL, 'STEP5 local cols')
    sub(cells, i5, STEP5_OLD_HEAD, STEP5_NEW_HEAD, 'STEP5 loop')
    sub(cells, i5, STEP5_OLD_DRV, STEP5_NEW_DRV, 'STEP5 drivers')
    print(f"  cell {i5}: STEP 5 rewired onto CV_FOLDS")

    i6 = find(cells, 'STEP 6 — Branches 202 and 1739')
    sub(cells, i6, STEP6_OLD, STEP6_NEW, 'STEP6 holdout')
    sub(cells, i6, STEP6_OLD_LOOP, STEP6_NEW_LOOP, 'STEP6 loop')
    print(f"  cell {i6}: STEP 6 now uses the notebook's real hold-out")

    # STEP 3 keeps its logic but now exports the decomposition so the narrative markdown
    # can be regenerated from the executed numbers instead of being hand-maintained.
    i3 = find(cells, 'STEP 3 — Concentrated or spread')
    sub(cells, i3,
        "CV_WM_MATRIX, _g, CV_BRANCH_EFFECT, CV_FOLD_EFFECT, _pb, _pf, _pi = _decompose('wmape')\n"
        "_, _, _, _, _pb2, _pf2, _pi2 = _decompose('r2')",
        "CV_WM_MATRIX, _g, CV_BRANCH_EFFECT, CV_FOLD_EFFECT, _pb, _pf, _pi = _decompose('wmape')\n"
        "_, _, _, _, _pb2, _pf2, _pi2 = _decompose('r2')\n"
        "_ANOVA_WMAPE = (_pb, _pf, _pi)      # (branch %, fold %, interaction %)\n"
        "_ANOVA_R2    = (_pb2, _pf2, _pi2)",
        'STEP3 anova export')
    sub(cells, i3,
        "_r = _cum[-1]['WMAPE_std'] / _BWMS",
        "_r = _cum[-1]['WMAPE_std'] / _BWMS\n"
        "CV_CUM_EXCL, CV_CUM5_RATIO = _cum, _r\n"
        "CV_BASE_R2_STD, CV_BASE_WM_STD = _BR2S, _BWMS",
        'STEP3 cumulative export')
    print(f"  cell {i3}: STEP 3 exports the variance decomposition and exclusion ladder")

    # Steps 1, 4 and 7 read only shared frames, so they need no edit - but their stored
    # outputs are now stale.
    for needle in ('STEP 1 — Diagnosis', 'STEP 4 — Visualising', 'STEP 7 — Verdict'):
        j = find(cells, needle)
        cells[j]['outputs'], cells[j]['execution_count'] = [], None

    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"Done. Backup: {NB}.pre_s70shared.bak")


if __name__ == '__main__':
    main()
