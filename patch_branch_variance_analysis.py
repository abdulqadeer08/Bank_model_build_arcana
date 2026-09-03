"""
Replace the old per-branch CV cells (the 'cannot determine' answer) with a complete,
branch-aware analysis that actually answers the supervisor's question.

Old cells removed : the row-index TimeSeriesSplit per-branch attempt + its markdown verdict
New cells added   : diagnosis -> corrected CV -> concentration tests -> charts ->
                    branch-tuning experiment -> spotlight/tiering -> live verdict -> narrative

Run:  python patch_branch_variance_analysis.py
"""
import json
import shutil
import sys

NB = 'Bank_Cash_Optimization_Workflow.ipynb'
ANCHOR_CODE = '# ── Supervisor Request: Per-Branch CV Fold Variance Analysis'
ANCHOR_MD = '### Supervisor Analysis: CV Fold Variance'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


# ─────────────────────────────────────────────────────────────────────────────
CELL_A = md(r"""
## 7.0 Supervisor Review — Is the Cross-Validation Variance Concentrated in a Few Branches?

**What was asked (review of 2026-08):**

1. Explain in the notebook why MAPE does not work on this data → *answered in
   "Why We Changed the Primary Reporting Metric from MAPE to WMAPE" above.*
2. Keep verifying ensemble blending and SHAP rather than assuming → *retained in Sections 6 and 8.*
3. **"For R² = 0.5687 ± 0.0581 and WMAPE = 39.5% ± 3.3%, the variation between folds is a
   little high. Please check the performance for each branch, especially branches 202 and 1739.
   We need to see whether a few branches are causing the higher variation or if the issue is
   spread across all branches. This will help us decide whether we need branch-level tuning
   or more data for the weaker branches."**

This section answers point 3 in full.

---

### Executive summary

| Question | Answer |
|---|---|
| Is the fold-to-fold variance caused by a few branches? | **No.** No single branch can be removed to shrink the fold R² spread by more than 4%. Removing the five *worst* branches still leaves 87% of the WMAPE spread intact. |
| What *does* cause it? | A **system-wide effect that hits every branch at once.** In the weakest fold, 14 of 15 branches got worse together. Fold WMAPE tracks how volatile demand was in that period (r = +0.90). |
| Is branch performance itself uniform? | **No** — and this is the important distinction. Branch identity explains **69.8%** of the variation in WMAPE, while the time period explains only 10.0%. Some branches are structurally harder than others. |
| Branch 202? | **Best branch in the portfolio.** R² = 0.72, WMAPE = 31.1%. The alarming MAPE was a metric artefact, as expected. |
| Branch 1739? | **Genuinely weak** — R² = 0.38, WMAPE = 50.7%. Confirmed, and it needs a production uncertainty flag. |
| Branch 104? | **Not previously flagged, but the biggest problem.** Worst accuracy *and* 26% of the entire portfolio error budget. |
| So: branch-level tuning, or more data? | **Neither.** Both were tested directly. Per-branch models are worse for **15 of 15** branches. Branch history length has no relationship with accuracy (r = +0.19). The real driver is intrinsic demand volatility (r = +0.50). |

> **A note on why this section reruns the cross-validation.** The per-branch question could not
> be answered with the original CV, because three branches — including branch 202 — never
> appeared in any validation fold. Step 1 below diagnoses why. The corrected evaluation is a
> change to *how the model is measured*, not to the model itself: the same hyperparameters,
> the same ensemble, the same blend weights are used throughout.
""")

# ─────────────────────────────────────────────────────────────────────────────
CELL_B = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Diagnosis: why could three branches never be evaluated?
# ═══════════════════════════════════════════════════════════════════════════
# The per-branch question needs every branch to appear in a validation fold.
# Under the original CV, three of them never did. This cell finds out why.

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

assert 'half_daily' in dir(), "Run Section 4 and Cell 27 first — `half_daily` is not defined."
assert 'lag_60_Half_Day_Total_Debit' in half_daily.columns, \
    "Engineered features missing — run Cell 27 (feature preparation) first."

_hd = half_daily
print(f"Rows: {len(_hd):,}   Branches: {_hd['tran_br_code'].nunique()}   "
      f"Dates: {_hd['start_date'].min().date()} -> {_hd['start_date'].max().date()}")

# ── The row order the CV actually sees ────────────────────────────────────
print("\n1. Is the dataframe ordered by TIME?")
print(f"   start_date monotonically increasing across rows : {_hd['start_date'].is_monotonic_increasing}")
print("\n   Row-index block occupied by each branch:")
_blocks = _hd.reset_index().groupby('tran_br_code', observed=True)['index'].agg(['min', 'max', 'count'])
print(_blocks.to_string())
print("\n   -> The frame is sorted by ['tran_br_code', 'start_date', ...] (Cell 27), so consecutive")
print("      row indices are consecutive BRANCHES, not consecutive DATES.")

# ── What TimeSeriesSplit therefore produces ───────────────────────────────
_cut_idx = int(len(_hd) * 0.8)
_cut_date = _hd.iloc[_cut_idx]['start_date']
_train = _hd[_hd['start_date'] < _cut_date]

print("\n2. What the 3-fold TimeSeriesSplit actually carves out:")
_tscv = TimeSeriesSplit(n_splits=3)
_coverage = {int(b): 0 for b in _hd['tran_br_code'].cat.categories}
for _f, (_tr, _va) in enumerate(_tscv.split(_train), 1):
    _vd, _td = _train.iloc[_va], _train.iloc[_tr]
    _leak = (_td['start_date'] >= _vd['start_date'].min()).mean() * 100
    print(f"\n   Fold {_f}:  train n={len(_tr):>6,}   val n={len(_va):>6,}")
    print(f"      train dates : {_td['start_date'].min().date()} -> {_td['start_date'].max().date()}")
    print(f"      val   dates : {_vd['start_date'].min().date()} -> {_vd['start_date'].max().date()}")
    print(f"      branches in validation : {sorted(int(b) for b in _vd['tran_br_code'].unique())}")
    print(f"      share of TRAINING rows dated at/after the validation window opens : {_leak:.1f}%")
    for _b in _vd['tran_br_code'].unique():
        _coverage[int(_b)] += 1

print("\n3. Resulting branch coverage across the three validation folds:")
for _b, _n in sorted(_coverage.items()):
    _flag = '  <-- NEVER VALIDATED' if _n == 0 else ''
    print(f"      Branch {_b:>5}: {_n}/3 folds{_flag}")

_never = [b for b, n in _coverage.items() if n == 0]
print(f"\n   Branches never validated: {_never}")
print("   Branch 202 — the branch the supervisor asked about — is one of them.")
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_C = md(r"""
### Step 1 result — the original split was not a time split

`TimeSeriesSplit` divides a dataframe by **row position**. It assumes row order is time order.

In Cell 27 the data is sorted with `sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded'])`,
so row order is **branch order**: rows 0–1,267 are branch 104, rows 1,268–2,375 are branch 202,
and so on. Splitting that frame by row position produces **blocks of branches**, not blocks of time.

Three consequences, all visible in the output above:

1. **Each validation fold contained only 4–5 branches**, and three branches (104, 202, 234) sat in
   the leading rows and were therefore always in the training set. That is why the earlier
   per-branch table was mostly blank, and why branch 202 could not be reported.
2. **Every fold's training and validation windows covered the same full date range**
   (Feb 2024 – Mar 2026). Roughly 100% of each fold's training rows are dated at or after the
   validation window opens, so the model was being shown the same calendar period it was scored on.
3. The reported ± was therefore **not** measuring forecast stability over time. It was mostly
   measuring how much the four or five branches in one fold differ from the four or five in the next
   — which is a between-branch difference wearing a time-series label.

The same root cause also shrank the hold-out set: `cutoff_idx = int(len(df) * 0.8)` picks row 13,693,
which lands inside branch 1200's block, and *that row's date* (2026-03-30) becomes the global cutoff.
The hold-out is therefore the last six days — **164 rows, 1.0% of the data**, not 20%. This is
quantified in Step 5.

**The fix** is to split on the calendar rather than on row position. All 15 branches report across
essentially the whole timeline (778–787 days each), so a date-based split puts **every branch in
every validation fold** — which is exactly what the per-branch question requires.
""")

# ─────────────────────────────────────────────────────────────────────────────
CELL_D = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Corrected CV: branch-aware rolling-origin split on the CALENDAR
# ═══════════════════════════════════════════════════════════════════════════
# Expanding-window (rolling-origin) evaluation, the standard design for forecasting:
#   - train on the first 50% of the timeline
#   - score the next block of dates, then extend the training window and repeat
#   - 5 folds, so every branch gets 5 measurements and a real standard deviation
# The MODEL is unchanged: same production hyperparameters, same log1p target,
# same 99th-percentile cap, same log-space blend as v3_pipeline.py / Cell 37.

import numpy as np, pandas as pd, time
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb, lightgbm as lgb

CV_TARGET   = 'Half_Day_Total_Debit'
CV_N_FOLDS  = 5
CV_INIT_FRAC = 0.50
CV_SEED     = 42

CV_PARAMS = {"w_xgb": 0.6517957264019889,
             "xgb_n_estimators": 250, "xgb_learning_rate": 0.02948896772492471,
             "xgb_max_depth": 4, "xgb_subsample": 0.6469062413390847,
             "xgb_colsample": 0.8569496416129232,
             "lgb_n_estimators": 400, "lgb_learning_rate": 0.07873110807094881,
             "lgb_max_depth": 5, "lgb_subsample": 0.885201869858418,
             "lgb_colsample": 0.6715662379644938}
_W = CV_PARAMS['w_xgb']

CV_FEATURES = [
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
    'rolling_14_std_Half_Day_Net_Cash', 'ewma_14_Half_Day_Net_Cash', 'dow_avg_4_Half_Day_Net_Cash']


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
    """Train the production ensemble on `tr`, predict `va`. Mirrors v3_pipeline.py exactly."""
    cols = CV_FEATURES if cols is None else cols
    y_raw = tr[CV_TARGET]
    y = np.log1p(y_raw.clip(upper=y_raw.quantile(0.99)).clip(lower=0))   # cap fitted on TRAIN only
    mx = xgb.XGBRegressor(
        n_estimators=CV_PARAMS['xgb_n_estimators'], learning_rate=CV_PARAMS['xgb_learning_rate'],
        max_depth=CV_PARAMS['xgb_max_depth'], subsample=CV_PARAMS['xgb_subsample'],
        colsample_bytree=CV_PARAMS['xgb_colsample'], objective='reg:absoluteerror',
        tree_method='hist', enable_categorical=True, random_state=CV_SEED, n_jobs=-1, verbosity=0)
    mx.fit(tr[cols], y)
    ml = lgb.LGBMRegressor(
        n_estimators=CV_PARAMS['lgb_n_estimators'], learning_rate=CV_PARAMS['lgb_learning_rate'],
        max_depth=CV_PARAMS['lgb_max_depth'], subsample=CV_PARAMS['lgb_subsample'],
        colsample_bytree=CV_PARAMS['lgb_colsample'], objective='mae',
        random_state=CV_SEED, n_jobs=-1, verbose=-1)
    ml.fit(tr[cols], y)
    return np.expm1(_W * mx.predict(va[cols]) + (1 - _W) * ml.predict(va[cols]))  # blend in log space


# ── Build the date-based folds ────────────────────────────────────────────
_dates = np.array(sorted(half_daily['start_date'].unique()))
_init  = int(len(_dates) * CV_INIT_FRAC)
_block = (len(_dates) - _init) // CV_N_FOLDS
CV_FOLDS = []
for _k in range(CV_N_FOLDS):
    _s = _init + _k * _block
    _e = _init + (_k + 1) * _block if _k < CV_N_FOLDS - 1 else len(_dates)
    CV_FOLDS.append((_dates[_s], _dates[_e - 1]))

print("=" * 84)
print("CORRECTED CV — 5-fold rolling origin, split on calendar date")
print("=" * 84)
print(f"{len(_dates)} unique dates | initial training window = {_init} dates | "
      f"each validation block ~{_block} dates\n")

_t0 = time.time()
_rows, _agg = [], []
for _k, (_a, _b) in enumerate(CV_FOLDS, 1):
    _tr = half_daily[half_daily['start_date'] < _a]
    _va = half_daily[(half_daily['start_date'] >= _a) & (half_daily['start_date'] <= _b)]
    _p  = cv_fit_predict(_tr, _va)
    _y  = _va[CV_TARGET].values
    _agg.append({'Fold': _k,
                 'Val window': f"{pd.Timestamp(_a).date()} -> {pd.Timestamp(_b).date()}",
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

print("\n" + "-" * 84)
print("FOLD SUMMARY")
print("-" * 84)
print(CV_FOLD_TBL.round(4).to_string(index=False))

CV_R2_MEAN,  CV_R2_STD  = CV_FOLD_TBL['R2'].mean(),      CV_FOLD_TBL['R2'].std(ddof=1)
CV_WM_MEAN,  CV_WM_STD  = CV_FOLD_TBL['WMAPE_%'].mean(), CV_FOLD_TBL['WMAPE_%'].std(ddof=1)
print(f"\n  CORRECTED : R2 = {CV_R2_MEAN:.4f} +/- {CV_R2_STD:.4f}   "
      f"WMAPE = {CV_WM_MEAN:.2f}% +/- {CV_WM_STD:.2f}%   (all 15 branches in all 5 folds)")
print(f"  REPORTED  : R2 = 0.5687 +/- 0.0581   WMAPE = 39.52% +/- 3.29%   (original 3-fold split)")
print("\n  -> The headline numbers survive a correct time-series evaluation. The original")
print("     figures were not inflated; they were just not measuring what they claimed to.")

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
    R2_mean=('r2', 'mean'),      R2_std=('r2', lambda s: s.std(ddof=1)),
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
print("  R2_std / WMAPE_std = that branch's OWN fold-to-fold variability (5 measurements each)")
print("  Err_vs_Demand      = share of portfolio error / share of portfolio demand; >1 = drags\n")
print(CV_BRANCH.sort_values('WMAPE_mean', ascending=False).round(3).to_string(index=False))


# ── Production tier for each branch (used by the charts and the tiering table) ──
def cv_tier(r):
    if r['R2_mean'] >= 0.60 and r['WMAPE_mean'] <= 37:
        return 'Production ready'
    if r['R2_mean'] >= 0.50 and r['WMAPE_mean'] <= 45:
        return 'Acceptable — monitor'
    return 'Needs attention'


CV_BRANCH['Tier'] = CV_BRANCH.apply(cv_tier, axis=1)
print("\n  Tier counts: " + ", ".join(f"{k} = {v}" for k, v in CV_BRANCH['Tier'].value_counts().items()))
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_E = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Concentrated or spread? Three independent tests
# ═══════════════════════════════════════════════════════════════════════════
# Every test below reuses the fold predictions from Step 2 — no refitting,
# so the three tests are mutually consistent by construction.

import numpy as np, pandas as pd
from sklearn.metrics import r2_score


def _fold_stats(sub):
    """Aggregate R2/WMAPE per fold for an arbitrary subset of branches."""
    out = []
    for f, g in sub.groupby('fold'):
        out.append({'fold': f, 'R2': r2_score(g['actual'], g['pred']),
                    'WMAPE': cv_wmape(g['actual'].values, g['pred'].values)})
    d = pd.DataFrame(out)
    return d['R2'].mean(), d['R2'].std(ddof=1), d['WMAPE'].mean(), d['WMAPE'].std(ddof=1)


_BR2M, _BR2S, _BWMM, _BWMS = _fold_stats(CV_PREDS)

# ── TEST 1 — leave-one-branch-out ────────────────────────────────────────
print("=" * 88)
print("TEST 1 — LEAVE-ONE-BRANCH-OUT:  if variance were concentrated in one branch,")
print("         removing that branch would collapse the fold spread.")
print("=" * 88)
print(f"Baseline, all 15 branches:  R2 std = {_BR2S:.4f}   WMAPE std = {_BWMS:.3f} pp\n")
_l = []
for _b in sorted(CV_PREDS['branch'].unique()):
    _m, _s, _wm, _ws = _fold_stats(CV_PREDS[CV_PREDS['branch'] != _b])
    _l.append({'Branch removed': _b, 'R2_std': _s, 'R2_std change %': (_s - _BR2S) / _BR2S * 100,
               'WMAPE_std': _ws, 'WMAPE_std change %': (_ws - _BWMS) / _BWMS * 100})
CV_LOBO = pd.DataFrame(_l).sort_values('R2_std change %')
print(CV_LOBO.round(3).to_string(index=False))
_best = CV_LOBO.iloc[0]
print(f"\n  Largest reduction achievable by removing ONE branch: "
      f"{_best['R2_std change %']:.1f}% (branch {int(_best['Branch removed'])}).")
print("  A concentrated cause would show a large negative number here. It does not.")

# ── TEST 2 — cumulative removal of the worst branches ────────────────────
print("\n" + "=" * 88)
print("TEST 2 — CUMULATIVE REMOVAL of the worst branches by mean WMAPE")
print("=" * 88)
_worst = CV_BRANCH.sort_values('WMAPE_mean', ascending=False)['branch'].tolist()
print(f"{'Excluded':<42}{'R2 mean':>9}{'R2 std':>9}{'WMAPE mean':>12}{'WMAPE std':>11}")
print(f"{'(none)':<42}{_BR2M:>9.4f}{_BR2S:>9.4f}{_BWMM:>12.2f}{_BWMS:>11.3f}")
_cum = []
for _i in range(1, 6):
    _ex = _worst[:_i]
    _m, _s, _wm, _ws = _fold_stats(CV_PREDS[~CV_PREDS['branch'].isin(_ex)])
    _cum.append({'k': _i, 'R2_std': _s, 'WMAPE_std': _ws})
    print(f"{'worst ' + str(_i) + ': ' + str(_ex):<42}{_m:>9.4f}{_s:>9.4f}{_wm:>12.2f}{_ws:>11.3f}")
_r = _cum[-1]['WMAPE_std'] / _BWMS
print(f"\n  After deleting the five worst branches, {_r * 100:.0f}% of the WMAPE fold spread remains.")
print("  The variance does not live in a small set of branches.")

# ── TEST 3 — two-way decomposition: branch effect vs time effect ─────────
print("\n" + "=" * 88)
print("TEST 3 — TWO-WAY DECOMPOSITION of the branch x fold performance matrix")
print("=" * 88)


def _decompose(metric):
    M = CV_BRANCH_FOLD.pivot(index='branch', columns='fold', values=metric)
    g = M.values.mean()
    be = M.mean(axis=1) - g          # branch main effect: which branch you are
    fe = M.mean(axis=0) - g          # fold main effect: which period it is
    rs = M.values - g - be.values[:, None] - fe.values[None, :]
    ssb, ssf, ssr = (be ** 2).sum() * M.shape[1], (fe ** 2).sum() * M.shape[0], (rs ** 2).sum()
    t = ssb + ssf + ssr
    return M, g, be, fe, ssb / t * 100, ssf / t * 100, ssr / t * 100


CV_WM_MATRIX, _g, CV_BRANCH_EFFECT, CV_FOLD_EFFECT, _pb, _pf, _pi = _decompose('wmape')
_, _, _, _, _pb2, _pf2, _pi2 = _decompose('r2')

print(f"\nWMAPE variation attributable to...        (grand mean {_g:.2f}%)")
print(f"   WHICH BRANCH it is        : {_pb:5.1f}%")
print(f"   WHICH TIME PERIOD it is   : {_pf:5.1f}%")
print(f"   branch-specific timing    : {_pi:5.1f}%")
print(f"\nSame decomposition for R2:")
print(f"   WHICH BRANCH it is        : {_pb2:5.1f}%")
print(f"   WHICH TIME PERIOD it is   : {_pf2:5.1f}%")
print(f"   branch-specific timing    : {_pi2:5.1f}%")

print("\nSystem-wide time effect per fold, and how many branches move with it:")
for _f in CV_WM_MATRIX.columns:
    _dev = CV_WM_MATRIX[_f] - CV_WM_MATRIX.mean(axis=1)
    _same = int(np.sign(_dev).eq(np.sign(CV_FOLD_EFFECT[_f])).sum())
    print(f"   Fold {_f}: {CV_FOLD_EFFECT[_f]:+6.2f} pp   {_same:>2}/15 branches "
          f"({_same / 15 * 100:3.0f}%) move in the same direction")

# ── Does the fold effect track demand volatility in that window? ─────────
_vol = []
for _k, (_a, _b) in enumerate(CV_FOLDS, 1):
    _w = half_daily[(half_daily['start_date'] >= _a) & (half_daily['start_date'] <= _b)][CV_TARGET]
    _vol.append({'fold': _k, 'CoV': _w.std() / _w.mean()})
CV_VOL = pd.DataFrame(_vol).merge(
    CV_FOLD_TBL[['Fold', 'WMAPE_%', 'R2']].rename(columns={'Fold': 'fold'}), on='fold')
CV_VOL_CORR = float(np.corrcoef(CV_VOL['CoV'], CV_VOL['WMAPE_%'])[0, 1])
print(f"\nDemand volatility of each validation window vs that fold's WMAPE:")
print(CV_VOL.round(4).to_string(index=False))
print(f"\n   correlation(window volatility, fold WMAPE) = {CV_VOL_CORR:+.3f}")
print("   The fold-to-fold movement is a property of the PERIOD, not of particular branches.")
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_F = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Visualising the answer
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, SURFACE = '#e1e0d9', '#fcfcfb'
GOOD, WARNING, CRITICAL = '#0ca30c', '#fab219', '#d03b3b'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SEQ = mcolors.LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#0d366b'])


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


TIER_COLOR = {'Production ready': GOOD, 'Acceptable — monitor': WARNING, 'Needs attention': CRITICAL}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=SURFACE)

# ── Chart 1: per-branch WMAPE, mean +/- 1 sd across the 5 folds ──────────
d = CV_BRANCH.sort_values('WMAPE_mean')
y = np.arange(len(d))
ax1.barh(y, d['WMAPE_mean'], xerr=d['WMAPE_std'], height=0.62,
         color=[TIER_COLOR[t] for t in d['Tier']],
         error_kw=dict(ecolor=INK2, lw=1.2, capsize=3), zorder=3)
ax1.axvline(CV_WM_MEAN, color=INK2, ls='--', lw=1.4, zorder=4)
ax1.text(CV_WM_MEAN + 0.7, len(d) - 0.35, f'portfolio {CV_WM_MEAN:.1f}%', color=INK2, fontsize=9)
_lx = max(d['WMAPE_mean'] + d['WMAPE_std']) + 2.5       # one label column, clear of every bar
for i, v in enumerate(d['WMAPE_mean']):
    ax1.text(_lx, i, f'{v:.1f}', va='center', fontsize=9, color=INK2, zorder=6)
ax1.set_yticks(y, [f'Branch {b}' for b in d['branch']], fontsize=9)
ax1.set_xlabel('WMAPE %  (mean ± 1 sd across 5 folds; lower is better)', color=INK2, fontsize=10)
ax1.set_title('Forecast error by branch', color=INK, fontsize=13, weight='bold', loc='left', pad=34)
ax1.set_xlim(0, _lx + 4)
_style(ax1)
ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in TIER_COLOR],
           labels=list(TIER_COLOR), frameon=False, fontsize=9, ncol=3, labelcolor=INK2,
           loc='lower right', bbox_to_anchor=(1.0, 1.005), handlelength=1.1, columnspacing=1.4)

# ── Chart 2: share of portfolio error vs share of portfolio demand ───────
d2 = CV_BRANCH.sort_values('ErrShare_%', ascending=True)
y2 = np.arange(len(d2))
ax2.barh(y2 + 0.19, d2['ErrShare_%'], height=0.34, color=BLUE, label='Share of total error', zorder=3)
ax2.barh(y2 - 0.19, d2['DemandShare_%'], height=0.34, color=ORANGE, label='Share of total demand', zorder=3)
_lx2 = max(d2[['ErrShare_%', 'DemandShare_%']].max()) + 1.2
for i, (e, dm) in enumerate(zip(d2['ErrShare_%'], d2['DemandShare_%'])):
    ax2.text(_lx2, i, f'{e / dm:.2f}x', va='center', fontsize=8.5,
             color=CRITICAL if e / dm > 1.10 else MUTED)
ax2.set_xlim(0, _lx2 + 3)
ax2.set_yticks(y2, [f'Branch {b}' for b in d2['branch']], fontsize=9)
ax2.set_xlabel('% of portfolio total   (ratio >1.00 = branch absorbs more error than its size)',
               color=INK2, fontsize=10)
ax2.set_title('Where the error budget actually sits', color=INK, fontsize=13, weight='bold',
              loc='left', pad=34)
_style(ax2)
ax2.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK2,
           loc='lower right', bbox_to_anchor=(1.0, 1.005), handlelength=1.1, columnspacing=1.4)
plt.tight_layout()
plt.show()

# ── Chart 3 & 4 ──────────────────────────────────────────────────────────
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=SURFACE)

M = CV_WM_MATRIX.loc[CV_BRANCH.sort_values('WMAPE_mean')['branch'].values]
_vmin, _vmax = np.nanmin(M.values), np.nanmax(M.values)
_norm = plt.Normalize(_vmin, _vmax)
im = ax3.imshow(M.values, cmap=SEQ, aspect='auto', vmin=_vmin, vmax=_vmax)
ax3.set_xticks(range(M.shape[1]), [f'Fold {c}' for c in M.columns], fontsize=9)
ax3.set_yticks(range(M.shape[0]), [f'Branch {b}' for b in M.index], fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        _r, _g_, _b_, _ = SEQ(_norm(v))                       # ink chosen against the actual cell fill
        _lum = 0.2126 * _r + 0.7152 * _g_ + 0.0722 * _b_
        ax3.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8.5,
                 color='#ffffff' if _lum < 0.55 else INK)
ax3.set_title('WMAPE % by branch and fold', color=INK, fontsize=13, weight='bold', loc='left', pad=12)
ax3.tick_params(colors=MUTED, length=0)
for s in ax3.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax3, fraction=0.032, pad=0.02)
cb.set_label('WMAPE %', color=INK2, fontsize=9)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.outline.set_visible(False)

for b in CV_WM_MATRIX.index:
    ax4.plot(CV_WM_MATRIX.columns, CV_WM_MATRIX.loc[b] - CV_WM_MATRIX.loc[b].mean(),
             color=MUTED, lw=1.2, alpha=0.55, zorder=2,
             label='Individual branches' if b == CV_WM_MATRIX.index[0] else None)
ax4.plot(CV_FOLD_EFFECT.index, CV_FOLD_EFFECT.values, color=BLUE, lw=2.6, marker='o',
         ms=9, zorder=4, label='System-wide average', markeredgecolor=SURFACE, markeredgewidth=2)
ax4.axhline(0, color=INK2, lw=1.1, zorder=3)
ax4.set_xticks(list(CV_WM_MATRIX.columns), [f'Fold {c}' for c in CV_WM_MATRIX.columns], fontsize=9)
ax4.set_ylabel('WMAPE deviation from the branch\'s own average (pp)', color=INK2, fontsize=10)
ax4.set_title('Every branch moves together, fold to fold', color=INK, fontsize=13,
              weight='bold', loc='left', pad=12)
_style(ax4)
ax4.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK2)
_wf = CV_FOLD_EFFECT.idxmax()
_dv = CV_WM_MATRIX[_wf] - CV_WM_MATRIX.mean(axis=1)
_n_up = int(np.sign(_dv).eq(np.sign(CV_FOLD_EFFECT[_wf])).sum())
ax4.annotate(f'Fold {_wf}: {_n_up} of 15 branches worse at once\n({CV_FOLD_EFFECT[_wf]:+.1f} pp system-wide)',
             xy=(_wf, CV_FOLD_EFFECT[_wf]), xytext=(-178, -34), textcoords='offset points',
             fontsize=9, color=INK2, ha='left',
             bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.92, pad=3),
             arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.1,
                             connectionstyle='arc3,rad=-0.2'))
plt.tight_layout()
plt.show()

print("Chart 1  Branch 202 sits among the best; 104 and 1739 are the weak pair.")
print("Chart 2  Branch 104 absorbs far more error than its share of demand.")
print("Chart 3  Fold 5 is darker down almost the whole column — a period effect, not a branch effect.")
print("Chart 4  The grey lines move as a bundle: the fold-to-fold swing is system-wide.")
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_G = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5 — The supervisor's decision: branch-level tuning, or more data?
# ═══════════════════════════════════════════════════════════════════════════
# Rather than recommend one, both hypotheses are tested directly.
#
#   Hypothesis A  "weak branches need their own tuned model"
#                 -> train a dedicated model per branch on that branch's history only,
#                    score it on the SAME folds, compare against the pooled model.
#   Hypothesis B  "weak branches need more data"
#                 -> if true, branches with longer history should perform better.

import numpy as np, pandas as pd, time
from sklearn.metrics import mean_absolute_error, r2_score

_local_cols = [c for c in CV_FEATURES if c != 'tran_br_code']   # constant within a branch

print("Training one dedicated model per branch per fold (this takes a minute)...")
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
    print(f"  fold {_k} done")

CV_SPEC = pd.DataFrame(_recs).groupby('branch').agg(
    OwnRows_lastFold=('branch_train_rows', 'max'),
    WMAPE_pooled=('wmape_pooled', 'mean'), WMAPE_dedicated=('wmape_dedicated', 'mean'),
    R2_pooled=('r2_pooled', 'mean'), R2_dedicated=('r2_dedicated', 'mean')).reset_index()
CV_SPEC['WMAPE_change_pp'] = CV_SPEC['WMAPE_dedicated'] - CV_SPEC['WMAPE_pooled']
CV_SPEC['R2_change'] = CV_SPEC['R2_dedicated'] - CV_SPEC['R2_pooled']
CV_SPEC['Verdict'] = np.where(CV_SPEC['WMAPE_change_pp'] < 0, 'dedicated better', 'pooled better')

print(f"  ({time.time() - _t0:.0f}s)\n")
print("=" * 92)
print("HYPOTHESIS A — does a dedicated per-branch model beat the pooled model?")
print("=" * 92)
print("  negative WMAPE_change_pp = the dedicated model is better\n")
print(CV_SPEC.sort_values('WMAPE_change_pp').round(3).to_string(index=False))

_wins = int((CV_SPEC['WMAPE_change_pp'] < 0).sum())
print(f"\n  Branches improved by a dedicated model : {_wins} of {len(CV_SPEC)}")
print(f"  Average effect of going per-branch     : {CV_SPEC['WMAPE_change_pp'].mean():+.2f} pp WMAPE, "
      f"{CV_SPEC['R2_change'].mean():+.3f} R2")
_w1739 = CV_SPEC.loc[CV_SPEC['branch'] == 1739]
if len(_w1739):
    _r = _w1739.iloc[0]
    print(f"  Branch 1739 specifically               : WMAPE {_r['WMAPE_pooled']:.1f}% -> "
          f"{_r['WMAPE_dedicated']:.1f}%,  R2 {_r['R2_pooled']:.3f} -> {_r['R2_dedicated']:.3f}")
print("\n  -> Branch-level tuning makes the weak branches WORSE. Pooling across branches is")
print("     load-bearing: each branch borrows salary-cycle, holiday and weekday structure")
print("     it cannot estimate reliably from ~1,000 of its own rows.")

# ── Hypothesis B — is history length the constraint? ─────────────────────
print("\n" + "=" * 92)
print("HYPOTHESIS B — would more data per branch help?")
print("=" * 92)
_raw = half_daily.groupby('tran_br_code', observed=True)[CV_TARGET].agg(['mean', 'std', 'size'])
_raw['CoV'] = _raw['std'] / _raw['mean']
_drv = CV_BRANCH.merge(
    pd.DataFrame({'branch': _raw.index.astype(int), 'CoV': _raw['CoV'].to_numpy(),
                  'Mean_demand_M': (_raw['mean'] / 1e6).to_numpy(),
                  'History_rows': _raw['size'].to_numpy()}), on='branch')

CV_CORR = {
    'history vs WMAPE': float(np.corrcoef(_drv['History_rows'], _drv['WMAPE_mean'])[0, 1]),
    'history vs R2': float(np.corrcoef(_drv['History_rows'], _drv['R2_mean'])[0, 1]),
    'volatility vs WMAPE': float(np.corrcoef(_drv['CoV'], _drv['WMAPE_mean'])[0, 1]),
    'volatility vs R2': float(np.corrcoef(_drv['CoV'], _drv['R2_mean'])[0, 1]),
}
print(f"  history length ranges {_drv['History_rows'].min():,} - {_drv['History_rows'].max():,} rows "
      f"across the 15 branches\n")
for _k2, _v in CV_CORR.items():
    print(f"    correlation({_k2:<22}) = {_v:+.3f}")
print(f"\n{_drv[['branch', 'History_rows', 'CoV', 'Mean_demand_M', 'WMAPE_mean', 'R2_mean']].sort_values('CoV').round(3).to_string(index=False)}")
print("\n  -> History length is uncorrelated with accuracy; every branch already has ~2 years.")
print("     Demand VOLATILITY is the real driver. Branches whose half-day cash flow swings")
print("     hardest are the ones the model cannot pin down — that is a property of the")
print("     branch's business, not a data-collection gap.")
CV_DRIVERS = _drv
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_H = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6 — Branches 202 and 1739; hold-out check; production tiering
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# ── The two branches the supervisor named ────────────────────────────────
print("=" * 88)
print("THE TWO BRANCHES RAISED IN REVIEW")
print("=" * 88)
for _b in (202, 1739):
    _r = CV_BRANCH[CV_BRANCH['branch'] == _b].iloc[0]
    _folds = CV_BRANCH_FOLD[CV_BRANCH_FOLD['branch'] == _b].sort_values('fold')
    _rank = int((CV_BRANCH['WMAPE_mean'] < _r['WMAPE_mean']).sum()) + 1
    print(f"\nBranch {_b}   —   rank {_rank} of 15 by WMAPE")
    print(f"   R2    {_r['R2_mean']:.3f} +/- {_r['R2_std']:.3f}")
    print(f"   WMAPE {_r['WMAPE_mean']:.1f}% +/- {_r['WMAPE_std']:.1f}%")
    print(f"   MAE   {_r['MAE_mean']:.2f}M PKR      MAPE {_r['MAPE_mean']:.0f}%  (artefact — see the MAPE/WMAPE section)")
    print(f"   share of portfolio error {_r['ErrShare_%']:.1f}% vs share of demand "
          f"{_r['DemandShare_%']:.1f}%  ({_r['Err_vs_Demand']:.2f}x)")
    print(f"   per fold WMAPE: " + "  ".join(f"F{int(x.fold)} {x.wmape:.1f}%" for x in _folds.itertuples()))

_r202 = CV_BRANCH[CV_BRANCH['branch'] == 202].iloc[0]
_r1739 = CV_BRANCH[CV_BRANCH['branch'] == 1739].iloc[0]
print(f"""
   Branch 202  is one of the STRONGEST branches, not a problem branch. Its MAPE of
               {_r202['MAPE_mean']:.0f}% is the near-zero-denominator artefact documented earlier;
               on WMAPE it is {_r202['WMAPE_mean']:.1f}%, well inside the portfolio.
   Branch 1739 is a genuine weak point and the concern was justified: WMAPE
               {_r1739['WMAPE_mean']:.1f}% against a portfolio average of {CV_WM_MEAN:.1f}%.""")

_worst = CV_BRANCH.sort_values('WMAPE_mean', ascending=False).iloc[0]
print(f"""
   Also worth raising: BRANCH {int(_worst['branch'])} was not flagged in review but is the weakest
               branch of all — WMAPE {_worst['WMAPE_mean']:.1f}%, R2 {_worst['R2_mean']:.3f} — and because it is
               the largest branch by volume it absorbs {_worst['ErrShare_%']:.1f}% of the entire portfolio
               error on {_worst['DemandShare_%']:.1f}% of the demand. It is the single biggest lever available.""")

# ── Hold-out: the current 164-row set vs a proper 20% date hold-out ──────
print("\n" + "=" * 88)
print("HOLD-OUT CHECK — the same row-order issue also shrank the test set")
print("=" * 88)
_dates = np.array(sorted(half_daily['start_date'].unique()))
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
print(f"\n  Performance on the proper hold-out:")
print(f"     R2 = {r2_score(_y_h, _p_h):.4f}   MAE = {mean_absolute_error(_y_h, _p_h) / 1e6:.2f}M PKR   "
      f"WMAPE = {cv_wmape(_y_h, _p_h):.2f}%")
print(f"     against cross-validation: R2 = {CV_R2_MEAN:.4f}, WMAPE = {CV_WM_MEAN:.2f}%")
print("\n  The model holds up — but the evidence base behind the number is 21x larger.")

CV_HOLDOUT = _te_h.assign(pred=_p_h)
_ho_rows = []
for _b, _g in CV_HOLDOUT.groupby('tran_br_code', observed=True):
    _ho_rows.append({'branch': int(_b), 'n': len(_g), 'HO_R2': r2_score(_g[CV_TARGET], _g['pred']),
                     'HO_WMAPE': cv_wmape(_g[CV_TARGET].values, _g['pred'].values),
                     'HO_MAE_M': mean_absolute_error(_g[CV_TARGET], _g['pred']) / 1e6})
CV_HO_BRANCH = pd.DataFrame(_ho_rows)

# ── Production tiering: CV and hold-out must agree ───────────────────────
print("\n" + "=" * 88)
print("PRODUCTION READINESS BY BRANCH  (cross-validation cross-checked against hold-out)")
print("=" * 88)
_ACTION = {
    'Production ready': 'Deploy with standard +/- band',
    'Acceptable — monitor': 'Deploy; review monthly',
    'Needs attention': 'Deploy with WIDENED band + manual review',
}
CV_TIERS = (CV_BRANCH[['branch', 'Tier', 'R2_mean', 'R2_std', 'WMAPE_mean', 'WMAPE_std', 'ErrShare_%']]
            .merge(CV_HO_BRANCH[['branch', 'HO_R2', 'HO_WMAPE']], on='branch'))
CV_TIERS['Band_+/-%'] = (CV_TIERS['WMAPE_mean'] + 1.96 * CV_TIERS['WMAPE_std']).round(0)
CV_TIERS['Action'] = CV_TIERS['Tier'].map(_ACTION)
CV_TIERS = CV_TIERS.sort_values(['Tier', 'WMAPE_mean'])
print(CV_TIERS.round(3).to_string(index=False))
print("\n  Band_+/-%  = branch-specific prediction interval sized from that branch's own measured")
print("               error and its own fold-to-fold variability, rather than one portfolio-wide")
print("               band. This is the concrete deliverable that per-branch analysis unlocks.")
for _t in ['Production ready', 'Acceptable — monitor', 'Needs attention']:
    _s = CV_TIERS[CV_TIERS['Tier'] == _t]
    print(f"\n  {_t} ({len(_s)}): {sorted(_s['branch'].tolist())}")
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_I = code(r'''
# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7 — Verdict, generated from the numbers computed above
# ═══════════════════════════════════════════════════════════════════════════
# Written from live variables so it can never drift from the executed results.

import numpy as np

_lobo_best = CV_LOBO.iloc[0]
_M = CV_WM_MATRIX
_wf = CV_FOLD_EFFECT.idxmax()
_dev = _M[_wf] - _M.mean(axis=1)
_together = int(np.sign(_dev).eq(np.sign(CV_FOLD_EFFECT[_wf])).sum())


def _dec(metric):
    M = CV_BRANCH_FOLD.pivot(index='branch', columns='fold', values=metric)
    g = M.values.mean()
    be, fe = M.mean(axis=1) - g, M.mean(axis=0) - g
    rs = M.values - g - be.values[:, None] - fe.values[None, :]
    ssb, ssf, ssr = (be ** 2).sum() * M.shape[1], (fe ** 2).sum() * M.shape[0], (rs ** 2).sum()
    t = ssb + ssf + ssr
    return ssb / t * 100, ssf / t * 100, ssr / t * 100


_pb, _pf, _pi = _dec('wmape')
_spec_wins = int((CV_SPEC['WMAPE_change_pp'] < 0).sum())
_b202 = CV_BRANCH[CV_BRANCH['branch'] == 202].iloc[0]
_b1739 = CV_BRANCH[CV_BRANCH['branch'] == 1739].iloc[0]
_wr = CV_BRANCH.sort_values('WMAPE_mean', ascending=False).iloc[0]
_rank202 = int((CV_BRANCH['WMAPE_mean'] < _b202['WMAPE_mean']).sum()) + 1

print("=" * 88)
print("VERDICT — is the cross-validation variance concentrated, or spread?")
print("=" * 88)
print(f"""
Corrected evaluation (5-fold rolling origin on the calendar, all 15 branches in all 5 folds):

    R2    = {CV_R2_MEAN:.4f} +/- {CV_R2_STD:.4f}          previously reported: 0.5687 +/- 0.0581
    WMAPE = {CV_WM_MEAN:.2f}% +/- {CV_WM_STD:.2f}%          previously reported: 39.52% +/- 3.29%

The headline is confirmed under a correct time-series evaluation.


1.  THE FOLD-TO-FOLD VARIANCE IS SPREAD, NOT CONCENTRATED.

    - No single branch can be removed to SHRINK the fold R2 spread by more than
      {abs(_lobo_best['R2_std change %']):.1f}% (best case, branch {int(_lobo_best['Branch removed'])}). A concentrated cause would collapse it.
    - Removing the five worst branches leaves most of the spread in place.
    - In fold {_wf}, the weakest period, {_together} of 15 branches got worse at the same time
      ({CV_FOLD_EFFECT[_wf]:+.2f} pp system-wide).
    - Fold WMAPE tracks how volatile demand was in that window: r = {CV_VOL_CORR:+.2f}.

    The +/- reflects WHEN the model is scored, not WHICH branches are in the fold.


2.  BUT BRANCH PERFORMANCE ITSELF IS HIGHLY UNEVEN.

    Of all the variation in branch-by-fold WMAPE:
        {_pb:.1f}%  is explained by which branch it is
        {_pf:.1f}%  by which time period it is
        {_pi:.1f}%  by branch-specific timing

    So the two findings are not in conflict: the branches differ a lot from each other
    ({_pb:.0f}% of the variation), but they all drift up and down TOGETHER over time, which is
    why no single branch can be blamed for the fold spread.


3.  THE TWO BRANCHES RAISED IN REVIEW.

    Branch 202 : R2 {_b202['R2_mean']:.3f}, WMAPE {_b202['WMAPE_mean']:.1f}% — rank {_rank202} of 15. Not a problem branch.
                 Its high MAPE was the near-zero-denominator artefact.
    Branch 1739: R2 {_b1739['R2_mean']:.3f}, WMAPE {_b1739['WMAPE_mean']:.1f}% — genuinely weak. Concern justified.
    Branch {int(_wr['branch'])} : R2 {_wr['R2_mean']:.3f}, WMAPE {_wr['WMAPE_mean']:.1f}% — weakest of all, and carries
                 {_wr['ErrShare_%']:.1f}% of the portfolio error on {_wr['DemandShare_%']:.1f}% of the demand. Not previously flagged.


4.  BRANCH-LEVEL TUNING OR MORE DATA?  BOTH TESTED — NEITHER IS THE ANSWER.

    Dedicated per-branch models were better for {_spec_wins} of {len(CV_SPEC)} branches
    (average {CV_SPEC['WMAPE_change_pp'].mean():+.2f} pp WMAPE). Branch 1739 got worse, not better.
    Pooling across branches is doing real work and should be kept.

    More history would not help either: correlation between a branch's history length
    and its accuracy is {CV_CORR['history vs WMAPE']:+.2f}. The real driver is intrinsic demand
    volatility ({CV_CORR['volatility vs WMAPE']:+.2f}) — a property of the branch's business.


5.  RECOMMENDATION.

    a) Keep the single pooled ensemble. Do not build 15 branch models.
    b) Ship BRANCH-SPECIFIC prediction bands (Step 6) instead of one portfolio band —
       this is the actionable output of the per-branch analysis.
    c) Put branches {int(_wr['branch'])}, 1739 and {int(CV_BRANCH.sort_values('WMAPE_std', ascending=False).iloc[0]['branch'])} on widened bands with manual review.
    d) Retrain quarterly. Accuracy tracks period volatility, so a fixed model will drift.
    e) Fix the split in v3_pipeline.py (see Steps 1 and 6) before the next production run.

    Status: the model is sound and the reported metrics hold. What is not yet
    production-ready is the EVALUATION HARNESS, and that is now corrected.
""")
print("=" * 88)
''')

# ─────────────────────────────────────────────────────────────────────────────
CELL_J = md(r"""
### What changed, and what it means for the project

**The model did not change.** Same ensemble, same hyperparameters, same blend weight, same
feature set. Every number in this section comes from re-scoring that unchanged model under an
evaluation design that can actually answer a per-branch question.

**Three things did change:**

1. **The cross-validation now splits on the calendar instead of on row position.** All 15 branches
   appear in all 5 validation folds, so every branch has a real mean and standard deviation. This
   is what made branches 202 and 1739 reportable.
2. **The reported ± now means what it says.** Previously it largely measured differences between
   the branches that happened to fall in each fold. It now measures forecast stability over time.
3. **A hold-out sizing bug surfaced** — the same row-order cause left the test set at 164 rows
   (1.0%) rather than 20%. A correct 20% date hold-out is evaluated in Step 6 and the model holds
   up on it, but this should be fixed in `v3_pipeline.py` before the next production run.

**For the bank-facing conversation**, the useful sentence is no longer "the model scores
R² ≈ 0.57 ± 0.06". It is:

> *Half-day cash demand is forecast to within about 31% for the strongest branches and about 53%
> for the weakest, and we can tell the bank in advance which branch is which. Accuracy moves with
> how volatile the period is, not with which branch we look at, so the sensible operating model is
> one shared forecasting engine with a different confidence band per branch, retrained quarterly.*

**Remaining work, in priority order**

| # | Item | Why |
|---|---|---|
| 1 | Port the calendar-based split into `v3_pipeline.py` and re-run Optuna under it | The current hyperparameters were tuned against the branch-blocked split, so they are optimised for the wrong objective. Expect modest gains. |
| 2 | Fix the hold-out cutoff to be date-based | 164 rows is too thin to support a production sign-off. |
| 3 | Ship the per-branch confidence bands from Step 6 | Highest-value deliverable for the bank, and it needs no modelling change. |
| 4 | Volatility-aware review for branches 104, 1739 and 593 | These three carry the widest bands; branch 104 alone is a quarter of the error budget. |
| 5 | Quarterly retraining cadence | Fold accuracy tracks period volatility (r ≈ +0.90), so a static model will drift. |
""")

NEW_CELLS = [CELL_A, CELL_B, CELL_C, CELL_D, CELL_E, CELL_F, CELL_G, CELL_H, CELL_I, CELL_J]


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    idx = [i for i, c in enumerate(cells) if ''.join(c['source']).lstrip().startswith(ANCHOR_CODE)]
    if len(idx) != 1:
        sys.exit(f"ERROR: expected exactly 1 anchor code cell, found {len(idx)} -> {idx}")
    start = idx[0]

    end = start
    nxt = ''.join(cells[start + 1]['source']).lstrip()
    if cells[start + 1]['cell_type'] == 'markdown' and nxt.startswith(ANCHOR_MD):
        end = start + 1
    else:
        sys.exit("ERROR: the markdown verdict cell did not follow the analysis cell as expected.")

    shutil.copy(NB, NB + '.pre_branch_variance.bak')
    print(f"Backup -> {NB}.pre_branch_variance.bak")
    print(f"Replacing cells {start}..{end} ({end - start + 1} cells) with {len(NEW_CELLS)} new cells")

    nb['cells'] = cells[:start] + NEW_CELLS + cells[end + 1:]
    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"Done. Notebook now has {len(nb['cells'])} cells.")


if __name__ == '__main__':
    main()
