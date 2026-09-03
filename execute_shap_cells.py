"""
Re-execute the Section 8 (SHAP) cells and bake their outputs back into the notebook,
after 8.1 was rewritten to explain the deployed XGB+LGB ensemble.

Rebuilds the notebook state up to Cell 37 (Debit models trained on the same split), then
runs 8.1-8.4 in order and stores their stdout and figures.

Run:  python execute_shap_cells.py
"""
import base64
import io
import json
import shutil
import sys
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb

from execute_branch_variance_cells import build_half_daily, Recorder

NB = 'Bank_Cash_Optimization_Workflow.ipynb'
SECTION_8_MARKERS = ['8.1 SHAP', '8.2 Summary plot', '8.3 SHAP Dependence', '8.4 Waterfall plot']

BEST = {"w_xgb": 0.6517957264019889,
        "xgb_n_estimators": 250, "xgb_learning_rate": 0.02948896772492471,
        "xgb_max_depth": 4, "xgb_subsample": 0.6469062413390847,
        "xgb_colsample": 0.8569496416129232,
        "lgb_n_estimators": 400, "lgb_learning_rate": 0.07873110807094881,
        "lgb_max_depth": 5, "lgb_subsample": 0.885201869858418,
        "lgb_colsample": 0.6715662379644938}


def build_state():
    """Reproduce the notebook namespace as it stands after Cell 37 (Debit branch)."""
    half_daily = build_half_daily()
    feature_cols = [
        'tran_br_code', 'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count',
        'ewma_14_Txn_Count', 'Days_to_Salary', 'Days_Since_Salary', 'Weekday', 'Is_Weekend',
        'Month', 'Day', 'Is_Salary_Day', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End',
        'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit',
        'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
        'rolling_14_std_Half_Day_Total_Debit', 'ewma_14_Half_Day_Total_Debit',
        'dow_avg_4_Half_Day_Total_Debit',
        'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit',
        'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
        'rolling_14_std_Half_Day_Total_Credit', 'ewma_14_Half_Day_Total_Credit',
        'dow_avg_4_Half_Day_Total_Credit',
        'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash',
        'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash',
        'rolling_14_std_Half_Day_Net_Cash', 'ewma_14_Half_Day_Net_Cash',
        'dow_avg_4_Half_Day_Net_Cash']

    cutoff_idx = int(len(half_daily) * 0.8)
    cutoff_date = half_daily.iloc[cutoff_idx]['start_date']
    train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()
    test_df = half_daily[half_daily['start_date'] >= cutoff_date].copy()
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    y_raw = train_df['Half_Day_Total_Debit']
    y_tgt = np.log1p(y_raw.clip(upper=y_raw.quantile(0.99)).clip(lower=0))

    # exactly as Cell 37 constructs them, including the `colsample` name that XGBoost ignores
    xgb_p = {k.replace('xgb_', ''): v for k, v in BEST.items() if k.startswith('xgb_')}
    xgb_p.update(enable_categorical=True, random_state=42)
    lgb_p = {k.replace('lgb_', ''): v for k, v in BEST.items() if k.startswith('lgb_')}
    lgb_p.update(random_state=42, verbose=-1)

    xgb_model = xgb.XGBRegressor(**xgb_p, objective='reg:absoluteerror')
    xgb_model.fit(X_train, y_tgt)
    lgb_model = lgb.LGBMRegressor(**lgb_p, objective='mae')
    lgb_model.fit(X_train, y_tgt)

    cv_params = {'Half_Day_Total_Debit': {'w_xgb': BEST['w_xgb'], 'best_params': BEST},
                 'Half_Day_Net_Cash': {'w_xgb': 0.20730122208202095}}

    return {
        'half_daily': half_daily, 'feature_cols': feature_cols,
        'train_df': train_df, 'test_df': test_df, 'X_train': X_train, 'X_test': X_test,
        'xgb_model': xgb_model, 'lgb_model': lgb_model, 'cv_params': cv_params,
        # Cell 37 leaves w_xgb holding the LAST target's weight (Net Cash)
        'w_xgb': cv_params['Half_Day_Net_Cash']['w_xgb'],
        'np': np, 'pd': pd, 'plt': plt, 'shap': shap, 'xgb': xgb, 'lgb': lgb,
        '__name__': '__main__',
    }


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    targets = []
    for marker in SECTION_8_MARKERS:
        hits = [i for i, c in enumerate(cells)
                if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
        if len(hits) != 1:
            sys.exit(f"ERROR: expected 1 cell for '{marker}', found {hits}")
        targets.append(hits[0])
    print(f"Section 8 cells: {targets}")

    ns = build_state()
    print(f"State rebuilt (X_test {ns['X_test'].shape}); executing...")

    rec = Recorder()
    plt.show = lambda *a, **k: rec.add_figure()
    counter = max([c.get('execution_count') or 0 for c in cells if c['cell_type'] == 'code']) + 1

    real = sys.stdout
    for i in targets:
        rec.events, rec.buf = [], []
        print(f"  cell {i} ...", file=real, end='', flush=True)
        sys.stdout = rec
        try:
            exec(compile(''.join(cells[i]['source']), f'<cell {i}>', 'exec'), ns)
        except Exception:
            sys.stdout = real
            import traceback
            traceback.print_exc()
            sys.exit(f"ERROR executing cell {i}")
        finally:
            sys.stdout = real
        outs = rec.done()
        cells[i]['outputs'] = outs
        cells[i]['execution_count'] = counter
        counter += 1
        nf = sum(1 for o in outs if o['output_type'] == 'display_data')
        nt = sum(len(o['text']) for o in outs if o['output_type'] == 'stream')
        print(f" ok ({nt} lines, {nf} figures)", file=real)

    shutil.copy(NB, NB + '.pre_shap_exec.bak')
    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"\nOutputs baked in. Backup: {NB}.pre_shap_exec.bak")


if __name__ == '__main__':
    main()
