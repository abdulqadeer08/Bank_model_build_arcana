"""
Execute the whole notebook in a real Jupyter kernel and write the outputs back in place.

Runs from a clean working directory containing no model_data/ and no models/, so the kernel
takes exactly the code paths a fresh Google Colab runtime takes: it downloads the dataset
from Drive, builds `half_daily` in Section 4, and Section 6.4 falls back to its embedded
hyperparameters. That means this run also verifies the Colab fallback path.

A temporary final cell dumps the notebook's computed values to JSON so the narrative
markdown can be regenerated from the numbers the notebook actually produced; the cell is
removed again before the notebook is saved.

Run:  python execute_notebook_full.py
"""
import json
import os
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient

NB = os.path.abspath('Bank_Cash_Optimization_Workflow.ipynb')
WORK = os.path.abspath(os.path.join(
    r'C:\Users\Raza\AppData\Local\Temp\claude\d--bank\24934af7-bbef-4554-8e8d-25719e71ab0d\scratchpad',
    'nbrun'))
VALUES = os.path.join(WORK, 'notebook_values.json')

DUMP = r'''
# TEMPORARY — injected by execute_notebook_full.py, removed before saving.
import json as _json, numpy as _np, pandas as _pd


def _rec(df):
    return _json.loads(df.to_json(orient='records'))


_out = {
    'split': {
        'cutoff_date': str(cutoff_date.date()),
        'n_total': int(len(half_daily)), 'n_train': int(len(train_df)), 'n_test': int(len(test_df)),
        'test_pct': round(len(test_df) / len(half_daily) * 100, 2),
        'test_start': str(test_df['start_date'].min().date()),
        'test_end': str(test_df['start_date'].max().date()),
        'n_folds': int(len(CV_FOLDS)),
    },
    'cv_params': {t: {k: v for k, v in d.items() if k != 'ho_per_branch'} for t, d in cv_params.items()},
    'model_comparison': _rec(_pd.DataFrame(results)) if 'results' in dir() else [],
    's70': {
        'r2_mean': float(CV_R2_MEAN), 'r2_std': float(CV_R2_STD),
        'wm_mean': float(CV_WM_MEAN), 'wm_std': float(CV_WM_STD),
        'vol_corr': float(CV_VOL_CORR),
        'folds': _rec(CV_FOLD_TBL),
        'branch': _rec(CV_BRANCH),
        'lobo': _rec(CV_LOBO),
        'spec': _rec(CV_SPEC),
        'corr': {k: float(v) for k, v in CV_CORR.items()},
        'tiers': _rec(CV_TIERS),
        'fold_effect': {int(k): float(v) for k, v in CV_FOLD_EFFECT.items()},
        'anova_wmape': list(_ANOVA_WMAPE), 'anova_r2': list(_ANOVA_R2),
        'cum_excl': CV_CUM_EXCL, 'cum5_ratio': float(CV_CUM5_RATIO),
        'base_r2_std': float(CV_BASE_R2_STD), 'base_wm_std': float(CV_BASE_WM_STD),
        'wm_matrix': {int(b): {int(f): float(v) for f, v in row.items()}
                      for b, row in CV_WM_MATRIX.iterrows()},
        'holdout': {
            'n': int(len(CV_HOLDOUT)),
            'r2': float(r2_score(CV_HOLDOUT[CV_TARGET], CV_HOLDOUT['pred'])),
            'mae_M': float(mean_absolute_error(CV_HOLDOUT[CV_TARGET], CV_HOLDOUT['pred'])) / 1e6,
            'wmape': float(cv_wmape(CV_HOLDOUT[CV_TARGET].values, CV_HOLDOUT['pred'].values)),
        },
        'spec_wins': int((CV_SPEC['WMAPE_change_pp'] < 0).sum()),
        'spec_mean_change': float(CV_SPEC['WMAPE_change_pp'].mean()),
        # share of each branch's scored rows with near-zero actual demand (<100K PKR),
        # measured on exactly the rows the MAPE/WMAPE figures are computed from
        'nearzero': _rec(CV_PREDS.assign(nz=(CV_PREDS['actual'] < 1e5).astype(float))
                         .groupby('branch', as_index=False)['nz'].mean()
                         .assign(nz=lambda d: d['nz'] * 100)),
        'holdout_branch': _rec(CV_HO_BRANCH),
    },
}
with open(r'__VALUES__', 'w') as _f:
    _json.dump(_out, _f, indent=2, default=float)
print('notebook values dumped')
'''


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    print(f"clean workdir : {WORK}")
    print(f"contents      : {os.listdir(WORK)}   (no model_data/, no models/ — as in Colab)")

    nb = nbformat.read(NB, as_version=4)
    n_cells = len(nb.cells)
    dump = nbformat.v4.new_code_cell(source=DUMP.replace('__VALUES__', VALUES.replace('\\', '\\\\')))
    dump.pop('id', None)
    nb.cells.append(dump)

    client = NotebookClient(
        nb, timeout=3600, kernel_name='python3', allow_errors=False,
        resources={'metadata': {'path': WORK}},
    )
    print(f"executing {n_cells} cells (+1 temporary dump cell)...")
    t0 = time.time()
    try:
        client.execute()
    except Exception as e:
        # surface the offending cell rather than a bare traceback
        for i, c in enumerate(nb.cells):
            if c.cell_type != 'code':
                continue
            for o in c.get('outputs', []):
                if o.get('output_type') == 'error':
                    print(f"\nFAILED at cell {i}:")
                    print('\n'.join(c.source.splitlines()[:6]))
                    print('  ' + '\n  '.join(o.get('traceback', [])[-8:]))
                    sys.exit(1)
        raise
    print(f"executed in {time.time() - t0:.0f}s")

    nb.cells = nb.cells[:n_cells]        # drop the temporary dump cell
    for c in nb.cells:
        c.pop('id', None)                # notebook is nbformat 4.0, which rejects cell ids
    nbformat.validate(nb)

    shutil.copy(NB, NB + '.pre_fullexec.bak')
    nbformat.write(nb, NB)
    print(f"outputs written back to {os.path.basename(NB)} (backup: .pre_fullexec.bak)")

    if os.path.exists(VALUES):
        v = json.load(open(VALUES))
        print(f"values dumped -> {VALUES}")
        s = v['s70']
        print(f"  split      : {v['split']['n_train']:,} train / {v['split']['n_test']:,} test "
              f"({v['split']['test_pct']}%), cutoff {v['split']['cutoff_date']}")
        print(f"  CV         : R2 {s['r2_mean']:.4f} +/- {s['r2_std']:.4f} | "
              f"WMAPE {s['wm_mean']:.2f}% +/- {s['wm_std']:.2f}%")
        print(f"  hold-out   : R2 {s['holdout']['r2']:.4f} | WMAPE {s['holdout']['wmape']:.2f}% "
              f"({s['holdout']['n']:,} rows)")
        print(f"  specialist : {s['spec_wins']} of {len(s['spec'])} branches improved")
    else:
        print("WARNING: values file not written")


if __name__ == '__main__':
    main()
