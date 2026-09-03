"""
Execute the Section 7.0 analysis cells and bake their outputs into the notebook,
so the notebook opens with tables and charts already visible (matching the rest
of the file, which ships executed).

The namespace is built exactly as the notebook builds it up to Cell 27, so the
executed results are identical to what a Colab run of the full notebook produces.

Run:  python execute_branch_variance_cells.py
"""
import base64
import io
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NB = 'Bank_Cash_Optimization_Workflow.ipynb'
FIRST_MARKER = '#  STEP 1 — Diagnosis'
LAST_MARKER = '#  STEP 7 — Verdict'

FEATURE_TARGETS = ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']


def build_half_daily():
    """Reproduce the notebook state after Cell 27 (identical to v3_pipeline.py)."""
    df = pd.read_csv('model_data/half_daily_features.csv')
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
    for t in FEATURE_TARGETS:
        df[f'rolling_14_std_{t}'] = df.groupby('tran_br_code')[t].transform(
            lambda x: x.shift(1).rolling(14, min_periods=2).std()).fillna(0)
        df[f'ewma_14_{t}'] = df.groupby('tran_br_code')[t].transform(
            lambda x: x.shift(1).ewm(span=14, adjust=False).mean()).fillna(0)
        df[f'dow_avg_4_{t}'] = df.groupby(['tran_br_code', 'Weekday', 'AM_PM_Encoded'])[t].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)
    df['tran_br_code'] = df['tran_br_code'].astype('category')
    df['Weekday'] = df['Weekday'].astype('category')
    df['Month'] = df['Month'].astype('category')
    return df


class Recorder(io.TextIOBase):
    """Collects stdout and figures as an ordered list of notebook outputs."""

    def __init__(self):
        self.events = []
        self.buf = []

    def write(self, s):
        self.buf.append(s)
        return len(s)

    def flush_text(self):
        if self.buf:
            txt = ''.join(self.buf)
            self.buf = []
            if txt:
                self.events.append({'output_type': 'stream', 'name': 'stdout',
                                    'text': txt.splitlines(keepends=True)})

    def add_figure(self):
        self.flush_text()
        b = io.BytesIO()
        fig = plt.gcf()
        fig.savefig(b, format='png', dpi=110, facecolor='#fcfcfb', bbox_inches='tight')
        plt.close('all')
        self.events.append({
            'output_type': 'display_data',
            'data': {'image/png': base64.b64encode(b.getvalue()).decode('ascii'),
                     'text/plain': ['<Figure size 1600x650 with 2 Axes>']},
            'metadata': {}})

    def done(self):
        self.flush_text()
        return self.events


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    idx = [i for i, c in enumerate(cells)
           if c['cell_type'] == 'code' and FIRST_MARKER in ''.join(c['source'])]
    jdx = [i for i, c in enumerate(cells)
           if c['cell_type'] == 'code' and LAST_MARKER in ''.join(c['source'])]
    if len(idx) != 1 or len(jdx) != 1:
        sys.exit(f"ERROR: could not bracket the new section (start={idx}, end={jdx})")
    lo, hi = idx[0], jdx[0]
    targets = [i for i in range(lo, hi + 1) if cells[i]['cell_type'] == 'code']
    print(f"Executing {len(targets)} code cells (notebook indices {targets})")

    ns = {'half_daily': build_half_daily(), '__name__': '__main__'}
    print(f"half_daily built: {ns['half_daily'].shape}")

    rec = Recorder()
    plt.show = lambda *a, **k: rec.add_figure()

    counter = max([c.get('execution_count') or 0 for c in cells if c['cell_type'] == 'code']) + 1

    real_stdout = sys.stdout
    for i in targets:
        src = ''.join(cells[i]['source'])
        rec.events, rec.buf = [], []
        print(f"  cell {i} ...", file=real_stdout, end='', flush=True)
        sys.stdout = rec
        try:
            exec(compile(src, f'<cell {i}>', 'exec'), ns)
        except Exception:
            sys.stdout = real_stdout
            import traceback
            traceback.print_exc()
            sys.exit(f"ERROR while executing notebook cell {i}")
        finally:
            sys.stdout = real_stdout
        outs = rec.done()
        cells[i]['outputs'] = outs
        cells[i]['execution_count'] = counter
        n_fig = sum(1 for o in outs if o['output_type'] == 'display_data')
        n_txt = sum(len(o['text']) for o in outs if o['output_type'] == 'stream')
        print(f" ok  ({n_txt} lines, {n_fig} figures)", file=real_stdout)
        counter += 1

    shutil.copy(NB, NB + '.pre_execute.bak')
    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"\nOutputs baked into {NB} (backup: {NB}.pre_execute.bak)")


if __name__ == '__main__':
    main()
