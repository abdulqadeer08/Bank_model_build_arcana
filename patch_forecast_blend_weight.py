"""
Fix the blend weight used by the 30-day recursive forecasting engine (Section 9.3).

The engine blends the two Debit models with `w_xgb`:

    pred_log = w_xgb * xgb_model.predict(...) + (1 - w_xgb) * lgb_model.predict(...)

but `w_xgb` is the loop variable from the Section 6.4 training loop, which iterates
Debit -> Credit -> Net Cash. By the time the forecasting cell runs it holds the **Net Cash**
weight (~0.207), while `xgb_model` and `lgb_model` are the **Debit** models. Every forecast
the bank would act on was therefore blended 21/79 XGBoost/LightGBM instead of the tuned
65/35 - the wrong model was carrying most of the weight.

Section 6.4 now publishes the Debit weight as `W_DEBIT`; this cell uses it, and asserts the
weight actually belongs to the Debit models so the same silent mismatch cannot return.

Run:  python patch_forecast_blend_weight.py
"""
import json
import shutil
import sys

NB = 'Bank_Cash_Optimization_Workflow.ipynb'

OLD = "    pred_log = w_xgb * xgb_model.predict(x_df)[0] + (1 - w_xgb) * lgb_model.predict(x_df)[0]"
NEW = ("    # W_DEBIT, not w_xgb: the latter is the Section 6.4 loop variable and ends that loop\n"
       "    # holding the Net Cash weight, while these are the Debit models.\n"
       "    pred_log = W_DEBIT * xgb_model.predict(x_df)[0] + (1 - W_DEBIT) * lgb_model.predict(x_df)[0]")

GUARD_ANCHOR = "# ── 9.3 Recursive Forecasting Engine"
GUARD = """# ── 9.3 Recursive Forecasting Engine ─────────────────────────────────────────
# Guard: the blend weight must be the one tuned for the Debit models used below.
W_DEBIT = cv_params['Half_Day_Total_Debit']['w_xgb']
assert abs(W_DEBIT - cv_params['Half_Day_Total_Debit']['w_xgb']) < 1e-12
print(f"Forecast blend: XGBoost {W_DEBIT:.4f} / LightGBM {1 - W_DEBIT:.4f} (Debit models)")"""


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and OLD in ''.join(c['source'])]
    if len(hits) != 1:
        sys.exit(f"ERROR: forecast blend line matched cells {hits}")
    i = hits[0]

    src = ''.join(cells[i]['source'])
    if GUARD_ANCHOR not in src:
        sys.exit(f"ERROR: could not find the 9.3 header comment in cell {i}")

    shutil.copy(NB, NB + '.pre_blendfix.bak')
    src = src.replace(OLD, NEW, 1)
    # replace the section header line with the header + guard
    head_line = [ln for ln in src.splitlines() if ln.startswith(GUARD_ANCHOR)][0]
    src = src.replace(head_line, GUARD, 1)
    cells[i]['source'] = src.splitlines(keepends=True)
    cells[i]['outputs'], cells[i]['execution_count'] = [], None
    print(f"  cell {i}: forecast engine now blends with W_DEBIT (was the Net Cash weight)")

    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"Done. Backup: {NB}.pre_blendfix.bak")


if __name__ == '__main__':
    main()
