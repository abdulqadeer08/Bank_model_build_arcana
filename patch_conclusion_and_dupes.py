"""
Two housekeeping fixes so the notebook is internally consistent after Section 7.0:

1. Cell 45 ("CONCLUSION") is a MARKDOWN cell containing Python source, so it renders
   with a literal print(...) wrapper and a stray escaped-newline pair. It also declares
   the model "production-ready" on the strength of the 3-fold split that Section 7.0
   shows was branch-blocked. Rewritten as real markdown, numbers preserved, with the
   production-ready claim reconciled against the corrected evaluation.

2. The last two cells of the notebook are byte-identical duplicates of cells 45 and 40,
   appended as markdown by an earlier patch script. Cell 78 renders as a wall of raw
   Python. Both are removed; the originals stay in place.

Run:  python patch_conclusion_and_dupes.py
"""
import json
import shutil
import sys

NB = 'Bank_Cash_Optimization_Workflow.ipynb'

NEW_CONCLUSION = r"""
### Conclusion — Model Comparison (Debit target)

The XGBoost + LightGBM Ensemble (V3) is the strongest of the four candidates.

**Evaluation protocol**

| Setting | Value |
|---|---|
| Cross-validation | 3-fold `TimeSeriesSplit` (see the correction in Section 7.0) |
| Optuna trials | 50 (TPESampler, seed = 42) |
| Objective | `reg:absoluteerror` (XGBoost) + `mae` (LightGBM) |
| Branch encoding | `tran_br_code` as categorical |
| Target transform | `log1p` (Debit, Credit) / raw (Net Cash) |
| Evaluation actuals | raw, uncapped |

**Cross-validated performance**

| Metric | Value |
|---|---|
| R² | 0.5687 ± 0.0581 |
| MAE | 8.825M ± 3.252M PKR |
| RMSE | 17.506M PKR |
| MAPE | 826.2%  *(metric artefact — see the MAPE → WMAPE section above)* |
| SMAPE | 51.9% |
| **WMAPE** | **39.5%**  ← the figure to quote in bank presentations |

**Against the baselines**

| Model | R² | WMAPE |
|---|---|---|
| 14-Day Rolling Mean baseline | 0.2964 | 82.25% |
| Prophet | −0.0169 | 89.61% |
| LightGBM standalone | 0.5411 | 45.18% |
| **XGB + LGB Ensemble V3** | **0.5687** | **39.5%** |

**Why cross-validated rather than a single split.** A single train/test split with ~150 test
rows proved sensitive to small dataset changes, with R² swinging across runs
(0.6785 → 0.6628 → 0.4838 → 0.4770). A fold-averaged figure is stable and will not lurch
if more rows arrive later.

> **Production readiness — read Section 7.0 before quoting these numbers.**
> The 3-fold split above turned out to divide the data by *branch* rather than by *date*,
> because the dataframe is sorted branch-first. Section 7.0 re-runs the evaluation on a
> calendar-based rolling-origin split in which all 15 branches appear in all 5 folds, and
> the headline holds: **R² = 0.577 ± 0.053, WMAPE = 41.1% ± 3.0%**. The model is therefore
> sound. What was not production-ready was the *evaluation harness*, and Section 7.0
> corrects it and reports per-branch readiness tiers.

*Note: the lower R² on Net Cash (~0.24) is expected — it is the difference of two
independently noisy signals (Credit minus Debit).*
"""


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    # ── 1. Locate and rewrite the conclusion cell ────────────────────────────
    concl = [i for i, c in enumerate(cells)
             if c['cell_type'] == 'markdown'
             and ''.join(c['source']).lstrip().startswith('# ══ CONCLUSION')
             and 'print(' in ''.join(c['source'])]
    if not concl:
        sys.exit("ERROR: conclusion cell not found")
    print(f"Conclusion-style cells found at: {concl}")

    # ── 2. Identify duplicates: same source appearing earlier in the notebook ─
    seen, dupes = {}, []
    for i, c in enumerate(cells):
        key = ''.join(c['source']).strip()
        if not key:
            continue
        if key in seen:
            dupes.append((i, seen[key]))
        else:
            seen[key] = i
    trailing = [(i, orig) for i, orig in dupes if i >= len(cells) - 2]
    print(f"Duplicate cells at the tail (index, duplicate-of): {trailing}")
    if len(trailing) != 2:
        sys.exit(f"ERROR: expected exactly 2 trailing duplicates, found {len(trailing)}")

    shutil.copy(NB, NB + '.pre_conclusion.bak')

    # Rewrite the first (real, in-place) conclusion cell
    target = concl[0]
    cells[target]['source'] = NEW_CONCLUSION.strip('\n').splitlines(keepends=True)
    print(f"Rewrote cell {target} as proper markdown")

    # Drop the trailing duplicates (highest index first)
    for i, orig in sorted(trailing, reverse=True):
        print(f"Removing duplicate cell {i} (copy of cell {orig})")
        del cells[i]

    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"Done. Notebook now has {len(cells)} cells. Backup: {NB}.pre_conclusion.bak")


if __name__ == '__main__':
    main()
