"""
Close the remaining gaps against the supervisor's review.

FIX A  The MAPE-vs-WMAPE section still illustrated its argument with per-branch figures
       taken from the old 164-row hold-out, which now contradict Section 7.0. Replaced
       with the corrected cross-validation figures, plus rank-correlation evidence that
       MAPE ranks branches by their near-zero share rather than by model accuracy.

FIX B  Review point 2 praised "the zero-discrepancy check in the SHAP waterfall", but no
       such check existed anywhere in the notebook, and Section 8 explained ONLY the
       XGBoost half of the deployed ensemble. Section 8.1 now builds the blended
       XGB+LGB explanation and proves its additivity; 8.2 and 8.4 use the blend.

FIX C  Records a hyperparameter bug found while doing FIX B: Cell 37 and v3_pipeline.py
       build their parameter dicts with k.replace('xgb_',''), turning `xgb_colsample`
       into `colsample` - not a valid XGBoost/LightGBM argument, so both tuned
       column-subsampling values are silently discarded.

Run:  python patch_review_gaps.py
"""
import json
import shutil
import sys

NB = 'Bank_Cash_Optimization_Workflow.ipynb'

# ─────────────────────────────────────────────────────────────────────────────
# FIX A — replace the evidence block inside the MAPE→WMAPE markdown cell
# ─────────────────────────────────────────────────────────────────────────────
OLD_A_START = 'From our data:'
OLD_A_END = '#### Why WMAPE Fixes This'

NEW_A = r"""From our data — every figure below comes from the corrected cross-validation in
Section 7.0 (5 folds, all 15 branches, 8,661 scored rows):

| Branch | Near-zero half-days (<100K PKR) | MAPE | rank by MAPE | WMAPE | rank by WMAPE |
|---|---|---|---|---|---|
| 659 | 3.95% | **763%** | 15th — *worst* | 35.8% | 6th — *better than average* |
| 1200 | 0.75% | 485% | 14th | 39.3% | 8th |
| 394 | 0.72% | 290% | 13th | **30.6%** | **2nd — near best** |
| 287 | 4.24% | 256% | 12th | 41.7% | 10th |
| 1376 | 1.31% | 238% | 11th | 36.8% | 7th |
| 1739 | 0.38% | 202% | 10th | **50.7%** | 14th |
| 104 | **0.00%** | 159% | 9th | **52.9%** | **15th — *worst*** |
| 202 | 0.57% | 116% | 5th | 31.1% | 3rd |
| 1297 | 0.16% | 71% | 1st — *best* | 29.7% | 1st — *best* |

**The two metrics barely agree on the ordering.** Across all 15 branches the rank correlation
between MAPE and WMAPE is only **ρ = +0.18**.

**And MAPE is ranking them by the wrong thing.** MAPE's ordering tracks a branch's share of
near-zero half-days at **ρ = +0.60**, while WMAPE's does not (**ρ = −0.26**). MAPE is largely
measuring *how many quiet half-days a branch has*, not how well the model predicts it — exactly
the failure mode described above.

#### Two concrete examples

**Branch 659 — MAPE says worst, WMAPE says better than average.** It has the second-highest share
of near-zero half-days (3.95%). Those few rows inflate its MAPE to 763%, the worst of all 15,
while its WMAPE of 35.8% is comfortably better than the 41.1% portfolio figure and its R² is 0.65.
There is nothing wrong with this branch.

**Branch 104 — MAPE says mid-table, WMAPE says worst.** It has *no* near-zero half-days at all, so
its MAPE (159%) looks unremarkable and it ranks 9th. Yet it is genuinely the weakest branch in the
portfolio — WMAPE 52.9%, R² 0.35 — and it absorbs 26% of the entire error budget (Section 7.0).
**MAPE hides the one branch that most needs attention.**

#### The branch raised in review — Branch 202

Branch 202 is one of the **strongest** branches by every meaningful measure:

- **R² = 0.722** — the highest of all 15 branches
- **WMAPE = 31.1%** — 3rd-lowest, well inside the portfolio
- **MAE = 6.74M PKR**

Yet its **MAPE = 116%** (and 260% on the older, much smaller hold-out) would suggest it is one of
the worst-predicted. That is not a model failure; it is a metric failure, driven by the 0.57% of
its half-days with near-zero actual demand.

---

"""

# ─────────────────────────────────────────────────────────────────────────────
# FIX B — Section 8: explain the deployed ensemble, and verify it
# ─────────────────────────────────────────────────────────────────────────────
NEW_81 = r'''# ── 8.1 SHAP for the DEPLOYED ENSEMBLE, with a zero-discrepancy check ─────────
# The production forecast is a weighted blend of XGBoost and LightGBM, so explaining
# only the XGBoost half would describe a model the bank never actually runs. SHAP values
# are additive, so the correct explanation of a weighted blend is the same weighted blend
# of the two models' SHAP values. This cell builds that blended explanation and then
# PROVES it reconstructs the ensemble's own output, rather than assuming it does.

import numpy as np

# Read the Debit blend weight explicitly: the loop variable `w_xgb` left over from Cell 37
# holds the weight of the LAST target in that loop (Net Cash), not Debit.
W_DEBIT = cv_params['Half_Day_Total_Debit']['w_xgb']
print(f"Deployed Debit ensemble: XGBoost {W_DEBIT:.4f} + LightGBM {1 - W_DEBIT:.4f}")
if abs(w_xgb - W_DEBIT) > 1e-9:
    print(f"  (note: the loop variable w_xgb currently holds {w_xgb:.4f} — the Net Cash "
          f"weight — so it is deliberately not used here)")

X_test_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)

explainer_xgb = shap.TreeExplainer(xgb_model)
explainer_lgb = shap.TreeExplainer(lgb_model)
sv_xgb = np.asarray(explainer_xgb.shap_values(X_test_sample))
sv_lgb = np.asarray(explainer_lgb.shap_values(X_test_sample))
print(f"Per-model SHAP computed: XGBoost {sv_xgb.shape}, LightGBM {sv_lgb.shape}")

# Blended explanation of the deployed ensemble
shap_values = W_DEBIT * sv_xgb + (1 - W_DEBIT) * sv_lgb
base_value_blend = (W_DEBIT * float(explainer_xgb.expected_value)
                    + (1 - W_DEBIT) * float(explainer_lgb.expected_value))
explainer = explainer_xgb        # retained so any later cell referencing `explainer` still works

# ── Zero-discrepancy check ───────────────────────────────────────────────────
# SHAP's additivity guarantee: base value + sum of feature contributions must reproduce
# the model's output EXACTLY. Checked in log1p space, because that is the space the models
# are trained and blended in.
recon = base_value_blend + shap_values.sum(axis=1)
actual = (W_DEBIT * xgb_model.predict(X_test_sample)
          + (1 - W_DEBIT) * lgb_model.predict(X_test_sample))
disc = np.abs(recon - actual)

print("\n" + "=" * 76)
print("ZERO-DISCREPANCY CHECK — blended SHAP vs the ensemble's own prediction")
print("=" * 76)
print(f"  rows checked                  : {len(disc)}")
print(f"  max  |reconstructed - actual| : {disc.max():.3e}   (log1p space)")
print(f"  mean |reconstructed - actual| : {disc.mean():.3e}")

TOL = 1e-4
assert disc.max() < TOL, f"SHAP additivity violated: max discrepancy {disc.max():.3e} >= {TOL}"
print(f"\n  PASS — every row reconstructs to within {TOL:.0e}. The residual is float rounding")
print("  inside XGBoost's own SHAP implementation, shown per model below:")
for nm, sv, ex, m in (('XGBoost', sv_xgb, explainer_xgb, xgb_model),
                      ('LightGBM', sv_lgb, explainer_lgb, lgb_model)):
    d = np.abs(float(ex.expected_value) + sv.sum(axis=1) - m.predict(X_test_sample))
    print(f"    {nm:<9} alone : max discrepancy {d.max():.3e}")
print("  Blending introduces no additional error, so the SHAP plots below explain exactly")
print("  the model that is deployed.")

# Why the check is in log space, quantified
_pkr_direct = np.expm1(actual)
_pkr_naive = np.expm1(base_value_blend) + (np.expm1(base_value_blend + shap_values)
                                           - np.expm1(base_value_blend)).sum(axis=1)
_gap = np.median(np.abs(_pkr_naive - _pkr_direct) / np.maximum(_pkr_direct, 1)) * 100
print(f"\n  Why log1p space: the models predict log1p(debit) and the ensemble blends those")
print(f"  log-space outputs before expm1 is applied, so additivity holds exactly there.")
print(f"  Converting one prediction to PKR is fine (expm1 of the total), but splitting that")
print(f"  PKR figure across features needs a rescaling step, because")
print(f"  expm1(sum of parts) != sum of expm1(parts) — naively doing so would misstate the")
print(f"  total by a median of {_gap:.1f}% on this sample.")
print(f"\nSHAP values computed for {len(X_test_sample)} hold-out samples (blended ensemble).")
'''

OLD_82_TITLE = "plt.title('SHAP Feature Importance - XGBoost V3 (Debit Prediction)', fontsize=13, weight='bold')"
NEW_82_TITLE = "plt.title('SHAP Feature Importance - XGB + LGB Ensemble V3 (Debit Prediction)', fontsize=13, weight='bold')"

OLD_84 = """base_value = explainer.expected_value

print("Explaining a single prediction in plain terms:")
print(f"  Base value (avg log-debit): {base_value:.4f}")
print(f"  Predicted log-debit:        {base_value + sample_shap.sum():.4f}")
print(f"  Predicted debit (PKR):      {np.expm1(base_value + sample_shap.sum()):,.0f}")
print()
"""

NEW_84 = """base_value = base_value_blend          # blended ensemble base, not the XGBoost-only one

print("Explaining a single ENSEMBLE prediction in plain terms:")
print(f"  Base value (avg log-debit): {base_value:.4f}")
print(f"  Predicted log-debit:        {base_value + sample_shap.sum():.4f}")
print(f"  Predicted debit (PKR):      {np.expm1(base_value + sample_shap.sum()):,.0f}")

# Zero-discrepancy check for THIS waterfall: the bars must sum back to the model's output
_row_actual = (W_DEBIT * xgb_model.predict(X_test_sample.iloc[[idx]])[0]
               + (1 - W_DEBIT) * lgb_model.predict(X_test_sample.iloc[[idx]])[0])
_row_disc = abs((base_value + sample_shap.sum()) - _row_actual)
print(f"  Ensemble's own output:      {_row_actual:.4f}")
print(f"  Discrepancy for this row:   {_row_disc:.3e}  -> bars reconcile to the prediction")
print()
"""

OLD_84_TITLE = "plt.title('SHAP Waterfall - Single Prediction Breakdown', fontsize=13, weight='bold')"
NEW_84_TITLE = "plt.title('SHAP Waterfall - Single Ensemble Prediction Breakdown', fontsize=13, weight='bold')"

# ─────────────────────────────────────────────────────────────────────────────
# FIX C — record the colsample bug in the Section 7.0 remaining-work table
# ─────────────────────────────────────────────────────────────────────────────
OLD_C = ("| 5 | Quarterly retraining cadence | Fold accuracy tracks period volatility "
         "(r ≈ +0.90), so a static model will drift. |")
NEW_C = (OLD_C + "\n| 6 | Fix the hyperparameter name bug in `v3_pipeline.py` and Cell 37 | "
         "Both build their parameter dicts with `k.replace('xgb_', '')`, so `xgb_colsample` "
         "becomes `colsample` — not a valid XGBoost or LightGBM argument. **Both tuned "
         "column-subsampling values are silently discarded** and default to 1.0, during Optuna "
         "search as well as final training. Measured impact on the corrected CV is small "
         "(R² 0.5769 with the tuned values applied vs 0.5771 without), so no conclusion in this "
         "notebook changes — but one of the five tuned XGBoost dimensions was a no-op, and the "
         "search should be re-run once the names are corrected. |")


def sub(cells, idx, old, new, label):
    src = ''.join(cells[idx]['source'])
    if old not in src:
        sys.exit(f"ERROR [{label}]: pattern not found in cell {idx}")
    cells[idx]['source'] = src.replace(old, new, 1).splitlines(keepends=True)
    print(f"  {label}: cell {idx} updated")


def find(cells, pred, label):
    hits = [i for i, c in enumerate(cells) if pred(c)]
    if len(hits) != 1:
        sys.exit(f"ERROR: expected 1 cell for {label}, found {hits}")
    return hits[0]


def main():
    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    shutil.copy(NB, NB + '.pre_review_gaps.bak')

    # FIX A
    i_mape = find(cells, lambda c: c['cell_type'] == 'markdown'
                  and ''.join(c['source']).lstrip().startswith('### Why We Changed'), 'MAPE cell')
    src = ''.join(cells[i_mape]['source'])
    a, b = src.find(OLD_A_START), src.find(OLD_A_END)
    if a < 0 or b < 0 or b <= a:
        sys.exit("ERROR [FIX A]: could not bracket the evidence block")
    cells[i_mape]['source'] = (src[:a] + NEW_A + src[b:]).splitlines(keepends=True)
    print(f"  FIX A: cell {i_mape} evidence block replaced")

    # FIX B
    i81 = find(cells, lambda c: c['cell_type'] == 'code'
               and '8.1 SHAP TreeExplainer' in ''.join(c['source']), '8.1')
    cells[i81]['source'] = NEW_81.strip('\n').splitlines(keepends=True)
    cells[i81]['outputs'], cells[i81]['execution_count'] = [], None
    print(f"  FIX B: cell {i81} (8.1) rewritten for the blended ensemble")

    i82 = find(cells, lambda c: c['cell_type'] == 'code'
               and '8.2 Summary plot' in ''.join(c['source']), '8.2')
    sub(cells, i82, OLD_82_TITLE, NEW_82_TITLE, 'FIX B')

    i84 = find(cells, lambda c: c['cell_type'] == 'code'
               and '8.4 Waterfall plot' in ''.join(c['source']), '8.4')
    sub(cells, i84, OLD_84, NEW_84, 'FIX B')
    sub(cells, i84, OLD_84_TITLE, NEW_84_TITLE, 'FIX B')

    # FIX C
    i_work = find(cells, lambda c: c['cell_type'] == 'markdown' and OLD_C in ''.join(c['source']),
                  'remaining-work table')
    sub(cells, i_work, OLD_C, NEW_C, 'FIX C')

    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"\nDone. Backup: {NB}.pre_review_gaps.bak")
    print("NOTE: cells 8.1-8.4 now need re-executing to refresh their outputs.")


if __name__ == '__main__':
    main()
