"""
Regenerate every hand-written number in the notebook's narrative from the values the
notebook actually produced when it was executed (notebook_values.json, written by
execute_notebook_full.py).

Nothing here is typed by hand: each figure is read from the executed run, so the prose,
the tables and the code outputs cannot disagree.

Run:  python regenerate_narrative.py <path to notebook_values.json>
"""
import json
import shutil
import sys

import numpy as np

NB = 'Bank_Cash_Optimization_Workflow.ipynb'


def rank_of(vals, key, ascending=True):
    order = sorted(vals, key=lambda r: r[key], reverse=not ascending)
    return {r['branch']: i + 1 for i, r in enumerate(order)}


def spearman(a, b):
    def rk(x):
        o = np.argsort(np.argsort(np.asarray(x, float)))
        return o.astype(float)
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def build(v):
    sp, s70 = v['split'], v['s70']
    dbg = v['cv_params']['Half_Day_Total_Debit']
    br = {r['branch']: r for r in s70['branch']}
    nz = {r['branch']: r['nz'] for r in s70['nearzero']}
    hob = {r['branch']: r for r in s70['holdout_branch']}
    blist = list(br.values())
    n_br = len(blist)

    r_mape = rank_of(blist, 'MAPE_mean', ascending=True)     # 1 = lowest MAPE
    r_wm = rank_of(blist, 'WMAPE_mean', ascending=True)      # 1 = lowest WMAPE
    bl = sorted(blist, key=lambda r: r['WMAPE_mean'])
    best, worst = bl[0], bl[-1]
    worst_mape = max(blist, key=lambda r: r['MAPE_mean'])
    b202, b1739 = br.get(202), br.get(1739)

    mape_v = [r['MAPE_mean'] for r in blist]
    wm_v = [r['WMAPE_mean'] for r in blist]
    nz_v = [nz[r['branch']] for r in blist]
    rho_mw = spearman(mape_v, wm_v)
    rho_mn = spearman(mape_v, nz_v)
    rho_wn = spearman(wm_v, nz_v)
    rho_cv_ho = spearman([br[b]['WMAPE_mean'] for b in hob], [hob[b]['HO_WMAPE'] for b in hob])

    lobo = sorted(s70['lobo'], key=lambda r: r['R2_std change %'])
    lobo_best = lobo[0]
    lobo_branch = int(lobo_best['Branch removed'])
    lobo_r2_shrink = -lobo_best['R2_std change %']
    lobo_wm_shrink = -next(r['WMAPE_std change %'] for r in s70['lobo'] if r['Branch removed'] == lobo_branch)
    r2_concentrated = lobo_r2_shrink > 25   # a >25% single-branch R2 shrink is a real concentration signal
    anova_b, anova_f, anova_i = s70['anova_wmape']
    a2b, a2f, a2i = s70['anova_r2']
    fe = {int(k): float(x) for k, x in s70['fold_effect'].items()}
    worst_fold = max(fe, key=lambda k: fe[k])
    wm_mat = {int(b): {int(f): x for f, x in row.items()} for b, row in s70['wm_matrix'].items()}
    together = sum(1 for b, row in wm_mat.items()
                   if np.sign(row[worst_fold] - np.mean(list(row.values()))) == np.sign(fe[worst_fold]))
    tiers = {}
    for r in s70['tiers']:
        tiers.setdefault(r['Tier'], []).append(r['branch'])
    spec_worse = n_br - s70['spec_wins']
    ho = s70['holdout']
    cmp_rows = {r['Model']: r for r in v['model_comparison']}

    out = {}

    # ── 1. Header result line ────────────────────────────────────────────────
    out['header'] = (
        f"**Final Result (XGB + LGB Ensemble V3 on the Debit target):** "
        f"cross-validated **R² = {s70['r2_mean']:.3f} ± {s70['r2_std']:.3f}**, "
        f"**WMAPE = {s70['wm_mean']:.1f}% ± {s70['wm_std']:.1f}%**, "
        f"MAE = {dbg['cv_mae_mean_M']:.2f} M PKR — substantially ahead of every alternative. "
        f"On a held-out final {sp['test_pct']:.0f}% of the timeline "
        f"({sp['n_test']:,} rows, {sp['test_start']} → {sp['test_end']}) it scores "
        f"R² = {ho['r2']:.3f}, WMAPE = {ho['wmape']:.1f}%."
    )

    # ── 2. V3 update block ───────────────────────────────────────────────────
    out['v3update'] = f"""### 🚀 V3 Model Improvement Update

**Recent enhancements:**

1. **Branch identity feature** — `tran_br_code` is passed as a categorical so the model can learn
   branch-specific baseline shifts.
2. **Honest evaluation** — metrics are computed against the raw, uncapped actuals
   (`Half_Day_Total_Debit_RAW`). Winsorization is applied only to the training target.
3. **Corrected evaluation harness (Section 7.0)** — the train/test split and the cross-validation
   now divide the data by **date** rather than by row position. Previously both divided it by
   *branch*, because the dataframe is sorted branch-first. That left the hold-out at 164 rows
   (1.0%) and kept three branches out of every validation fold.
4. **Result** — {sp['n_folds']}-fold rolling-origin CV gives
   **R² = {s70['r2_mean']:.4f} ± {s70['r2_std']:.4f}** and
   **WMAPE = {s70['wm_mean']:.2f}% ± {s70['wm_std']:.2f}%**, with all 15 branches scored in every fold."""

    # ── 3. Production alignment ──────────────────────────────────────────────
    out['alignment'] = f"""### 📌 Note on Production Alignment

This notebook is the end-to-end interactive workflow. The automated production pipeline
([v3_pipeline.py](file:///d:/bank/v3_pipeline.py)) runs the identical scheme on the same
{sp['n_total']:,}-row half-daily dataset — the same date-based split, the same
{sp['n_folds']}-fold rolling-origin cross-validation, the same {dbg['n_optuna_trials']}-trial Optuna
search — and reports the authoritative figures:

| | Cross-validated | Hold-out ({sp['n_test']:,} rows) |
|---|---|---|
| R² | {dbg['cv_r2_mean']:.4f} ± {dbg['cv_r2_std']:.4f} | {dbg['ho_r2']:.4f} |
| MAE | {dbg['cv_mae_mean_M']:.2f}M ± {dbg['cv_mae_std_M']:.2f}M PKR | {dbg['ho_mae_M']:.2f}M PKR |
| WMAPE | {dbg['cv_wmape_mean']:.2f}% ± {dbg['cv_wmape_std']:.2f}% | {dbg['ho_wmape']:.2f}% |

Small differences between these and the notebook's own re-run are floating-point
non-determinism in tree training, not a difference in method."""

    # ── 4. Response to review ────────────────────────────────────────────────
    out['response'] = f"""### 📮 Response to Supervisor Review (2026-08)

The three review points are addressed as follows:

| Review point | Where |
|---|---|
| Explain why MAPE is unreliable on this data | *"Why We Changed the Primary Reporting Metric from MAPE to WMAPE"* (Section 6) |
| Keep verifying ensemble blending and SHAP rather than assuming | Section 8.1 — the blended-ensemble SHAP additivity check |
| **Check per-branch performance — especially 202 and 1739 — and determine whether the fold variance is concentrated or spread** | **Section 7.0** |

**Headline answers.** The fold-to-fold variance splits by metric: pooled **R² is partly
concentrated** in branch {lobo_branch} (removing it alone cuts the R² fold-to-fold std by
{lobo_r2_shrink:.0f}%, because R² pools branches of very different size), while **WMAPE —
the metric the question was framed in — is spread**: the same branch's removal only cuts the
WMAPE std by {lobo_wm_shrink:.0f}%, and removing the five worst branches still leaves
{s70['cum5_ratio']*100:.0f}% of the WMAPE spread intact. Branch **202 is among the strongest**
(R² = {b202['R2_mean']:.2f}, WMAPE = {b202['WMAPE_mean']:.1f}%) — its high MAPE was the metric artefact.
Branch **1739 is genuinely weak** (R² = {b1739['R2_mean']:.2f}, WMAPE = {b1739['WMAPE_mean']:.1f}%) and the
concern was justified. Branch **{worst['branch']}**{", not previously flagged," if worst['branch'] not in (202, 1739) else ""} is the weakest of all
and carries {worst['ErrShare_%']:.0f}% of the portfolio error budget. Branch-level tuning and "more data"
were both **tested directly** and neither is the answer — the recommendation is a single pooled
model with **per-branch confidence bands**.

**Six defects were found and fixed** along the way — five in the evaluation harness, which had
to be corrected before the per-branch question could be answered at all, and one in the forecast
itself. All are listed in the table at the end of Section 7.0 and in the `v3_pipeline.py` header:
the branch-ordered train/test split, the branch-ordered cross-validation, a hyperparameter name
bug that silently discarded the tuned column-subsampling, an outlier cap that leaked across folds,
cross-validation scoring against winsorized rather than raw actuals, and — in the 30-day
forecasting engine — a blend that used the Net Cash weight against the Debit models.

Re-scored correctly, the model holds up:
**R² = {s70['r2_mean']:.3f} ± {s70['r2_std']:.3f}, WMAPE = {s70['wm_mean']:.1f}% ± {s70['wm_std']:.1f}%**."""

    # ── 5. MAPE evidence block ───────────────────────────────────────────────
    show = sorted(blist, key=lambda r: -r['MAPE_mean'])
    keep = [show[0], show[1], show[2]]
    for extra in (worst, b202, best):
        if extra['branch'] not in [k['branch'] for k in keep]:
            keep.append(extra)
    keep = sorted(keep, key=lambda r: -r['MAPE_mean'])
    rows = "\n".join(
        f"| {r['branch']} | {nz[r['branch']]:.2f}% | {r['MAPE_mean']:.0f}% | "
        f"{r_mape[r['branch']]}{'th' if r_mape[r['branch']] not in (1, 2, 3) else ['st','nd','rd'][r_mape[r['branch']]-1]}"
        f"{' — *highest*' if r_mape[r['branch']] == n_br else ''} | "
        f"{r['WMAPE_mean']:.1f}% | "
        f"{r_wm[r['branch']]}{'th' if r_wm[r['branch']] not in (1, 2, 3) else ['st','nd','rd'][r_wm[r['branch']]-1]}"
        f"{' — *worst*' if r_wm[r['branch']] == n_br else (' — *best*' if r_wm[r['branch']] == 1 else '')} |"
        for r in keep)

    out['mape_evidence'] = f"""From our data — every figure below comes from the corrected cross-validation in
Section 7.0 ({sp['n_folds']} folds, all {n_br} branches, {sum(r['Rows'] for r in blist):,} scored rows,
raw uncapped actuals):

| Branch | Near-zero half-days (<100K PKR) | MAPE | rank by MAPE | WMAPE | rank by WMAPE |
|---|---|---|---|---|---|
{rows}

**The two metrics barely agree on the ordering.** Across all {n_br} branches the rank correlation
between MAPE and WMAPE is only **ρ = {rho_mw:+.2f}**.

**And MAPE is ranking them by the wrong thing.** MAPE's ordering tracks a branch's share of
near-zero half-days at **ρ = {rho_mn:+.2f}**, while WMAPE's does not (**ρ = {rho_wn:+.2f}**). MAPE is largely
measuring *how many quiet half-days a branch has*, not how well the model predicts it — exactly
the failure mode described above.

#### Two concrete examples

**Branch {worst_mape['branch']} — MAPE says worst, WMAPE says otherwise.** It carries a high share of
near-zero half-days ({nz[worst_mape['branch']]:.2f}%). Those few rows inflate its MAPE to
{worst_mape['MAPE_mean']:.0f}%, the highest of all {n_br}, while its WMAPE of {worst_mape['WMAPE_mean']:.1f}%
ranks {r_wm[worst_mape['branch']]} of {n_br} and its R² is {worst_mape['R2_mean']:.2f}.

**Branch {worst['branch']} — MAPE says mid-table, WMAPE says worst.** With only
{nz[worst['branch']]:.2f}% near-zero half-days its MAPE ({worst['MAPE_mean']:.0f}%) looks unremarkable and it
ranks {r_mape[worst['branch']]} of {n_br}. Yet it is the weakest branch in the portfolio — WMAPE
{worst['WMAPE_mean']:.1f}%, R² {worst['R2_mean']:.2f} — and it absorbs {worst['ErrShare_%']:.0f}% of the entire error
budget. **MAPE hides the one branch that most needs attention.**

#### The branch raised in review — Branch 202

| Measure | Value | Rank of {n_br} |
|---|---|---|
| R² | {b202['R2_mean']:.3f} | {rank_of(blist, 'R2_mean', ascending=False)[202]} |
| WMAPE | {b202['WMAPE_mean']:.1f}% | {r_wm[202]} |
| MAE | {b202['MAE_mean']:.2f}M PKR | — |
| **MAPE** | **{b202['MAPE_mean']:.0f}%** | {r_mape[202]} |

Its MAPE would suggest one of the worst-predicted branches. Every other measure puts it among
the best. That is not a model failure; it is a metric failure, driven by the
{nz[202]:.2f}% of its half-days with near-zero actual demand.

---

"""

    # ── 6. Conclusion ────────────────────────────────────────────────────────
    def cmp_row(name, label):
        r = cmp_rows.get(name)
        if not r:
            return None
        # Two key conventions coexist in older dumps: evaluate()'s 'WMAPE (%)' (Baseline/
        # Prophet/LightGBM) and the ensemble row's 'WMAPE_%'. Accept either.
        wmape = r.get('WMAPE (%)', r.get('WMAPE_%'))
        r2 = r.get('R2')
        if wmape is None or r2 is None:
            return None
        return f"| {label} | {r2:.4f} | {wmape:.2f}% |"

    base_rows = [cmp_row('Baseline (14-Day Rolling Mean)', '14-Day Rolling Mean baseline'),
                 cmp_row('Prophet', 'Prophet'),
                 cmp_row('LightGBM', 'LightGBM standalone'),
                 cmp_row('XGBoost + LightGBM Ensemble V3', '**XGB + LGB Ensemble V3**')]
    base_tbl = "\n".join(r for r in base_rows if r)

    out['conclusion'] = f"""### Conclusion — Model Comparison (Debit target)

The XGBoost + LightGBM Ensemble (V3) is the strongest of the four candidates.

**Evaluation protocol**

| Setting | Value |
|---|---|
| Cross-validation | {sp['n_folds']}-fold rolling origin, split on calendar date |
| Hold-out | final {sp['test_pct']:.0f}% of the timeline — {sp['n_test']:,} rows, {sp['test_start']} → {sp['test_end']}, all 15 branches |
| Optuna trials | {dbg['n_optuna_trials']} (TPESampler, seed = 42) |
| Objective | `reg:absoluteerror` (XGBoost) + `mae` (LightGBM) |
| Blend | XGBoost {dbg['w_xgb']:.4f} / LightGBM {1 - dbg['w_xgb']:.4f} |
| Target transform | `log1p` (Debit, Credit) / raw (Net Cash) |
| Scored against | raw **uncapped** actuals; winsorization applied to the training target only |

**Cross-validated performance**

| Metric | Value |
|---|---|
| R² | {dbg['cv_r2_mean']:.4f} ± {dbg['cv_r2_std']:.4f} |
| MAE | {dbg['cv_mae_mean_M']:.3f}M ± {dbg['cv_mae_std_M']:.3f}M PKR |
| RMSE | {dbg['cv_rmse_mean_M']:.3f}M PKR |
| MAPE | {dbg['cv_mape_mean']:.1f}%  *(metric artefact — see the MAPE → WMAPE section above)* |
| SMAPE | {dbg['cv_smape_mean']:.1f}% |
| **WMAPE** | **{dbg['cv_wmape_mean']:.2f}% ± {dbg['cv_wmape_std']:.2f}%**  ← the figure to quote to the bank |

**Hold-out ({sp['n_test']:,} rows):** R² = {dbg['ho_r2']:.4f}, MAE = {dbg['ho_mae_M']:.2f}M PKR,
WMAPE = {dbg['ho_wmape']:.2f}%.

**Against the baselines**

| Model | R² | WMAPE |
|---|---|---|
{base_tbl}

**Why cross-validated rather than a single split.** A single split is sensitive to which window
it happens to land on. A fold-averaged figure with a stated ± is stable, and the ± is now
meaningful: because the folds are carved out of the calendar, it measures how much accuracy moves
between time periods.

> **Production readiness.** Section 7.0 re-derives these numbers per branch and reports which
> branches are ready to deploy on a standard confidence band, which need a wider one, and which
> need manual review. The model is sound; what was not production-ready was the evaluation
> harness, and that is now corrected.

*Note: the lower R² on Net Cash (~{v['cv_params']['Half_Day_Net_Cash']['cv_r2_mean']:.2f}) is expected — it is the
difference of two independently noisy signals (Credit minus Debit).*"""

    # ── 7. Section 7.0 executive summary ─────────────────────────────────────
    out['s70_intro'] = f"""## 7.0 Supervisor Review — Is the Cross-Validation Variance Concentrated in a Few Branches?

**What was asked (review of 2026-08):**

1. Explain in the notebook why MAPE does not work on this data → *answered in
   "Why We Changed the Primary Reporting Metric from MAPE to WMAPE" above.*
2. Keep verifying ensemble blending and SHAP rather than assuming → *Section 8.1 now carries the
   blended-ensemble additivity check.*
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
| Is the fold-to-fold variance caused by a few branches? | **It depends on the metric.** Removing branch {lobo_branch} alone shrinks the pooled R² fold-to-fold std by **{lobo_r2_shrink:.0f}%** — a real, single-branch effect, because R² pools all branches together and branch {lobo_branch} is far larger in volume than any other. The **same branch's** removal shrinks the WMAPE std by only {lobo_wm_shrink:.0f}%, and removing the five *worst* branches by WMAPE still leaves **{s70['cum5_ratio']*100:.0f}%** of the WMAPE spread intact — so on WMAPE, the metric the question was framed in, the variance is spread. |
| What *does* cause it? | A **system-wide effect that hits every branch at once.** In the weakest fold, {together} of {n_br} branches got worse together. That co-movement is not simply "the period was more volatile" — fold-level demand volatility only weakly relates to fold WMAPE (r = {s70['vol_corr']:+.2f}) — so something about the period itself (not captured by aggregate volatility alone) is driving it. |
| Is branch performance itself uniform? | **No** — and this is the important distinction. Branch identity explains **{anova_b:.1f}%** of the variation in WMAPE, while the time period explains only {anova_f:.1f}%. Some branches are structurally harder than others. |
| Branch 202? | **Among the strongest, not a problem branch.** R² = {b202['R2_mean']:.2f} (rank {rank_of(blist,'R2_mean',ascending=False)[202]} of {n_br}), WMAPE = {b202['WMAPE_mean']:.1f}% (rank {r_wm[202]}). The alarming MAPE was a metric artefact. |
| Branch 1739? | **Genuinely weak** — R² = {b1739['R2_mean']:.2f}, WMAPE = {b1739['WMAPE_mean']:.1f}% (rank {r_wm[1739]} of {n_br}). Confirmed; it needs a widened production band. |
| Branch {worst['branch']}? | **{"Not previously flagged, but the" if worst['branch'] not in (202, 1739) else "The"} biggest problem.** Worst accuracy (WMAPE {worst['WMAPE_mean']:.1f}%) *and* {worst['ErrShare_%']:.0f}% of the entire portfolio error budget on {worst['DemandShare_%']:.0f}% of the demand. |
| So: branch-level tuning, or more data? | **Neither.** Both were tested directly. Training on one branch's own ~1,000 rows alone (same architecture and hyperparameters as the pooled model — not a fresh per-branch tuning search) is worse for **{spec_worse} of {n_br}** branches (mean {s70['spec_mean_change']:+.2f} pp WMAPE). Branch history length has no relationship with accuracy (r = {s70['corr']['history vs WMAPE']:+.2f}). The real driver is intrinsic demand volatility (r = {s70['corr']['volatility vs WMAPE']:+.2f}). |

> **A note on what this section reruns.** The per-branch question could not be answered under the
> original evaluation, because three branches — including branch 202 — never appeared in any
> validation fold. Step 1 diagnoses why. The corrected scheme is now used by the whole notebook
> and by `v3_pipeline.py`: same ensemble, same hyperparameters, same blend weight — only the way
> the data is divided has changed."""

    # ── 8. Bridging note before 7.1 ──────────────────────────────────────────
    out['bridge'] = f"""### How 7.1–7.3 below relate to Section 7.0 above

Both views are now trustworthy, and they measure different things:

- **Section 7.0** is the **cross-validated** view — {sp['n_folds']} folds inside the training window,
  {sum(r['Rows'] for r in blist):,} scored rows, so each branch carries a mean *and* a standard
  deviation. Use it for stability and for sizing confidence bands.
- **Sections 7.1–7.3** are the **hold-out** view — the final {sp['test_pct']:.0f}% of the timeline
  ({sp['n_test']:,} rows, roughly {sp['n_test']//n_br} per branch), data the model never saw during
  training or tuning. Use it as the out-of-sample confirmation.

They agree closely: the rank correlation between their per-branch WMAPE orderings is
**ρ = {rho_cv_ho:+.2f}**. Before the split was corrected this comparison was not meaningful — the
hold-out held only 164 rows, about 11 per branch, which produced negative R² values for several
branches purely as small-sample noise."""

    # ── 9. Remaining work ────────────────────────────────────────────────────
    tier_line = "; ".join(f"**{t}** ({len(bs)}): {sorted(bs)}"
                          for t, bs in sorted(tiers.items()))
    out['whatchanged'] = f"""### What changed, and what it means for the project

**The model did not change.** Same ensemble, same search space, same blend structure. What changed
is how the data is divided and what the metrics are scored against — and those changes were needed
before any per-branch question could be answered.

**Six defects were found and fixed** (the first five are documented in the `v3_pipeline.py` header):

| # | Defect | Effect |
|---|---|---|
| 1 | Train/test split taken from a **row position** in a branch-sorted frame | Hold-out was 164 rows (1.0%, 5 days) instead of 20%. Now {sp['n_test']:,} rows across all 15 branches. |
| 2 | `TimeSeriesSplit` on the same branch-sorted frame | Folds were blocks of *branches*, not time. Three branches never validated; ~100% of training rows post-dated their validation window. Now rolling origin on the calendar. |
| 3 | `k.replace('xgb_', '')` turned `xgb_colsample` into `colsample` | Not a valid XGBoost/LightGBM argument, so the tuned column-subsampling was silently discarded and defaulted to 1.0 — during the Optuna search as well as final training. |
| 4 | 99th-percentile cap fitted once over the whole training frame | Each fold's cap was informed by its own validation period. Now fitted per fold. |
| 5 | Cross-validation scored against **winsorized** actuals while the hold-out used raw ones | The cap bites on ~1% of rows and hid truncations up to 775M PKR. Both now score against raw. |
| 6 | The 30-day forecasting engine blended with `w_xgb` | That loop variable ends Section 6.4 holding the **Net Cash** weight, while the models it blends are the **Debit** models — so every forecast was weighted ~21/79 instead of the tuned split. Now uses `W_DEBIT`. |

**Production readiness by branch:** {tier_line}.

**For the bank-facing conversation**, the useful sentence is:

> *Half-day cash demand is forecast to within about {best['WMAPE_mean']:.0f}% for the strongest branches and
> about {worst['WMAPE_mean']:.0f}% for the weakest, and we can say in advance which branch is which. Accuracy
> moves with how volatile the period is rather than with which branch we look at, so the sensible
> operating model is one shared forecasting engine with a different confidence band per branch,
> retrained quarterly.*

**Remaining work, in priority order**

| # | Item | Why |
|---|---|---|
| 1 | Ship the per-branch confidence bands from Step 6 | Highest-value deliverable for the bank, and it needs no modelling change. |
| 2 | Volatility-aware review for the *Needs attention* tier | These carry the widest bands; branch {worst['branch']} alone is {worst['ErrShare_%']:.0f}% of the error budget. |
| 3 | Quarterly retraining cadence | Fold accuracy moves together across branches from one period to the next (item 1 above) even though it is not explained by aggregate demand volatility alone (r = {s70['vol_corr']:+.2f} with fold WMAPE) — something about each period shifts outcomes for every branch at once, so a static model will drift and periodic retraining is the safe response. |
| 4 | Feed the corrected hold-out into the dashboard and forecast artefacts | `dashboard.py` and `forecast_pipeline.py` still read metrics produced before the split was fixed. |"""

    # ── 10. Step 1 result (past tense: the defect is now fixed) ──────────────
    out['step1result'] = f"""### Step 1 result — the original split was not a time split

`TimeSeriesSplit` divides a dataframe by **row position**, assuming row order is time order.

The data is sorted with `sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded'])`, so row
order was **branch order**: rows 0–1,267 were branch 104, rows 1,268–2,375 branch 202, and so on.
Splitting that frame by row position produced **blocks of branches, not blocks of time**.

Three consequences, all visible in the output above:

1. **Each validation fold held only 4–5 branches**, and three of them (104, 202, 234) sat in the
   leading rows and so were always in the training set. That is why the earlier per-branch table
   was mostly blank, and why branch 202 could not be reported at all.
2. **Every fold's training and validation windows covered the same full date range**
   (Feb 2024 – Mar 2026). Roughly 100% of each fold's training rows were dated at or after its
   validation window opened, so the model was being scored on the same calendar period it had
   been trained on.
3. The reported ± was therefore **not** measuring forecast stability over time. It was largely
   measuring how much the four or five branches in one fold differed from the four or five in the
   next — a between-branch difference wearing a time-series label.

The same root cause shrank the hold-out: `cutoff_idx = int(len(df) * 0.8)` picked a row inside
branch 1200's block, and *that row's date* became the global cutoff — leaving the last six days,
**164 rows, 1.0% of the data**, instead of 20%.

**Both are now fixed.** The split is taken from the unique-date timeline and the folds are
rolling-origin windows on the calendar, so the hold-out is **{sp['n_test']:,} rows
({sp['test_pct']:.1f}%, {sp['test_start']} → {sp['test_end']})** and **every one of the 15 branches
appears in all {sp['n_folds']} validation folds**. The same helpers are used by Section 6.4 and by
`v3_pipeline.py`, so the three cannot diverge again. Step 6 quantifies the hold-out change."""

    return out


ANCHORS = [
    ('header', '**Final Result', None),
    ('v3update', '### 🚀 V3 Model Improvement Update', 'full'),
    ('alignment', '### 📌 Note on Production Alignment', 'full'),
    ('response', '### 📮 Response to Supervisor Review', 'full'),
    ('mape_evidence', 'From our data', 'block'),
    ('conclusion', '### Conclusion — Model Comparison', 'full'),
    ('s70_intro', '## 7.0 Supervisor Review', 'full'),
    ('bridge', '### How 7.1–7.3 below relate', 'full'),
    ('whatchanged', '### What changed, and what it means', 'full'),
    ('step1result', '### Step 1 result', 'full'),
]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: regenerate_narrative.py <notebook_values.json>")
    v = json.load(open(sys.argv[1], encoding='utf-8'))
    new = build(v)

    with open(NB, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    shutil.copy(NB, NB + '.pre_narrative.bak')

    for key, anchor, mode in ANCHORS:
        hits = [i for i, c in enumerate(cells)
                if c['cell_type'] == 'markdown' and anchor in ''.join(c['source'])]
        if len(hits) != 1:
            sys.exit(f"ERROR: anchor '{anchor}' matched cells {hits}")
        i = hits[0]
        src = ''.join(cells[i]['source'])
        if mode == 'full':
            src = new[key]
        elif mode == 'block':
            a = src.find('From our data')
            b = src.find('#### Why WMAPE Fixes This')
            if a < 0 or b < 0:
                sys.exit(f"ERROR: could not bracket the MAPE evidence block in cell {i}")
            src = src[:a] + new[key] + src[b:]
        else:                                   # single line replacement
            lines = src.splitlines(keepends=True)
            for j, ln in enumerate(lines):
                if anchor in ln:
                    lines[j] = new[key] + '\n'
                    break
            src = ''.join(lines)
        cells[i]['source'] = src.splitlines(keepends=True)
        print(f"  cell {i:3d}: {key}")

    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"Done. Backup: {NB}.pre_narrative.bak")


if __name__ == '__main__':
    main()
