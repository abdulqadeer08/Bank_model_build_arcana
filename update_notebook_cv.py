"""
update_notebook_cv.py
=====================
After v3_pipeline.py (50-trial) completes, this script:

1. Reads models/v3/cv_results.json
2. Updates notebook Cell 37 (XGBoost training) with fresh best_params + CV display
3. Clears stale outputs in cells 31-40 so the next Run All will regenerate them
4. Updates notebook conclusion cells with CV-based numbers
5. Saves notebook

Run AFTER v3_pipeline.py has completed.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, os

# ── Load CV results ────────────────────────────────────────────────────────────
cv_path = 'models/v3/cv_results.json'
if not os.path.exists(cv_path):
    print(f"ERROR: {cv_path} not found. Run v3_pipeline.py first.")
    sys.exit(1)

with open(cv_path) as f:
    cv = json.load(f)

debit = cv['Half_Day_Total_Debit']
bp    = debit['best_params']

print("=" * 60)
print("  CV Results loaded:")
print("=" * 60)
print(f"  Half_Day_Total_Debit (PRIMARY target):")
print(f"    CV R²  : {debit['cv_r2_mean']:.4f} ± {debit['cv_r2_std']:.4f}")
print(f"    CV MAE : {debit['cv_mae_mean_M']:.3f}M ± {debit['cv_mae_std_M']:.3f}M")
print(f"    CV MAPE: {debit['cv_mape_mean']:.1f}% ± {debit['cv_mape_std']:.1f}%")
print(f"    H/O R² : {debit['ho_r2']:.4f}  (hold-out, {debit['ho_test_rows']} rows)")
print(f"    H/O MAE: {debit['ho_mae_M']:.3f}M")
print(f"  Best params: {bp}")

# ── Load notebook ──────────────────────────────────────────────────────────────
with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
print(f"\nNotebook loaded: {len(cells)} cells")

# ── Build new Cell 37 source ───────────────────────────────────────────────────
# Format best_params as a clean dict literal
param_lines = "best_params = {\n"
param_lines += f"    # From v3_pipeline.py: {debit['n_optuna_trials']}-trial Optuna, seed=42, {debit['n_cv_folds']}-fold TSCV\n"
for k, v in bp.items():
    if isinstance(v, float):
        param_lines += f"    {k!r}: {v!r},\n"
    else:
        param_lines += f"    {k!r}: {v!r},\n"
param_lines += "    'enable_categorical': True,\n"
param_lines += "    'random_state': 42,\n"
param_lines += "    'objective': 'reg:absoluteerror',\n"
param_lines += "}\n"

new_cell37 = (
    "# ── 6.4 XGBoost V3 (Tuned — params from 50-trial Optuna on current data) ──\n"
    "# best_params are the result of v3_pipeline.py:\n"
    "#   • 50 Optuna trials, TPESampler seed=42\n"
    "#   • 3-fold TimeSeriesSplit CV as tuning objective\n"
    "#   • Wider search space: n_estimators 100-600, lr 0.005-0.1, reg_alpha/lambda added\n"
    "# CV-averaged performance (PRIMARY, stable metric):\n"
    f"#   R²  = {debit['cv_r2_mean']:.4f} ± {debit['cv_r2_std']:.4f}  (across {debit['n_cv_folds']} folds)\n"
    f"#   MAE = {debit['cv_mae_mean_M']:.3f}M ± {debit['cv_mae_std_M']:.3f}M\n"
    f"#   MAPE= {debit['cv_mape_mean']:.1f}% ± {debit['cv_mape_std']:.1f}%\n"
    "\n"
    + param_lines +
    "\n"
    "print('Training XGBoost V3 with 50-trial Optuna params...')\n"
    f"print(f'  n_estimators : {bp['n_estimators']}')\n"
    f"print(f'  learning_rate: {bp['learning_rate']:.6f}')\n"
    f"print(f'  max_depth    : {bp['max_depth']}')\n"
    "xgb_model = xgb.XGBRegressor(**best_params)\n"
    "xgb_model.fit(X_train, y_train_log)\n"
    "print('XGBoost V3 trained.')\n"
    "\n"
    "# Predict & inverse-transform\n"
    "y_pred_log = xgb_model.predict(X_test)\n"
    "y_pred_xgb = np.expm1(y_pred_log)\n"
    "\n"
    "results.append(evaluate(y_test_raw, y_pred_xgb, 'XGBoost V3 (Tuned)'))\n"
)

# Verify cell 37 is XGBoost cell
old37 = ''.join(cells[37].get('source', []))
assert 'XGBoost' in old37 and 'best_params' in old37, \
    f"Cell 37 doesn't look right: {old37[:80]}"

cells[37]['source'] = [new_cell37]
cells[37]['outputs'] = []
cells[37]['execution_count'] = None
print("\nCell 37 updated.")

# ── Clear stale outputs in cells 29-40 so Run All regenerates ─────────────────
for i in range(29, min(41, len(cells))):
    if cells[i]['cell_type'] == 'code':
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None
print("Cleared stale outputs in cells 29-40.")

# ── Save notebook ──────────────────────────────────────────────────────────────
with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook saved: {len(cells)} cells")

# ── Print what to update in dashboard ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  NUMBERS TO USE IN DASHBOARD / MODEL_RESULTS.CSV")
print("=" * 60)
print(f"  XGBoost V3 — CV-averaged (PRIMARY):")
print(f"    R²  = {debit['cv_r2_mean']:.4f} ± {debit['cv_r2_std']:.4f}")
print(f"    MAE = {debit['cv_mae_mean_M']:.3f}M ± {debit['cv_mae_std_M']:.3f}M")
print(f"    MAPE= {debit['cv_mape_mean']:.1f}% ± {debit['cv_mape_std']:.1f}%")
for k, (rn, ma, mp) in enumerate(zip(
        debit['cv_fold_r2'], debit['cv_fold_mae_M'], debit['cv_fold_mape']), 1):
    print(f"    Fold {k}: R²={rn:.4f}  MAE={ma:.3f}M  MAPE={mp:.1f}%")
print(f"\n  XGBoost V3 — Hold-out test ({debit['ho_test_rows']} rows, supplemental):")
print(f"    R²  = {debit['ho_r2']:.4f}")
print(f"    MAE = {debit['ho_mae_M']:.3f}M")
print(f"    MAPE= {debit['ho_mape']:.1f}%")
print(f"    SMAPE={debit['ho_smape']:.1f}%")
print(f"    WMAPE={debit['ho_wmape']:.1f}%")
