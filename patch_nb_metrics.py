"""
patch_nb_metrics.py — Programmatically add SMAPE/WMAPE to the notebook.

Modifies Bank_Cash_Optimization_Workflow.ipynb cells:
  Cell 29: evaluate() helper — add SMAPE/WMAPE computation and return values
  Cell 39: comparison table — automatically picks up new columns (no change needed)
  Cell 48: per-branch metrics — add SMAPE/WMAPE to the metrics dict
  Cell 50: business recommendations — add SMAPE/WMAPE to display columns

Run once:  python patch_nb_metrics.py
Then:      Restart Session > Run All in the notebook
"""

import json
import sys
import shutil

NB_PATH = 'Bank_Cash_Optimization_Workflow.ipynb'
BACKUP_PATH = 'Bank_Cash_Optimization_Workflow.ipynb.bak'

# Load notebook
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Backup
shutil.copy2(NB_PATH, BACKUP_PATH)
print(f"Backup saved to {BACKUP_PATH}")

# ============================================================================
# CELL 29: evaluate() helper — add SMAPE and WMAPE
# ============================================================================
CELL_29_NEW = [
    "# \u2500\u2500 Helper: evaluation metrics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "from metrics_utils import mape as calc_mape, smape as calc_smape, wmape as calc_wmape\n",
    "\n",
    "def evaluate(y_true, y_pred, model_name):\n",
    '    """Compute R2, MAE, MAPE, SMAPE, and WMAPE and return as a dict."""\n',
    "    mae  = mean_absolute_error(y_true, y_pred)\n",
    "    r2   = r2_score(y_true, y_pred)\n",
    "    mape_val = calc_mape(y_true, y_pred)\n",
    "    smape_val, smape_exc = calc_smape(y_true, y_pred)\n",
    "    wmape_val = calc_wmape(y_true, y_pred)\n",
    '    print(f"\\n-- {model_name} --")\n',
    '    print(f"   R2    = {r2:.4f}")\n',
    '    print(f"   MAE   = {mae/1e6:.2f} M PKR")\n',
    '    print(f"   MAPE  = {mape_val:.2f}%")\n',
    '    print(f"   SMAPE = {smape_val:.2f}%")\n',
    '    print(f"   WMAPE = {wmape_val:.2f}%  << recommended for bank presentations")\n',
    "    return {'Model': model_name, 'R2': round(r2, 4),\n",
    "            'MAE (M PKR)': round(mae / 1e6, 2), 'MAPE (%)': round(mape_val, 2),\n",
    "            'SMAPE (%)': round(smape_val, 2), 'WMAPE (%)': round(wmape_val, 2)}\n",
    "\n",
    "results = []   # collect dicts for the comparison table\n",
]

nb['cells'][29]['source'] = CELL_29_NEW
print("Patched Cell 29 (evaluate helper)")

# ============================================================================
# CELL 48: per-branch metrics — add SMAPE and WMAPE
# ============================================================================
CELL_48_NEW = [
    "# \u2500\u2500 7.1 Per-branch metrics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "test_eval = test_df[['tran_br_code']].copy()\n",
    "test_eval['y_actual']    = y_test_raw\n",
    "test_eval['y_predicted'] = y_pred_xgb\n",
    "\n",
    "branch_metrics = []\n",
    "for branch, grp in test_eval.groupby('tran_br_code'):\n",
    "    ya = grp['y_actual'].values\n",
    "    yp = grp['y_predicted'].values\n",
    "    mae  = mean_absolute_error(ya, yp)\n",
    "    r2   = r2_score(ya, yp) if len(ya) > 1 else np.nan\n",
    "    mape_val = calc_mape(ya, yp)\n",
    "    smape_val, _ = calc_smape(ya, yp)\n",
    "    wmape_val = calc_wmape(ya, yp)\n",
    "    avg_actual = ya.mean() / 1e6\n",
    "    branch_metrics.append({\n",
    "        'Branch': int(branch),\n",
    "        'Avg_Demand_M': round(avg_actual, 2),\n",
    "        'MAE_M':  round(mae / 1e6, 2),\n",
    "        'MAPE_%': round(mape_val, 1) if not np.isnan(mape_val) else 0.0,\n",
    "        'SMAPE_%': round(smape_val, 1) if not np.isnan(smape_val) else 0.0,\n",
    "        'WMAPE_%': round(wmape_val, 1) if not np.isnan(wmape_val) else 0.0,\n",
    "        'R2':     round(r2, 4),\n",
    "    })\n",
    "\n",
    "branch_df = pd.DataFrame(branch_metrics).sort_values('Avg_Demand_M', ascending=False)\n",
    "print('Per-Branch Evaluation (sorted by demand volume):')\n",
    "print(branch_df.to_string(index=False))\n",
    "\n",
    "best_branch  = branch_df.sort_values('MAE_M').iloc[0]\n",
    "worst_branch = branch_df.sort_values('MAE_M').iloc[-1]\n",
    "print(f'\\nBest  predicted branch: {int(best_branch[\"Branch\"])}  (MAE = {best_branch[\"MAE_M\"]}M PKR)')\n",
    "print(f'Worst predicted branch: {int(worst_branch[\"Branch\"])} (MAE = {worst_branch[\"MAE_M\"]}M PKR)')\n",
]

nb['cells'][48]['source'] = CELL_48_NEW
print("Patched Cell 48 (per-branch metrics)")

# ============================================================================
# CELL 50: business recommendations — add SMAPE/WMAPE to display
# ============================================================================
CELL_50_NEW = [
    "# \u2500\u2500 7.3 Business recommendations with trust levels \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "print('=' * 80)\n",
    "print('  BUSINESS RECOMMENDATIONS - BRANCH-WISE CASH STRATEGY')\n",
    "print('=' * 80)\n",
    "\n",
    "# Calculate MAE percentiles across branches to assign confidence tiers robustly\n",
    "p33 = branch_df['MAE_M'].quantile(0.33)\n",
    "p66 = branch_df['MAE_M'].quantile(0.66)\n",
    "\n",
    "recommendations = []\n",
    "for _, row in branch_df.iterrows():\n",
    "    mape_v = row['MAPE_%']\n",
    "    smape_v = row.get('SMAPE_%', 0.0)\n",
    "    wmape_v = row.get('WMAPE_%', 0.0)\n",
    "    mae  = row['MAE_M']\n",
    "    avg  = row['Avg_Demand_M']\n",
    "\n",
    "    # Safety buffer based on MAE percentiles to avoid MAPE distortion from near-zero days\n",
    "    if mae <= p33:\n",
    "        buffer_pct, trust, action = 5,  'HIGH',     'Use model directly for replenishment'\n",
    "    elif mae <= p66:\n",
    "        buffer_pct, trust, action = 12, 'MEDIUM',   'Add safety buffer, monitor weekly'\n",
    "    else:\n",
    "        buffer_pct, trust, action = 20, 'LOW', 'Use with caution, manual override advised'\n",
    "\n",
    "    safe_amount = avg * (1 + buffer_pct / 100)\n",
    "    recommendations.append({\n",
    "        'Branch': int(row['Branch']),\n",
    "        'Avg_Demand_M': avg,\n",
    "        'MAE_M': mae,\n",
    "        'MAPE_%': mape_v,\n",
    "        'SMAPE_%': smape_v,\n",
    "        'WMAPE_%': wmape_v,\n",
    "        'Trust_Level': trust,\n",
    "        'Buffer_%': buffer_pct,\n",
    "        'Recommended_M': round(safe_amount, 2),\n",
    "        'Action': action,\n",
    "    })\n",
    "\n",
    "rec_df = pd.DataFrame(recommendations).sort_values('Avg_Demand_M', ascending=False)\n",
    "print(rec_df[['Branch','Avg_Demand_M','MAE_M','MAPE_%','SMAPE_%','WMAPE_%','Trust_Level','Buffer_%','Recommended_M']].to_string(index=False))\n",
    "print()\n",
    "for _, r in rec_df.iterrows():\n",
    "    print(f'  Branch {int(r[\"Branch\"]):>5} -> {r[\"Action\"]}')\n",
    "print('=' * 80)\n",
]

nb['cells'][50]['source'] = CELL_50_NEW
print("Patched Cell 50 (business recommendations)")

# ============================================================================
# Save
# ============================================================================
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nNotebook patched and saved: {NB_PATH}")
print("Next step: Open the notebook -> Restart Session -> Run All")
