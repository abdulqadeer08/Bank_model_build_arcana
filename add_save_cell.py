"""
add_save_cell.py
Inserts a new code cell after Cell 39 in the notebook that saves 
the comparison results to models/model_results.csv with dashboard-compatible
column names.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import copy

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# The new cell content
new_cell_source = [
    "# ── Save model_results.csv (for dashboard) ────────────────────────────────\n",
    "# Convert the comparison DataFrame to dashboard-compatible column names and save.\n",
    "import os as _os\n",
    "_os.makedirs('models', exist_ok=True)\n",
    "\n",
    "# Rename columns to match what dashboard.py expects\n",
    "_col_map = {\n",
    "    'MAE (M PKR)': 'MAE_M',\n",
    "    'MAPE (%)': 'MAPE',\n",
    "    'SMAPE (%)': 'SMAPE',\n",
    "    'WMAPE (%)': 'WMAPE',\n",
    "}\n",
    "model_results_df = comparison.rename(columns=_col_map).copy()\n",
    "# Add MAE in raw PKR for completeness\n",
    "if 'MAE_M' in model_results_df.columns:\n",
    "    model_results_df['MAE'] = (model_results_df['MAE_M'] * 1e6).round(2)\n",
    "model_results_df.to_csv('models/model_results.csv', index=False)\n",
    "print('Saved models/model_results.csv')\n",
    "print(model_results_df[['Model', 'R2', 'MAE_M', 'MAPE', 'SMAPE', 'WMAPE']].to_string(index=False))\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

# Insert after Cell 39 (index 39), so at position 40
cells.insert(40, new_cell)

print(f"Total cells before: {len(nb['cells'])}")

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Reload and verify
with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

print(f"Total cells after: {len(nb2['cells'])}")
print(f"\nNew cell 40 source:")
print(''.join(nb2['cells'][40].get('source', [])))
