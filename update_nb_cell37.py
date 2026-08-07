"""
update_nb_cell37.py
===================
After run_v3_fresh.py completes, this script:
1. Reads the fresh best_params from models/v3/best_params_debit.json
2. Updates notebook Cell 35 to use the same capping logic as v3_pipeline.py
3. Updates notebook Cell 37 with the exact best_params from current Optuna run
4. Saves the notebook

Run this AFTER run_v3_fresh.py has completed.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, os

# Load the fresh best_params from Optuna run
if not os.path.exists('models/v3/best_params_debit.json'):
    print("ERROR: models/v3/best_params_debit.json not found.")
    print("Run run_v3_fresh.py first.")
    sys.exit(1)

with open('models/v3/best_params_debit.json', 'r') as f:
    best_params = json.load(f)

print("Loaded fresh best_params from models/v3/best_params_debit.json:")
for k, v in best_params.items():
    print(f"  {k}: {v!r}")

# Build the new Cell 37 source
new_cell37_src = [
    "# ── 6.4 XGBoost V3 (Tuned — params from current Optuna run on current data) ──\n",
    "# These best_params are the result of running v3_pipeline.py fresh (20 Optuna trials,\n",
    "# TPESampler seed=42, TimeSeriesSplit n_splits=3, objective='reg:absoluteerror').\n",
    "# They are automatically updated by update_nb_cell37.py after each v3_pipeline run.\n",
    "import json as _json\n",
    "# Load params from file so they stay in sync with v3_pipeline output:\n",
    "try:\n",
    "    with open('models/v3/best_params_debit.json') as _f:\n",
    "        best_params = _json.load(_f)\n",
    "    print(f'Loaded best_params from models/v3/best_params_debit.json')\n",
    "except FileNotFoundError:\n",
    "    # Fallback hardcoded (updated " + str(__import__('datetime').date.today()) + "):\n",
]

# Add the hardcoded fallback dict
param_lines = "    best_params = {\n"
for k, v in best_params.items():
    param_lines += f"        {k!r}: {v!r},\n"
param_lines += "    }\n"

new_cell37_src.append(param_lines)
new_cell37_src += [
    "    print(f'Using hardcoded fallback best_params')\n",
    "\n",
    "print('Training XGBoost V3 model with current production hyperparams...')\n",
    "print(f'  n_estimators: {best_params[\"n_estimators\"]}')\n",
    "print(f'  learning_rate: {best_params[\"learning_rate\"]:.6f}')\n",
    "print(f'  max_depth: {best_params[\"max_depth\"]}')\n",
    "xgb_model = xgb.XGBRegressor(**best_params)\n",
    "xgb_model.fit(X_train, y_train_log)\n",
    "print('XGBoost V3 model trained.')\n",
    "\n",
    "# Predict & inverse-transform\n",
    "y_pred_log = xgb_model.predict(X_test)\n",
    "y_pred_xgb = np.expm1(y_pred_log)\n",
    "\n",
    "results.append(evaluate(y_test_raw, y_pred_xgb, 'XGBoost V3 (Tuned)'))\n",
]

print("\nNew Cell 37 source:")
print(''.join(new_cell37_src))

# Load notebook
with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Verify cell 37 is the XGBoost training cell
cell37_src = ''.join(cells[37].get('source', []))
if 'XGBoost' not in cell37_src or 'best_params' not in cell37_src:
    print(f"ERROR: Cell 37 doesn't look like the XGBoost cell. First line: {cell37_src.split(chr(10))[0]}")
    sys.exit(1)

print(f"\nOld Cell 37 first line: {cell37_src.split(chr(10))[0]}")

# Update Cell 37
cells[37]['source'] = new_cell37_src
cells[37]['outputs'] = []  # Clear stale outputs
cells[37]['execution_count'] = None

# Save notebook
with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook saved. Cell 37 updated with fresh Optuna params.")

# Verify by re-reading
with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
new_src = ''.join(nb2['cells'][37].get('source', []))
print(f"New Cell 37 first line: {new_src.split(chr(10))[0]}")
print(f"Total cells: {len(nb2['cells'])}")
