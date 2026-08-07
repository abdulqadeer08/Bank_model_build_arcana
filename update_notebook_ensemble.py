import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, os

cv_path = 'models/v3/cv_results.json'
if not os.path.exists(cv_path):
    print(f"ERROR: {cv_path} not found. Run v3_pipeline.py first.")
    sys.exit(1)

with open(cv_path) as f:
    cv = json.load(f)

debit = cv['Half_Day_Total_Debit']
bp = debit['best_params']
w_xgb = debit['w_xgb']

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

new_cell37 = (
    "# ── 6.4 XGBoost + LightGBM Ensemble (Tuned via Optuna + CV) ──\n"
    "# These parameters and blend weights were optimized simultaneously using a 3-fold\n"
    "# TimeSeriesSplit on the expanded feature set (EWMA, Month Boundaries, Days_Since_Salary).\n"
    f"# CV-averaged performance:\n"
    f"#   R²  = {debit['cv_r2_mean']:.4f} ± {debit['cv_r2_std']:.4f}\n"
    f"#   MAE = {debit['cv_mae_mean_M']:.3f}M ± {debit['cv_mae_std_M']:.3f}M\n"
    f"#   MAPE= {debit['cv_mape_mean']:.1f}% ± {debit['cv_mape_std']:.1f}%\n"
    "\n"
    f"best_params = {json.dumps(bp, indent=4)}\n"
    f"w_xgb = {w_xgb:.4f}\n"
    "\n"
    "xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}\n"
    "xgb_params['enable_categorical'] = True\n"
    "xgb_params['random_state'] = 42\n"
    "lgb_params = {k.replace('lgb_', ''): v for k, v in best_params.items() if k.startswith('lgb_')}\n"
    "lgb_params['random_state'] = 42\n"
    "lgb_params['verbose'] = -1\n"
    "\n"
    "print(f'Training XGBoost and LightGBM Ensemble (w_xgb={w_xgb:.2f})...')\n"
    "xgb_model = xgb.XGBRegressor(**xgb_params, objective='reg:absoluteerror')\n"
    "xgb_model.fit(X_train, y_train_log)\n"
    "lgb_model = lgb.LGBMRegressor(**lgb_params, objective='mae')\n"
    "lgb_model.fit(X_train, y_train_log)\n"
    "print('Ensemble trained.')\n"
    "\n"
    "# Predict & inverse-transform\n"
    "y_pred_log = w_xgb * xgb_model.predict(X_test) + (1 - w_xgb) * lgb_model.predict(X_test)\n"
    "y_pred_ens = np.expm1(y_pred_log)\n"
    "\n"
    "results.append(evaluate(y_test_raw, y_pred_ens, 'XGBoost + LightGBM Ensemble'))\n"
)

cells[37]['source'] = [new_cell37]
cells[37]['outputs'] = []
cells[37]['execution_count'] = None

# Clear stale outputs
for i in range(29, min(41, len(cells))):
    if cells[i]['cell_type'] == 'code':
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with ensemble params and code.")
