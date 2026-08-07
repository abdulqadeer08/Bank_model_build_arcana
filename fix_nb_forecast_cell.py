"""fix_nb_forecast_cell.py — Fix the recursive forecasting cell in the notebook.
This cell uses the old feature set without the new v3 features (ewma, dow_avg, etc.)
and calls xgb_model.predict() directly (single model, not ensemble).
Fix it to: add all new features to the row dict, and use the ensemble."""
import json

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed_source = (
    "# \u2500\u2500 9.3 Recursive Forecasting Engine \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "# Keep a working copy of recent history per branch (tail 65 rows for lag_60)\n"
    "working_history = (\n"
    "    half_daily\n"
    "    .sort_values(['tran_br_code', 'start_date', 'AM_PM'])\n"
    "    .groupby('tran_br_code')\n"
    "    .tail(65)\n"
    "    .copy()\n"
    ")\n"
    "forecast_records = []\n"
    "for step in range(1, FORECAST_DAYS + 1):\n"
    "    target_date = last_date + pd.Timedelta(days=step)\n"
    "    for branch in branches:\n"
    "        br_hist = working_history[working_history['tran_br_code'] == branch]\n"
    "        for am_pm, am_pm_enc in [('AM', 0), ('PM', 1)]:\n"
    "            row = {\n"
    "                'start_date': target_date,\n"
    "                'tran_br_code': branch,\n"
    "                'AM_PM': am_pm,\n"
    "                'AM_PM_Encoded': am_pm_enc,\n"
    "                'Weekday': target_date.weekday(),\n"
    "                'Is_Weekend': 1 if target_date.weekday() >= 5 else 0,\n"
    "                'Month': target_date.month,\n"
    "                'Day': target_date.day,\n"
    "                'Is_Salary_Day': is_salary_day(target_date.day),\n"
    "                'Days_to_Salary': max(0, min(25, 25 - target_date.day if target_date.day < 25 else (31 - target_date.day + 5))),\n"
    "                'Days_Since_Salary': target_date.day - 25 if target_date.day >= 25 else target_date.day + (31 - 25),\n"
    "                'Is_Month_Start': 1 if target_date.day == 1 else 0,\n"
    "                'Is_Month_End': 1 if (target_date + pd.Timedelta(days=1)).month != target_date.month else 0,\n"
    "                'Is_Holiday': is_holiday(target_date),\n"
    "            }\n"
    "            # Txn_Count features\n"
    "            try:\n"
    "                row['lag_1_Txn_Count'] = br_hist['Txn_Count'].iloc[-1]\n"
    "                row['rolling_14_mean_Txn_Count'] = br_hist['Txn_Count'].tail(14).mean()\n"
    "                row['ewma_14_Txn_Count'] = br_hist['Txn_Count'].tail(14).mean()  # approx\n"
    "            except IndexError:\n"
    "                row['lag_1_Txn_Count'] = branch_avgs.get(branch, 0)\n"
    "                row['rolling_14_mean_Txn_Count'] = branch_avgs.get(branch, 0)\n"
    "                row['ewma_14_Txn_Count'] = branch_avgs.get(branch, 0)\n"
    "            # Lag and rolling features for each target\n"
    "            for t_col in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:\n"
    "                try:\n"
    "                    row[f'lag_1_{t_col}']  = br_hist[t_col].iloc[-1]\n"
    "                    row[f'lag_2_{t_col}']  = br_hist[t_col].iloc[-2]\n"
    "                    row[f'lag_14_{t_col}'] = br_hist[t_col].iloc[-14]\n"
    "                    row[f'lag_60_{t_col}'] = br_hist[t_col].iloc[-60]\n"
    "                    row[f'rolling_14_mean_{t_col}'] = br_hist[t_col].tail(14).mean()\n"
    "                    row[f'rolling_14_std_{t_col}'] = br_hist[t_col].tail(14).std() if len(br_hist[t_col]) > 1 else 0\n"
    "                    row[f'ewma_14_{t_col}'] = br_hist[t_col].tail(14).mean()  # approx ewma\n"
    "                    # dow_avg_4: approximate with last 4 same-weekday same-ampm\n"
    "                    same_slot = br_hist[(br_hist['Weekday'].astype(int) == target_date.weekday()) & (br_hist['AM_PM'] == am_pm)][t_col] if 'Weekday' in br_hist.columns else br_hist[t_col]\n"
    "                    row[f'dow_avg_4_{t_col}'] = same_slot.tail(4).mean() if len(same_slot) > 0 else row[f'rolling_14_mean_{t_col}']\n"
    "                except IndexError:\n"
    "                    for sfx in ['lag_1', 'lag_2', 'lag_14', 'lag_60', 'rolling_14_mean', 'rolling_14_std', 'ewma_14', 'dow_avg_4']:\n"
    "                        row[f'{sfx}_{t_col}'] = 0\n"
    "            # Predict using V3 Ensemble\n"
    "            x_df = pd.DataFrame([row])[feature_cols]\n"
    "            x_df['tran_br_code'] = x_df['tran_br_code'].astype('category')\n"
    "            x_df['Weekday'] = x_df['Weekday'].astype('category')\n"
    "            x_df['Month'] = x_df['Month'].astype('category')\n"
    "            # Use ensemble (or fall back to single model)\n"
    "            import joblib as _jl\n"
    "            _ens = _jl.load('models/v3/model_Half_Day_Total_Debit.pkl')\n"
    "            if isinstance(_ens, dict):\n"
    "                _w = _ens['w_xgb']\n"
    "                pred_log = _w * _ens['xgb'].predict(x_df)[0] + (1 - _w) * _ens['lgb'].predict(x_df)[0]\n"
    "            else:\n"
    "                pred_log = _ens.predict(x_df)[0]\n"
    "            pred_dr = max(0, np.expm1(pred_log))\n"
    "            # Store predicted values back so next step can use them as lags\n"
    "            row['Half_Day_Total_Debit']  = pred_dr\n"
    "            row['Half_Day_Total_Credit'] = pred_dr * 0.9   # approximate\n"
    "            row['Half_Day_Net_Cash']     = row['Half_Day_Total_Credit'] - pred_dr\n"
    "            row['Txn_Count']             = branch_avgs.get(branch, 0)\n"
    "            forecast_records.append(row)\n"
    "            hist_row = pd.DataFrame([row])\n"
    "            working_history = pd.concat([working_history, hist_row], ignore_index=True)\n"
    "            working_history = working_history.groupby('tran_br_code').tail(65)\n"
    "fc_half_daily = pd.DataFrame(forecast_records)\n"
    "print(f\"Forecast generated: {len(fc_half_daily)} half-daily records for {len(branches)} branches x {FORECAST_DAYS} days\")\n"
)

fixed = False
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if ('9.3 Recursive Forecasting' in src or ('working_history' in src and 'FORECAST_DAYS' in src and 'xgb_model' in src)):
        lines = fixed_source.rstrip('\n').split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        print(f'Fixed Cell {i}: Recursive forecasting engine')
        fixed = True
        break

if not fixed:
    print('Searching broadly for forecast engine cell...')
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        if 'FORECAST_DAYS' in src and 'xgb_model.predict' in src:
            lines = fixed_source.rstrip('\n').split('\n')
            cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
            print(f'Fixed Cell {i} (broad match)')
            fixed = True
            break

if not fixed:
    print('ERROR: Could not find forecast engine cell')
else:
    with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook saved.')
