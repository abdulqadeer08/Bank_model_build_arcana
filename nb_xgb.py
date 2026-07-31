
# --- CELL 26 ---
# ── 5. Chronological train-test split ─────────────────────────────────────────
feature_cols = [
    'AM_PM_Encoded', 'lag_1_Txn_Count', 'rolling_14_mean_Txn_Count', 'Days_to_Salary', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    # Debit lags
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit',
    'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit',
    'rolling_14_mean_Half_Day_Total_Debit', 'rolling_14_std_Half_Day_Total_Debit',
    # Credit lags
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit',
    'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit',
    'rolling_14_mean_Half_Day_Total_Credit', 'rolling_14_std_Half_Day_Total_Credit',
    # Net-cash lags
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash',
    'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash',
    'rolling_14_mean_Half_Day_Net_Cash', 'rolling_14_std_Half_Day_Net_Cash',
]

# Fix: Calculate cutoff date based on unique dates to ensure an actual time-split across all branches
unique_dates = np.sort(half_daily['start_date'].unique())
cutoff_idx = int(len(unique_dates) * 0.8)
cutoff_date = pd.to_datetime(unique_dates[cutoff_idx])

train_df = half_daily[half_daily['start_date'] <  cutoff_date].copy()
test_df  = half_daily[half_daily['start_date'] >= cutoff_date].copy()

# Fix: Sort by time to ensure TimeSeriesSplit in Optuna works correctly (past -> future)
train_df = train_df.sort_values(['start_date', 'AM_PM']).reset_index(drop=True)
test_df = test_df.sort_values(['start_date', 'AM_PM']).reset_index(drop=True)

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

print(f"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Train shape : {X_train.shape}")
print(f"Test  shape : {X_test.shape}")
# --- CELL 34 ---
# ── 6.3 LightGBM ─────────────────────────────────────────────────────────────
y_train_raw = train_df['Half_Day_Total_Debit'].values
y_test_raw  = test_df['Half_Day_Total_Debit_RAW'].values  # Evaluate on RAW, uncapped data!

# log1p transformation (same as XGBoost V3)
y_train_log = np.log1p(np.clip(y_train_raw, 0, None))

lgb_model = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 63,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

lgb_model.fit(X_train, y_train_log)
print("LightGBM model trained.")

# Predict & inverse-transform
y_pred_lgb_log = lgb_model.predict(X_test)
y_pred_lgb = np.expm1(y_pred_lgb_log)

results.append(evaluate(y_test_raw, y_pred_lgb, 'LightGBM'))
# --- CELL 36 ---
# ── 6.4 XGBoost V3 (Dynamic Optuna Tuning for MAE) ───────────────────────────

print('Tuning XGBoost hyperparameters for MAE...')
def tune_xgboost(X_tr, y_tr):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42,
            'objective': 'reg:absoluteerror'
        }
        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for train_index, test_index in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[train_index], X_tr.iloc[test_index]
            y_t, y_v = y_tr[train_index], y_tr[test_index]
            model = xgb.XGBRegressor(enable_categorical=True, **params)
            model.fit(X_t, y_t, verbose=False)
            preds = model.predict(X_v)
            maes.append(mean_absolute_error(y_v, preds))
        return np.mean(maes)

    study = optuna.create_study(direction='minimize')
    # Run a quick 10 trials for demonstration (increase n_trials for production)
    study.optimize(objective, n_trials=10)
    return study.best_params

# Run tuner
best_params = tune_xgboost(X_train, y_train_log)
best_params['random_state'] = 42
best_params['objective'] = 'reg:absoluteerror'
print(f'Best params found: {best_params}')

# Train final model with best params
xgb_model = xgb.XGBRegressor(enable_categorical=True, **best_params)
xgb_model.fit(X_train, y_train_log)
print('XGBoost V3 model trained.')

# Predict & inverse-transform
y_pred_log = xgb_model.predict(X_test)
y_pred_xgb = np.expm1(y_pred_log)

results.append(evaluate(y_test_raw, y_pred_xgb, 'XGBoost V3 (Tuned)'))

# --- CELL 49 ---
# ── 8.3 SHAP Dependence Plot (top feature) ───────────────────────────────────
mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
top_feature = mean_shap.index[0]

fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot(top_feature, shap_values, X_test_sample, show=False, ax=ax)
ax.set_title(f'SHAP Dependence Plot - {top_feature}', fontsize=13, weight='bold')
plt.tight_layout()
plt.show()
# --- CELL 50 ---
# ── 8.4 Waterfall plot for a single prediction ────────────────────────────────
# Pick the first test sample and explain it
idx = 0
sample_row = X_test_sample.iloc[idx]
sample_shap = shap_values[idx]
base_value = explainer.expected_value

print("Explaining a single prediction in plain terms:")
print(f"  Base value (avg log-debit): {base_value:.4f}")
print(f"  Predicted log-debit:        {base_value + sample_shap.sum():.4f}")
print(f"  Predicted debit (PKR):      {np.expm1(base_value + sample_shap.sum()):,.0f}")
print()

# Show waterfall
shap_explanation = shap.Explanation(
    values=sample_shap,
    base_values=base_value,
    data=sample_row.values,
    feature_names=feature_cols
)
plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_explanation, show=False)
plt.title('SHAP Waterfall - Single Prediction Breakdown', fontsize=13, weight='bold')
plt.tight_layout()
plt.show()
# --- CELL 55 ---
# ── 9.3 Recursive Forecasting Engine ──────────────────────────────────────────
# Keep a working copy of recent history per branch (tail 65 rows for lag_60)
working_history = (
    half_daily
    .sort_values(['tran_br_code', 'start_date', 'AM_PM'])
    .groupby('tran_br_code')
    .tail(65)
    .copy()
)

forecast_records = []

for step in range(1, FORECAST_DAYS + 1):
    target_date = last_date + pd.Timedelta(days=step)

    for branch in branches:
        br_hist = working_history[working_history['tran_br_code'] == branch]

        for am_pm, am_pm_enc in [('AM', 0), ('PM', 1)]:
            row = {
                'start_date': target_date,
                'tran_br_code': branch,
                'AM_PM': am_pm,
                'AM_PM_Encoded': am_pm_enc,
                
                'Weekday': target_date.weekday(),
                'Is_Weekend': 1 if target_date.weekday() >= 5 else 0,
                'Month': target_date.month,
                'Day': target_date.day,
                'Is_Salary_Day': is_salary_day(target_date.day),
                'Days_to_Salary': max(0, min(25, 25 - target_date.day if target_date.day < 25 else (31 - target_date.day + 5))),
                'Is_Holiday': is_holiday(target_date),
            }

            # Fetch lags from the working history queue

            try:
                row['lag_1_Txn_Count'] = br_hist['Txn_Count'].iloc[-1]
                row['rolling_14_mean_Txn_Count'] = br_hist['Txn_Count'].tail(14).mean()
            except IndexError:
                row['lag_1_Txn_Count'] = branch_avgs.get(branch, 0)
                row['rolling_14_mean_Txn_Count'] = branch_avgs.get(branch, 0)
            for t_col in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
                try:
                    row[f'lag_1_{t_col}']  = br_hist[t_col].iloc[-1]
                    row[f'lag_2_{t_col}']  = br_hist[t_col].iloc[-2]
                    row[f'lag_14_{t_col}'] = br_hist[t_col].iloc[-14]
                    row[f'lag_60_{t_col}'] = br_hist[t_col].iloc[-60]
                    row[f'rolling_14_mean_{t_col}'] = br_hist[t_col].tail(14).mean()
                    row[f'rolling_14_std_{t_col}'] = br_hist[t_col].tail(14).std() if len(br_hist[t_col]) > 1 else 0
                except IndexError:
                    row[f'lag_1_{t_col}']  = 0
                    row[f'lag_2_{t_col}']  = 0
                    row[f'lag_14_{t_col}'] = 0
                    row[f'lag_60_{t_col}'] = 0
                    row[f'rolling_14_mean_{t_col}'] = 0
                    row[f'rolling_14_std_{t_col}'] = 0

            # Predict using XGBoost V3 (log-space -> inverse)
            x_df = pd.DataFrame([row])[feature_cols]
            pred_log = xgb_model.predict(x_df)[0]
            pred_dr = max(0, np.expm1(pred_log))

            # Store predicted values back so next step can use them as lags
            row['Half_Day_Total_Debit']  = pred_dr
            row['Half_Day_Total_Credit'] = pred_dr * 0.9   # approximate
            row['Half_Day_Net_Cash']     = row['Half_Day_Total_Credit'] - pred_dr
            row['Txn_Count']             = branch_avgs.get(branch, 0)

            forecast_records.append(row)

            # Append to working history and keep it trimmed
            hist_row = pd.DataFrame([row])
            working_history = pd.concat([working_history, hist_row], ignore_index=True)
            working_history = working_history.groupby('tran_br_code').tail(65)

fc_half_daily = pd.DataFrame(forecast_records)
print(f"Forecast generated: {len(fc_half_daily)} half-daily records for {len(branches)} branches x {FORECAST_DAYS} days")