Cell 19
# ── 4.1 Half-daily aggregation ────────────────────────────────────────────────
df['AM_PM'] = np.where(df['txn_hour'] < 12, 'AM', 'PM')

half_daily = df.groupby(['start_date', 'AM_PM', 'tran_br_code']).agg(
    Half_Day_Total_Debit  = ('TOTAL_DR', 'sum'),
    Half_Day_Total_Credit = ('TOTAL_CR', 'sum'),
    Txn_Count             = ('TOTAL_DR', 'count'),
    Weekday               = ('Weekday', 'first'),
    Is_Weekend            = ('Is_Weekend', 'first'),
    Month                 = ('start_date', lambda x: x.dt.month.iloc[0]),
    Day                   = ('start_date', lambda x: x.dt.day.iloc[0]),
).reset_index()

half_daily['Half_Day_Net_Cash'] = half_daily['Half_Day_Total_Credit'] - half_daily['Half_Day_Total_Debit']
half_daily['AM_PM_Encoded'] = np.where(half_daily['AM_PM'] == 'AM', 0, 1)

# Save RAW target for honest evaluation before capping
half_daily['Half_Day_Total_Debit_RAW'] = half_daily['Half_Day_Total_Debit']

# Outlier capping (Winsorization at 99th percentile per branch)
def cap_outliers(group):
    # Calculate 99th percentile ONLY on the first 80% (train set) to prevent data leakage!
    cutoff = int(len(group) * 0.8)
    cap = group['Half_Day_Total_Debit'].iloc[:cutoff].quantile(0.99)
    group['Half_Day_Total_Debit'] = group['Half_Day_Total_Debit'].clip(upper=cap)
    return group
half_daily = half_daily.groupby('tran_br_code', group_keys=False).apply(cap_outliers)


print(f"Half-daily records: {len(half_daily)}")
half_daily.head()
Cell 34
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
