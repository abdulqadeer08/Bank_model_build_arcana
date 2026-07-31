import json
import codecs

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 1. Fix feature_cols missing tran_br_code
        if 'feature_cols = [' in source and 'AM_PM_Encoded' in source and 'tran_br_code' not in source:
            source = source.replace("'AM_PM_Encoded',", "'tran_br_code', 'AM_PM_Encoded',")
            if "half_daily['tran_br_code'] = " not in source:
                source += "\nhalf_daily['tran_br_code'] = half_daily['tran_br_code'].astype('category')\n"
            
        # 2. Fix train-test split logic to match v3_pipeline
        if 'unique_dates = np.sort(half_daily[' in source:
            new_split = '''# Match v3_pipeline.py exact split logic
half_daily = half_daily.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)
cutoff_idx = int(len(half_daily) * 0.8)
cutoff_date = half_daily.iloc[cutoff_idx]['start_date']

train_df = half_daily[half_daily['start_date'] < cutoff_date].copy()
test_df = half_daily[half_daily['start_date'] >= cutoff_date].copy()

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

print(f"Cutoff date : {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Train shape : {X_train.shape}")
print(f"Test  shape : {X_test.shape}")
'''
            source = new_split
            
        # 3. Fix XGBoost training to use hardcoded best params instead of inline optuna
        if 'def tune_xgboost' in source and 'study.optimize' in source:
            new_xgb = '''# ── 6.4 XGBoost V3 (Tuned matching production) ───────────────────────────
# Use exact hyperparams from v3_pipeline.py to match production
best_params = {
    'colsample_bytree': 0.6082643854734315,
    'learning_rate': 0.07184164822290748,
    'max_depth': 4,
    'min_child_weight': 5,
    'n_estimators': 450,
    'subsample': 0.6543262151830934,
    'enable_categorical': True,
    'random_state': 42
}

print('Training XGBoost V3 model with production hyperparams...')
xgb_model = xgb.XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train_log)
print('XGBoost V3 model trained.')

# Predict & inverse-transform
y_pred_log = xgb_model.predict(X_test)
y_pred_xgb = np.expm1(y_pred_log)

results.append(evaluate(y_test_raw, y_pred_xgb, 'XGBoost V3 (Tuned)'))
'''
            source = new_xgb
            
        cell['source'] = [line + '\n' for line in source.split('\n')]
        cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
        if cell['source'] and cell['source'][-1].endswith('\n'):
            cell['source'][-1] = cell['source'][-1][:-1]
            
with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched successfully!")
