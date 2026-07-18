import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import optuna
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Starting V3 Advanced Pipeline (Optuna + Log Transform)...")

# 1. Load Preprocessed Data (from V2)
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
print(f"Loaded {len(df)} rows of data.")

feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

# Chronological split
cutoff_idx = int(len(df) * 0.8)
cutoff_date = df.iloc[cutoff_idx]['start_date']

train_df = df[df['start_date'] < cutoff_date]
test_df = df[df['start_date'] >= cutoff_date]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

os.makedirs('models/v3', exist_ok=True)
os.makedirs('eda_plots/v3', exist_ok=True)

models = {}

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:
    print(f"\n{'='*50}\nTraining Advanced Model for: {target}\n{'='*50}")
    
    y_train_raw = train_df[target]
    y_test_raw = test_df[target]
    
    # 2. Target Transformation (Log1p for stabilizing variance)
    # Note: Net Cash can be negative, so we use a shift if needed.
    # To keep it simple and safe for net cash, we won't log-transform Net Cash, only Credit and Debit (which are strictly >= 0)
    
    use_log = target != 'Half_Day_Net_Cash'
    
    if use_log:
        y_train = np.log1p(y_train_raw.clip(lower=0)) # Ensure no negatives just in case
    else:
        y_train = y_train_raw
        
    # 3. Hyperparameter Tuning with Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42
        }
        
        # We use TimeSeriesSplit for robust tuning
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        errors = []
        for train_index, val_index in tscv.split(X_train):
            X_t, X_v = X_train.iloc[train_index], X_train.iloc[val_index]
            y_t, y_v = y_train.iloc[train_index], y_train.iloc[val_index]
            
            model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            
            preds = model.predict(X_v)
            if use_log:
                preds_orig = np.expm1(preds)
                y_v_orig = np.expm1(y_v)
                error = mean_absolute_error(y_v_orig, preds_orig)
            else:
                error = mean_absolute_error(y_v, preds)
            errors.append(error)
            
        return np.mean(errors)

    print(f"Running Optuna Optimization for {target} (20 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    print(f"Best Optuna Parameters: {best_params}")
    
    # 4. Train Final Model on all training data with best params
    print("Training final model with best parameters...")
    final_model = xgb.XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    models[target] = final_model
    
    # Save Model
    joblib.dump(final_model, f'models/v3/model_{target}.pkl')
    
    # 5. Evaluate
    y_pred_transformed = final_model.predict(X_test)
    
    # Reverse transformation
    if use_log:
        y_pred = np.expm1(y_pred_transformed)
    else:
        y_pred = y_pred_transformed
        
    # Enforce 0 for Sundays
    y_pred = np.where(X_test['Weekday'] == 6, 0, y_pred)
    
    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2 = r2_score(y_test_raw, y_pred)
    
    print(f"\nFinal Metrics on Test Set for {target}:")
    print(f"  MAE: {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R2: {r2:,.4f}")
    
    if use_log:
        print(f"  MAPE: {mape(y_test_raw, y_pred):.2f}%")

# Save a config indicating we use log transform
config = {'targets_using_log1p': ['Half_Day_Total_Debit', 'Half_Day_Total_Credit']}
with open('models/v3/v3_config.json', 'w') as f:
    json.dump(config, f)

print("\nV3 Pipeline completed successfully. Models saved to models/v3/")
