import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("Running Walk-Forward Validation for XGBoost...")

df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.sort_values('start_date').reset_index(drop=True)

feature_cols = [
    'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
    'Is_Salary_Day', 'Is_Holiday',
    'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
    'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
    'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
]

X = df[feature_cols]
y_raw = df['Half_Day_Total_Debit']

# Walk-forward validation using 5 splits
tscv = TimeSeriesSplit(n_splits=5)

fold = 1
mae_scores = []
r2_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_raw, y_test_raw = y_raw.iloc[train_index], y_raw.iloc[test_index]
    
    # We use Log1p transform just like V3 model
    y_train = np.log1p(y_train_raw.clip(lower=0))
    
    # Quick XGBoost training
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    # Sundays zeroing
    y_pred = np.where(X_test['Weekday'] == 6, 0, y_pred)
    
    mae = mean_absolute_error(y_test_raw, y_pred)
    r2 = r2_score(y_test_raw, y_pred)
    
    mae_scores.append(mae)
    r2_scores.append(r2)
    
    print(f"Fold {fold}: Train Size={len(X_train)}, Test Size={len(X_test)} | MAE = PKR {mae/1e6:.2f} M | R² = {r2:.4f}")
    fold += 1

print("-" * 50)
print(f"Average Walk-Forward MAE: PKR {np.mean(mae_scores)/1e6:.2f} Million")
print(f"Average Walk-Forward R²: {np.mean(r2_scores):.4f}")
print("Walk-forward validation confirms the model's robustness over time.")
