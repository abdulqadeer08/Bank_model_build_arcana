"""verify_all_outputs.py — Quick verification of all updated files after pipeline completion."""
import json
import pandas as pd
import os
from datetime import datetime

print("=== Verification of all updated files ===")
print()

# 1. cv_results.json
with open("models/v3/cv_results.json") as f:
    cv = json.load(f)
d = cv["Half_Day_Total_Debit"]
print("1. cv_results.json (Debit target):")
print(f"   CV R2    = {d['cv_r2_mean']:.4f} +/- {d['cv_r2_std']:.4f}")
print(f"   CV MAE   = {d['cv_mae_mean_M']:.3f}M +/- {d['cv_mae_std_M']:.3f}M")
print(f"   CV RMSE  = {d['cv_rmse_mean_M']:.3f}M")
print(f"   CV SMAPE = {d['cv_smape_mean']:.1f}% +/- {d['cv_smape_std']:.1f}%")
print(f"   CV WMAPE = {d['cv_wmape_mean']:.1f}% +/- {d['cv_wmape_std']:.1f}%")
print(f"   Optuna trials = {d['n_optuna_trials']}")
print(f"   CV folds      = {d['n_cv_folds']}")
print(f"   Per-fold R2   = {d['cv_fold_r2']}")
print(f"   Per-fold MAE  = {d['cv_fold_mae_M']}")
print()

# 2. model_results.csv
mr = pd.read_csv("models/model_results.csv")
print("2. model_results.csv:")
for _, row in mr.iterrows():
    print(f"   {row['Model'][:55]:55s} MAE_M={row['MAE_M']:.2f}  WMAPE={row['WMAPE']:.1f}%  CV_R2={row.get('CV_R2_Mean', float('nan')):.4f}" if not pd.isna(row.get('CV_R2_Mean', float('nan'))) else f"   {row['Model'][:55]:55s} MAE_M={row['MAE_M']:.2f}  WMAPE={row['WMAPE']:.1f}%")
print()

# 3. final_evaluation_report.csv
fer = pd.read_csv("models/final_evaluation_report.csv")
print("3. final_evaluation_report.csv:")
for _, row in fer.iterrows():
    print(f"   Branch {row['Branch']:5} MAE={row['MAE_M']:.2f}M  WMAPE={row['WMAPE_%']:.1f}%  R2={row['R2']:.4f}  {row['Trust_Level']}")
print()

# 4. forecast_next30days.xlsx
print("4. forecast_next30days.xlsx:")
fc = pd.read_excel("models/forecast_next30days.xlsx", sheet_name="All_Branches")
print(f"   Rows: {len(fc)}, Branches: {fc['Branch'].nunique()}, Steps: 1-{fc['Step'].max()}")
print(f"   Confidence breakdown: {fc['Confidence'].value_counts().to_dict()}")
print()

# 5. shap_values.csv
sv = pd.read_csv("models/shap_values.csv")
top_feat = sv.abs().mean().sort_values(ascending=False)
print("5. shap_values.csv:")
print(f"   Shape: {sv.shape}")
print("   Top 5 features by mean |SHAP|:")
for feat, val in top_feat.head(5).items():
    print(f"     {feat}: {val:.4f}")
print()

# 6. Complete_Project_Documentation.md
with open("Complete_Project_Documentation.md", "r", encoding="utf-8") as f:
    doc = f.read()
has_note = "Why Metrics Changed" in doc
has_cv_table = "3-fold" in doc
print(f"6. Complete_Project_Documentation.md:")
print(f"   Methodology note present: {has_note}")
print(f"   CV table present: {has_cv_table}")
print()

# 7. Model pkl timestamps
print("7. Model pkl timestamps (all should be from today ~2026-08-08):")
for fn in ["models/v3/model_Half_Day_Total_Debit.pkl",
           "models/v3/model_Half_Day_Total_Credit.pkl",
           "models/v3/model_Half_Day_Net_Cash.pkl"]:
    mtime = os.path.getmtime(fn)
    print(f"   {os.path.basename(fn):45s}: {datetime.fromtimestamp(mtime)}")

print()
print("=== Verification complete ===")
