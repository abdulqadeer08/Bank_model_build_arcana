"""
ADF Stationarity Test — Investigating High MAPE (132.7%)
=========================================================
Tests whether Half_Day_Total_Debit is stationary at:
  (a) Overall level (all branches combined, sorted by date)
  (b) Per-branch level (15 branches individually)
  (c) On log1p-transformed target

Also generates time-series plots for visual inspection.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
import os

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# 1. Load Data (identical to v3_pipeline.py)
# ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)

print(f"Loaded {len(df)} rows, {df['tran_br_code'].nunique()} branches")
print(f"Date range: {df['start_date'].min()} to {df['start_date'].max()}")

os.makedirs('eda_plots/v3', exist_ok=True)

TARGET = 'Half_Day_Total_Debit'

# ──────────────────────────────────────────────────────────────────────
# 2. ADF Test Helper
# ──────────────────────────────────────────────────────────────────────
def run_adf(series, label=""):
    """Run ADF test and return dict with results."""
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return {'label': label, 'adf_stat': np.nan, 'p_value': np.nan, 
                'verdict': 'INSUFFICIENT DATA', 'n_obs': len(series_clean)}
    
    result = adfuller(series_clean, autolag='AIC')
    adf_stat = result[0]
    p_value = result[1]
    verdict = 'STATIONARY' if p_value < 0.05 else 'NON-STATIONARY'
    
    return {
        'label': label,
        'adf_stat': round(adf_stat, 4),
        'p_value': round(p_value, 6),
        'verdict': verdict,
        'n_obs': len(series_clean)
    }

# ──────────────────────────────────────────────────────────────────────
# 3. Overall ADF Tests (all branches combined, sorted by date)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("OVERALL ADF TESTS (All branches combined, sorted by date)")
print("="*60)

overall_series = df.sort_values('start_date')[TARGET]
overall_log = np.log1p(overall_series.clip(lower=0))

overall_raw_result = run_adf(overall_series, "Overall (Raw)")
overall_log_result = run_adf(overall_log, "Overall (log1p)")

for r in [overall_raw_result, overall_log_result]:
    print(f"  {r['label']:30s} | ADF={r['adf_stat']:8.4f} | p={r['p_value']:.6f} | {r['verdict']} | n={r['n_obs']}")

# ──────────────────────────────────────────────────────────────────────
# 4. Per-Branch ADF Tests
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PER-BRANCH ADF TESTS")
print("="*60)

branches = sorted(df['tran_br_code'].unique())
all_results = []

for br in branches:
    br_df = df[df['tran_br_code'] == br].sort_values(['start_date', 'AM_PM_Encoded'])
    raw_series = br_df[TARGET]
    log_series = np.log1p(raw_series.clip(lower=0))
    
    raw_res = run_adf(raw_series, f"Branch {br} (Raw)")
    log_res = run_adf(log_series, f"Branch {br} (log1p)")
    
    all_results.append({
        'Branch': br,
        'ADF_Stat_Raw': raw_res['adf_stat'],
        'p_value_Raw': raw_res['p_value'],
        'Verdict_Raw': raw_res['verdict'],
        'ADF_Stat_Log': log_res['adf_stat'],
        'p_value_Log': log_res['p_value'],
        'Verdict_Log': log_res['verdict'],
        'n_obs': raw_res['n_obs']
    })

# Add overall results
all_results.insert(0, {
    'Branch': 'OVERALL',
    'ADF_Stat_Raw': overall_raw_result['adf_stat'],
    'p_value_Raw': overall_raw_result['p_value'],
    'Verdict_Raw': overall_raw_result['verdict'],
    'ADF_Stat_Log': overall_log_result['adf_stat'],
    'p_value_Log': overall_log_result['p_value'],
    'Verdict_Log': overall_log_result['verdict'],
    'n_obs': overall_raw_result['n_obs']
})

results_df = pd.DataFrame(all_results)

# Pretty-print table
print(f"\n{'Branch':>10} | {'ADF p (raw)':>12} | {'ADF p (log1p)':>14} | {'Verdict (raw)':>16} | {'Verdict (log1p)':>18}")
print("-" * 85)
for _, row in results_df.iterrows():
    print(f"{str(row['Branch']):>10} | {row['p_value_Raw']:>12.6f} | {row['p_value_Log']:>14.6f} | {row['Verdict_Raw']:>16} | {row['Verdict_Log']:>18}")

# Summary
n_raw_stationary = (results_df[results_df['Branch'] != 'OVERALL']['Verdict_Raw'] == 'STATIONARY').sum()
n_log_stationary = (results_df[results_df['Branch'] != 'OVERALL']['Verdict_Log'] == 'STATIONARY').sum()
n_branches = len(branches)

print(f"\nSUMMARY:")
print(f"  Raw target:  {n_raw_stationary}/{n_branches} branches are stationary (p < 0.05)")
print(f"  Log1p target: {n_log_stationary}/{n_branches} branches are stationary (p < 0.05)")

# Save results
results_df.to_csv('models/adf_test_results.csv', index=False)
print(f"\nResults saved to models/adf_test_results.csv")

# ──────────────────────────────────────────────────────────────────────
# 5. Visual Plots — Overall Time Series (Raw + Log-Transformed)
# ──────────────────────────────────────────────────────────────────────
print("\nGenerating stationarity plots...")

# Aggregate to daily for cleaner visualization
daily_overall = df.groupby('start_date')[TARGET].sum().sort_index()
daily_overall_log = np.log1p(daily_overall.clip(lower=0))

fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=100)

# Plot 1: Raw series
axes[0].plot(daily_overall.index, daily_overall.values / 1e6, color='#3B82F6', linewidth=0.8, alpha=0.8)
axes[0].set_title(f'Overall Daily {TARGET} (Raw) — All Branches Combined', fontsize=14, fontweight='bold')
axes[0].set_ylabel('PKR (Millions)', fontsize=12)
axes[0].set_xlabel('')
axes[0].grid(True, alpha=0.3)
# Add rolling mean to show trend
if len(daily_overall) > 30:
    rolling_mean = daily_overall.rolling(30).mean() / 1e6
    axes[0].plot(daily_overall.index, rolling_mean.values, color='#EF4444', linewidth=2, label='30-day Rolling Mean')
    axes[0].legend(fontsize=10)

# Plot 2: Log-transformed series
axes[1].plot(daily_overall_log.index, daily_overall_log.values, color='#10B981', linewidth=0.8, alpha=0.8)
axes[1].set_title(f'Overall Daily {TARGET} (log1p Transformed)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('log1p(PKR)', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].grid(True, alpha=0.3)
if len(daily_overall_log) > 30:
    rolling_mean_log = daily_overall_log.rolling(30).mean()
    axes[1].plot(daily_overall_log.index, rolling_mean_log.values, color='#EF4444', linewidth=2, label='30-day Rolling Mean')
    axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('eda_plots/v3/adf_overall_timeseries.png', bbox_inches='tight')
plt.close()
print("Saved: eda_plots/v3/adf_overall_timeseries.png")

# Plot per-branch (grid of small multiples)
n_branches_plot = len(branches)
ncols = 5
nrows = (n_branches_plot + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), dpi=80)
axes_flat = axes.flatten()

for idx, br in enumerate(branches):
    ax = axes_flat[idx]
    br_df = df[df['tran_br_code'] == br].sort_values(['start_date', 'AM_PM_Encoded'])
    br_daily = br_df.groupby('start_date')[TARGET].sum().sort_index()
    
    ax.plot(br_daily.index, br_daily.values / 1e6, linewidth=0.6, alpha=0.7)
    
    # Get this branch's verdict
    br_row = results_df[results_df['Branch'] == br].iloc[0]
    color = '#10B981' if br_row['Verdict_Raw'] == 'STATIONARY' else '#EF4444'
    ax.set_title(f"Br {br} (p={br_row['p_value_Raw']:.4f})", fontsize=10, color=color, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.2)

# Hide unused subplots
for idx in range(len(branches), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle('Per-Branch Time Series with ADF p-values (Green=Stationary, Red=Non-Stationary)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_plots/v3/adf_per_branch_timeseries.png', bbox_inches='tight')
plt.close()
print("Saved: eda_plots/v3/adf_per_branch_timeseries.png")

print("\n" + "="*60)
print("ADF STATIONARITY ANALYSIS COMPLETE")
print("="*60)
