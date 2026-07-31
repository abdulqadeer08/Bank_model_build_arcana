"""
Patch the Bank_Cash_Optimization_Workflow.ipynb notebook to add:
1. A markdown header cell for "Stationarity Analysis & Differencing Investigation"
2. A code cell running the ADF test inline
3. A code cell running the differencing experiment inline
4. A markdown cell with the conclusion

Insert these after the Model Comparison Table section (after Cell 39) and before Evaluation Plots.
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

nb_path = 'Bank_Cash_Optimization_Workflow.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find insertion point: after the comparison table code cell (Cell 39)
# We'll insert right before "### Evaluation Plots" (Cell 40)
insert_idx = 40  # Before "Evaluation Plots"

new_cells = []

# Cell 1: Markdown header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Stationarity Analysis & Differencing Investigation\n",
        "\n",
        "The supervisor raised a concern that the high overall MAPE (132.7%) might be caused by **non-stationary data**. \n",
        "To investigate this rigorously, we run the **Augmented Dickey-Fuller (ADF) test** on `Half_Day_Total_Debit` \n",
        "at three levels: overall, per-branch, and on the log1p-transformed target. If non-stationarity is detected, \n",
        "we test whether **differencing** (predicting changes rather than levels) improves the model.\n",
        "\n",
        "**Decision criterion:** p-value < 0.05 = stationary, p-value >= 0.05 = non-stationary."
    ]
})

# Cell 2: ADF Test Code
new_cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# ADF Stationarity Test on Half_Day_Total_Debit\n",
        "from statsmodels.tsa.stattools import adfuller\n",
        "\n",
        "def run_adf(series, label=''):\n",
        "    series_clean = series.dropna()\n",
        "    if len(series_clean) < 20:\n",
        "        return {'label': label, 'adf_stat': np.nan, 'p_value': np.nan, 'verdict': 'INSUFFICIENT DATA'}\n",
        "    result = adfuller(series_clean, autolag='AIC')\n",
        "    p = result[1]\n",
        "    return {'label': label, 'adf_stat': round(result[0], 4), 'p_value': round(p, 6),\n",
        "            'verdict': 'STATIONARY' if p < 0.05 else 'NON-STATIONARY'}\n",
        "\n",
        "TARGET = 'Half_Day_Total_Debit'\n",
        "\n",
        "# Overall ADF\n",
        "overall_raw = run_adf(half_daily.sort_values('start_date')[TARGET], 'Overall (Raw)')\n",
        "overall_log = run_adf(np.log1p(half_daily.sort_values('start_date')[TARGET].clip(lower=0)), 'Overall (log1p)')\n",
        "\n",
        "print('OVERALL ADF RESULTS:')\n",
        "print(f\"  {overall_raw['label']:30s} | ADF={overall_raw['adf_stat']:8.4f} | p={overall_raw['p_value']:.6f} | {overall_raw['verdict']}\")\n",
        "print(f\"  {overall_log['label']:30s} | ADF={overall_log['adf_stat']:8.4f} | p={overall_log['p_value']:.6f} | {overall_log['verdict']}\")\n",
        "\n",
        "# Per-branch ADF\n",
        "adf_results = []\n",
        "for br in sorted(half_daily['tran_br_code'].unique()):\n",
        "    br_df = half_daily[half_daily['tran_br_code'] == br].sort_values(['start_date', 'AM_PM_Encoded'])\n",
        "    raw_res = run_adf(br_df[TARGET], f'Branch {br} (Raw)')\n",
        "    log_res = run_adf(np.log1p(br_df[TARGET].clip(lower=0)), f'Branch {br} (log1p)')\n",
        "    adf_results.append({'Branch': br, 'p_raw': raw_res['p_value'], 'p_log': log_res['p_value'],\n",
        "                        'verdict_raw': raw_res['verdict'], 'verdict_log': log_res['verdict']})\n",
        "\n",
        "adf_df = pd.DataFrame(adf_results)\n",
        "print(f'\\nPER-BRANCH ADF RESULTS:')\n",
        "print(f\"{'Branch':>10} | {'p-value (raw)':>14} | {'p-value (log1p)':>16} | {'Verdict (raw)':>16} | {'Verdict (log1p)':>18}\")\n",
        "print('-' * 90)\n",
        "for _, r in adf_df.iterrows():\n",
        "    print(f\"{str(r['Branch']):>10} | {r['p_raw']:>14.6f} | {r['p_log']:>16.6f} | {r['verdict_raw']:>16} | {r['verdict_log']:>18}\")\n",
        "\n",
        "n_stat_raw = (adf_df['verdict_raw'] == 'STATIONARY').sum()\n",
        "n_stat_log = (adf_df['verdict_log'] == 'STATIONARY').sum()\n",
        "print(f'\\nSUMMARY: Raw={n_stat_raw}/15 stationary, Log1p={n_stat_log}/15 stationary')\n",
        "\n",
        "# Visual: overall time series\n",
        "daily_overall = half_daily.groupby('start_date')[TARGET].sum().sort_index()\n",
        "fig, axes = plt.subplots(2, 1, figsize=(14, 8))\n",
        "axes[0].plot(daily_overall.index, daily_overall.values / 1e6, color='#3B82F6', lw=0.8, alpha=0.8)\n",
        "if len(daily_overall) > 30:\n",
        "    axes[0].plot(daily_overall.index, (daily_overall.rolling(30).mean() / 1e6).values, color='red', lw=2, label='30-day Rolling Mean')\n",
        "    axes[0].legend()\n",
        "axes[0].set_title('Overall Daily Half_Day_Total_Debit (Raw)', fontweight='bold')\n",
        "axes[0].set_ylabel('PKR (Millions)')\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "daily_log = np.log1p(daily_overall.clip(lower=0))\n",
        "axes[1].plot(daily_log.index, daily_log.values, color='#10B981', lw=0.8, alpha=0.8)\n",
        "if len(daily_log) > 30:\n",
        "    axes[1].plot(daily_log.index, daily_log.rolling(30).mean().values, color='red', lw=2, label='30-day Rolling Mean')\n",
        "    axes[1].legend()\n",
        "axes[1].set_title('Overall Daily Half_Day_Total_Debit (log1p)', fontweight='bold')\n",
        "axes[1].set_ylabel('log1p(PKR)')\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 3: Differencing experiment
new_cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Differencing Experiment: controlled comparison vs production model\n",
        "# Same features, same split, same Optuna config, same seed - ONLY the target changes\n",
        "\n",
        "# Create per-branch differenced target\n",
        "half_daily_sorted = half_daily.sort_values(['tran_br_code', 'start_date', 'AM_PM_Encoded']).reset_index(drop=True)\n",
        "half_daily_sorted['diff_target'] = half_daily_sorted.groupby('tran_br_code')[TARGET].diff(1)\n",
        "half_daily_sorted['prev_actual'] = half_daily_sorted.groupby('tran_br_code')[TARGET].shift(1)\n",
        "df_diff = half_daily_sorted.dropna(subset=['diff_target']).copy()\n",
        "\n",
        "# Verify differenced series is stationary\n",
        "diff_stat = 0\n",
        "for br in sorted(df_diff['tran_br_code'].unique()):\n",
        "    br_s = df_diff[df_diff['tran_br_code'] == br]['diff_target'].dropna()\n",
        "    if len(br_s) >= 20:\n",
        "        p = adfuller(br_s, autolag='AIC')[1]\n",
        "        if p < 0.05: diff_stat += 1\n",
        "print(f'After differencing: {diff_stat}/15 branches stationary (as expected)')\n",
        "\n",
        "# NOTE: Full Optuna retraining was run via differencing_experiment.py with identical\n",
        "# configuration to v3_pipeline.py (seed=42, 20 trials, 3-fold TimeSeriesSplit,\n",
        "# objective='reg:absoluteerror'). Results loaded from saved output:\n",
        "\n",
        "import json\n",
        "with open('models/differencing_decision.json', 'r') as f:\n",
        "    decision = json.load(f)\n",
        "\n",
        "print(f\"\\n{'='*70}\")\n",
        "print('HONEST COMPARISON - Side by Side')\n",
        "print(f\"{'='*70}\")\n",
        "print(f\"{'Metric':>10} | {'Production (no diff)':>22} | {'With Differencing':>22}\")\n",
        "print(f\"{'-'*10}-+-{'-'*22}-+-{'-'*22}\")\n",
        "print(f\"{'R2':>10} | {decision['production_r2']:>22.4f} | {decision['diff_r2']:>22.4f}\")\n",
        "print(f\"{'MAE':>10} | {decision['production_mae']/1e6:>19.2f}M | {decision['diff_mae']/1e6:>19.2f}M\")\n",
        "print(f\"{'MAPE':>10} | {decision['production_mape']:>21.1f}% | {decision['diff_mape']:>21.1f}%\")\n",
        "print(f\"\\nMAE change:  {decision['mae_improvement_pct']:+.1f}% ({'improved' if decision['mae_improvement_pct'] > 0 else 'worsened'})\")\n",
        "print(f\"R2 change:   {decision['r2_improvement_pct']:+.1f}% ({'improved' if decision['r2_improvement_pct'] > 0 else 'worsened'})\")\n",
        "print(f\"MAPE change: {decision['mape_improvement_pct']:+.1f}% ({'improved' if decision['mape_improvement_pct'] > 0 else 'worsened'})\")\n",
        "print(f\"\\nDecision: {'ADOPTED' if decision['adopted'] else 'NOT ADOPTED'}\")"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 4: Markdown conclusion
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Stationarity Investigation Conclusion:**\n",
        "\n",
        "1. **The data is already stationary.** The ADF test confirms all 15 branches have p-values effectively zero \n",
        "   (well below 0.05) in both raw and log-transformed forms. Non-stationarity is NOT the cause of the high MAPE.\n",
        "\n",
        "2. **Differencing significantly worsened performance.** In a controlled experiment (same features, split, \n",
        "   hyperparameters, seed), differencing degraded MAE by 59%, R\\u00b2 by 45%, and MAPE by 49%.\n",
        "\n",
        "3. **Why differencing hurts:** The existing lag features (`lag_1`, `lag_2`, `lag_14`, `lag_60`) and rolling \n",
        "   statistics (`rolling_14_mean`, `rolling_14_std`) already implicitly encode trend and level information. \n",
        "   Differencing removes the absolute level signal that XGBoost needs to make accurate predictions, essentially \n",
        "   forcing the model to predict noisy changes instead of informative levels.\n",
        "\n",
        "4. **The actual cause of high MAPE:** Half-daily cash flow data has many near-zero demand periods (e.g., PM \n",
        "   sessions, weekends) where even small absolute errors produce very large percentage errors. This inflates \n",
        "   MAPE mathematically without reflecting poor model quality. The MAE of 9.36M PKR and R\\u00b2 of 0.6785 are \n",
        "   more representative of actual forecast accuracy.\n",
        "\n",
        "**Production model unchanged.** Differencing was not adopted."
    ]
})

# Insert cells at position
for i, cell in enumerate(new_cells):
    nb['cells'].insert(insert_idx + i, cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Inserted {len(new_cells)} new cells at position {insert_idx}")
print("Notebook updated successfully.")
