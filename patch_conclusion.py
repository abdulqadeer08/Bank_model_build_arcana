import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
nb = json.load(open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8'))

# Find and update the Conclusion cell (Cell 66)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown' and c['source']:
        if '## Conclusion' in c['source'][0]:
            # Append stationarity investigation note
            c['source'].extend([
                "\n",
                "### 4. Stationarity Investigation\n",
                "A formal ADF (Augmented Dickey-Fuller) stationarity test was conducted on `Half_Day_Total_Debit` to investigate whether non-stationary data was causing the high MAPE (132.7%). **All 15 branches were found to be stationary** (p-values effectively zero in both raw and log1p forms). A controlled differencing experiment confirmed that differencing **worsened** all metrics significantly (MAE: 9.36M to 14.88M, R\u00b2: 0.6785 to 0.3750). The existing lag/rolling features already capture temporal dependencies, making explicit differencing redundant. The high MAPE is attributable to near-zero demand periods inflating percentage errors, not non-stationarity. Production model unchanged.\n"
            ])
            print(f"Updated Conclusion cell at index {i}")
            break

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook saved.")
