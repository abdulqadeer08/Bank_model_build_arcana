import json

new_code = """# ── 7.3 Business recommendations with trust levels ────────────────────────────
print('=' * 70)
print('  BUSINESS RECOMMENDATIONS - BRANCH-WISE CASH STRATEGY')
print('=' * 70)

# Calculate MAE percentiles across branches to assign confidence tiers robustly
p33 = branch_df['MAE_M'].quantile(0.33)
p66 = branch_df['MAE_M'].quantile(0.66)

recommendations = []
for _, row in branch_df.iterrows():
    mape = row['MAPE_%']
    mae  = row['MAE_M']
    avg  = row['Avg_Demand_M']

    # Safety buffer based on MAE percentiles to avoid MAPE distortion from near-zero days
    if mae <= p33:
        buffer_pct, trust, action = 5,  'HIGH',     'Use model directly for replenishment'
    elif mae <= p66:
        buffer_pct, trust, action = 12, 'MEDIUM',   'Add safety buffer, monitor weekly'
    else:
        buffer_pct, trust, action = 20, 'MODERATE', 'Use with caution, manual override advised'

    safe_amount = avg * (1 + buffer_pct / 100)
    recommendations.append({
        'Branch': int(row['Branch']),
        'Avg_Demand_M': avg,
        'MAE_M': mae,
        'MAPE_%': mape,
        'Trust_Level': trust,
        'Buffer_%': buffer_pct,
        'Recommended_M': round(safe_amount, 2),
        'Action': action,
    })

rec_df = pd.DataFrame(recommendations).sort_values('Avg_Demand_M', ascending=False)
print(rec_df[['Branch','Avg_Demand_M','MAE_M','MAPE_%','Trust_Level','Buffer_%','Recommended_M']].to_string(index=False))
print()
for _, r in rec_df.iterrows():
    print(f'  Branch {int(r["Branch"]):>5} -> {r["Action"]}')
print('=' * 70)"""

for nb_file in ['Bank_Cash_Optimization_Workflow.ipynb', '06_Evaluation_Final_Report.ipynb']:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code' and 'Trust_Level' in ''.join(cell['source']):
                # replace cell source
                cell['source'] = [line + '\n' for line in new_code.split('\n')]
                if cell['source'] and cell['source'][-1].endswith('\n'):
                    cell['source'][-1] = cell['source'][-1][:-1]
                
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f'Patched {nb_file}')
    except Exception as e:
        print(f'Failed on {nb_file}: {e}')
