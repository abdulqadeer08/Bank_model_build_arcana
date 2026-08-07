import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8'))
# Find the cell that saves model_results or comparison
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'to_csv' in src or 'model_results' in src or 'branch_df.to_csv' in src:
        print(f'=== CELL {i} (type={cell["cell_type"]}) ===')
        print(src[:3000])
        print('=== END ===\n')
