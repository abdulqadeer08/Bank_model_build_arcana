import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8'))
for i in [25, 27, 29, 31, 33, 35, 37]:
    if i < len(nb['cells']):
        print(f"=== CELL {i} ===")
        print(''.join(nb['cells'][i]['source']))
        print("=== END ===\n")
