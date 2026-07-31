import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
nb = json.load(open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
# Find the Conclusion cell
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and c['source']:
        if '## Conclusion' in c['source'][0]:
            print(f"Found Conclusion at Cell {i}")
            print("Current content:")
            for line in c['source']:
                print(f"  {line.rstrip()}")
            break
