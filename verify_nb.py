import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
nb = json.load(open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
for i in range(38, 48):
    c = cells[i]
    ct = c['cell_type']
    src = c['source']
    first_line = src[0][:120].strip() if src else '(none)'
    first_line = first_line.encode('ascii', 'replace').decode('ascii')
    print(f"Cell {i}: {ct} | {first_line}")
