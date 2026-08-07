import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# Print full source of cells 27, 35, 37, 40
for i in [27, 35, 37, 40]:
    cell = cells[i]
    src = ''.join(cell.get('source', []))
    print(f'\n{"="*70}')
    print(f'=== Cell {i} ({cell["cell_type"]}) FULL SOURCE ===')
    print(f'{"="*70}')
    print(src)
