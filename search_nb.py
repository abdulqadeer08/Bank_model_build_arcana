import json
import sys
import codecs

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if '0.6785' in src or '0.6628' in src or '0.4838' in src or '0.4770' in src:
        print(f'Old metrics found in cell {i}:')
        print(src[:200])
