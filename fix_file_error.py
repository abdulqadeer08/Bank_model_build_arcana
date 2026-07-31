"""Fix: inline differencing_decision.json into notebook cell 42 so it works in Colab."""
import json

NB_PATH = 'Bank_Cash_Optimization_Workflow.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][42]
src = ''.join(cell['source'])

old_block = (
    "import json\n"
    "with open('models/differencing_decision.json', 'r') as f:\n"
    "    decision = json.load(f)"
)

new_block = (
    "# Inline results (no external file dependency)\n"
    "decision = {\n"
    '    "adopted": False,\n'
    '    "production_mae": 9360000.0,\n'
    '    "production_r2": 0.6785,\n'
    '    "production_mape": 132.7,\n'
    '    "diff_mae": 14882723.495243903,\n'
    '    "diff_r2": 0.3749700341756034,\n'
    '    "diff_mape": 197.68256691863832,\n'
    '    "diff_rmse": 36942638.388570525,\n'
    '    "mae_improvement_pct": -59.0034561457682,\n'
    '    "r2_improvement_pct": -44.73544079946892,\n'
    '    "mape_improvement_pct": -48.969530458657374\n'
    "}"
)

if old_block not in src:
    print("ERROR: Could not find the target block in cell 42")
    raise SystemExit(1)

new_src = src.replace(old_block, new_block)
cell['source'] = [new_src]

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK - Patched cell 42: inlined differencing_decision.json data")
