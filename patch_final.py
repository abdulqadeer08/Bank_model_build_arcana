import json

def replace_in_notebook(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            for old, new in replacements:
                if old in source:
                    source = source.replace(old, new)
                    modified = True
            
            if modified:
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source'] and cell['source'][-1].endswith('\n'):
                    cell['source'][-1] = cell['source'][-1][:-1]
                
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f'Patched {filepath}')

replacements = [
    ("buffer_pct, trust, action = 20, 'MODERATE', 'Use with caution, manual override advised'", "buffer_pct, trust, action = 20, 'LOW', 'Use with caution, manual override advised'"),
    ("n_estimators      = 200", "n_estimators      = 500"),
    ("max_depth         = 4", "max_depth         = 6"),
    ("lgb.LGBMRegressor(", "lgb.LGBMRegressor(random_state=42, "),
    ("xgb.XGBRegressor(", "xgb.XGBRegressor(random_state=42, ")
]

replace_in_notebook('Bank_Cash_Optimization_Workflow.ipynb', replacements)
replace_in_notebook('06_Evaluation_Final_Report.ipynb', replacements)

# Also fix generate_report.py
with open('generate_report.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's remove emojis in generate_report.py to be safe and rename MODERATE to LOW
text = text.replace("return '🟢 HIGH'", "return 'HIGH'")
text = text.replace("return '🟡 MEDIUM'", "return 'MEDIUM'")
text = text.replace("return '🟠 MODERATE'", "return 'LOW'")
text = text.replace("return '🔴 LOW'", "return 'LOW'")

with open('generate_report.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Patched generate_report.py')
