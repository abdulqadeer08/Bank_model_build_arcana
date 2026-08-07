import json
with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Recursive Forecasting Engine' in ''.join(cell['source']):
        source = cell['source']
        new_source = []
        skip = False
        for line in source:
            if 'import joblib as _jl' in line:
                new_source.append("            # Use ensemble in memory\n")
                new_source.append("            pred_log = w_xgb * xgb_model.predict(x_df)[0] + (1 - w_xgb) * lgb_model.predict(x_df)[0]\n")
                skip = True
            elif skip and 'pred_log =' in line and 'else:' not in line:
                continue
            elif skip and 'pred_dr =' in line:
                skip = False
                new_source.append(line)
            elif not skip:
                new_source.append(line)
        cell['source'] = new_source
        break

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Patched recursive forecasting engine cell.')
