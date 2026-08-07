import json

with open('models/v3/cv_results.json', 'r') as f:
    cv_params = json.load(f)

with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Format the dict as python code
dict_str = json.dumps(cv_params, indent=4)
# Split the json string by newlines to insert them properly into the notebook cell source list
lines = [line + '\n' for line in dict_str.split('\n')]
lines[0] = 'cv_params = ' + lines[0]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "with open('models/v3/cv_results.json', 'r') as f:" in line:
                # Replace this line with the dict lines
                new_source = source[:i] + lines + source[i+2:] # skip the next line too: cv_params = json.load(f)
                cell['source'] = new_source
                break

with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Embedded cv_results.json into the notebook.')
