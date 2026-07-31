import json

def fix_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source']
            for i, line in enumerate(src):
                if 'y_train_raw = train_df[\'Half_Day_Total_Debit\'].values' in line:
                    src[i] = "cap_val = train_df['Half_Day_Total_Debit'].quantile(0.99)\n"
                    src.insert(i+1, "y_train_raw = train_df['Half_Day_Total_Debit'].clip(upper=cap_val).values\n")
                    break
            cell['source'] = src
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_notebook('Bank_Cash_Optimization_Workflow.ipynb')
fix_notebook('06_Evaluation_Final_Report.ipynb')
print('Notebooks fixed.')
