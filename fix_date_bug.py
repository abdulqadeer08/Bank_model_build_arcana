import json

def fix_notebook():
    with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    def find_cell(substring):
        for idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if substring in source:
                    return idx, source
        return -1, ""

    # Fix the cutoff_date object type for strftime
    idx, src = find_cell("cutoff_idx = int(len(unique_dates) * 0.8)")
    if idx != -1:
        new_src = src.replace(
            "cutoff_date = unique_dates[cutoff_idx]",
            "cutoff_date = pd.to_datetime(unique_dates[cutoff_idx])"
        )
        nb['cells'][idx]['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]]

    with open('Bank_Cash_Optimization_Workflow.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    fix_notebook()
