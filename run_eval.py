import json

def extract_and_run():
    with open('Bank_Cash_Optimization_Workflow.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_lines = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    continue
                code_lines.append(line.replace('plt.show()', 'pass'))
    
    full_code = ''.join(code_lines)
    
    # Run the code in a clean namespace
    namespace = {}
    try:
        exec(full_code, namespace)
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == '__main__':
    extract_and_run()
