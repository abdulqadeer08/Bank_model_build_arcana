import json
import sys

files = ['02_Feature_Engineering.ipynb', '03_Model_Preparation.ipynb', '04_Model_Training.ipynb']
with open('extracted_code.py', 'w', encoding='utf-8') as out:
    for f in files:
        out.write(f'\n# --- {f} ---\n')
        with open(f, 'r', encoding='utf-8') as infile:
            nb = json.load(infile)
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    out.write(''.join(cell.get('source', [])) + '\n\n')
print("Extraction complete.")
