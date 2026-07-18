import re

with open('forecast_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Locate compute_uncertainty
old_compute = """    def compute_uncertainty(row):
        step = row['Step']
        uncertainty_pct = 0.05 + (step - 1) * (0.20 / forecast_days)"""

new_compute = """    # Load evaluation report for V3 MAE/MAPE
    try:
        eval_report = pd.read_csv('models/final_evaluation_report.csv')
        branch_mape = eval_report.set_index('Branch')['MAPE_%'].to_dict()
    except Exception:
        branch_mape = {}

    def compute_uncertainty(row):
        step = row['Step']
        br = row['Branch']
        # Use branch MAPE as baseline, fallback to 10% if not found. Cap base at 15%.
        base_uncertainty = min(branch_mape.get(br, 10.0) / 100.0, 0.15)
        # Add temporal uncertainty (0 to 15% over 30 days)
        uncertainty_pct = base_uncertainty + (step - 1) * (0.15 / forecast_days)"""

code = code.replace(old_compute, new_compute)

with open('forecast_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("forecast_pipeline.py updated!")
