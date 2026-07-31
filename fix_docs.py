import re

with open('Complete_Project_Documentation.md', 'r', encoding='utf-8') as f:
    text = f.read()

justification = """### Why XGBoost Over LightGBM (Despite a Marginal Performance Gap)
During final benchmarking with a fixed random seed (random_state=42) for full 
reproducibility, LightGBM showed a marginal edge over XGBoost V3:
- LightGBM: R² = 0.7199, MAE = 8.94M PKR
- XGBoost V3: R² = 0.7128, MAE = 9.53M PKR
(roughly a 6% relative difference in MAE)

Despite this, XGBoost V3 was retained as the production model for three reasons:
1. The entire explainability layer (SHAP TreeExplainer, the 'Why This Amount?' 
   dashboard feature, and all waterfall chart logic) was built and validated around 
   XGBoost's tree structure.
2. The performance difference is marginal relative to the model's overall prediction 
   uncertainty (MAE of ~9-10M against average daily branch volumes of 20-60M).
3. Switching to LightGBM would require re-validating the entire confidence-tier 
   system, SHAP explainability pipeline, and forecast generation logic — a 
   disproportionate cost for a ~6% marginal gain.

LightGBM's competitive performance is noted here transparently as a candidate for 
future iteration or as an ensemble component.

"""

text = text.replace('### Why XGBoost?', justification + '### Why XGBoost?')

with open('Complete_Project_Documentation.md', 'w', encoding='utf-8') as f:
    f.write(text)

with open('dashboard.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the old dataframe setup
old_df = """        benchmark_df = pd.DataFrame({
            "Model": ["Prophet (Baseline)", "XGBoost V3 (Tuned)"],
            "R² Score": [0.0478, 0.7147],
            "MAE (M PKR)": [19.16, 9.37],
            "MAPE (%)": [663.15, 137.24]
        })"""

new_df = """        benchmark_df = pd.DataFrame({
            "Model": ["Baseline (14-Day Rolling Mean)", "Prophet", "LightGBM", "XGBoost V3 (Tuned)"],
            "R² Score": [0.1706, 0.0478, 0.7199, 0.7128],
            "MAE (M PKR)": [18.25, 19.16, 8.94, 9.53],
            "MAPE (%)": [552.31, 663.15, 127.65, 138.48]
        })"""

text = text.replace(old_df, new_df)

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done editing.')
