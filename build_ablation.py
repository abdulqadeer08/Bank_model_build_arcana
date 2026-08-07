import os

with open('v3_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Change save path to avoid overwriting production models
code = code.replace('models/v3/', 'models/ablation/')
if not os.path.exists('models/ablation'):
    os.makedirs('models/ablation')

# Remove the newer features from feature_cols
new_features = [
    'ewma_14_Txn_Count', 'Days_Since_Salary', 'Is_Month_Start', 'Is_Month_End',
    'ewma_14_Half_Day_Total_Debit', 'dow_avg_4_Half_Day_Total_Debit',
    'ewma_14_Half_Day_Total_Credit', 'dow_avg_4_Half_Day_Total_Credit',
    'ewma_14_Half_Day_Net_Cash', 'dow_avg_4_Half_Day_Net_Cash'
]
for feat in new_features:
    code = code.replace("'" + feat + "', ", "")
    code = code.replace(", '" + feat + "'", "")
    code = code.replace("'" + feat + "'", "")

# Let's run just for Debit to save time, since Debit is the primary metric mentioned (R2 0.5687/WMAPE 39.5%)
code = code.replace("for target in ['Half_Day_Total_Debit', 'Half_Day_Total_Credit', 'Half_Day_Net_Cash']:", "for target in ['Half_Day_Total_Debit']:")

with open('ablation_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Created ablation_pipeline.py.')
