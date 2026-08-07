import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('dashboard.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_justification = False

for line in lines:
    if "st.markdown(\"**Model:** XGBoost V3" in line and "0.6785" in line:
        line = "        st.markdown(\"**Model:** XGBoost + LightGBM Ensemble (TimeSeries Tuned)  \\n**WMAPE:** _see Model Performance_\")\n"

    if "<b style='color:#60A5FA; font-size:15px;'>⚖️ Production Choice Justification" in line:
        in_justification = True
        new_lines.append(line)
        new_lines.append("        The production model has been upgraded to an <b>XGBoost + LightGBM Ensemble</b> model to incorporate recent feedback and address dataset drift. By simultaneously tuning both models and their blend weights via Optuna with TimeSeriesSplit Cross-Validation, we achieve a highly robust forecast that is less susceptible to the variance of a single test set window. Features have been expanded to include Exponential Moving Averages (EWMA) and days-since-salary effects.\n")
        continue

    if in_justification:
        if "</div>" in line:
            in_justification = False
            new_lines.append(line)
        continue

    # Update title
    if "XGBoost V3 — Cross-Validated Performance" in line:
        line = line.replace("XGBoost V3", "XGBoost + LightGBM Ensemble")
        
    new_lines.append(line)

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("dashboard.py patched.")
