import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

print("Generating Train-Test Split Diagram...")
df = pd.read_csv('model_data/half_daily_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

# We aggregate total branch transactions per day just for visualization
daily_df = df.groupby('start_date').size().reset_index(name='count')
daily_df = daily_df.sort_values('start_date').reset_index(drop=True)

cutoff_idx = int(len(daily_df) * 0.8)
train_df = daily_df.iloc[:cutoff_idx]
test_df = daily_df.iloc[cutoff_idx:]

plt.figure(figsize=(12, 6))
plt.plot(train_df['start_date'], train_df['count'], color='#1f77b4', label='Training Data (80%)', alpha=0.8)
plt.plot(test_df['start_date'], test_df['count'], color='#ff7f0e', label='Testing Data (20%)', alpha=0.8)

# Highlight regions
plt.axvspan(train_df['start_date'].min(), train_df['start_date'].max(), color='#1f77b4', alpha=0.1)
plt.axvspan(test_df['start_date'].min(), test_df['start_date'].max(), color='#ff7f0e', alpha=0.1)

plt.axvline(x=test_df['start_date'].min(), color='red', linestyle='--', linewidth=2, label='Split Point')

plt.title('Chronological Train-Test Split (Time Series Approach)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Activity Count', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs('eda_plots', exist_ok=True)
output_path = 'eda_plots/13_train_test_split.png'
plt.savefig(output_path, dpi=300)
print(f"Diagram saved to {output_path}")
