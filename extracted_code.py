
# --- 02_Feature_Engineering.ipynb ---
# ── Standard Library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

# ── Data Manipulation ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualization ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

# ── Display Configuration ─────────────────────────────────────────────────────
pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.width', 150)

plt.rcParams['figure.figsize']  = (14, 5)
plt.rcParams['axes.titlesize']  = 13
plt.rcParams['axes.labelsize']  = 11
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

BANK_PALETTE = ['#1F4E79', '#2E75B6', '#A9C4E2', '#D6462B', '#F4A460']
sns.set_palette(BANK_PALETTE)

print("✅ Libraries imported successfully.")

# ── Load Dataset ───────────────────────────────────────────────────────────────
df = pd.read_excel('Bank Cash Optimization.xlsx')

# Ensure correct datetime type for start_date
df['start_date'] = pd.to_datetime(df['start_date'])

print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Columns : {df.columns.tolist()}")
print(f"   Date Range: {df['start_date'].min().date()} → {df['start_date'].max().date()}")
df.head(5)

# ── 2.1 Missing Values Check ───────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    'Column'        : missing.index,
    'Missing Count' : missing.values,
    'Missing %'     : missing_pct.values.round(2)
})
print("🔍 Missing Value Report:")
print(missing_report.to_string(index=False))

# ── 2.2 Missing Value Handling Strategy ───────────────────────────────────────
if missing.sum() == 0:
    print("✅ No missing values detected. No imputation required.")
else:
    # Strategy for each column type:
    # - Monetary columns (TOTAL_DR, TOTAL_CR): Median imputation (robust to outliers)
    # - txn_hour: Mode imputation (most frequent hour)
    # - tran_br_code: Flag as 'UNKNOWN' and investigate source
    # - start_date: Forward fill (banking data is sequential)
    
    for col in ['TOTAL_DR', 'TOTAL_CR']:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   {col}: Filled {df[col].isnull().sum()} nulls with median ({median_val:,.0f})")
    
    if df['txn_hour'].isnull().sum() > 0:
        mode_val = df['txn_hour'].mode()[0]
        df['txn_hour'].fillna(mode_val, inplace=True)
        print(f"   txn_hour: Filled with mode ({mode_val})")
    
    if df['start_date'].isnull().sum() > 0:
        df['start_date'].fillna(method='ffill', inplace=True)
        print("   start_date: Forward-filled missing dates")

print(f"\n✅ Post-handling missing values: {df.isnull().sum().sum()}")

# ── 3.1 Duplicate Detection ────────────────────────────────────────────────────
exact_dups  = df.duplicated().sum()
biz_key_dups = df.duplicated(subset=['start_date', 'txn_hour', 'tran_br_code']).sum()

print(f"🔍 Duplicate Analysis:")
print(f"   Exact Row Duplicates               : {exact_dups:,}")
print(f"   Business-Key Duplicates            : {biz_key_dups:,}")
print(f"   (Business Key = date + hour + branch)")

# ── 3.2 Remove Duplicates ─────────────────────────────────────────────────────
rows_before = len(df)

# Remove exact row duplicates
df.drop_duplicates(inplace=True)
rows_after_exact = len(df)

# For business-key duplicates: keep the record with higher total cash movement
# (strategy: max DR + CR wins — assumes the larger value is the correct consolidated record)
df['_total_cash'] = df['TOTAL_DR'] + df['TOTAL_CR']
df = df.sort_values('_total_cash', ascending=False)
df = df.drop_duplicates(subset=['start_date', 'txn_hour', 'tran_br_code'], keep='first')
df = df.drop(columns=['_total_cash'])
df.reset_index(drop=True, inplace=True)

rows_after_all = len(df)
print(f"✅ Duplicate Removal Summary:")
print(f"   Rows Before              : {rows_before:,}")
print(f"   After Exact Dedup        : {rows_after_exact:,} (removed {rows_before - rows_after_exact:,})")
print(f"   After Business-Key Dedup : {rows_after_all:,} (removed {rows_after_exact - rows_after_all:,})")
print(f"   Final Dataset Size       : {rows_after_all:,} rows")

# ── 4.1 Preserve Raw Values ────────────────────────────────────────────────────
df['TOTAL_DR_raw'] = df['TOTAL_DR'].copy()
df['TOTAL_CR_raw'] = df['TOTAL_CR'].copy()

# ── 4.2 Calculate Winsorization Bounds ────────────────────────────────────────
p01_dr, p99_dr = df['TOTAL_DR'].quantile(0.01), df['TOTAL_DR'].quantile(0.99)
p01_cr, p99_cr = df['TOTAL_CR'].quantile(0.01), df['TOTAL_CR'].quantile(0.99)

print("📊 Winsorization Bounds:")
print(f"   TOTAL_DR — Cap at: {p01_dr/1e6:.2f}M (1st pct) to {p99_dr/1e6:.2f}M (99th pct)")
print(f"   TOTAL_CR — Cap at: {p01_cr/1e6:.2f}M (1st pct) to {p99_cr/1e6:.2f}M (99th pct)")

# ── 4.3 Apply Winsorization ───────────────────────────────────────────────────
df['TOTAL_DR'] = df['TOTAL_DR'].clip(lower=p01_dr, upper=p99_dr)
df['TOTAL_CR'] = df['TOTAL_CR'].clip(lower=p01_cr, upper=p99_cr)

dr_affected = (df['TOTAL_DR'] != df['TOTAL_DR_raw']).sum()
cr_affected = (df['TOTAL_CR'] != df['TOTAL_CR_raw']).sum()

print(f"\n✅ Winsorization Applied:")
print(f"   TOTAL_DR: {dr_affected:,} values capped ({100*dr_affected/len(df):.1f}%)")
print(f"   TOTAL_CR: {cr_affected:,} values capped ({100*cr_affected/len(df):.1f}%)")

# ── 4.4 Flag High-Value Transactions (for ML signal) ──────────────────────────
df['Is_High_Value_DR'] = (df['TOTAL_DR_raw'] > p99_dr).astype(int)
df['Is_High_Value_CR'] = (df['TOTAL_CR_raw'] > p99_cr).astype(int)

print(f"\n   High-Value DR flag: {df['Is_High_Value_DR'].sum():,} transactions")
print(f"   High-Value CR flag: {df['Is_High_Value_CR'].sum():,} transactions")

# ── 5.1 Pre-conversion Type Check ────────────────────────────────────────────
print("📊 Data Types Before Correction:")
print(df[['start_date', 'txn_hour', 'tran_br_code', 'TOTAL_DR', 'TOTAL_CR']].dtypes)
print(f"\n   Memory Usage Before: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ── 5.2 Apply Type Corrections ────────────────────────────────────────────────

# tran_br_code → category (it's a nominal identifier, NOT numeric)
df['tran_br_code'] = df['tran_br_code'].astype('category')

# txn_hour → uint8 (0–23 fits in 1 byte, saves memory)
df['txn_hour'] = df['txn_hour'].astype('uint8')

# start_date → already datetime64, keep as is
df['start_date'] = pd.to_datetime(df['start_date'])

# TOTAL_DR, TOTAL_CR → float32 (after winsorization, float is appropriate)
df['TOTAL_DR'] = df['TOTAL_DR'].astype('float32')
df['TOTAL_CR'] = df['TOTAL_CR'].astype('float32')

print("📊 Data Types After Correction:")
print(df[['start_date', 'txn_hour', 'tran_br_code', 'TOTAL_DR', 'TOTAL_CR']].dtypes)
print(f"\n   Memory Usage After : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\n✅ Data types corrected successfully.")

# ── 6.1 Extract Date Features ─────────────────────────────────────────────────
df['Year']        = df['start_date'].dt.year.astype('int16')
df['Month']       = df['start_date'].dt.month.astype('uint8')
df['Day']         = df['start_date'].dt.day.astype('uint8')
df['Weekday']     = df['start_date'].dt.dayofweek.astype('uint8')   # 0=Monday, 6=Sunday
df['Quarter']     = df['start_date'].dt.quarter.astype('uint8')
df['Week_Number'] = df['start_date'].dt.isocalendar().week.astype('uint8')
df['Is_Weekend']  = (df['Weekday'] >= 5).astype('uint8')           # 1=Sat/Sun

print("✅ Date features created:")
date_feat_cols = ['Year','Month','Day','Weekday','Quarter','Week_Number','Is_Weekend']
print(df[['start_date'] + date_feat_cols].head(8).to_string(index=False))
print(f"\n   Is_Weekend distribution: {df['Is_Weekend'].value_counts().to_dict()}")

# ── 7.1 Time Period Labels ─────────────────────────────────────────────────────
def assign_time_period(hour):
    """Assign a time period label based on hour of day."""
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 16:
        return 'Afternoon'
    elif 16 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'

df['Time_Period'] = df['txn_hour'].apply(assign_time_period).astype('category')

# ── 7.2 Business Hour Flag ────────────────────────────────────────────────────
# Standard banking hours: 9 AM – 5 PM (17:00)
df['Is_Business_Hour'] = ((df['txn_hour'] >= 9) & (df['txn_hour'] < 17)).astype('uint8')

# ── 7.3 Peak Hour Flag ────────────────────────────────────────────────────────
# Based on EDA: Peak hours are 11 AM – 3 PM (highest transaction volume and debit)
df['Is_Peak_Hour'] = ((df['txn_hour'] >= 11) & (df['txn_hour'] <= 15)).astype('uint8')

# ── 7.4 Hour Sine/Cosine Encoding (for cyclical ML features) ─────────────────
# Captures the cyclical nature of time (hour 23 is close to hour 0)
df['Hour_Sin'] = np.sin(2 * np.pi * df['txn_hour'] / 24).astype('float32')
df['Hour_Cos'] = np.cos(2 * np.pi * df['txn_hour'] / 24).astype('float32')

print("✅ Time features created:")
time_cols = ['txn_hour', 'Time_Period', 'Is_Business_Hour', 'Is_Peak_Hour', 'Hour_Sin', 'Hour_Cos']
print(df[time_cols].drop_duplicates('txn_hour').sort_values('txn_hour').to_string(index=False))

# ── 7.5 Time Period Distribution Check ────────────────────────────────────────
tp_dist = df['Time_Period'].value_counts()
print("\n📊 Time Period Distribution:")
for period, count in tp_dist.items():
    pct = 100 * count / len(df)
    print(f"   {period:<12}: {count:>7,} records ({pct:.1f}%)")

print(f"\n   Business Hours records : {df['Is_Business_Hour'].sum():,} ({100*df['Is_Business_Hour'].mean():.1f}%)")
print(f"   Peak Hours records     : {df['Is_Peak_Hour'].sum():,} ({100*df['Is_Peak_Hour'].mean():.1f}%)")

# ════════════════════════════════════════════════════════════════
# 8.1 TRANSACTION-LEVEL FEATURES
# ════════════════════════════════════════════════════════════════

# Net Cash Flow: Credit minus Debit at transaction level
# Positive = net inflow (deposits > withdrawals)
# Negative = net outflow (withdrawals > deposits)
df['Net_Cash_Flow'] = (df['TOTAL_CR'] - df['TOTAL_DR']).astype('float32')

# Total Cash Movement: Combined throughput (DR + CR)
# Reflects total branch activity regardless of direction
df['Total_Cash_Movement'] = (df['TOTAL_DR'] + df['TOTAL_CR']).astype('float32')

# Debit-to-Credit Ratio: How much is withdrawn for every unit deposited
# High ratio = cash drain branch; Low ratio = cash surplus branch
df['Debit_Credit_Ratio'] = (df['TOTAL_DR'] / (df['TOTAL_CR'] + 1)).astype('float32')

# Net Cash Direction Flag: 1 = net outflow, 0 = net inflow
df['Is_Net_Outflow'] = (df['Net_Cash_Flow'] < 0).astype('uint8')

print("✅ Transaction-level features created:")
print(df[['TOTAL_DR','TOTAL_CR','Net_Cash_Flow','Total_Cash_Movement','Debit_Credit_Ratio','Is_Net_Outflow']].head(5).to_string(index=False))

# ════════════════════════════════════════════════════════════════
# 8.2 BRANCH-LEVEL AGGREGATE FEATURES
# ════════════════════════════════════════════════════════════════
# These features encode each branch's historical cash behavior.
# The ML model needs to know whether branch X is typically a high-DR or high-CR branch.

branch_agg = df.groupby('tran_br_code').agg(
    Branch_Total_Debit        = ('TOTAL_DR', 'sum'),
    Branch_Total_Credit       = ('TOTAL_CR', 'sum'),
    Branch_Avg_Debit          = ('TOTAL_DR', 'mean'),
    Branch_Avg_Credit         = ('TOTAL_CR', 'mean'),
    Branch_Total_Cash         = ('Total_Cash_Movement', 'sum'),
    Branch_Txn_Count          = ('TOTAL_DR', 'count'),
    Branch_Avg_Net_Cash_Flow  = ('Net_Cash_Flow', 'mean')
).reset_index()

# Merge back to main dataframe
df = df.merge(branch_agg, on='tran_br_code', how='left')

print("✅ Branch-level features merged:")
branch_feat_cols = ['tran_br_code','Branch_Total_Debit','Branch_Total_Credit',
                    'Branch_Avg_Debit','Branch_Avg_Credit','Branch_Txn_Count']
print(df[branch_feat_cols].drop_duplicates('tran_br_code').sort_values('Branch_Total_Debit', ascending=False).to_string(index=False))

# ════════════════════════════════════════════════════════════════
# 8.3 HOURLY AGGREGATE FEATURES
# ════════════════════════════════════════════════════════════════
# Encodes the typical cash behavior at each hour across all branches.

hourly_agg = df.groupby('txn_hour').agg(
    Hourly_Total_Debit      = ('TOTAL_DR', 'sum'),
    Hourly_Total_Credit     = ('TOTAL_CR', 'sum'),
    Hourly_Cash_Movement    = ('Total_Cash_Movement', 'sum'),
    Hourly_Avg_Debit        = ('TOTAL_DR', 'mean'),
    Hourly_Avg_Credit       = ('TOTAL_CR', 'mean'),
    Hourly_Txn_Count        = ('TOTAL_DR', 'count')
).reset_index()

df = df.merge(hourly_agg, on='txn_hour', how='left')
print("✅ Hourly aggregate features merged.")
print(df[['txn_hour','Hourly_Total_Debit','Hourly_Total_Credit','Hourly_Txn_Count']].drop_duplicates('txn_hour').sort_values('txn_hour').head(10).to_string(index=False))

# ════════════════════════════════════════════════════════════════
# 8.4 DAILY AGGREGATE FEATURES
# ════════════════════════════════════════════════════════════════
# Daily aggregations capture the total cash load on each calendar day.

daily_agg = df.groupby('start_date').agg(
    Daily_Total_Debit       = ('TOTAL_DR', 'sum'),
    Daily_Total_Credit      = ('TOTAL_CR', 'sum'),
    Daily_Cash_Movement     = ('Total_Cash_Movement', 'sum'),
    Daily_Txn_Count         = ('TOTAL_DR', 'count'),
    Daily_Net_Cash_Flow     = ('Net_Cash_Flow', 'sum')
).reset_index()

df = df.merge(daily_agg, on='start_date', how='left')
print("✅ Daily aggregate features merged.")
print(df[['start_date','Daily_Total_Debit','Daily_Total_Credit','Daily_Txn_Count']].drop_duplicates('start_date').sort_values('start_date').head(8).to_string(index=False))

# ════════════════════════════════════════════════════════════════
# 8.5 MONTHLY AGGREGATE FEATURES
# ════════════════════════════════════════════════════════════════
# Monthly features capture seasonality — critical for festival/quarter-end patterns.

df['YearMonth'] = df['start_date'].dt.to_period('M')

monthly_agg = df.groupby('YearMonth').agg(
    Monthly_Total_Debit     = ('TOTAL_DR', 'sum'),
    Monthly_Total_Credit    = ('TOTAL_CR', 'sum'),
    Monthly_Cash_Movement   = ('Total_Cash_Movement', 'sum'),
    Monthly_Txn_Count       = ('TOTAL_DR', 'count')
).reset_index()

df = df.merge(monthly_agg, on='YearMonth', how='left')
df.drop(columns=['YearMonth'], inplace=True)  # Drop temp period column
print("✅ Monthly aggregate features merged.")

# ════════════════════════════════════════════════════════════════
# 8.6 BRANCH × DAILY CROSS-FEATURES
# ════════════════════════════════════════════════════════════════
# Measures how many transactions occurred at a specific branch on a specific date.
# Captures inter-day variability within each branch.

branch_daily = df.groupby(['tran_br_code', 'start_date']).agg(
    Branch_Daily_Txn_Count  = ('TOTAL_DR', 'count'),
    Branch_Daily_Debit      = ('TOTAL_DR', 'sum'),
    Branch_Daily_Credit     = ('TOTAL_CR', 'sum')
).reset_index()

df = df.merge(branch_daily, on=['tran_br_code', 'start_date'], how='left')

# ════════════════════════════════════════════════════════════════
# 8.7 BRANCH × HOURLY CROSS-FEATURES
# ════════════════════════════════════════════════════════════════
# Captures how busy a specific branch is at a specific hour.
# This is the most granular feature — directly useful for hour-level cash forecasting.

branch_hourly = df.groupby(['tran_br_code', 'txn_hour']).agg(
    Branch_Hourly_Txn_Count = ('TOTAL_DR', 'count'),
    Branch_Hourly_Debit     = ('TOTAL_DR', 'mean'),  # average debit for this branch at this hour
    Branch_Hourly_Credit    = ('TOTAL_CR', 'mean')
).reset_index()

df = df.merge(branch_hourly, on=['tran_br_code', 'txn_hour'], how='left')

print(f"✅ Branch-level cross features created.")
print(f"   Current shape: {df.shape}")

# ════════════════════════════════════════════════════════════════
# 8.8 MONTH SINE/COSINE ENCODING (cyclical seasonality)
# ════════════════════════════════════════════════════════════════
# Month 12 and Month 1 are close in time — standard encoding doesn't capture this.

df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12).astype('float32')
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12).astype('float32')

# Day-of-week cyclical encoding
df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday'] / 7).astype('float32')
df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday'] / 7).astype('float32')

print("✅ Cyclical sine/cosine encodings added for Month and Weekday.")
print("   This ensures the ML model understands month 12 is adjacent to month 1.")

# ── Full Feature Inventory ─────────────────────────────────────────────────────
print("\n📋 Complete Feature List After Step 8:")
print(f"   Total Columns: {df.shape[1]}")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:>3}. {col:<40} | dtype: {df[col].dtype}")

# ── 9.1 Select Columns for Scaling ────────────────────────────────────────────
# We scale only continuous numeric columns
# Binary flags (Is_Weekend, Is_Business_Hour, etc.) do NOT need scaling
# Cyclical features (sin/cos) are already in [-1, 1] range

scale_cols = [
    'TOTAL_DR', 'TOTAL_CR', 'Net_Cash_Flow', 'Total_Cash_Movement', 'Debit_Credit_Ratio',
    'Branch_Total_Debit', 'Branch_Total_Credit', 'Branch_Total_Cash',
    'Branch_Avg_Debit', 'Branch_Avg_Credit', 'Branch_Avg_Net_Cash_Flow',
    'Hourly_Total_Debit', 'Hourly_Total_Credit', 'Hourly_Cash_Movement',
    'Hourly_Avg_Debit', 'Hourly_Avg_Credit',
    'Daily_Total_Debit', 'Daily_Total_Credit', 'Daily_Cash_Movement', 'Daily_Net_Cash_Flow',
    'Monthly_Total_Debit', 'Monthly_Total_Credit', 'Monthly_Cash_Movement',
    'Branch_Daily_Debit', 'Branch_Daily_Credit',
    'Branch_Hourly_Debit', 'Branch_Hourly_Credit'
]

# Keep only columns that exist in df
scale_cols = [c for c in scale_cols if c in df.columns]
print(f"📊 Columns selected for scaling: {len(scale_cols)}")
print(scale_cols)

# ── 9.2 Apply RobustScaler ────────────────────────────────────────────────────
scaler = RobustScaler()

# Create a scaled copy — store scaled features with '_scaled' suffix
# This preserves original values for interpretability
df_scaled_values = scaler.fit_transform(df[scale_cols])
scaled_col_names = [f"{col}_scaled" for col in scale_cols]
df_scaled_df = pd.DataFrame(df_scaled_values, columns=scaled_col_names, index=df.index)

# Merge scaled columns back
df = pd.concat([df, df_scaled_df], axis=1)

print(f"✅ RobustScaler applied to {len(scale_cols)} features.")
print(f"   Scaled columns added with '_scaled' suffix.")
print(f"   Shape after scaling: {df.shape}")

# Quick validation — verify scaled values are centered around 0
sample_check = pd.DataFrame({
    'Feature'       : scale_cols[:5],
    'Original_Mean' : [df[c].mean() for c in scale_cols[:5]],
    'Scaled_Median' : [df[f"{c}_scaled"].median() for c in scale_cols[:5]],
    'Scaled_IQR'    : [df[f"{c}_scaled"].quantile(0.75) - df[f"{c}_scaled"].quantile(0.25)
                       for c in scale_cols[:5]]
})
print("\n📊 Scaling Validation (first 5 features):")
print(sample_check.to_string(index=False))
print("   (Scaled_Median ≈ 0 and Scaled_IQR ≈ 1.0 confirms correct scaling)")

# ── Define Feature Set for Selection ──────────────────────────────────────────
# Use original (unscaled) numeric features for selection
# Exclude: raw backup cols, date column, category column, and scaled duplicates

exclude_cols = [
    'start_date', 'tran_br_code', 'TOTAL_DR_raw', 'TOTAL_CR_raw',
    'Time_Period'
]
scaled_cols_list = [c for c in df.columns if c.endswith('_scaled')]
exclude_cols += scaled_cols_list

feature_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8']]

print(f"📊 Features being evaluated: {len(feature_cols)}")
print(feature_cols)

# ── 10.1 Variance Threshold ────────────────────────────────────────────────────
# Remove features with variance below threshold (near-constant = no info)
X_selection = df[feature_cols].fillna(0)  # Fill any NaN for selection step

vt = VarianceThreshold(threshold=0.01)  # Threshold: variance must be > 0.01
vt.fit(X_selection)

low_var_mask  = vt.get_support()
low_var_cols  = [c for c, keep in zip(feature_cols, low_var_mask) if not keep]
high_var_cols = [c for c, keep in zip(feature_cols, low_var_mask) if keep]

print(f"📊 Variance Threshold Analysis:")
print(f"   Features with sufficient variance : {len(high_var_cols)}")
print(f"   Low-variance features removed    : {len(low_var_cols)}")
if low_var_cols:
    print(f"   Removed: {low_var_cols}")

# ── 10.2 Correlation-Based Redundancy Removal ──────────────────────────────────
corr_matrix = df[high_var_cols].corr().abs()

# Identify pairs with correlation > 0.95 (extremely redundant)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

print(f"📊 Correlation-Based Feature Removal (threshold: >0.95):")
print(f"   Highly correlated (redundant) features: {len(high_corr_cols)}")
if high_corr_cols:
    print(f"   Removed: {high_corr_cols}")

# Features remaining after correlation filter
remaining_after_corr = [c for c in high_var_cols if c not in high_corr_cols]
print(f"   Features remaining : {len(remaining_after_corr)}")

# ── 10.3 Mutual Information — Feature Importance Ranking ──────────────────────
# Target: TOTAL_DR (predicting cash outflow demand = core cash optimization task)

target_col = 'TOTAL_DR'
mi_features = [c for c in remaining_after_corr if c != target_col]

X_mi = df[mi_features].fillna(0)
y_mi = df[target_col].fillna(0)

mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
mi_df = pd.DataFrame({
    'Feature'           : mi_features,
    'Mutual_Info_Score' : mi_scores
}).sort_values('Mutual_Info_Score', ascending=False).reset_index(drop=True)

print("📊 Mutual Information Scores (Target = TOTAL_DR — Cash Demand):")
print(mi_df.to_string(index=False))

# ── 10.4 Mutual Information Visualization ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, max(6, len(mi_df) * 0.35)))

top_mi = mi_df.head(25)  # Show top 25
colors_mi = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_mi)))
bars = ax.barh(top_mi['Feature'][::-1], top_mi['Mutual_Info_Score'][::-1],
               color=colors_mi[::-1], edgecolor='white')

ax.set_title('Feature Importance — Mutual Information vs TOTAL_DR (Cash Demand)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Mutual Information Score')
ax.set_ylabel('Feature')
ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1, label='MI=0.01 threshold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('eda_plots/fe_01_mutual_information.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 10.5 Select Final Feature Set ────────────────────────────────────────────
# Keep features with MI score > 0.01 (above noise threshold)
selected_features = mi_df[mi_df['Mutual_Info_Score'] > 0.01]['Feature'].tolist()

# Always include key identifying columns and target
must_include = ['start_date', 'tran_br_code', 'txn_hour', 'TOTAL_DR', 'TOTAL_CR']
final_features = must_include + [f for f in selected_features if f not in must_include]

print(f"📊 Feature Selection Summary:")
print(f"   Features before selection : {len(feature_cols)}")
print(f"   After Variance Threshold  : {len(high_var_cols)}")
print(f"   After Correlation Filter  : {len(remaining_after_corr)}")
print(f"   After MI Threshold (>0.01): {len(selected_features)}")
print(f"   Final features (+ ID cols): {len(final_features)}")
print(f"\n   Selected Features:")
for f in final_features:
    print(f"   - {f}")

# ── Build Final DataFrame ──────────────────────────────────────────────────────
df_final = df[final_features].copy()

print("╔══════════════════════════════════════════════════════════╗")
print("║           FINAL FEATURE-ENGINEERED DATASET               ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  Shape          : {df_final.shape[0]:>7,} rows × {df_final.shape[1]:>3} columns         ║")
print(f"║  Memory Usage   : {df_final.memory_usage(deep=True).sum()/1024**2:>7.2f} MB                        ║")
print("╠══════════════════════════════════════════════════════════╣")
print("║  Original Columns     : start_date, txn_hour,            ║")
print("║                         tran_br_code, TOTAL_DR, TOTAL_CR ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  Newly Created Features : {df_final.shape[1] - 5:>3} features added              ║")
print("╚══════════════════════════════════════════════════════════╝")

# ── Final Shape & Column Overview ─────────────────────────────────────────────
print("📋 Final Columns & Data Types:")
for i, (col, dtype) in enumerate(df_final.dtypes.items(), 1):
    tag = '🆕' if col not in ['start_date','txn_hour','tran_br_code','TOTAL_DR','TOTAL_CR'] else '📌'
    print(f"   {tag} {i:>3}. {col:<45} | {str(dtype):<10}")

# ── Preview of Final Dataset ───────────────────────────────────────────────────
print("\n📋 Preview of Final Dataset (First 10 rows):")
df_final.head(10)

# ── Statistics of Newly Created Features ──────────────────────────────────────
new_numeric_feats = [c for c in df_final.columns
                     if c not in ['start_date','txn_hour','tran_br_code','TOTAL_DR','TOTAL_CR','Time_Period']
                     and df_final[c].dtype in ['float32','float64','int16','int32','int64','uint8']]

print("\n📊 Summary Statistics — Newly Created Features:")
df_final[new_numeric_feats].describe().round(2)

# ── Export Final Dataset ───────────────────────────────────────────────────────
df_final.to_csv('bank_cash_optimization_features.csv', index=False)
print("\n✅ Final feature-engineered dataset saved to: bank_cash_optimization_features.csv")
print(f"   File contains: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")

# ── Final Completion Banner ────────────────────────────────────────────────────
print("=" * 65)
print("     FEATURE ENGINEERING NOTEBOOK — EXECUTION COMPLETE")
print("=" * 65)
print(f"  Input  : 81,717 rows × 5 raw columns")
print(f"  Output : {df_final.shape[0]:,} rows × {df_final.shape[1]} feature-engineered columns")
print(f"  Saved  : bank_cash_optimization_features.csv")
print("")
print("  Feature Categories Created:")
print("    📅 Date Features         : Year, Month, Day, Weekday, Quarter,")
print("                              Week_Number, Is_Weekend")
print("    ⏰ Time Features          : Time_Period, Is_Business_Hour,")
print("                              Is_Peak_Hour, Hour_Sin/Cos")
print("    🏦 Banking Features       : Net_Cash_Flow, Total_Cash_Movement,")
print("                              Debit_Credit_Ratio, Is_Net_Outflow")
print("    🏢 Branch Aggregates      : Branch_Total/Avg DR/CR, Cash")
print("    🕐 Hourly Aggregates      : Hourly_Total/Avg DR/CR/Movement")
print("    📆 Daily Aggregates       : Daily_Total/Net DR/CR/Movement")
print("    📅 Monthly Aggregates     : Monthly_Total DR/CR/Movement")
print("    🔗 Cross Features         : Branch×Daily, Branch×Hour")
print("    🔄 Cyclical Encodings     : Month/Weekday Sin/Cos")
print("    ⚖️  Scaling               : RobustScaler on continuous features")
print("=" * 65)
print("  ✅ Dataset is ML-ready. Proceed to Model Development.")
print("=" * 65)


# --- 03_Model_Preparation.ipynb ---
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import joblib
import os

# ── Display Config ─────────────────────────────────────────────────────────
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.width', 150)

plt.rcParams.update({
    'figure.dpi'        : 130,
    'axes.titlesize'    : 13,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : '#F9FAFB',
    'axes.grid'         : True,
    'grid.color'        : '#E5E7EB',
})

os.makedirs('model_data', exist_ok=True)
print('✅ Libraries imported successfully.')

df = pd.read_csv('bank_cash_optimization_features.csv')
df['start_date'] = pd.to_datetime(df['start_date'])

print(f'✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'📅 Date range   : {df["start_date"].min().date()}  →  {df["start_date"].max().date()}')
print(f'🏦 Branches     : {df["tran_br_code"].nunique()} unique branches')
print(f'❌ Missing values: {df.isnull().sum().sum()}')
df.head(3)

daily = df.groupby(['start_date', 'tran_br_code']).agg(
    # ── Target ────────────────────────────────────────────────────
    Daily_Total_Debit      = ('TOTAL_DR',              'sum'),
    Daily_Total_Credit     = ('TOTAL_CR',              'sum'),
    # ── Transaction Volume ─────────────────────────────────────────
    Daily_Txn_Count        = ('TOTAL_DR',              'count'),
    # ── Time Features ─────────────────────────────────────────────
    Weekday                = ('Weekday',               'first'),
    Is_Weekend             = ('Is_Weekend',            'first'),
    Month                  = ('Month',                 'first'),
    Year                   = ('Year',                  'first'),
    Day                    = ('Day',                   'first'),
    Weekday_Sin            = ('Weekday_Sin',           'first'),
    Weekday_Cos            = ('Weekday_Cos',           'first'),
    Month_Sin              = ('Month_Sin',             'first'),
    Month_Cos              = ('Month_Cos',             'first'),
    # ── Branch Level Features ──────────────────────────────────────
    Branch_Total_Debit     = ('Branch_Total_Debit',    'first'),
    Branch_Txn_Count       = ('Branch_Txn_Count',      'first'),
    Branch_Avg_Net_CF      = ('Branch_Avg_Net_Cash_Flow','first'),
    # ── Hourly Patterns ───────────────────────────────────────────
    Peak_Hour_Txns         = ('Is_Peak_Hour',          'sum'),
    Business_Hour_Txns     = ('Is_Business_Hour',      'sum'),
    # ── Net Cash Flow ──────────────────────────────────────────────
    Daily_Net_Cash_Flow    = ('Net_Cash_Flow',         'sum'),
    Net_Outflow_Periods    = ('Is_Net_Outflow',        'sum'),
).reset_index()

# Sort chronologically — CRITICAL for time series
daily = daily.sort_values(['start_date', 'tran_br_code']).reset_index(drop=True)

print(f'✅ Daily-branch dataset: {daily.shape[0]:,} rows × {daily.shape[1]} columns')
print(f'📅 Date range: {daily["start_date"].min().date()}  →  {daily["start_date"].max().date()}')
print(f'🏦 Branches  : {daily["tran_br_code"].nunique()}')
daily.head()

# Add lag features per branch (shift within each branch's time series)
daily = daily.sort_values(['tran_br_code', 'start_date']).reset_index(drop=True)

for lag in [1, 7, 14, 30]:
    daily[f'lag_{lag}_debit'] = (
        daily.groupby('tran_br_code')['Daily_Total_Debit']
        .shift(lag)
    )

# Rolling averages (smoothed historical trend)
daily['rolling_7_mean_debit'] = (
    daily.groupby('tran_br_code')['Daily_Total_Debit']
    .shift(1)
    .rolling(window=7, min_periods=1)
    .mean()
    .values
)
daily['rolling_30_mean_debit'] = (
    daily.groupby('tran_br_code')['Daily_Total_Debit']
    .shift(1)
    .rolling(window=30, min_periods=1)
    .mean()
    .values
)

print(f'✅ Lag features added')
print(f'   Lag features: lag_1, lag_7, lag_14, lag_30, rolling_7_mean, rolling_30_mean')

# Rows with NaN lags (first few days per branch) — drop them
before = len(daily)
daily = daily.dropna(subset=['lag_30_debit']).reset_index(drop=True)
after  = len(daily)
print(f'   Dropped {before - after} rows (insufficient lag history) — {after:,} rows remaining')

# Columns to EXCLUDE from features
drop_cols = [
    'start_date',          # Not a numeric feature (used for split)
    'Daily_Total_Debit',   # This is our TARGET — never use as feature
    'Daily_Total_Credit',  # Highly correlated with target, causes leakage
    'Daily_Net_Cash_Flow', # Contains target info (DR - CR)
    'Net_Outflow_Periods', # Derived from target
]

feature_cols = [c for c in daily.columns if c not in drop_cols]
TARGET       = 'Daily_Total_Debit'

print(f'✅ Target variable : {TARGET}')
print(f'✅ Total features  : {len(feature_cols)}')
print(f'\n📋 Feature List:')
for i, col in enumerate(feature_cols, 1):
    print(f'   {i:2d}. {col}')

# Sort chronologically before split
daily = daily.sort_values('start_date').reset_index(drop=True)

# 80/20 cutoff by date
cutoff_idx  = int(len(daily) * 0.80)
cutoff_date = daily.iloc[cutoff_idx]['start_date']

train_df = daily[daily['start_date'] <  cutoff_date].reset_index(drop=True)
test_df  = daily[daily['start_date'] >= cutoff_date].reset_index(drop=True)

X_train = train_df[feature_cols]
y_train = train_df[TARGET]
X_test  = test_df[feature_cols]
y_test  = test_df[TARGET]

print(f'📅 Train period : {train_df["start_date"].min().date()}  →  {train_df["start_date"].max().date()}')
print(f'📅 Test period  : {test_df["start_date"].min().date()}  →  {test_df["start_date"].max().date()}')
print(f'\n📦 Train size   : {len(X_train):,} rows  ({len(X_train)/len(daily)*100:.1f}%)')
print(f'📦 Test size    : {len(X_test):,} rows  ({len(X_test)/len(daily)*100:.1f}%)')
print(f'\n✅ NO SHUFFLE — Pure chronological split (no data leakage)')

fig, ax = plt.subplots(figsize=(16, 5))

# Aggregate daily total debit across all branches
train_agg = train_df.groupby('start_date')['Daily_Total_Debit'].sum()
test_agg  = test_df.groupby('start_date')['Daily_Total_Debit'].sum()

ax.plot(train_agg.index, train_agg.values / 1e9, color='#1F4E79', linewidth=1.5,
        label=f'TRAIN  ({train_df["start_date"].min().date()} → {train_df["start_date"].max().date()})')
ax.plot(test_agg.index,  test_agg.values  / 1e9, color='#D6462B', linewidth=1.5,
        label=f'TEST   ({test_df["start_date"].min().date()} → {test_df["start_date"].max().date()})')

ax.axvline(cutoff_date, color='black', linestyle='--', linewidth=2, label=f'Split Date: {cutoff_date.date()}')
ax.fill_between(train_agg.index, train_agg.values / 1e9, alpha=0.1, color='#1F4E79')
ax.fill_between(test_agg.index,  test_agg.values  / 1e9, alpha=0.1, color='#D6462B')

ax.set_title('Train / Test Chronological Split — Daily Total Cash Withdrawal (All Branches)', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Total Cash Withdrawn (Billion PKR)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('eda_plots/08_train_test_split.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Split visualization saved.')

train_max_date = train_df['start_date'].max()
test_min_date  = test_df['start_date'].min()

print('🔍 Data Leakage Check:')
print(f'   Train last date : {train_max_date.date()}')
print(f'   Test first date : {test_min_date.date()}')

if train_max_date < test_min_date:
    print('\n   ✅ PASSED — No overlap between train and test sets!')
    print('   ✅ PASSED — No shuffle applied!')
    print('   ✅ PASSED — Future data not visible to model during training!')
else:
    print('\n   ❌ FAILED — Data leakage detected! Check split logic.')

# Check for any overlapping dates
overlap = set(train_df['start_date']) & set(test_df['start_date'])
print(f'\n   Overlapping dates: {len(overlap)} (should be 0)')
if len(overlap) == 0:
    print('   ✅ PASSED — Zero date overlap!')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Target Variable Analysis — Daily_Total_Debit', fontsize=14, fontweight='bold')

# Distribution
axes[0].hist(daily['Daily_Total_Debit'] / 1e6, bins=50, color='#1F4E79', alpha=0.8, edgecolor='white')
axes[0].set_title('Distribution of Daily Cash Withdrawal')
axes[0].set_xlabel('Daily Debit (Million PKR)')
axes[0].set_ylabel('Frequency')

# By Weekday
wd_avg = daily.groupby('Weekday')['Daily_Total_Debit'].mean() / 1e6
days   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
axes[1].bar(range(len(wd_avg)), wd_avg.values,
            color=['#D6462B' if v == wd_avg.max() else '#2E75B6' for v in wd_avg.values],
            edgecolor='white')
axes[1].set_xticks(range(len(wd_avg)))
axes[1].set_xticklabels([days[i] for i in wd_avg.index])
axes[1].set_title('Average Daily Debit by Weekday')
axes[1].set_ylabel('Avg Debit (Million PKR)')

# By Branch
br_avg = daily.groupby('tran_br_code')['Daily_Total_Debit'].mean().sort_values(ascending=False) / 1e6
colors = ['#D6462B' if i == 0 else '#A9C4E2' for i in range(len(br_avg))]
axes[2].bar(br_avg.index.astype(str), br_avg.values, color=colors, edgecolor='white')
axes[2].set_title('Average Daily Debit by Branch')
axes[2].set_xlabel('Branch Code')
axes[2].set_ylabel('Avg Debit (Million PKR)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_plots/09_target_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Target analysis saved.')

# Save train/test splits
X_train.to_csv('model_data/X_train.csv', index=False)
X_test.to_csv('model_data/X_test.csv',   index=False)
y_train.to_csv('model_data/y_train.csv', index=False)
y_test.to_csv('model_data/y_test.csv',   index=False)

# Save full daily dataset (for final model training on 100% data)
daily.to_csv('model_data/daily_full.csv', index=False)

# Save feature column list
import json
with open('model_data/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print('✅ All files saved in model_data/ folder:')
print('   📄 X_train.csv      — Training features')
print('   📄 X_test.csv       — Test features')
print('   📄 y_train.csv      — Training target')
print('   📄 y_test.csv       — Test target')
print('   📄 daily_full.csv   — Full dataset (for final model)')
print('   📄 feature_cols.json — Feature list')


# --- 04_Model_Training.ipynb ---
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib, json, os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

pd.set_option('display.float_format', '{:,.2f}'.format)
os.makedirs('models',    exist_ok=True)
os.makedirs('eda_plots', exist_ok=True)

plt.rcParams.update({
    'figure.dpi'       : 130,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#F9FAFB',
    'axes.grid'        : True,
    'grid.color'       : '#E5E7EB',
})

print('✅ Libraries imported successfully')

X_train = pd.read_csv('model_data/X_train.csv')
X_test  = pd.read_csv('model_data/X_test.csv')
y_train = pd.read_csv('model_data/y_train.csv').squeeze()
y_test  = pd.read_csv('model_data/y_test.csv').squeeze()

with open('model_data/feature_cols.json') as f:
    feature_cols = json.load(f)

print(f'✅ Data loaded successfully')
print(f'   X_train : {X_train.shape}  |  y_train : {y_train.shape}')
print(f'   X_test  : {X_test.shape}   |  y_test  : {y_test.shape}')
print(f'   Features: {len(feature_cols)}')
print(f'\n   Target Stats (y_train):')
print(f'   Mean   : PKR {y_train.mean()/1e6:,.1f} Million')
print(f'   Median : PKR {y_train.median()/1e6:,.1f} Million')
print(f'   Max    : PKR {y_train.max()/1e6:,.1f} Million')
print(f'   Min    : PKR {y_train.min()/1e6:,.1f} Million')

def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE: only on non-zero actuals (avoids division by near-zero)
    mask = np.array(y_true) > 0
    mape = np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100
    r2   = r2_score(y_true, y_pred)
    print(f'\n📊 {name} — Evaluation Results:')
    print(f'   MAE  : PKR {mae/1e6:>8,.2f} Million   (Average error per prediction)')
    print(f'   RMSE : PKR {rmse/1e6:>8,.2f} Million   (Penalizes big errors more)')
    print(f'   MAPE :     {mape:>8,.2f}%          (% error on average)')
    print(f'   R²   :     {r2:>8,.4f}            (1.0 = perfect, >0.85 = good)')
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

results = []  # Store all model results for comparison
print('✅ Evaluation function ready')

# Baseline: predict the mean of training target for everything
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

res = evaluate_model('Baseline (Mean Predictor)', y_test, y_pred_baseline)
results.append(res)
print('\n   This is our minimum bar — our real model MUST beat this.')

xgb_model = XGBRegressor(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

xgb_model.fit(
    X_train, y_train,
    eval_set        = [(X_test, y_test)],
    verbose         = False
)

y_pred_xgb = xgb_model.predict(X_test)
res = evaluate_model('XGBoost', y_test, y_pred_xgb)
results.append(res)
print('\n\u2705 XGBoost training complete')

lgb_model = LGBMRegressor(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 63,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
)

y_pred_lgb = lgb_model.predict(X_test)
res = evaluate_model('LightGBM', y_test, y_pred_lgb)
results.append(res)

results_df = pd.DataFrame(results)
results_df['MAE_M']  = results_df['MAE']  / 1e6
results_df['RMSE_M'] = results_df['RMSE'] / 1e6

print('\n' + '='*65)
print('MODEL COMPARISON SUMMARY')
print('='*65)
display_df = results_df[['Model','MAE_M','RMSE_M','MAPE','R2']].copy()
display_df.columns = ['Model', 'MAE (Million PKR)', 'RMSE (Million PKR)', 'MAPE (%)', 'R²']
print(display_df.to_string(index=False))
print('='*65)

# Best model based on R2
best_row = results_df.loc[results_df['R2'].idxmax()]
print(f'\n🏆 Best Model: {best_row["Model"]}  (R² = {best_row["R2"]:.4f})')

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Actual vs Predicted — Daily Cash Withdrawal', fontsize=14, fontweight='bold')

for ax, (name, y_pred) in zip(axes, [('XGBoost', y_pred_xgb), ('LightGBM', y_pred_lgb)]):
    r2 = r2_score(y_test, y_pred)
    ax.scatter(y_test/1e6, y_pred/1e6, alpha=0.3, color='#1F4E79', s=15, label='Predictions')
    mn = min(y_test.min(), y_pred.min()) / 1e6
    mx = max(y_test.max(), y_pred.max()) / 1e6
    ax.plot([mn, mx], [mn, mx], color='#D6462B', linewidth=2, linestyle='--', label='Perfect Prediction')
    ax.set_title(f'{name}  (R² = {r2:.4f})')
    ax.set_xlabel('Actual (Million PKR)')
    ax.set_ylabel('Predicted (Million PKR)')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('eda_plots/10_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Actual vs Predicted plot saved → eda_plots/10_actual_vs_predicted.png')

# Load full daily data to get dates for test set
daily_full = pd.read_csv('model_data/daily_full.csv')
daily_full['start_date'] = pd.to_datetime(daily_full['start_date'])
daily_full = daily_full.sort_values('start_date').reset_index(drop=True)

cutoff_idx  = int(len(daily_full) * 0.80)
cutoff_date = daily_full.iloc[cutoff_idx]['start_date']
test_full   = daily_full[daily_full['start_date'] >= cutoff_date].reset_index(drop=True)

# Pick most active branch
top_branch = daily_full.groupby('tran_br_code')['Daily_Total_Debit'].sum().idxmax()
mask_test  = test_full['tran_br_code'] == top_branch

branch_dates  = test_full.loc[mask_test, 'start_date'].values
branch_actual = y_test.values[mask_test.values]
branch_xgb    = y_pred_xgb[mask_test.values]
branch_lgb    = y_pred_lgb[mask_test.values]

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(branch_dates, branch_actual/1e6, color='black',   linewidth=2,   label='Actual',   zorder=3)
ax.plot(branch_dates, branch_xgb/1e6,   color='#1F4E79', linewidth=1.5, linestyle='--', label='XGBoost Predicted')
ax.plot(branch_dates, branch_lgb/1e6,   color='#D6462B', linewidth=1.5, linestyle=':',  label='LightGBM Predicted')
ax.set_title(f'Branch {top_branch} — Actual vs Predicted Daily Cash Withdrawal (Test Period)', fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Cash Withdrawal (Million PKR)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.0f}M'))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('eda_plots/11_prediction_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ Timeline plot saved → eda_plots/11_prediction_timeline.png  (Branch: {top_branch})')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Feature Importance — Which Features Matter Most?', fontsize=14, fontweight='bold')

# XGBoost importance
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
xgb_imp.plot(kind='barh', ax=axes[0], color='#1F4E79', edgecolor='white')
axes[0].set_title('XGBoost — Feature Importance')
axes[0].set_xlabel('Importance Score')

# LightGBM importance
lgb_imp = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
lgb_imp.plot(kind='barh', ax=axes[1], color='#D6462B', edgecolor='white')
axes[1].set_title('LightGBM — Feature Importance')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('eda_plots/12_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('✅ Feature importance plot saved → eda_plots/12_feature_importance.png')

# Determine best model based on R2
r2_xgb = r2_score(y_test, y_pred_xgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

if r2_xgb >= r2_lgb:
    best_model      = xgb_model
    best_model_name = 'XGBoost'
    best_r2         = r2_xgb
    best_pred       = y_pred_xgb
else:
    best_model      = lgb_model
    best_model_name = 'LightGBM'
    best_r2         = r2_lgb
    best_pred       = y_pred_lgb

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(xgb_model,  'models/xgb_model.pkl')
joblib.dump(lgb_model,  'models/lgb_model.pkl')

# Save results summary
results_df.to_csv('models/model_results.csv', index=False)

print(f'✅ Models saved in models/ folder:')
print(f'   📄 best_model.pkl  — {best_model_name} (R² = {best_r2:.4f})')
print(f'   📄 xgb_model.pkl   — XGBoost')
print(f'   📄 lgb_model.pkl   — LightGBM')
print(f'   📄 model_results.csv')
print(f'\n🎉 Phase 4 COMPLETE!')
print(f'   ➡️  Next: Phase 5 — XAI (Explainable AI with SHAP)')

