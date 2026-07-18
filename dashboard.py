"""
Phase 8 — Bank Cash Optimization Dashboard (Enhanced)
Includes: Cash Need Calendar, SHAP-based Explanation, Branch Forecast
Sir's Requirement: Daily cash requirement by branch + SHAP reasoning
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import joblib, json, os, shap, warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Cash Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 60%,#16213e 100%); }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e1e3a,#252550);
    border:1px solid #3a3a6c; border-radius:14px; padding:18px;
    box-shadow:0 6px 24px rgba(0,0,0,.5);
}
div[data-testid="metric-container"] label { color:#9999cc !important; font-size:11px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color:#e0e0ff !important; font-size:26px !important; font-weight:700 !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d0d1f,#1a1a2e); border-right:1px solid #2a2a4a; }
h1 { color:#a78bfa !important; font-weight:700 !important; }
h2 { color:#818cf8 !important; }
h3 { color:#6ee7b7 !important; }
.explain-box {
    background:linear-gradient(135deg,#1a1a3e,#1e2a3a);
    border:1px solid #3a5a7c; border-radius:14px;
    padding:20px; margin:12px 0;
}
.reason-item { padding:6px 0; border-bottom:1px solid #2a2a4a; }
.stTabs [data-baseweb="tab-list"] { background:#1a1a2e; border-radius:10px; gap:4px; padding:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#9999cc; border-radius:8px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:linear-gradient(135deg,#7c3aed,#4338ca); color:white !important; }
.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#4338ca); color:white;
    border:none; border-radius:10px; font-weight:600; padding:10px 24px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor':'#0f0f1a','axes.facecolor':'#1a1a2e',
    'axes.edgecolor':'#444466','axes.labelcolor':'#c0c0d0',
    'xtick.color':'#c0c0d0','ytick.color':'#c0c0d0',
    'text.color':'#e0e0f0','grid.color':'#2a2a3e',
    'grid.linestyle':'--','grid.alpha':0.5,'font.family':'DejaVu Sans',
})

# ─── Load Resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('model_data/daily_full.csv', parse_dates=['start_date'])
    return df.sort_values(['tran_br_code','start_date']).reset_index(drop=True)

@st.cache_data
def load_forecast():
    fc = pd.read_excel('models/forecast_next30days.xlsx', sheet_name='All_Branches')
    fc['Date'] = pd.to_datetime(fc['Date'])
    return fc

@st.cache_data
def load_shap():
    sv = pd.read_csv('models/shap_values.csv')
    with open('models/shap_expected_value.json') as f:
        ev = json.load(f)
    return sv, ev['expected_value']

@st.cache_data
def load_features():
    # Using V3 feature columns
    return [
        'AM_PM_Encoded', 'Txn_Count', 'Weekday', 'Is_Weekend', 'Month', 'Day',
        'Is_Salary_Day', 'Is_Holiday',
        'lag_1_Half_Day_Total_Debit', 'lag_2_Half_Day_Total_Debit', 'lag_14_Half_Day_Total_Debit', 'lag_60_Half_Day_Total_Debit', 'rolling_14_mean_Half_Day_Total_Debit',
        'lag_1_Half_Day_Total_Credit', 'lag_2_Half_Day_Total_Credit', 'lag_14_Half_Day_Total_Credit', 'lag_60_Half_Day_Total_Credit', 'rolling_14_mean_Half_Day_Total_Credit',
        'lag_1_Half_Day_Net_Cash', 'lag_2_Half_Day_Net_Cash', 'lag_14_Half_Day_Net_Cash', 'lag_60_Half_Day_Net_Cash', 'rolling_14_mean_Half_Day_Net_Cash'
    ]

@st.cache_data
def load_forecast_features():
    ff = pd.read_csv('models/forecast_features.csv')
    ff['start_date'] = pd.to_datetime(ff['start_date'])
    return ff

@st.cache_data
def load_branch_metrics():
    return pd.read_csv('models/branch_metrics.csv')

@st.cache_data
def load_eval_report():
    return pd.read_csv('models/final_evaluation_report.csv')

model          = load_model()
df             = load_data()
forecast_df    = load_forecast()
shap_values, base_value = load_shap()
feature_cols   = load_features()
forecast_feats = load_forecast_features()
branch_metrics = load_branch_metrics()
eval_report    = load_eval_report()

BRANCHES  = sorted(df['tran_br_code'].unique().tolist())
LAST_DATE = df['start_date'].max()
BRANCH_AVGS = df.groupby('tran_br_code')[['Daily_Txn_Count','Branch_Total_Debit',
    'Branch_Txn_Count','Branch_Avg_Net_CF','Peak_Hour_Txns','Business_Hour_Txns']].mean()

WEEKDAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
CONF_COLORS   = {'HIGH':'#2ecc71','MEDIUM':'#f39c12','LOW':'#e74c3c'}

# ─── Feature human-readable names ────────────────────────────────────────────
FEATURE_LABELS = {
    'rolling_14_mean_Half_Day_Total_Debit' : ('📅 14-Day Average',    'Average withdrawal over the past 14 days'),
    'lag_1_Half_Day_Total_Debit'           : ('⏮️ Last Half-Day Withdrawal', 'Cash withdrawal amount from the previous half-day'),
    'lag_2_Half_Day_Total_Debit'           : ('⏮️ Yesterday Withdrawal', 'Cash withdrawal amount from yesterday same time'),
    'lag_14_Half_Day_Total_Debit'          : ('📅 14 Days Ago',        'Cash withdrawal amount from exactly two weeks ago'),
    'lag_60_Half_Day_Total_Debit'          : ('📅 60 Days Ago',        'Cash withdrawal amount from exactly two months ago'),
    'rolling_14_mean_Half_Day_Total_Credit': ('📅 14-Day Avg Deposit', 'Average deposit over the past 14 days'),
    'lag_1_Half_Day_Total_Credit'          : ('⏮️ Last Half-Day Deposit', 'Deposit amount from the previous half-day'),
    'lag_2_Half_Day_Total_Credit'          : ('⏮️ Yesterday Deposit', 'Deposit amount from yesterday same time'),
    'lag_14_Half_Day_Total_Credit'         : ('📅 14 Days Ago Deposit','Deposit amount from exactly two weeks ago'),
    'lag_60_Half_Day_Total_Credit'         : ('📅 60 Days Ago Deposit','Deposit amount from exactly two months ago'),
    'rolling_14_mean_Half_Day_Net_Cash'    : ('📅 14-Day Avg Net Cash','Average net cash over the past 14 days'),
    'lag_1_Half_Day_Net_Cash'              : ('⏮️ Last Half-Day Net Cash','Net cash amount from the previous half-day'),
    'lag_2_Half_Day_Net_Cash'              : ('⏮️ Yesterday Net Cash', 'Net cash amount from yesterday same time'),
    'lag_14_Half_Day_Net_Cash'             : ('📅 14 Days Ago Net Cash','Net cash amount from exactly two weeks ago'),
    'lag_60_Half_Day_Net_Cash'             : ('📅 60 Days Ago Net Cash','Net cash amount from exactly two months ago'),
    'Txn_Count'             : ('🔢 Transactions',      'Total number of transactions'),
    'Is_Salary_Day'         : ('💰 Salary Day',        'Whether the day is near salary day'),
    'Is_Holiday'            : ('🎉 Holiday',           'Whether the day is a public holiday'),
    'Weekday'               : ('📆 Day (Weekday)',     'Day of the week'),
    'Is_Weekend'            : ('🏖️ Weekend',           'Whether the day is a weekend'),
    'Month'                 : ('🗓️ Month',             'Month of the year'),
    'Day'                   : ('🔢 Date',              'Date of the month'),
    'AM_PM_Encoded'         : ('☀️/🌙 Time of Day',    'Morning (AM) or Afternoon (PM)'),
}

def get_shap_for_row(X_row_am, X_row_pm):
    """Compute SHAP values for AM and PM, sum linear impacts, and return total Daily SHAP."""
    explainer = shap.TreeExplainer(model)
    
    # AM
    sv_am = explainer.shap_values(X_row_am)[0]
    pred_log_am = model.predict(X_row_am)[0]
    pred_am = np.expm1(pred_log_am)
    base_log = explainer.expected_value
    base_val = np.expm1(base_log)
    
    diff_am = pred_am - base_val
    sum_abs_am = sum(abs(sv_am))
    sv_linear_am = (sv_am / sum_abs_am) * diff_am if sum_abs_am != 0 else sv_am * 0
    
    # PM
    sv_pm = explainer.shap_values(X_row_pm)[0]
    pred_log_pm = model.predict(X_row_pm)[0]
    pred_pm = np.expm1(pred_log_pm)
    
    diff_pm = pred_pm - base_val
    sum_abs_pm = sum(abs(sv_pm))
    sv_linear_pm = (sv_pm / sum_abs_pm) * diff_pm if sum_abs_pm != 0 else sv_pm * 0
    
    # Aggregate
    total_shap = sv_linear_am + sv_linear_pm
    total_base = base_val * 2
    
    return total_shap, total_base, pred_am + pred_pm

def build_explanation(shap_vals, feature_vals, feat_names, base_val, prediction, branch, date):
    """Return markdown explanation of why this prediction was made."""
    pairs = sorted(zip(shap_vals, feat_names), key=lambda x: abs(x[0]), reverse=True)[:6]

    lines = []
    lines.append(f"### 🔍 Explanation: Branch **{branch}** on **{date}**\n")
    lines.append(f"**Base prediction** (average of all branches/days): **PKR {base_val/1e6:.1f}M**\n")
    lines.append(f"**Final prediction**: **PKR {prediction:.1f}M**\n")
    lines.append("---\n#### Top Reasons:\n")

    for sv, fn in pairs:
        label, desc = FEATURE_LABELS.get(fn, (fn, fn))
        direction   = "⬆️ INCREASED" if sv > 0 else "⬇️ DECREASED"
        color_word  = "more" if sv > 0 else "less"
        sv_m        = sv / 1e6
        val         = feature_vals.get(fn, '?')
        if isinstance(val, float) and abs(val) > 1000:
            val_str = f"{val/1e6:.1f}M PKR"
        elif isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)

        lines.append(f"- {direction} **{label}** — {desc}\n"
                     f"  - Value: `{val_str}` → Model predicted **{color_word}**  \n"
                     f"  - Impact: `{sv_m:+.2f}M PKR`\n")

    lines.append("---\n")
    diff = prediction - base_val/1e6
    if diff > 0:
        lines.append(f"✅ **Net Result:** Base ({base_val/1e6:.1f}M) + combined effect of features = **{prediction:.1f}M PKR**\n"
                     f"  _(Features INCREASED prediction by {diff:.1f}M PKR)_\n")
    else:
        lines.append(f"✅ **Net Result:** Base ({base_val/1e6:.1f}M) + combined effect of features = **{prediction:.1f}M PKR**\n"
                     f"  _(Features DECREASED prediction by {abs(diff):.1f}M PKR)_\n")
    return "\n".join(lines)

def get_top_shap_reasons_str(X_row, shap_values_row):
    """Get a short plain text explanation of top 2 SHAP features for the calendar view."""
    pairs = sorted(zip(shap_values_row, feature_cols, X_row.values[0]), key=lambda x: abs(x[0]), reverse=True)
    reasons = []
    for sv, fn, fv in pairs[:2]:
        if abs(sv) < 0.1: continue
        label = FEATURE_LABELS.get(fn, (fn, fn))[0]
        label = label.split(' ', 1)[1] if ' ' in label else label  # Remove emoji if present, or keep it short
        
        # Override for clearer plain language
        if fn == 'Day' and (fv <= 5 or fv >= 27):
            label = "Salary Day"
        elif fn == 'Is_Weekend' and fv == 1:
            label = "Weekend"
            
        sign = "⬆️" if sv > 0 else "⬇️"
        reasons.append(f"{sign} {label} ({abs(sv)/1e6:.1f}M)")
    return " | ".join(reasons) if reasons else "Normal Pattern"

def get_forecast_features(branch, target_date):
    """Fetch AM and PM feature rows for a future date from the pre-generated forecast features."""
    target_date = pd.to_datetime(target_date)
    rows = forecast_feats[(forecast_feats['tran_br_code'] == branch) & (forecast_feats['start_date'] == target_date)]
    if len(rows) == 0:
        return None, None
    
    am_row = rows[rows['AM_PM'] == 'AM'].iloc[0].to_dict()
    pm_row = rows[rows['AM_PM'] == 'PM'].iloc[0].to_dict()
    return am_row, pm_row

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Cash Intelligence")
    st.markdown("*Bank Cash Optimization System*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📅 Cash Need Calendar",
        "🔍 Why This Amount? (SHAP)",
        "🔮 Branch Forecast",
        "⚖️ Model Comparison (Benchmark)",
        "🕹️ What-If Simulator",
        "📊 Overview",
        "📈 Model Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** XGBoost V3 (TimeSeries Tuned)  \n**R²:** 0.6334  \n**MAE:** 9.52M PKR")
    st.markdown(f"**Data till:** {LAST_DATE.date()}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CASH NEED CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
if page == "📅 Cash Need Calendar":
    st.title("📅 Cash Need Calendar")
    st.markdown("*Daily cash requirement by branch — 30-day view*")
    st.markdown("---")

    view = st.radio("View Type", ["🏦 Per Branch (Daily)", "📊 All Branches Heatmap"], horizontal=True)

    if view == "🏦 Per Branch (Daily)":
        sel_br = st.selectbox("Select Branch", BRANCHES)
        br_fc  = forecast_df[forecast_df['Branch']==sel_br].sort_values('Date').copy()
        
        with st.spinner("Analyzing SHAP reasons for the calendar..."):
            top_drivers = []
            for dt in br_fc['Date']:
                am_row, pm_row = get_forecast_features(sel_br, dt)
                if am_row is None:
                    top_drivers.append("Normal Pattern")
                    continue
                X_am = pd.DataFrame([am_row])[feature_cols]
                X_pm = pd.DataFrame([pm_row])[feature_cols]
                
                sv_total, _, _ = get_shap_for_row(X_am, X_pm)
                
                # Use AM row features as proxy for feature values display
                top_drivers.append(get_top_shap_reasons_str(X_am, sv_total))
                
            br_fc['Top_Drivers'] = top_drivers

        br_fc['DayName'] = br_fc['Date'].dt.strftime('%a')
        br_fc['DateStr'] = br_fc['Date'].dt.strftime('%d %b')
        br_fc['Week']    = ((br_fc['Step']-1) // 7) + 1

        st.subheader(f"🏦 Branch {sel_br} — 30-Day Cash Calendar")

        # --- KPI Cards ---
        br_metrics = branch_metrics[branch_metrics['Branch']==sel_br]
        accuracy = 100 - br_metrics['MAPE_%'].values[0] if len(br_metrics)>0 else 0
        avg_debit = df[df['tran_br_code']==sel_br]['Daily_Total_Debit'].mean()
        net_cf = BRANCH_AVGS.loc[sel_br, 'Branch_Avg_Net_CF']
        avg_credit = avg_debit + net_cf  # Approximate cash in
        
        st.markdown(f"""
        <div style="display:flex; gap:12px; margin-bottom: 24px;">
            <div style="flex:1; background:linear-gradient(135deg, #3a1c1c, #e74c3c); padding:16px; border-radius:12px; border:1px solid #ff7979; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="color:#ffd0d0; font-size:12px; font-weight:600; text-transform:uppercase;">Avg Daily Debit (Out)</div>
                <div style="color:white; font-size:24px; font-weight:bold;">{avg_debit/1e6:.1f}M</div>
            </div>
            <div style="flex:1; background:linear-gradient(135deg, #1c3a24, #2ecc71); padding:16px; border-radius:12px; border:1px solid #58d68d; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="color:#d0ffd0; font-size:12px; font-weight:600; text-transform:uppercase;">Avg Daily Credit (In)</div>
                <div style="color:white; font-size:24px; font-weight:bold;">{avg_credit/1e6:.1f}M</div>
            </div>
            <div style="flex:1; background:linear-gradient(135deg, #1c283a, #3498db); padding:16px; border-radius:12px; border:1px solid #5dade2; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="color:#d0eaff; font-size:12px; font-weight:600; text-transform:uppercase;">Net Cash Flow</div>
                <div style="color:white; font-size:24px; font-weight:bold;">{net_cf/1e6:+.1f}M</div>
            </div>
            <div style="flex:1; background:linear-gradient(135deg, #2c3e50, #95a5a6); padding:16px; border-radius:12px; border:1px solid #bdc3c7; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="color:#e0e0e0; font-size:12px; font-weight:600; text-transform:uppercase;">Model Accuracy</div>
                <div style="color:white; font-size:24px; font-weight:bold;">{accuracy:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Week-wise grouped cards
        for wk in [1,2,3,4]:
            wk_data = br_fc[br_fc['Week']==wk]
            conf_label = "🟢 HIGH Confidence" if wk==1 else ("🟡 MEDIUM Confidence" if wk==2 else "🔴 LOW Confidence")
            st.markdown(f"**Week {wk} — {conf_label}**")
            cols = st.columns(min(len(wk_data), 7))
            for ci, (_, row) in enumerate(wk_data.iterrows()):
                with cols[ci % 7]:
                    bg = "#1a3a1a" if row['Confidence']=='HIGH' else ("#3a2a00" if row['Confidence']=='MEDIUM' else "#3a1a1a")
                    border = "#2ecc71" if row['Confidence']=='HIGH' else ("#f39c12" if row['Confidence']=='MEDIUM' else "#e74c3c")
                    st.markdown(f"""
                    <div style='background:{bg};border:1px solid {border};border-radius:10px;padding:10px;text-align:center;margin:4px 0;height:120px;display:flex;flex-direction:column;justify-content:center;'>
                        <div style='color:#888;font-size:11px'>{row['DayName']}</div>
                        <div style='color:#e0e0ff;font-weight:700;font-size:13px'>{row['DateStr']}</div>
                        <div style='color:#a78bfa;font-size:18px;font-weight:700'>{row['Predicted_M']:.0f}M</div>
                        <div style='color:#bbb;font-size:9px;margin-top:4px;min-height:22px;line-height:1.2;'><i>{row['Top_Drivers']}</i></div>
                        <div style='color:#666;font-size:9px;margin-top:auto;'>±{row['Uncertainty_Pct']:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("")

        # Timeline chart
        fig, ax = plt.subplots(figsize=(14, 4))
        x = np.arange(len(br_fc))
        colors_bar = [CONF_COLORS[c] for c in br_fc['Confidence']]
        ax.bar(x, br_fc['Predicted_M'], color=colors_bar, alpha=0.85, edgecolor='#0f0f1a', width=0.7)
        ax.plot(x, br_fc['Predicted_M'], color='white', linewidth=1.5, marker='o', markersize=4, zorder=5)
        ax.fill_between(x, br_fc['Lower_M'], br_fc['Upper_M'], alpha=0.15, color='#7c3aed')
        xticks = [f"{r['DayName']}\n{r['DateStr']}" for _, r in br_fc.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, fontsize=7, rotation=45)
        ax.set_ylabel('Cash (Million PKR)', color='#c0c0d0')
        ax.set_title(f'Branch {sel_br} — Daily Cash Need (Next 30 Days)', color='#e0e0f0', fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.0f}M'))
        
        # Add Annotations for Salary Days
        for i, r in br_fc.reset_index(drop=True).iterrows():
            if r['Date'].day in [1, 2, 3, 4, 5, 27, 28, 29, 30, 31]:
                ax.axvline(x=i, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.6)
                if i == 0 or br_fc.iloc[i-1]['Date'].day not in [1, 2, 3, 4, 5, 27, 28, 29, 30, 31]:
                    ax.text(i, ax.get_ylim()[1]*0.95, '💰 Salary Day', color='#e74c3c', fontsize=9, rotation=90, va='top', ha='right')

        patches = [mpatches.Patch(color=c, label=f'{l}') for l,c in CONF_COLORS.items()]
        ax.legend(handles=patches, facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#c0c0d0')
        ax.grid(True, axis='y', alpha=0.4)
        st.pyplot(fig); plt.close()

        # Table with Highlighting and SHAP Explanations
        disp = br_fc[['DateStr','DayName','Predicted_M','Top_Drivers','Lower_M','Upper_M','Uncertainty_Pct','Confidence']].copy()
        disp.columns = ['Date','Day','Predicted (M PKR)','Why? (Top Drivers)','Lower (M)','Upper (M)','Uncertainty %','Confidence']
        
        def highlight_risk(row):
            # Highlight Light Red for LOW confidence or High Demand (Top 20%)
            high_demand_thresh = br_fc['Predicted_M'].quantile(0.8)
            if row['Confidence'] == 'LOW':
                return ['background-color: rgba(231, 76, 60, 0.25)'] * len(row)
            elif row['Predicted (M PKR)'] > high_demand_thresh:
                return ['background-color: rgba(243, 156, 18, 0.2)'] * len(row)
            return [''] * len(row)

        st.dataframe(disp.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

        # Download
        csv = disp.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, f"branch_{sel_br}_cash_calendar.csv", "text/csv")

    else:  # All Branches Heatmap
        st.subheader("📊 All Branches — Cash Need Heatmap (Next 30 Days)")
        heat = forecast_df.pivot_table(index='Branch', columns='Date', values='Predicted_M')
        heat.columns = [d.strftime('%d %b') if hasattr(d,'strftime') else str(d) for d in pd.to_datetime(heat.columns)]
        heat.index = heat.index.astype(str)

        # Show every 3 days
        step_cols = heat.columns[::3]
        heat_disp = heat[step_cols]

        fig, ax = plt.subplots(figsize=(20, 7))
        fig.patch.set_facecolor('#0f0f1a')
        sns.heatmap(heat_disp, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
                    linewidths=0.5, linecolor='#0f0f1a',
                    annot_kws={'size':9,'color':'#0f0f1a'},
                    cbar_kws={'label':'Cash Need (M PKR)','shrink':0.8})
        ax.set_title('Branch × Date Cash Need Heatmap (M PKR) — Darker = More Cash',
                     color='#e0e0f0', fontsize=14, fontweight='bold')
        ax.set_xlabel('Forecast Date', color='#c0c0d0', fontsize=11)
        ax.set_ylabel('Branch', color='#c0c0d0', fontsize=11)
        ax.tick_params(colors='#c0c0d0')
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('#c0c0d0')
        cbar.ax.tick_params(colors='#c0c0d0')
        st.pyplot(fig); plt.close()

        st.info("💡 **Tip:** Darker color = More cash required by the branch on that day. This can be used directly for replenishment scheduling.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: WHY THIS AMOUNT? (SHAP EXPLANATION)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Why This Amount? (SHAP)":
    st.title("🔍 Why This Amount?")
    st.markdown("*SHAP-based explanation — how the model made this prediction*")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sel_br   = st.selectbox("Select Branch", BRANCHES)
    with col2:
        fc_dates = forecast_df[forecast_df['Branch']==sel_br]['Date'].dt.date.tolist()
        sel_date = st.selectbox("Select Date (Forecast Days)", fc_dates)

    if st.button("🔍 Explain — Why This Amount?", use_container_width=True):
        with st.spinner("Running SHAP analysis..."):
            target_dt  = pd.Timestamp(sel_date)
            feat_row_am, feat_row_pm = get_forecast_features(sel_br, target_dt)
            
            if feat_row_am is None:
                st.error("Feature data not found for this date. Run forecast pipeline to generate features.")
                st.stop()
                
            X_row_am = pd.DataFrame([feat_row_am])[feature_cols]
            X_row_pm = pd.DataFrame([feat_row_pm])[feature_cols]
            
            sv, bv, pred_val = get_shap_for_row(X_row_am, X_row_pm)
            feat_row = feat_row_am  # for display values

            # Get forecast row for confidence info
            fc_row = forecast_df[(forecast_df['Branch']==sel_br) &
                                 (forecast_df['Date'].dt.date == sel_date)]
            conf = fc_row['Confidence'].values[0] if len(fc_row) > 0 else 'N/A'
            unc  = fc_row['Uncertainty_Pct'].values[0] if len(fc_row) > 0 else 0
            lower= fc_row['Lower_M'].values[0] if len(fc_row) > 0 else 0
            upper= fc_row['Upper_M'].values[0] if len(fc_row) > 0 else 0

        # ── Main Prediction Box ───────────────────────────────────────────────
        weekday_name = WEEKDAY_NAMES[target_dt.weekday()]
        conf_color = {'HIGH':'#2ecc71','MEDIUM':'#f39c12','LOW':'#e74c3c'}.get(conf,'#888')
        conf_icon  = {'HIGH':'🟢','MEDIUM':'🟡','LOW':'🔴'}.get(conf,'⚪')

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1a3e,#1e2a3a);
                    border:2px solid #7c3aed; border-radius:16px; padding:24px; margin:12px 0;
                    text-align:center;'>
            <div style='color:#9999cc;font-size:14px'>Branch {sel_br} — {weekday_name}, {target_dt.strftime('%d %B %Y')}</div>
            <div style='color:#a78bfa;font-size:48px;font-weight:700;margin:8px 0'>{pred_val/1e6:.1f}M PKR</div>
            <div style='color:#c0c0d0;font-size:14px'>Cash withdrawal predicted</div>
            <div style='color:{conf_color};margin-top:10px;font-size:16px'>
                {conf_icon} {conf} Confidence &nbsp;|&nbsp; Range: {lower:.1f}M – {upper:.1f}M PKR &nbsp;|&nbsp; ±{unc:.0f}% uncertainty
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── SHAP Waterfall ────────────────────────────────────────────────────
        st.subheader("📊 Feature Contributions (Waterfall)")

        # Build sorted SHAP pairs
        pairs = sorted(zip(sv, feature_cols, [feat_row.get(f,0) for f in feature_cols]),
                       key=lambda x: abs(x[0]), reverse=True)[:8]
        feat_names_disp = [FEATURE_LABELS.get(fn,(fn,''))[0] for _,fn,_ in pairs]
        shap_vals_disp  = [s/1e6 for s,_,_ in pairs]
        bar_colors      = ['#2ecc71' if s>0 else '#e74c3c' for s in shap_vals_disp]

        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(feat_names_disp))
        ax.barh(y_pos, shap_vals_disp, color=bar_colors, edgecolor='#0f0f1a', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names_disp, fontsize=10, color='#e0e0f0')
        ax.axvline(0, color='white', linewidth=1.5)
        ax.set_xlabel('SHAP Value (Million PKR impact on prediction)', color='#c0c0d0')
        ax.set_title(f'Why Branch {sel_br} needs {pred_val/1e6:.1f}M on {sel_date}?',
                     color='#e0e0f0', fontweight='bold')
        for i, (val, (sv_raw,_,_)) in enumerate(zip(shap_vals_disp, pairs)):
            ax.text(val + (0.2 if val >= 0 else -0.2), i,
                    f'{val:+.1f}M', va='center', ha='left' if val>=0 else 'right',
                    fontsize=9, color='#e0e0f0', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        st.pyplot(fig); plt.close()

        # ── Plain Language Explanation ────────────────────────────────────────
        st.markdown("---")
        st.subheader("📝 Plain Language Explanation")

        base_m = bv / 1e6
        pred_m = pred_val / 1e6

        st.markdown(f"""
        <div class='explain-box'>
            <h4 style='color:#a78bfa'>🏦 Branch {sel_br} requires <span style='color:#6ee7b7'>{pred_m:.1f}M PKR</span> on {sel_date} ({weekday_name})</h4>
            <p style='color:#9999cc;font-size:13px'>Base amount (average prediction): <b style='color:#e0e0f0'>{base_m:.1f}M PKR</b></p>
            <hr style='border-color:#2a2a4a'>
            <p style='color:#c0c0d0;font-weight:600'>Top Reasons:</p>
        """, unsafe_allow_html=True)

        for sv_val, fn, fv in pairs[:5]:
            label, desc = FEATURE_LABELS.get(fn, (fn, fn))
            sv_m = sv_val / 1e6
            direction = "⬆️ INCREASE" if sv_m > 0 else "⬇️ DECREASE"
            color = "#2ecc71" if sv_m > 0 else "#e74c3c"
            impact_word = "increased" if sv_m > 0 else "decreased"
            if isinstance(fv, float) and abs(fv) > 1000:
                fv_str = f"{fv/1e6:.1f}M PKR"
            else:
                fv_str = f"{fv:.1f}" if isinstance(fv, float) else str(fv)

            st.markdown(f"""
            <div class='reason-item'>
                <span style='color:{color};font-weight:700'>{direction} {abs(sv_m):.1f}M PKR</span>
                &nbsp;—&nbsp;
                <span style='color:#a78bfa;font-weight:600'>{label}</span>
                <br>
                <span style='color:#888;font-size:12px'>{desc} → Value: <code>{fv_str}</code> → {impact_word} prediction</span>
            </div>
            """, unsafe_allow_html=True)

        diff = pred_m - base_m
        diff_word = f"INCREASED by {diff:.1f}M" if diff > 0 else f"DECREASED by {abs(diff):.1f}M"
        st.markdown(f"""
            <hr style='border-color:#2a2a4a'>
            <p style='color:#c0c0d0'>
                Base: <b style='color:#e0e0f0'>{base_m:.1f}M</b> + Features <b style='color:#6ee7b7'>{diff_word}</b>
                = <b style='color:#a78bfa;font-size:18px'>{pred_m:.1f}M PKR</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: BRANCH FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Branch Forecast":
    st.title("🔮 Branch-Level Forecast")
    st.markdown("---")

    c1, c2 = st.columns([2,1])
    with c1:
        sel_br = st.selectbox("Branch", BRANCHES)
    with c2:
        hist_d = st.slider("Past days", 30, 120, 60, 10)

    br_hist = df[df['tran_br_code']==sel_br].sort_values('start_date').tail(hist_d)
    br_fc   = forecast_df[forecast_df['Branch']==sel_br].sort_values('Date')

    wk1 = br_fc[br_fc['Step']<=7]['Predicted_M'].mean()
    wk2 = br_fc[(br_fc['Step']>7)&(br_fc['Step']<=14)]['Predicted_M'].mean()
    wk3 = br_fc[br_fc['Step']>14]['Predicted_M'].mean()
    tot = br_fc['Predicted_M'].sum()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Week 1 Avg", f"{wk1:.1f}M",   "🟢 HIGH conf")
    k2.metric("Week 2 Avg", f"{wk2:.1f}M",   "🟡 MEDIUM")
    k3.metric("Week 3-4 Avg", f"{wk3:.1f}M", "🔴 LOW conf")
    k4.metric("30-Day Total", f"{tot:.0f}M",  "PKR")

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(br_hist['start_date'], br_hist['Daily_Total_Debit']/1e6,
            color='#95a5a6', linewidth=1.8, label='Historical', alpha=0.9)
    ax.axvline(LAST_DATE, color='#f39c12', linewidth=2, linestyle='--', label='Forecast Start')
    for conf, grp in br_fc.groupby('Confidence', sort=False):
        c = CONF_COLORS[conf]
        ax.plot(grp['Date'], grp['Predicted_M'], color=c, linewidth=2.5, label=f'{conf} Conf')
        ax.fill_between(grp['Date'], grp['Lower_M'], grp['Upper_M'], color=c, alpha=0.18)
    ax.set_title(f'Branch {sel_br} — Cash Withdrawal Forecast', color='#e0e0f0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Million PKR', color='#c0c0d0')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.0f}M'))
    ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#c0c0d0')
    ax.grid(True, alpha=0.4)
    st.pyplot(fig); plt.close()

    disp = br_fc[['Date','Predicted_M','Lower_M','Upper_M','Uncertainty_Pct','Confidence']].copy()
    disp['Date'] = disp['Date'].dt.strftime('%Y-%m-%d (%a)')
    st.dataframe(disp, use_container_width=True, hide_index=True)
    csv = disp.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, f"branch_{sel_br}_forecast.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Overview":
    st.title("📊 Cash Optimization — Overview")
    st.markdown(f"*Forecast: **{forecast_df['Date'].min().date()}** to **{forecast_df['Date'].max().date()}***")
    st.markdown("---")

    tot30  = forecast_df['Predicted_M'].sum()
    wk1avg = forecast_df[forecast_df['Step']<=7]['Predicted_M'].mean()
    top_br = int(forecast_df.groupby('Branch')['Predicted_M'].mean().idxmax())
    top_avg= forecast_df.groupby('Branch')['Predicted_M'].mean().max()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("30-Day Need (All)", f"{tot30/1000:.2f}B PKR")
    c2.metric("Week 1 Daily Avg", f"{wk1avg:.1f}M PKR")
    c3.metric("Highest Branch", f"Br {top_br}", f"{top_avg:.1f}M/day")
    c4.metric("Branches Covered", f"{len(BRANCHES)}")
    st.markdown("---")

    br_tot = forecast_df.groupby('Branch')['Predicted_M'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(10,6))
    colors = plt.cm.plasma(np.linspace(0.3,0.9,len(br_tot)))
    bars = ax.barh(br_tot.index.astype(str), br_tot.values, color=colors, edgecolor='#0f0f1a')
    ax.set_xlabel("30-Day Total (M PKR)", color='#c0c0d0')
    ax.set_ylabel("Branch", color='#c0c0d0')
    ax.set_title("Branch-wise 30-Day Cash Requirement", color='#e0e0f0', fontweight='bold')
    for bar, val in zip(bars, br_tot.values):
        ax.text(val+5, bar.get_y()+bar.get_height()/2, f'{val:.0f}M', va='center', fontsize=9, color='#c0c0d0')
    ax.grid(True, axis='x', alpha=0.4)
    st.pyplot(fig); plt.close()

    st.subheader("Branch Priority Summary")
    summ = forecast_df.groupby('Branch')['Predicted_M'].agg(['mean','sum']).round(1)
    summ.columns = ['Daily Avg (M)','30-Day Total (M)']
    summ = summ.reset_index().sort_values('30-Day Total (M)', ascending=False)
    summ['Priority'] = summ['Daily Avg (M)'].apply(
        lambda x: '🔴 CRITICAL' if x>60 else ('🟡 HIGH' if x>40 else '🟢 NORMAL'))
    st.dataframe(summ, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation")
    st.markdown("---")

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("MAE",  "9.52M PKR", "Tuned via TimeSeriesSplit")
    m2.metric("RMSE", "14.80M PKR", "")
    m3.metric("MAPE", "55.4%", "")
    m4.metric("R²",   "0.6334", "Strong ✓")
    m5.metric("Model","XGBoost V3","Best of all")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🏦 Per-Branch", "📊 Plots", "💼 Recommendations"])
    with tab1:
        bm = branch_metrics.copy()
        bm['Quality'] = bm['MAPE_%'].apply(
            lambda x:'🟢 Excellent' if x<30 else ('🟡 Good' if x<60 else ('🟠 Moderate' if x<100 else '🔴 Poor')))
        st.dataframe(bm, use_container_width=True, hide_index=True)
    with tab2:
        plots = {
            "Plot 18 — Branch MAE & R²":     "eda_plots/18_branch_wise_evaluation.png",
            "Plot 19 — Error Distribution":  "eda_plots/19_error_distribution.png",
            "Plot 20 — Residuals":           "eda_plots/20_residuals_plot.png",
            "Plot 21 — Top Branch Timeline": "eda_plots/21_top_branches_timeline.png",
            "Plot 22 — All Branches Forecast":"eda_plots/22_all_branches_forecast.png",
            "Plot 23 — Confidence Ribbons":  "eda_plots/23_confidence_ribbons.png",
            "Plot 24 — Heatmap":             "eda_plots/24_forecast_heatmap.png",
        }
        chosen = st.selectbox("Select Plot", list(plots.keys()))
        path = plots[chosen]
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning(f"Not found: {path}")
    with tab3:
        er = eval_report[['Branch','Avg_Demand_M','MAE_M','MAPE_%','Trust_Level','Buffer_%','Recommended_M']].copy()
        st.dataframe(er, use_container_width=True, hide_index=True)
        st.info("Buffer Strategy: HIGH → 5% | MEDIUM → 12% | LOW → 30%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🕹️ What-If Simulator":
    st.title("🕹️ What-If Simulator")
    st.markdown("*Real-time AI scenario testing. Change variables and see how cash demand reacts.*")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Scenario Variables")
        sel_br = st.selectbox("Select Branch", BRANCHES)
        
        fc_dates = forecast_df[forecast_df['Branch']==sel_br]['Date'].dt.date.tolist()
        sel_date = st.selectbox("Select Target Date", fc_dates)
        target_dt = pd.Timestamp(sel_date)
        
        # Get baseline features
        base_feat_am, base_feat_pm = get_forecast_features(sel_br, target_dt)
        if base_feat_am is None:
            st.error("Feature data not found. Please select a valid date.")
            st.stop()
        
        is_holiday = st.checkbox("Is Public Holiday?", value=bool(base_feat_am.get('Is_Holiday', False)))
        is_salary = st.checkbox("Is Salary Day?", value=bool(base_feat_am.get('Is_Salary_Day', False)))
        
        # Sliders for continuous variables
        st.markdown("**Historical Volume Adjustments**")
        mult_14 = st.slider("14-Day Avg Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Simulating AI Model..."):
                # Apply changes to both AM and PM
                sim_feat_am = base_feat_am.copy()
                sim_feat_pm = base_feat_pm.copy()
                
                for sf in [sim_feat_am, sim_feat_pm]:
                    if 'Is_Holiday' in sf: sf['Is_Holiday'] = int(is_holiday)
                    if 'Is_Salary_Day' in sf: sf['Is_Salary_Day'] = int(is_salary)
                    if 'rolling_14_mean_Half_Day_Total_Debit' in sf: sf['rolling_14_mean_Half_Day_Total_Debit'] *= mult_14
                
                sim_model = model  # model is already V3
                
                # Baseline Prediction
                X_base_am = pd.DataFrame([base_feat_am])[feature_cols]
                X_base_pm = pd.DataFrame([base_feat_pm])[feature_cols]
                base_pred = np.expm1(sim_model.predict(X_base_am)[0]) + np.expm1(sim_model.predict(X_base_pm)[0])
                
                # Simulated Prediction
                X_sim_am = pd.DataFrame([sim_feat_am])[feature_cols]
                X_sim_pm = pd.DataFrame([sim_feat_pm])[feature_cols]
                sim_pred = np.expm1(sim_model.predict(X_sim_am)[0]) + np.expm1(sim_model.predict(X_sim_pm)[0])
                
                diff = sim_pred - base_pred
                pct_change = (diff / base_pred) * 100 if base_pred > 0 else 0
                
                # Render Results
                with col2:
                    st.subheader("📊 Simulation Results")
                    st.markdown(f"Scenario for **Branch {sel_br}** on **{target_dt.strftime('%A, %d %B %Y')}**")
                    
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Original Prediction", f"{base_pred/1e6:.1f}M")
                    sc2.metric("Simulated Prediction", f"{sim_pred/1e6:.1f}M", f"{diff/1e6:+.1f}M ({pct_change:+.1f}%)", delta_color="inverse")
                    
                    st.markdown("---")
                    st.markdown("### Why did it change?")
                    
                    if diff > 0:
                        st.success(f"The simulation caused an INCREASE of {diff/1e6:.1f} Million PKR.")
                    elif diff < 0:
                        st.info(f"The simulation caused a DECREASE of {abs(diff)/1e6:.1f} Million PKR.")
                    else:
                        st.warning("The simulation caused NO CHANGE in the predicted amount.")
                        
                    st.markdown("*Note: The model intelligently weights these features. For example, declaring a holiday on a weekend might have a different impact than on a weekday.*")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: MODEL COMPARISON (BENCHMARK)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Model Comparison (Benchmark)":
    st.title("⚖️ Model Comparison (Benchmark)")
    st.markdown("*Comparing Prophet baseline with our production XGBoost Model.*")
    
    st.markdown("""
    <div style='background-color:rgba(243,156,18,0.15); border-left:4px solid #f39c12; padding:12px; border-radius:4px; margin-bottom:20px;'>
        <b style='color:#f39c12;'>⚠️ Important Note:</b> This page is strictly for comparison and validation. Production forecasts (Branch Forecast tab) are generated using the XGBoost V3 model, which yields higher accuracy (R² = 0.63). Prophet is included here as a cross-check benchmark.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    try:
        prophet_fc = pd.read_csv('models/prophet_forecast.csv')
        prophet_fc['ds'] = pd.to_datetime(prophet_fc['ds'])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            sel_br = st.selectbox("Select Branch", prophet_fc['Branch'].unique())
            
        br_fc_prophet = prophet_fc[prophet_fc['Branch'] == sel_br].sort_values('ds')
        
        # Overlay XGBoost forecast
        xgb_fc = forecast_df[forecast_df['Branch'] == sel_br].sort_values('Date')
        
        st.subheader(f"Future Cash Forecast (Branch {sel_br}) - Prophet vs XGBoost")
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Prophet
        ax.plot(br_fc_prophet['ds'], br_fc_prophet['yhat']/1e6, color='#06b6d4', linewidth=2.5, label='Prophet (Baseline)')
        ax.fill_between(br_fc_prophet['ds'], br_fc_prophet['yhat_lower']/1e6, br_fc_prophet['yhat_upper']/1e6, color='#06b6d4', alpha=0.15)
        
        # XGBoost
        ax.plot(xgb_fc['Date'], xgb_fc['Predicted_M'], color='#a78bfa', linewidth=2.5, linestyle='--', label='XGBoost (Production)')
        ax.fill_between(xgb_fc['Date'], xgb_fc['Lower_M'], xgb_fc['Upper_M'], color='#a78bfa', alpha=0.15)
        
        ax.set_title(f'Model Comparison: Prophet vs XGBoost (Next 30 Days)', color='#e0e0f0', fontweight='bold')
        ax.set_ylabel('Million PKR', color='#c0c0d0')
        ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#c0c0d0')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
        
        st.markdown("---")
        st.subheader("🔍 Time-Series Components Decomposition (Prophet)")
        st.markdown("*Prophet explicitly separates the overall trend from weekly patterns.*")
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(br_fc_prophet['ds'], br_fc_prophet['trend']/1e6, color='#8b5cf6', linewidth=2)
            ax.set_title('Macro Trend (Is cash demand generally rising?)', color='#e0e0f0')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()
            
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            # Extract one week of data to show the weekly pattern cleanly
            weekly = br_fc_prophet.head(14).copy()
            weekly['DayName'] = weekly['ds'].dt.day_name()
            # Plot against day name
            ax.bar(weekly['DayName'], weekly['weekly']/1e6, color='#10b981')
            ax.set_title('Weekly Seasonality (Which days are busiest?)', color='#e0e0f0')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig); plt.close()
            
    except Exception as e:
        st.warning("Prophet Forecast data not found. Please run `ts_pipeline.py` first.")
        st.code(str(e))
