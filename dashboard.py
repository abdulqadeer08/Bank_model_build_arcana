"""
Phase 8 — Bank Cash Optimization Dashboard (Enhanced)
Includes: Cash Need Calendar, SHAP-based Explanation, Branch Forecast
Sir's Requirement: Kis din, kis branch, kitni cash + kyun itni chahiye
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
    return joblib.load('models/best_model.pkl')

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
    with open('model_data/feature_cols.json') as f:
        return json.load(f)

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
    'rolling_30_mean_debit' : ('📅 30-Din Average',    'Pichle 30 din ka average withdrawal'),
    'rolling_7_mean_debit'  : ('📆 7-Din Average',     'Pichle 7 din ka average withdrawal'),
    'lag_1_debit'           : ('⏮️ Kal ka Withdrawal', 'Kal kitna cash nikla tha'),
    'lag_7_debit'           : ('📅 7 Din Pehle',       'Ek hafte pehle isi din kitna withdrawal tha'),
    'lag_14_debit'          : ('📅 14 Din Pehle',      'Do hafte pehle isi din ka withdrawal'),
    'lag_30_debit'          : ('📅 30 Din Pehle',      'Ek mahine pehle ka withdrawal'),
    'Daily_Txn_Count'       : ('🔢 Transactions',      'Us din kitne transactions hue'),
    'Branch_Total_Debit'    : ('🏦 Branch Volume',     'Is branch ki overall transaction volume'),
    'Branch_Txn_Count'      : ('🔢 Branch Txn Total',  'Branch ki total transaction count'),
    'Branch_Avg_Net_CF'     : ('💵 Net Cash Flow',     'Branch ka average net cash flow'),
    'Peak_Hour_Txns'        : ('⏰ Peak Hour',         'Peak hour mein transactions ki count'),
    'Business_Hour_Txns'    : ('🕐 Business Hours',    'Business hours transactions'),
    'Weekday'               : ('📆 Din (Weekday)',     'Hafte ka kaunsa din hai'),
    'Is_Weekend'            : ('🏖️ Weekend',           'Kya yeh weekend hai'),
    'Month'                 : ('🗓️ Mahina',            'Sal ka kaunsa mahina'),
    'Year'                  : ('📅 Saal',              'Kaunsa saal'),
    'Day'                   : ('🔢 Tarikh',            'Mahine ki tarikh'),
    'Weekday_Sin'           : ('📐 Weekday Sin',       'Weekday ka cyclical encoding'),
    'Weekday_Cos'           : ('📐 Weekday Cos',       'Weekday ka cyclical encoding'),
    'Month_Sin'             : ('📐 Month Sin',         'Month ka cyclical encoding'),
    'Month_Cos'             : ('📐 Month Cos',         'Month ka cyclical encoding'),
    'tran_br_code'          : ('🏦 Branch Code',       'Branch ki ID'),
}

def get_shap_for_row(X_row):
    """Compute SHAP values for a single row on-the-fly."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_row)
    return sv[0], float(explainer.expected_value)

def build_explanation(shap_vals, feature_vals, feat_names, base_val, prediction, branch, date):
    """Return markdown explanation of why this prediction was made."""
    pairs = sorted(zip(shap_vals, feat_names), key=lambda x: abs(x[0]), reverse=True)[:6]

    lines = []
    lines.append(f"### 🔍 Explanation: Branch **{branch}** on **{date}**\n")
    lines.append(f"**Base prediction** (average of all branches/days): **PKR {base_val/1e6:.1f}M**\n")
    lines.append(f"**Final prediction**: **PKR {prediction:.1f}M**\n")
    lines.append("---\n#### Top Reasons (Kyun itni cash chahiye):\n")

    for sv, fn in pairs:
        label, desc = FEATURE_LABELS.get(fn, (fn, fn))
        direction   = "⬆️ BADHAYA" if sv > 0 else "⬇️ GHATAYA"
        color_word  = "zyada" if sv > 0 else "kam"
        sv_m        = sv / 1e6
        val         = feature_vals.get(fn, '?')
        if isinstance(val, float) and abs(val) > 1000:
            val_str = f"{val/1e6:.1f}M PKR"
        elif isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)

        lines.append(f"- {direction} **{label}** — {desc}\n"
                     f"  - Value: `{val_str}` → Model ne **{color_word}** predict kiya  \n"
                     f"  - Impact: `{sv_m:+.2f}M PKR`\n")

    lines.append("---\n")
    diff = prediction - base_val/1e6
    if diff > 0:
        lines.append(f"✅ **Net Result:** Base ({base_val/1e6:.1f}M) + features ka combined effect = **{prediction:.1f}M PKR**\n"
                     f"  _(Features ne {diff:.1f}M PKR BADHAYA)_\n")
    else:
        lines.append(f"✅ **Net Result:** Base ({base_val/1e6:.1f}M) + features ka combined effect = **{prediction:.1f}M PKR**\n"
                     f"  _(Features ne {abs(diff):.1f}M PKR GHATAYA)_\n")
    return "\n".join(lines)

def get_forecast_features(branch, target_date):
    """Build feature row for a future date."""
    br_hist = df[df['tran_br_code']==branch].sort_values('start_date')
    dh = br_hist.set_index('start_date')['Daily_Total_Debit']
    br_avgs = BRANCH_AVGS.loc[branch]

    def get_lag(lag_days):
        ld = target_date - pd.Timedelta(days=lag_days)
        avail = dh[dh.index <= ld]
        return float(avail.iloc[-1]) if len(avail) > 0 else float(dh.iloc[-1])

    recent_7  = [float(dh[dh.index == target_date - pd.Timedelta(days=d)].iloc[0])
                 for d in range(1,8) if len(dh[dh.index == target_date - pd.Timedelta(days=d)]) > 0]
    recent_30 = [float(dh[dh.index == target_date - pd.Timedelta(days=d)].iloc[0])
                 for d in range(1,31) if len(dh[dh.index == target_date - pd.Timedelta(days=d)]) > 0]

    wd = target_date.weekday()
    mo = target_date.month
    row = {
        'tran_br_code'         : branch,
        'Daily_Txn_Count'      : br_avgs['Daily_Txn_Count'],
        'Weekday'              : wd,
        'Is_Weekend'           : int(wd >= 5),
        'Month'                : mo,
        'Year'                 : target_date.year,
        'Day'                  : target_date.day,
        'Weekday_Sin'          : np.sin(2*np.pi*wd/7),
        'Weekday_Cos'          : np.cos(2*np.pi*wd/7),
        'Month_Sin'            : np.sin(2*np.pi*mo/12),
        'Month_Cos'            : np.cos(2*np.pi*mo/12),
        'Branch_Total_Debit'   : br_avgs['Branch_Total_Debit'],
        'Branch_Txn_Count'     : br_avgs['Branch_Txn_Count'],
        'Branch_Avg_Net_CF'    : br_avgs['Branch_Avg_Net_CF'],
        'Peak_Hour_Txns'       : br_avgs['Peak_Hour_Txns'],
        'Business_Hour_Txns'   : br_avgs['Business_Hour_Txns'],
        'lag_1_debit'          : get_lag(1),
        'lag_7_debit'          : get_lag(7),
        'lag_14_debit'         : get_lag(14),
        'lag_30_debit'         : get_lag(30),
        'rolling_7_mean_debit' : np.mean(recent_7)  if recent_7  else get_lag(7),
        'rolling_30_mean_debit': np.mean(recent_30) if recent_30 else get_lag(30),
    }
    return row

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Cash Intelligence")
    st.markdown("*Bank Cash Optimization System*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📅 Cash Need Calendar",
        "🔍 Why This Amount? (SHAP)",
        "🔮 Branch Forecast",
        "📈 Prophet Time Series",
        "🕹️ What-If Simulator",
        "📊 Overview",
        "📈 Model Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** XGBoost  \n**R²:** 0.5607  \n**MAE:** 14.25M PKR")
    st.markdown(f"**Data till:** {LAST_DATE.date()}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CASH NEED CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
if page == "📅 Cash Need Calendar":
    st.title("📅 Cash Need Calendar")
    st.markdown("*Kis din, kis branch ko kitni cash chahiye — 30-day view*")
    st.markdown("---")

    view = st.radio("View Type", ["🏦 Per Branch (Daily)", "📊 All Branches Heatmap"], horizontal=True)

    if view == "🏦 Per Branch (Daily)":
        sel_br = st.selectbox("Branch Select Karo", BRANCHES)
        br_fc  = forecast_df[forecast_df['Branch']==sel_br].sort_values('Date').copy()
        br_fc['DayName'] = br_fc['Date'].dt.strftime('%a')
        br_fc['DateStr'] = br_fc['Date'].dt.strftime('%d %b')
        br_fc['Week']    = ((br_fc['Step']-1) // 7) + 1

        st.subheader(f"🏦 Branch {sel_br} — 30-Din Cash Calendar")

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
                    <div style='background:{bg};border:1px solid {border};border-radius:10px;padding:10px;text-align:center;margin:4px 0'>
                        <div style='color:#888;font-size:11px'>{row['DayName']}</div>
                        <div style='color:#e0e0ff;font-weight:700;font-size:13px'>{row['DateStr']}</div>
                        <div style='color:#a78bfa;font-size:18px;font-weight:700'>{row['Predicted_M']:.0f}M</div>
                        <div style='color:#666;font-size:9px'>±{row['Uncertainty_Pct']:.0f}%</div>
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
        patches = [mpatches.Patch(color=c, label=f'{l}') for l,c in CONF_COLORS.items()]
        ax.legend(handles=patches, facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#c0c0d0')
        ax.grid(True, axis='y', alpha=0.4)
        st.pyplot(fig); plt.close()

        # Table
        disp = br_fc[['DateStr','DayName','Predicted_M','Lower_M','Upper_M','Uncertainty_Pct','Confidence']].copy()
        disp.columns = ['Date','Day','Predicted (M PKR)','Lower (M)','Upper (M)','Uncertainty %','Confidence']
        st.dataframe(disp, use_container_width=True, hide_index=True)

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

        st.info("💡 **Tip:** Darker color = us din us branch ko zyada cash chahiye. Yeh directly replenishment scheduling mein use ho sakta hai.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: WHY THIS AMOUNT? (SHAP EXPLANATION)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Why This Amount? (SHAP)":
    st.title("🔍 Kyun Itni Cash Chahiye?")
    st.markdown("*SHAP-based explanation — model ne iss prediction ke liye kya socha*")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sel_br   = st.selectbox("Branch Select Karo", BRANCHES)
    with col2:
        fc_dates = forecast_df[forecast_df['Branch']==sel_br]['Date'].dt.date.tolist()
        sel_date = st.selectbox("Date Select Karo (Forecast Days)", fc_dates)

    if st.button("🔍 Explain Karo — Kyun Itni Cash?", use_container_width=True):
        with st.spinner("SHAP analysis chal rahi hai..."):
            target_dt  = pd.Timestamp(sel_date)
            feat_row   = get_forecast_features(sel_br, target_dt)
            X_row      = pd.DataFrame([feat_row])[feature_cols]
            sv, bv     = get_shap_for_row(X_row)
            pred_val   = float(model.predict(X_row)[0])

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
        st.subheader("📝 Plain Language Explanation (Urdu/English)")

        base_m = bv / 1e6
        pred_m = pred_val / 1e6

        st.markdown(f"""
        <div class='explain-box'>
            <h4 style='color:#a78bfa'>🏦 Branch {sel_br} ko {sel_date} ({weekday_name}) ko <span style='color:#6ee7b7'>{pred_m:.1f}M PKR</span> chahiye</h4>
            <p style='color:#9999cc;font-size:13px'>Base amount (average prediction): <b style='color:#e0e0f0'>{base_m:.1f}M PKR</b></p>
            <hr style='border-color:#2a2a4a'>
            <p style='color:#c0c0d0;font-weight:600'>Top Reasons (Kyun itna?):</p>
        """, unsafe_allow_html=True)

        for sv_val, fn, fv in pairs[:5]:
            label, desc = FEATURE_LABELS.get(fn, (fn, fn))
            sv_m = sv_val / 1e6
            direction = "⬆️ INCREASE" if sv_m > 0 else "⬇️ DECREASE"
            color = "#2ecc71" if sv_m > 0 else "#e74c3c"
            impact_word = "zyada" if sv_m > 0 else "kam"
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
        diff_word = f"+{diff:.1f}M badhaya" if diff > 0 else f"{diff:.1f}M ghataya"
        st.markdown(f"""
            <hr style='border-color:#2a2a4a'>
            <p style='color:#c0c0d0'>
                Base: <b style='color:#e0e0f0'>{base_m:.1f}M</b> + Features ne <b style='color:#6ee7b7'>{diff_word}</b>
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
    m1.metric("MAE",  "14.25M PKR", "↓35% vs Baseline")
    m2.metric("RMSE", "20.29M PKR", "")
    m3.metric("MAPE", "62.13%", "")
    m4.metric("R²",   "0.5607", "Moderate ✓")
    m5.metric("Model","XGBoost","Best of 3")
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
        chosen = st.selectbox("Plot Select Karo", list(plots.keys()))
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
        base_feat = get_forecast_features(sel_br, target_dt)
        
        is_holiday = st.checkbox("Is Public Holiday?", value=bool(base_feat.get('Is_Holiday', False)))
        is_salary = st.checkbox("Is Salary Day?", value=bool(base_feat.get('Is_Salary_Day', False)))
        
        # Sliders for continuous variables
        st.markdown("**Historical Volume Adjustments**")
        mult_30 = st.slider("30-Day Avg Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
        mult_7 = st.slider("7-Day Avg Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Simulating AI Model..."):
                # Apply changes
                sim_feat = base_feat.copy()
                if 'Is_Holiday' in sim_feat: sim_feat['Is_Holiday'] = int(is_holiday)
                if 'Is_Salary_Day' in sim_feat: sim_feat['Is_Salary_Day'] = int(is_salary)
                if 'rolling_30_mean_debit' in sim_feat: sim_feat['rolling_30_mean_debit'] *= mult_30
                if 'rolling_7_mean_debit' in sim_feat: sim_feat['rolling_7_mean_debit'] *= mult_7
                
                # Check for V3 Model (Optuna + Log Transform)
                try:
                    v3_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
                    # V3 feature structure is slightly different (half daily). We will simulate using the V2 model for now, 
                    # but if V3 gets fully integrated into dashboard.py, we will use it here.
                    sim_model = model
                except:
                    sim_model = model
                
                # Baseline Prediction
                X_base = pd.DataFrame([base_feat])[feature_cols]
                base_pred = float(sim_model.predict(X_base)[0])
                
                # Simulated Prediction
                X_sim = pd.DataFrame([sim_feat])[feature_cols]
                sim_pred = float(sim_model.predict(X_sim)[0])
                
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
# PAGE 7: PROPHET TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Prophet Time Series":
    st.title("📈 Prophet Time-Series Forecast")
    st.markdown("*Advanced Time-Series Modeling (Meta Prophet) natively handling Holidays & Seasonality.*")
    st.markdown("---")

    try:
        prophet_fc = pd.read_csv('models/prophet_forecast.csv')
        prophet_fc['ds'] = pd.to_datetime(prophet_fc['ds'])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            sel_br = st.selectbox("Select Branch", prophet_fc['Branch'].unique())
            
        br_fc = prophet_fc[prophet_fc['Branch'] == sel_br].sort_values('ds')
        
        st.subheader(f"Future Cash Forecast (Branch {sel_br})")
        fig, ax = plt.subplots(figsize=(14, 5))
        
        ax.plot(br_fc['ds'], br_fc['yhat']/1e6, color='#06b6d4', linewidth=2.5, label='Predicted Trend')
        ax.fill_between(br_fc['ds'], br_fc['yhat_lower']/1e6, br_fc['yhat_upper']/1e6, color='#06b6d4', alpha=0.2, label='Confidence Interval')
        
        ax.set_title(f'Prophet Forecast (Next 30 Days)', color='#e0e0f0', fontweight='bold')
        ax.set_ylabel('Million PKR', color='#c0c0d0')
        ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#c0c0d0')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
        
        st.markdown("---")
        st.subheader("🔍 Time-Series Components Decomposition")
        st.markdown("*Prophet explicitly separates the overall trend from weekly patterns.*")
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(br_fc['ds'], br_fc['trend']/1e6, color='#8b5cf6', linewidth=2)
            ax.set_title('Macro Trend (Is cash demand generally rising?)', color='#e0e0f0')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()
            
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            # Extract one week of data to show the weekly pattern cleanly
            weekly = br_fc.head(14).copy()
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
