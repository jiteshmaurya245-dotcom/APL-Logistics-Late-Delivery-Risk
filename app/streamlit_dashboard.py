# ============================================================
# APL LOGISTICS — REDESIGNED STREAMLIT DASHBOARD v2.0
# Clean & Professional — Corporate Grade
# ============================================================
# HOW TO RUN — this is a Streamlit app, NOT a plain script.
# Do not click "Run" / "python streamlit_dashboard.py" in your IDE — that
# runs it in "bare mode" and it will crash (no real Streamlit session).
# From a terminal, from anywhere:
#   pip install -r requirements.txt
#   streamlit run "<path-to-repo>/app/streamlit_dashboard.py"
#   (place APL_Logistics.csv in the data/ folder first)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="APL Logistics · Risk Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# DESIGN SYSTEM — CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F5F6FA !important;
    color: #1A1D2E !important;
}

.main { background-color: #F5F6FA !important; }
.block-container { padding: 3rem 2rem 2rem 2rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { background: #1A1D2E !important; border-right: none !important; }
section[data-testid="stSidebar"] * { color: #E8EAF0 !important; }
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background: #2E3247 !important; }
section[data-testid="stSidebar"] label { color: #9DA3B8 !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
section[data-testid="stSidebar"] hr { border-color: #2E3247 !important; }

/* KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #E8EAF0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #2563EB);
    border-radius: 12px 12px 0 0;
}
.kpi-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7280; margin-bottom: 0.5rem; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #1A1D2E; line-height: 1; font-family: 'DM Mono', monospace; }
.kpi-sub { font-size: 12px; color: #9CA3AF; margin-top: 0.35rem; }
.kpi-badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-top: 0.4rem; }
.badge-red    { background: #FEE2E2; color: #DC2626; }
.badge-yellow { background: #FEF3C7; color: #D97706; }
.badge-green  { background: #D1FAE5; color: #059669; }
.badge-blue   { background: #DBEAFE; color: #2563EB; }

/* Section headers */
.section-title {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #6B7280; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px; background: #E8EAF0;
}

/* Chart cards */
.chart-card {
    background: #FFFFFF; border-radius: 12px;
    border: 1px solid #E8EAF0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    padding: 1.25rem;
}

/* Risk tier progress bars */
.tier-row { margin-bottom: 0.75rem; }
.tier-label { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 13px; }
.tier-bar-bg { background: #F3F4F6; border-radius: 4px; height: 8px; overflow: hidden; }
.tier-bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease; }

/* Action items */
.action-item {
    background: #FFFFFF; border-radius: 10px; border: 1px solid #E8EAF0;
    padding: 0.85rem 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: flex-start; gap: 10px;
    transition: border-color 0.2s ease;
}
.action-item:hover { border-color: #2563EB; }
.action-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 999px !important;
    padding: 5px !important;
    border: 1px solid #E8EAF0 !important;
    gap: 2px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px !important;
    padding: 9px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    border: none !important;
    background: transparent !important;
    transition: background 0.2s ease, color 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #1A1D2E !important;
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* Sidebar model metrics */
.model-metric {
    background: #2E3247; border-radius: 8px; padding: 10px 14px;
    margin-bottom: 8px; display: flex; justify-content: space-between;
    align-items: center;
}
.model-metric-label { font-size: 12px; color: #9DA3B8 !important; }
.model-metric-value { font-size: 16px; font-weight: 700; color: #FFFFFF !important; font-family: 'DM Mono', monospace; }

/* Buttons */
.stButton > button {
    background: #1A1D2E !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 10px 20px !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover { background: #2563EB !important; }

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* Dataframe */
.stDataFrame { border-radius: 10px !important; border: 1px solid #E8EAF0 !important; }

/* Download button */
.stDownloadButton > button {
    background: #F0F4FF !important; color: #2563EB !important;
    border: 1px solid #BFDBFE !important; border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Selectbox / input */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    border-radius: 8px !important;
    border-color: #E8EAF0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Page header */
.page-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; padding-bottom: 1rem;
    border-bottom: 1px solid #E8EAF0;
}
.page-title { font-size: 1.5rem; font-weight: 700; color: #1A1D2E; letter-spacing: -0.02em; }
.page-subtitle { font-size: 13px; color: #9CA3AF; margin-top: 2px; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #D1FAE5; color: #065F46; border-radius: 20px;
    padding: 5px 12px; font-size: 12px; font-weight: 600;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #10B981; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Gauge label */
.gauge-label { text-align: center; font-size: 12px; color: #6B7280; margin-top: -10px; }

/* Feature importance bars */
.fi-row { margin-bottom: 10px; }
.fi-name { font-size: 12px; color: #374151; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fi-bar-bg { background: #F3F4F6; border-radius: 3px; height: 6px; }
.fi-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #2563EB, #60A5FA); }

/* Sidebar logo area */
.sidebar-logo {
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid #2E3247;
    margin-bottom: 1rem;
}
.sidebar-logo-title { font-size: 16px; font-weight: 700; color: #FFFFFF !important; letter-spacing: -0.01em; }
.sidebar-logo-sub { font-size: 11px; color: #6B7280 !important; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD & TRAIN (cached)
# ─────────────────────────────────────────
DECISION_THRESHOLD = 0.35  # recalibrated cutoff for "predicted late" — see docs/project-brief.md
# Recall matters more than precision here (a missed late order costs more than
# an extra check-in), and 0.35 gives materially better recall than the default
# 0.5 cutoff on this model.

# Hyperparameters are deliberately depth/leaf-constrained. An earlier
# unconstrained RandomForestClassifier(n_estimators=100) with no max_depth
# grew to a 650MB pickle (full-purity leaves on 144K rows) — that blew past
# Streamlit Community Cloud's ~1GB memory limit during training and crashed
# the deployed app with no traceback (silent OOM kill). This configuration
# produces a ~10MB model with equal-or-better recall at the 0.35 threshold.
RF_PARAMS = dict(n_estimators=100, max_depth=12, min_samples_leaf=5,
                  class_weight='balanced', random_state=42, n_jobs=-1)

@st.cache_data(show_spinner=False)
def load_and_train():
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(DATA_DIR / "APL_Logistics.csv", encoding='latin1')
    drop_cols = [
        'Customer Fname','Customer Lname','Customer Street','Customer Zipcode',
        'Customer City','Customer State','Customer Country','Order City',
        'Order Country','Order State','Customer Id','Order Customer Id',
        'Category Id','Department Id','Latitude','Longitude','Product Name',
        'Delivery Status','Days for shipping (real)'
    ]
    df_raw = df.drop(columns=drop_cols).dropna().reset_index(drop=True)
    del df  # drop the pre-cleanup frame as soon as we're done with it
    df_ml  = df_raw.copy()
    df_ml['shipping_pressure']  = df_ml['Days for shipment (scheduled)'] / (df_ml['Order Item Quantity'] + 1)
    df_ml['discount_flag']      = (df_ml['Order Item Discount'] > 0).astype(int)
    df_ml['profit_flag']        = (df_ml['Order Profit Per Order'] < 0).astype(int)
    df_ml['high_discount']      = (df_ml['Order Item Discount Rate'] > 0.15).astype(int)
    df_ml['low_scheduled_days'] = (df_ml['Days for shipment (scheduled)'] <= 2).astype(int)

    # Split on the raw (pre-encoding) index. This produces the identical test
    # set as encoding-then-splitting (verified separately — same random_state
    # and stratify column fully determine the partition regardless of which
    # columns X has), but lets the fast path below encode ONLY the ~36K test
    # rows instead of one-hot-encoding and holding all 180K rows in memory
    # every session when 144K of them are never used in that path.
    idx_train, idx_test = train_test_split(
        df_ml.index, test_size=0.2, random_state=42, stratify=df_ml['Late_delivery_risk']
    )

    model_path, scaler_path, cols_path = DATA_DIR/"model.pkl", DATA_DIR/"scaler.pkl", DATA_DIR/"feature_cols.json"
    if model_path.exists() and scaler_path.exists() and cols_path.exists():
        # Fast path: load the pre-trained model instead of retraining on every
        # cold start, AND only encode the test slice actually needed for
        # scoring. Training only ever happens offline via scripts/train_model.py.
        with open(model_path, 'rb') as f: rf = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        cols_meta = json.load(open(cols_path))
        feature_cols, num_cols = cols_meta['feature_cols'], cols_meta['num_cols']

        df_ml_test = df_ml.loc[idx_test]
        df_enc_test = pd.get_dummies(df_ml_test, columns=df_ml_test.select_dtypes(include=['object','string']).columns.tolist(), drop_first=True, dtype=int)
        X_test = df_enc_test.drop(columns=['Late_delivery_risk'])
        y_test = df_enc_test['Late_delivery_risk']
        X_test_s = X_test.reindex(columns=feature_cols, fill_value=0).copy()
        X_test_s[num_cols] = scaler.transform(X_test_s[num_cols])
        del df_ml_test, df_enc_test, X_test
    else:
        # Fallback: train live (e.g. first run without committed artifacts).
        # This path DOES need the full encoded dataset to fit on the training
        # rows, so it's inherently heavier — but it should rarely run once the
        # pretrained artifacts are committed. Always uses the same
        # size-constrained RF_PARAMS — never the old unconstrained config
        # that caused the original OOM crash.
        df_enc = pd.get_dummies(df_ml, columns=df_ml.select_dtypes(include=['object','string']).columns.tolist(), drop_first=True, dtype=int)
        X = df_enc.drop(columns=['Late_delivery_risk'])
        y = df_enc['Late_delivery_risk']
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]
        y_train, y_test = y.loc[idx_train], y.loc[idx_test]
        num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
        scaler = StandardScaler()
        X_train_s = X_train.copy(); X_test_s = X_test.copy()
        X_train_s[num_cols] = scaler.fit_transform(X_train_s[num_cols])
        X_test_s[num_cols]  = scaler.transform(X_test_s[num_cols])
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train_s, y_train)
        feature_cols = X_train_s.columns.tolist()
        del df_enc, X, y, X_train, X_train_s

    probs = rf.predict_proba(X_test_s)[:,1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)
    metrics = {
        'auc':  round(roc_auc_score(y_test, probs), 4),
        'f1':   round(f1_score(y_test, preds), 4),
        'prec': round(precision_score(y_test, preds), 4),
        'rec':  round(recall_score(y_test, preds), 4),
    }
    scored = df_raw.loc[idx_test].copy().reset_index(drop=True)
    scored['late_prob']     = probs
    scored['risk_category'] = pd.cut(probs, bins=[0,0.35,0.65,1.0], labels=['Low','Medium','High'], include_lowest=True)
    scored['pred']          = preds
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return df_raw, scored, metrics, fi, rf, scaler, feature_cols, num_cols, y_test, probs

with st.spinner("Initialising risk intelligence engine..."):
    df_raw, scored, metrics, feat_imp, model, scaler, feature_cols, num_cols, y_test_full, probs_full = load_and_train()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-title">🚢 APL Logistics</div>
        <div class="sidebar-logo-sub">Risk Intelligence Platform · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Filters")
    shipping_modes = st.multiselect("Shipping Mode", ['Standard Class','Same Day','First Class','Second Class'],
                                    default=['Standard Class','Same Day','First Class','Second Class'])
    markets = st.multiselect("Market / Region", ['Pacific Asia','LATAM','USCA','Europe','Africa'],
                             default=['Pacific Asia','LATAM','USCA','Europe','Africa'])
    segments = st.multiselect("Customer Segment", ['Consumer','Corporate','Home Office'],
                              default=['Consumer','Corporate','Home Office'])
    risk_threshold = st.slider("High Risk Threshold", 0.50, 0.90, 0.65, 0.05,
                               help="Probability above which orders are flagged High Risk")

    st.markdown("---")
    st.markdown("##### Model Performance")
    for label, val in [("ROC-AUC", metrics['auc']), ("F1 Score", metrics['f1']),
                       ("Precision", metrics['prec']), ("Recall", metrics['rec'])]:
        st.markdown(f"""
        <div class="model-metric">
            <span class="model-metric-label">{label}</span>
            <span class="model-metric-value">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Random Forest · 100 estimators · 180K orders")

# ─────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────
flt = scored.copy()
if shipping_modes: flt = flt[flt['Shipping Mode'].isin(shipping_modes)]
if markets:        flt = flt[flt['Market'].isin(markets)]
if segments:       flt = flt[flt['Customer Segment'].isin(segments)]
flt['risk_category'] = pd.cut(flt['late_prob'], bins=[0,0.35,risk_threshold,1.0], labels=['Low','Medium','High'], include_lowest=True)

n_total  = len(flt)
n_high   = (flt['risk_category']=='High').sum()
n_medium = (flt['risk_category']=='Medium').sum()
n_low    = (flt['risk_category']=='Low').sum()

# Live rates — recomputed from the CURRENT filtered slice (flt) so every KPI,
# reference line, and panel stays consistent with the sidebar filters/threshold
# instead of showing stale numbers from the full unfiltered dataset.
avg_late_rate = flt['Late_delivery_risk'].mean()*100 if n_total > 0 else 0.0
high_rate   = flt[flt['risk_category']=='High']['Late_delivery_risk'].mean()*100 if n_high   > 0 else 0.0
medium_rate = flt[flt['risk_category']=='Medium']['Late_delivery_risk'].mean()*100 if n_medium > 0 else 0.0
low_rate    = flt[flt['risk_category']=='Low']['Late_delivery_risk'].mean()*100 if n_low    > 0 else 0.0

COLORS = {'High':'#DC2626','Medium':'#D97706','Low':'#059669','Blue':'#2563EB'}
CHART_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='DM Sans', color='#374151', size=12),
                    margin=dict(t=20,b=20,l=10,r=10))

# ─────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div>
    <div class="page-title">Late Delivery Risk Intelligence</div>
    <div class="page-subtitle">Predictive scoring for {n_total:,} shipments · Random Forest · AUC {metrics['auc']}</div>
  </div>
  <div class="status-pill"><span class="status-dot"></span>Model Active</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["  📊  Overview  ", "  🔍  Order Risk  ", "  🗺️  Region & Mode  ", "  ⚡  Action Panel  "])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    # ── KPI Row ──
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "Total Scored",    f"{n_total:,}",  "",              "#2563EB", "blue"),
        (c2, "High Risk",       f"{n_high:,}",   f"{n_high/n_total*100:.1f}% of orders", "#DC2626", "red"),
        (c3, "Medium Risk",     f"{n_medium:,}", f"{n_medium/n_total*100:.1f}% of orders","#D97706","yellow"),
        (c4, "Low Risk",        f"{n_low:,}",    f"{n_low/n_total*100:.1f}% of orders",  "#059669","green"),
        (c5, "High Risk Accuracy",f"{high_rate:.1f}%",        "actual late rate","#2563EB","blue"),
    ]
    for col, label, val, sub, accent, badge in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Donut + Tier Accuracy ──
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

    with col_a:
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        tier_vals = [n_low, n_medium, n_high]
        fig_donut = go.Figure(go.Pie(
            labels=['Low Risk','Medium Risk','High Risk'], values=tier_vals,
            hole=0.62,
            marker=dict(colors=['#059669','#D97706','#DC2626'], line=dict(color='#FFFFFF',width=3)),
            textinfo='percent', textfont=dict(size=12, family='DM Sans'),
            hovertemplate='<b>%{label}</b><br>%{value:,} orders<br>%{percent}<extra></extra>'
        ))
        fig_donut.add_annotation(text=f"<b>{n_total:,}</b><br><span style='font-size:10px'>orders</span>",
                                 x=0.5, y=0.5, showarrow=False, font=dict(size=16, family='DM Sans'))
        fig_donut.update_layout(**CHART_LAYOUT, height=260, showlegend=True,
                                legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center',
                                           font=dict(size=11)))
        st.plotly_chart(fig_donut, width='stretch')

    with col_b:
        st.markdown('<div class="section-title">Tier Validation</div>', unsafe_allow_html=True)
        tier_data = []
        for t, col_color in [('Low','#059669'),('Medium','#D97706'),('High','#DC2626')]:
            sub = flt[flt['risk_category']==t]
            rate = sub['Late_delivery_risk'].mean()*100 if len(sub)>0 else 0
            tier_data.append({'Tier':t,'Rate':round(rate,1),'Count':len(sub),'Color':col_color})
        td = pd.DataFrame(tier_data)
        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(
            x=td['Tier'], y=td['Rate'],
            marker_color=td['Color'].tolist(),
            marker_line_color='rgba(0,0,0,0)',
            text=td['Rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(size=12, family='DM Mono', color='#374151'),
            hovertemplate='<b>%{x}</b><br>Actual late rate: %{y:.1f}%<extra></extra>',
            width=0.45
        ))
        fig_tier.add_hline(y=avg_late_rate, line_dash='dot', line_color='#9CA3AF', line_width=1.5,
                           annotation_text=f'Avg {avg_late_rate:.1f}%', annotation_font_size=10,
                           annotation_font_color='#9CA3AF')
        fig_tier.update_layout(**CHART_LAYOUT, height=260,
                               yaxis=dict(range=[0,100], gridcolor='#F3F4F6', ticksuffix='%'),
                               xaxis=dict(showgrid=False))
        st.plotly_chart(fig_tier, width='stretch')

    with col_c:
        st.markdown('<div class="section-title">Risk Breakdown</div>', unsafe_allow_html=True)
        for tier, count, color in [('High Risk', n_high,'#DC2626'),('Medium Risk',n_medium,'#D97706'),('Low Risk',n_low,'#059669')]:
            pct = count/n_total*100
            st.markdown(f"""
            <div class="tier-row">
              <div class="tier-label">
                <span style="color:#374151;font-size:13px;">{tier}</span>
                <span style="color:#6B7280;font-size:12px;font-family:'DM Mono'">{count:,} &nbsp;·&nbsp; {pct:.1f}%</span>
              </div>
              <div class="tier-bar-bg">
                <div class="tier-bar-fill" style="width:{pct:.1f}%;background:{color};opacity:0.85;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for label, val, badge in [("ROC-AUC",f"{metrics['auc']}","blue"),("F1 Score",f"{metrics['f1']}","blue"),("Recall",f"{metrics['rec']*100:.1f}%","blue")]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;border-bottom:1px solid #F3F4F6;">
              <span style="font-size:12px;color:#6B7280">{label}</span>
              <span style="font-size:14px;font-weight:700;font-family:'DM Mono';color:#1A1D2E">{val}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Probability Dist + Feature Importance ──
    col_d, col_e = st.columns([1.4, 1])

    with col_d:
        st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=flt[flt['Late_delivery_risk']==0]['late_prob'], name='On-time',
            nbinsx=50, marker_color='#059669', opacity=0.7,
            hovertemplate='Prob: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        fig_hist.add_trace(go.Histogram(
            x=flt[flt['Late_delivery_risk']==1]['late_prob'], name='Late',
            nbinsx=50, marker_color='#DC2626', opacity=0.7,
            hovertemplate='Prob: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        fig_hist.add_vrect(x0=0, x1=0.35, fillcolor='#059669', opacity=0.04, line_width=0)
        fig_hist.add_vrect(x0=0.35, x1=risk_threshold, fillcolor='#D97706', opacity=0.04, line_width=0)
        fig_hist.add_vrect(x0=risk_threshold, x1=1, fillcolor='#DC2626', opacity=0.04, line_width=0)
        fig_hist.add_vline(x=0.35, line_dash='dash', line_color='#D97706', line_width=1.5,
                           annotation_text='Low/Med', annotation_font_size=10)
        fig_hist.add_vline(x=risk_threshold, line_dash='dash', line_color='#DC2626', line_width=1.5,
                           annotation_text='Med/High', annotation_font_size=10)
        fig_hist.update_layout(**CHART_LAYOUT, height=260, barmode='overlay',
                               xaxis=dict(title='Late Delivery Probability', showgrid=False, range=[0,1]),
                               yaxis=dict(title='Orders', gridcolor='#F3F4F6'),
                               legend=dict(orientation='h', y=1.05, x=1, xanchor='right', font=dict(size=11)))
        st.plotly_chart(fig_hist, width='stretch')

    with col_e:
        st.markdown('<div class="section-title">Top Risk Drivers</div>', unsafe_allow_html=True)
        top10 = feat_imp.head(10)
        max_fi = top10.max()
        labels_map = {
            'Order Profit Per Order':'Order Profit',
            'Benefit per order':'Benefit / Order',
            'Order Item Profit Ratio':'Profit Ratio',
            'Days for shipment (scheduled)':'Scheduled Days',
            'low_scheduled_days':'Tight Schedule ≤2d',
            'Order Item Total':'Item Total',
            'Sales per customer':'Customer Sales',
            'Order Item Discount':'Item Discount',
            'shipping_pressure':'Shipping Pressure',
            'Order Item Discount Rate':'Discount Rate'
        }
        for feat, val in top10.items():
            label = labels_map.get(feat, feat[:28])
            pct = val/max_fi*100
            st.markdown(f"""
            <div class="fi-row">
              <div class="fi-name">{label}</div>
              <div style="display:flex;align-items:center;gap:8px;">
                <div class="fi-bar-bg" style="flex:1">
                  <div class="fi-bar-fill" style="width:{pct:.0f}%"></div>
                </div>
                <span style="font-size:11px;font-family:'DM Mono';color:#6B7280;width:36px;text-align:right">{val:.3f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4: ROC Curve ──
    col_h, col_i = st.columns([1.3, 1])

    with col_h:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        from sklearn.metrics import roc_curve, auc as sk_auc
        fpr, tpr, roc_thresholds = roc_curve(y_test_full, probs_full)
        roc_auc_val = sk_auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name='Random Forest',
            line=dict(color='#378ADD', width=2.5),
            fill='tozeroy', fillcolor='rgba(55,138,221,0.08)',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines', name='Random baseline (AUC 0.5)',
            line=dict(color='#9CA3AF', width=1.5, dash='dash'),
            hoverinfo='skip'
        ))
        thresh_idx = (np.abs(roc_thresholds - DECISION_THRESHOLD)).argmin()
        fig_roc.add_trace(go.Scatter(
            x=[fpr[thresh_idx]], y=[tpr[thresh_idx]], mode='markers',
            marker=dict(color='#E24B4A', size=10, symbol='circle'),
            name=f'Operating point (threshold={DECISION_THRESHOLD})',
            hovertemplate=f'Decision threshold {DECISION_THRESHOLD}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
        ))
        fig_roc.update_layout(**CHART_LAYOUT, height=300,
                              xaxis=dict(title='False Positive Rate', range=[0,1], gridcolor='#F3F4F6'),
                              yaxis=dict(title='True Positive Rate', range=[0,1], gridcolor='#F3F4F6'),
                              legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center', font=dict(size=10)))
        st.plotly_chart(fig_roc, width='stretch')

    with col_i:
        st.markdown('<div class="section-title">Reading the ROC Curve</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:13px;color:#374151;line-height:1.9;">
          <div>Area under the curve (AUC): <b>{roc_auc_val:.4f}</b></div>
          <div style="margin-top:8px;">
            The curve shows the trade-off between catching true late deliveries
            (True Positive Rate) and raising false alarms (False Positive Rate)
            as the decision threshold changes.
          </div>
          <div style="margin-top:10px;padding-top:10px;border-top:1px solid #F3F4F6;">
            The red dot marks where the model currently operates, at a decision
            threshold of <b>{DECISION_THRESHOLD}</b> — recalibrated from the default
            0.5 because missing a genuinely late order costs more than an
            unnecessary check-in on a false alarm.
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — ORDER-LEVEL RISK
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Predict Risk for a New Order</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#6B7280;margin-bottom:1.5rem;">Enter order details to receive an instant late delivery risk score and recommended action.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Shipment**")
        inp_mode = st.selectbox("Shipping Mode", ['Standard Class','Same Day','First Class','Second Class'], label_visibility='collapsed')
        inp_days = st.slider("Scheduled Days", 0, 4, 2)
        inp_qty  = st.slider("Order Quantity", 1, 5, 2)
    with c2:
        st.markdown("**Customer**")
        inp_seg  = st.selectbox("Segment", ['Consumer','Corporate','Home Office'], label_visibility='collapsed')
        inp_mkt  = st.selectbox("Market", ['Pacific Asia','LATAM','USCA','Europe','Africa'])
        inp_dept = st.selectbox("Department", ['Footwear','Golf','Fan Shop','Apparel','Outdoors','Fitness','Book Shop','Technology','Health and Beauty ','Pet Shop','Discs Shop'])
    with c3:
        st.markdown("**Financials**")
        inp_profit  = st.number_input("Profit ($)", -500.0, 500.0, 50.0, 10.0)
        inp_benefit = st.number_input("Benefit ($)", -500.0, 500.0, 40.0, 10.0)
        inp_sales   = st.number_input("Sales ($)", 0.0, 2000.0, 200.0, 10.0)
    with c4:
        st.markdown("**Order**")
        inp_discount      = st.number_input("Discount ($)", 0.0, 100.0, 5.0, 1.0)
        inp_discount_rate = st.slider("Discount Rate", 0.0, 0.25, 0.05, 0.01)
        inp_status        = st.selectbox("Order Status", ['COMPLETE','ON_HOLD','PENDING_PAYMENT','PENDING','PROCESSING','SUSPECTED_FRAUD','CLOSED','PAYMENT_REVIEW','CANCELED'])

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("  🔮  Calculate Risk Score  ", width='stretch')

    if btn:
        input_dict = {col: 0 for col in feature_cols}
        input_dict.update({
            'Days for shipment (scheduled)': inp_days,
            'Order Item Quantity': inp_qty,
            'Order Item Discount Rate': inp_discount_rate,
            'Order Profit Per Order': inp_profit,
            'Benefit per order': inp_benefit,
            'Order Item Discount': inp_discount,
            'Sales per customer': inp_sales,
            'Sales': inp_sales,
            'Order Item Total': inp_sales - inp_discount,
            'Order Item Profit Ratio': inp_profit / max(inp_sales, 1),
            'shipping_pressure': inp_days / (inp_qty + 1),
            'discount_flag': 1 if inp_discount > 0 else 0,
            'profit_flag': 1 if inp_profit < 0 else 0,
            'high_discount': 1 if inp_discount_rate > 0.15 else 0,
            'low_scheduled_days': 1 if inp_days <= 2 else 0,
        })
        for mode in ['Standard Class','Second Class','Same Day']:
            k = f'Shipping Mode_{mode}'
            if k in input_dict: input_dict[k] = 1 if inp_mode == mode else 0
        for seg in ['Corporate','Home Office']:
            k = f'Customer Segment_{seg}'
            if k in input_dict: input_dict[k] = 1 if inp_seg == seg else 0
        for mkt in ['Europe','LATAM','Pacific Asia','USCA']:
            k = f'Market_{mkt}'
            if k in input_dict: input_dict[k] = 1 if inp_mkt == mkt else 0
        sk = f'Order Status_{inp_status}'
        if sk in input_dict: input_dict[sk] = 1
        dk = f'Department Name_{inp_dept}'
        if dk in input_dict: input_dict[dk] = 1

        inp_df = pd.DataFrame([input_dict])
        inp_df[num_cols] = scaler.transform(inp_df[num_cols])
        prob = model.predict_proba(inp_df)[0][1]

        if prob >= risk_threshold:
            tier, accent, badge_cls = 'HIGH RISK', '#DC2626', 'badge-red'
            action = "Immediate action required — proactively contact customer, consider rerouting"
            icon = "🔴"
        elif prob >= 0.35:
            tier, accent, badge_cls = 'MEDIUM RISK', '#D97706', 'badge-yellow'
            action = "Monitor closely — flag for daily check-in and prepare contingency"
            icon = "🟡"
        else:
            tier, accent, badge_cls = 'LOW RISK', '#059669', 'badge-green'
            action = "Standard processing — no intervention required"
            icon = "🟢"

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent}">
                <div class="kpi-label">Risk Category</div>
                <div class="kpi-value" style="font-size:1.4rem;color:{accent}">{icon} {tier}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{accent}">
                <div class="kpi-label">Late Delivery Probability</div>
                <div class="kpi-value">{prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:#6B7280">
                <div class="kpi-label">Scheduled Days</div>
                <div class="kpi-value">{inp_days}d</div>
                <div class="kpi-sub">{'⚠ Tight schedule' if inp_days<=2 else '✓ Normal schedule'}</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:#6B7280">
                <div class="kpi-label">Shipping Mode</div>
                <div class="kpi-value" style="font-size:1.1rem">{inp_mode}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:10px;
                    padding:12px 16px;margin-top:1rem;display:flex;align-items:center;gap:10px;">
          <span style="font-size:18px">💡</span>
          <span style="font-size:13px;color:#0369A1"><b>Recommended Action:</b> {action}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            delta={'reference': avg_late_rate, 'valueformat': '.1f', 'suffix':'%',
                   'increasing':{'color':'#DC2626'}, 'decreasing':{'color':'#059669'}},
            number={'suffix':'%', 'font':{'size':36,'family':'DM Mono','color':'#1A1D2E'}},
            title={'text':"Late Delivery Probability", 'font':{'size':13,'family':'DM Sans','color':'#6B7280'}},
            gauge={
                'axis':{'range':[0,100], 'tickcolor':'#9CA3AF', 'tickfont':{'size':10}},
                'bar':{'color':accent, 'thickness':0.25},
                'bgcolor':'#F9FAFB',
                'borderwidth':0,
                'steps':[
                    {'range':[0,35], 'color':'#F0FDF4'},
                    {'range':[35,int(risk_threshold*100)], 'color':'#FFFBEB'},
                    {'range':[int(risk_threshold*100),100], 'color':'#FEF2F2'}
                ],
                'threshold':{'line':{'color':accent,'width':3},'thickness':0.8,'value':prob*100}
            }
        ))
        fig_gauge.update_layout(**{**CHART_LAYOUT, 'paper_bgcolor': 'rgba(255,255,255,1)',
                                    'margin': dict(t=40,b=20,l=40,r=40)}, height=280)
        g_col = st.columns([1,2,1])
        with g_col[1]:
            st.plotly_chart(fig_gauge, width='stretch')

        # ── SHAP explanation for this prediction ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Why This Score? (SHAP Explanation)</div>', unsafe_allow_html=True)

        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(inp_df)
        if shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        labels_map_shap = {
            'Order Profit Per Order':'Order Profit', 'Benefit per order':'Benefit / Order',
            'Order Item Profit Ratio':'Profit Ratio', 'Days for shipment (scheduled)':'Scheduled Days',
            'low_scheduled_days':'Tight Schedule ≤2d', 'Order Item Total':'Item Total',
            'Sales per customer':'Customer Sales', 'Order Item Discount':'Item Discount',
            'shipping_pressure':'Shipping Pressure', 'Order Item Discount Rate':'Discount Rate',
        }

        shap_df = pd.DataFrame({'feature': feature_cols, 'shap_value': sv})
        shap_df['abs_val'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_val', ascending=False).head(8)
        shap_df['label'] = shap_df['feature'].apply(lambda f: labels_map_shap.get(f, f[:30]))

        max_abs = shap_df['abs_val'].max()
        for _, row in shap_df.iterrows():
            pct = (row['abs_val'] / max_abs * 100) if max_abs > 0 else 0
            color = '#E24B4A' if row['shap_value'] > 0 else '#1D9E75'
            direction = '↑ increases risk' if row['shap_value'] > 0 else '↓ decreases risk'
            st.markdown(f"""
            <div class="fi-row">
              <div class="fi-name">{row['label']}</div>
              <div style="display:flex;align-items:center;gap:8px;">
                <div class="fi-bar-bg" style="flex:1">
                  <div class="fi-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
                </div>
                <span style="font-size:11px;font-family:'DM Mono';color:{color};width:150px;text-align:right">{direction} ({row['shap_value']:+.3f})</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.caption("SHAP values show how much each feature pushed this specific order's risk score up or down, relative to the model's average prediction.")

# ════════════════════════════════════════════════════════════
# TAB 3 — REGION & MODE
# ════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Shipping Mode Risk</div>', unsafe_allow_html=True)
        mode_df = flt.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        mode_df.columns = ['Mode','Rate']
        mode_df['Rate%'] = (mode_df['Rate']*100).round(1)
        mode_df['Color'] = mode_df['Rate%'].apply(lambda x: '#DC2626' if x>70 else '#D97706' if x>50 else '#059669')
        mode_df = mode_df.sort_values('Rate%', ascending=False)
        fig_mode = go.Figure(go.Bar(
            x=mode_df['Mode'], y=mode_df['Rate%'],
            marker_color=mode_df['Color'].tolist(),
            marker_line_color='rgba(0,0,0,0)',
            text=mode_df['Rate%'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(size=12, family='DM Mono'),
            width=0.5,
            hovertemplate='<b>%{x}</b><br>Late rate: %{y:.1f}%<extra></extra>'
        ))
        fig_mode.add_hline(y=avg_late_rate, line_dash='dot', line_color='#9CA3AF', line_width=1,
                           annotation_text=f'Avg {avg_late_rate:.1f}%', annotation_font_size=10, annotation_font_color='#9CA3AF')
        fig_mode.update_layout(**CHART_LAYOUT, height=280,
                               yaxis=dict(range=[0,115], gridcolor='#F3F4F6', ticksuffix='%'),
                               xaxis=dict(showgrid=False))
        st.plotly_chart(fig_mode, width='stretch')

    with col_b:
        st.markdown('<div class="section-title">Market Risk</div>', unsafe_allow_html=True)
        mkt_df = flt.groupby('Market')['Late_delivery_risk'].mean().reset_index()
        mkt_df.columns = ['Market','Rate']
        mkt_df['Rate%'] = (mkt_df['Rate']*100).round(1)
        mkt_df = mkt_df.sort_values('Rate%')
        fig_mkt = go.Figure(go.Bar(
            y=mkt_df['Market'], x=mkt_df['Rate%'], orientation='h',
            marker_color='#2563EB', marker_line_color='rgba(0,0,0,0)',
            opacity=0.85,
            text=mkt_df['Rate%'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(size=12, family='DM Mono'),
            hovertemplate='<b>%{y}</b><br>Late rate: %{x:.1f}%<extra></extra>'
        ))
        fig_mkt.update_layout(**CHART_LAYOUT, height=280,
                              xaxis=dict(range=[0,70], gridcolor='#F3F4F6', ticksuffix='%'),
                              yaxis=dict(showgrid=False))
        st.plotly_chart(fig_mkt, width='stretch')

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Shipping Mode × Segment Heatmap</div>', unsafe_allow_html=True)
        hm = flt.groupby(['Shipping Mode','Customer Segment'])['late_prob'].mean().unstack().round(3)
        fig_hm = px.imshow(hm, color_continuous_scale=['#D1FAE5','#FEF3C7','#FEE2E2'],
                           text_auto='.2f', aspect='auto')
        fig_hm.update_coloraxes(showscale=False)
        fig_hm.update_layout(**CHART_LAYOUT, height=260)
        fig_hm.update_traces(textfont=dict(size=12, family='DM Mono'))
        st.plotly_chart(fig_hm, width='stretch')

    with col_d:
        st.markdown('<div class="section-title">Department Risk Ranking</div>', unsafe_allow_html=True)
        dept_df = flt.groupby('Department Name').agg(
            Rate=('Late_delivery_risk','mean'), Count=('Late_delivery_risk','count')
        ).reset_index()
        dept_df['Rate%'] = (dept_df['Rate']*100).round(1)
        dept_df = dept_df.sort_values('Rate%', ascending=False)
        dept_df['Color'] = dept_df['Rate%'].apply(lambda x: '#DC2626' if x>58 else '#D97706' if x>55 else '#059669')
        fig_dept = go.Figure(go.Bar(
            y=dept_df['Department Name'], x=dept_df['Rate%'], orientation='h',
            marker_color=dept_df['Color'].tolist(), marker_line_color='rgba(0,0,0,0)',
            text=dept_df['Rate%'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(size=11, family='DM Mono'),
            hovertemplate='<b>%{y}</b><br>Late rate: %{x:.1f}%<br>Orders: %{customdata:,}<extra></extra>',
            customdata=dept_df['Count']
        ))
        fig_dept.update_layout(**CHART_LAYOUT, height=320,
                               xaxis=dict(range=[0,70], gridcolor='#F3F4F6', ticksuffix='%'),
                               yaxis=dict(showgrid=False, tickfont=dict(size=11)))
        st.plotly_chart(fig_dept, width='stretch')

# ════════════════════════════════════════════════════════════
# TAB 4 — ACTION PANEL
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Operations Action Queue</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    for col, tier, count, rate, color, bg, action_text in [
        (a1,'🔴 High Risk', n_high, f'{high_rate:.1f}%', '#DC2626','#FEF2F2','Immediate intervention'),
        (a2,'🟡 Medium Risk',n_medium,f'{medium_rate:.1f}%','#D97706','#FFFBEB','Active monitoring'),
        (a3,'🟢 Low Risk',   n_low,  f'{low_rate:.1f}%',  '#059669','#F0FDF4','Standard processing'),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{color};background:{bg}">
                <div class="kpi-label" style="color:{color}">{tier}</div>
                <div class="kpi-value" style="color:{color}">{count:,}</div>
                <div class="kpi-sub">Actual late rate: {rate}</div>
                <div style="font-size:11px;color:{color};margin-top:6px;font-weight:600">{action_text}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        panel_tier = st.selectbox("Filter by Risk Tier", ['High','Medium','Low','All'])
    with fc2:
        top_n = st.slider("Number of orders to show", 10, 300, 100)
    with fc3:
        sort_col = st.selectbox("Sort by", ['Late Probability','Profit','Scheduled Days'])

    sort_map = {'Late Probability':'late_prob','Profit':'Order Profit Per Order','Scheduled Days':'Days for shipment (scheduled)'}

    action_df = flt.copy()
    if panel_tier != 'All': action_df = action_df[action_df['risk_category']==panel_tier]
    action_df = action_df.sort_values(sort_map[sort_col], ascending=(sort_col!='Late Probability')).head(top_n)

    display_cols = ['Shipping Mode','Market','Customer Segment','Department Name',
                    'Days for shipment (scheduled)','Order Item Quantity',
                    'Order Profit Per Order','late_prob','risk_category','Late_delivery_risk']
    disp = action_df[[c for c in display_cols if c in action_df.columns]].copy()
    disp['late_prob'] = (disp['late_prob']*100).round(1).astype(str)+'%'
    disp = disp.rename(columns={
        'late_prob':'Risk Prob','risk_category':'Risk Tier',
        'Late_delivery_risk':'Actual Late','Days for shipment (scheduled)':'Sched Days',
        'Order Profit Per Order':'Profit ($)','Order Item Quantity':'Qty'
    })

    st.dataframe(disp, width='stretch', height=380)

    dl1, dl2 = st.columns(2)
    with dl1:
        csv = action_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Orders CSV", data=csv,
                          file_name=f"apl_risk_{panel_tier.lower()}.csv",
                          mime='text/csv', width='stretch')
    with dl2:
        high_only = flt[flt['risk_category']=='High'].to_csv(index=False).encode('utf-8')
        st.download_button("🚨 Download ALL High-Risk Orders", data=high_only,
                          file_name="apl_high_risk_all.csv",
                          mime='text/csv', width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended Actions by Tier</div>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    actions = {
        ac1: ('🔴 High Risk', '#DC2626', [
            'Proactively contact customer before dispatch',
            'Evaluate rerouting or alternative carriers',
            'Escalate to operations manager immediately',
            'Prioritise in fulfilment queue',
            'Prepare proactive delay compensation offer'
        ]),
        ac2: ('🟡 Medium Risk', '#D97706', [
            'Flag order for daily status check-in',
            'Set internal threshold alerts on carrier',
            'Prepare customer communication template',
            'Monitor warehouse processing time',
            'Review scheduled days vs actual capacity'
        ]),
        ac3: ('🟢 Low Risk', '#059669', [
            'Process through standard workflow',
            'No customer intervention required',
            'Redirect saved capacity to High-risk orders',
            'Apply routine shipment tracking',
            'Normal SLA commitments apply'
        ])
    }
    for col, (title, color, items) in actions.items():
        with col:
            st.markdown(f'<p style="font-size:13px;font-weight:700;color:{color};margin-bottom:8px">{title}</p>', unsafe_allow_html=True)
            for item in items:
                st.markdown(f"""
                <div class="action-item">
                  <div class="action-dot" style="background:{color}"></div>
                  <span style="font-size:12px;color:#374151">{item}</span>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="border-top:1px solid #E8EAF0;padding-top:1rem;
            display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:12px;color:#9CA3AF">APL Logistics · Late Delivery Risk Intelligence Platform · v2.0</span>
  <span style="font-size:12px;color:#9CA3AF">Random Forest · {len(df_raw):,} orders · AUC {metrics['auc']}</span>
</div>
""", unsafe_allow_html=True)
