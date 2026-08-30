# ============================================================
# APL LOGISTICS — PHASE 5: STREAMLIT DASHBOARD
# Late Delivery Risk Prediction System
# ============================================================
# HOW TO RUN:
#   1. pip install streamlit pandas scikit-learn plotly
#   2. Place APL_Logistics.csv in the same folder
#   3. streamlit run streamlit_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="APL Logistics — Delay Risk Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #378ADD;
    }
    .metric-card-red   { border-left-color: #E24B4A !important; }
    .metric-card-green { border-left-color: #1D9E75 !important; }
    .metric-card-yellow{ border-left-color: #EF9F27 !important; }
    .risk-high   { background:#3a1a1a; color:#ff6b6b; padding:4px 10px; border-radius:12px; font-weight:600; font-size:13px; }
    .risk-medium { background:#3a2e10; color:#EF9F27; padding:4px 10px; border-radius:12px; font-weight:600; font-size:13px; }
    .risk-low    { background:#0f2e1e; color:#1D9E75; padding:4px 10px; border-radius:12px; font-weight:600; font-size:13px; }
    .section-header { font-size:1.1rem; font-weight:600; color:#ffffff; margin-bottom:0.5rem; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & MODEL TRAINING (cached)
# ============================================================
@st.cache_data
def load_and_train():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / 'APL_Logistics.csv', encoding='latin1')

    drop_cols = [
        'Customer Fname','Customer Lname','Customer Street','Customer Zipcode',
        'Customer City','Customer State','Customer Country','Order City',
        'Order Country','Order State','Customer Id','Order Customer Id',
        'Category Id','Department Id','Latitude','Longitude','Product Name',
        'Delivery Status','Days for shipping (real)'
    ]
    df_raw = df.drop(columns=drop_cols).dropna().reset_index(drop=True)

    df_ml = df_raw.copy()
    df_ml['shipping_pressure']   = df_ml['Days for shipment (scheduled)'] / (df_ml['Order Item Quantity'] + 1)
    df_ml['discount_flag']       = (df_ml['Order Item Discount'] > 0).astype(int)
    df_ml['profit_flag']         = (df_ml['Order Profit Per Order'] < 0).astype(int)
    df_ml['high_discount']       = (df_ml['Order Item Discount Rate'] > 0.15).astype(int)
    df_ml['low_scheduled_days']  = (df_ml['Days for shipment (scheduled)'] <= 2).astype(int)

    df_encoded = pd.get_dummies(
        df_ml,
        columns=df_ml.select_dtypes(include='object').columns.tolist(),
        drop_first=True, dtype=int
    )

    X = df_encoded.drop(columns=['Late_delivery_risk'])
    y = df_encoded['Late_delivery_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler()
    X_train_s = X_train.copy(); X_test_s = X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train_s[num_cols])
    X_test_s[num_cols]  = scaler.transform(X_test_s[num_cols])

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    probs = rf.predict_proba(X_test_s)[:, 1]
    preds = rf.predict(X_test_s)

    metrics = {
        'auc':       round(roc_auc_score(y_test, probs), 4),
        'f1':        round(f1_score(y_test, preds), 4),
        'precision': round(precision_score(y_test, preds), 4),
        'recall':    round(recall_score(y_test, preds), 4),
    }

    scored = df_raw.iloc[X_test.index].copy().reset_index(drop=True)
    scored['late_delivery_probability'] = probs
    scored['risk_category'] = pd.cut(
        probs, bins=[0, 0.35, 0.65, 1.0], labels=['Low', 'Medium', 'High']
    )
    scored['predicted_label'] = preds

    fi = pd.Series(rf.feature_importances_, index=X_train_s.columns).sort_values(ascending=False)

    return df_raw, scored, metrics, fi, rf, scaler, X_train_s.columns.tolist(), num_cols

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("🚀 Loading data and training model..."):
    df_raw, scored, metrics, feat_imp, model, scaler, feature_cols, num_cols = load_and_train()

# ============================================================
# SIDEBAR — FILTERS
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/APL_logo.svg/320px-APL_logo.svg.png",
                 width=140, use_container_width=False)
st.sidebar.markdown("## 🔧 Filters")
st.sidebar.markdown("---")

shipping_modes = st.sidebar.multiselect(
    "Shipping Mode",
    options=['Standard Class','Same Day','First Class','Second Class'],
    default=['Standard Class','Same Day','First Class','Second Class']
)

markets = st.sidebar.multiselect(
    "Market / Region",
    options=['Pacific Asia','LATAM','USCA','Europe','Africa'],
    default=['Pacific Asia','LATAM','USCA','Europe','Africa']
)

segments = st.sidebar.multiselect(
    "Customer Segment",
    options=['Consumer','Corporate','Home Office'],
    default=['Consumer','Corporate','Home Office']
)

risk_threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.50, max_value=0.90, value=0.65, step=0.05,
    help="Orders above this probability are flagged as High Risk"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("ROC-AUC",   metrics['auc'])
st.sidebar.metric("F1 Score",  metrics['f1'])
st.sidebar.metric("Precision", metrics['precision'])
st.sidebar.metric("Recall",    metrics['recall'])

# ============================================================
# APPLY FILTERS
# ============================================================
filtered = scored.copy()
if shipping_modes:
    filtered = filtered[filtered['Shipping Mode'].isin(shipping_modes)]
if markets:
    filtered = filtered[filtered['Market'].isin(markets)]
if segments:
    filtered = filtered[filtered['Customer Segment'].isin(segments)]

# Re-apply risk threshold from slider
filtered['risk_category'] = pd.cut(
    filtered['late_delivery_probability'],
    bins=[0, 0.35, risk_threshold, 1.0],
    labels=['Low', 'Medium', 'High']
)

n_high   = (filtered['risk_category'] == 'High').sum()
n_medium = (filtered['risk_category'] == 'Medium').sum()
n_low    = (filtered['risk_category'] == 'Low').sum()

# ============================================================
# HEADER
# ============================================================
st.markdown("# 🚚 APL Logistics — Late Delivery Risk Intelligence")
st.markdown("**Predictive system for flagging high-risk shipments before they are dispatched**")
st.markdown("---")

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Overview",
    "🔍 Order-Level Risk",
    "🗺️ Region & Mode Analysis",
    "⚡ Operations Action Panel"
])

# ============================================================
# TAB 1 — RISK OVERVIEW
# ============================================================
with tab1:
    st.markdown("### Overall Risk Distribution")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders Scored", f"{len(filtered):,}")
    with col2:
        st.metric("🔴 High Risk", f"{n_high:,}", delta=f"{n_high/len(filtered)*100:.1f}%")
    with col3:
        st.metric("🟡 Medium Risk", f"{n_medium:,}", delta=f"{n_medium/len(filtered)*100:.1f}%")
    with col4:
        st.metric("🟢 Low Risk", f"{n_low:,}", delta=f"{n_low/len(filtered)*100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Risk Tier Distribution")
        tier_counts = filtered['risk_category'].value_counts().reindex(['Low','Medium','High'])
        fig_donut = go.Figure(go.Pie(
            labels=['Low Risk','Medium Risk','High Risk'],
            values=tier_counts.values,
            hole=0.55,
            marker_colors=['#1D9E75','#EF9F27','#E24B4A'],
            textinfo='label+percent',
            textfont_size=13
        ))
        fig_donut.update_layout(
            showlegend=False, height=320,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown("#### Risk Tier Accuracy Validation")
        tier_accuracy = []
        for tier in ['Low','Medium','High']:
            sub = filtered[filtered['risk_category'] == tier]
            if len(sub) > 0:
                rate = sub['Late_delivery_risk'].mean() * 100
                tier_accuracy.append({'Tier': tier, 'Actual Late Rate (%)': round(rate, 1), 'Count': len(sub)})
        acc_df = pd.DataFrame(tier_accuracy)
        fig_acc = px.bar(
            acc_df, x='Tier', y='Actual Late Rate (%)',
            color='Tier',
            color_discrete_map={'Low':'#1D9E75','Medium':'#EF9F27','High':'#E24B4A'},
            text='Actual Late Rate (%)'
        )
        fig_acc.add_hline(y=54.8, line_dash="dash", line_color="gray",
                          annotation_text="Overall avg 54.8%", annotation_position="top right")
        fig_acc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_acc.update_layout(
            height=320, showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', yaxis=dict(range=[0,100]),
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("#### Probability Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=filtered[filtered['Late_delivery_risk']==0]['late_delivery_probability'],
        name='Actual On-time', nbinsx=50,
        marker_color='#1D9E75', opacity=0.7
    ))
    fig_hist.add_trace(go.Histogram(
        x=filtered[filtered['Late_delivery_risk']==1]['late_delivery_probability'],
        name='Actual Late', nbinsx=50,
        marker_color='#E24B4A', opacity=0.7
    ))
    fig_hist.add_vline(x=0.35, line_dash="dash", line_color="orange",
                       annotation_text="Low/Medium (0.35)")
    fig_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Medium/High ({risk_threshold})")
    fig_hist.update_layout(
        barmode='overlay', height=280,
        xaxis_title='Late Delivery Probability',
        yaxis_title='Number of Orders',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', legend=dict(orientation='h', y=1.1),
        margin=dict(t=30,b=10,l=10,r=10)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Top 15 Risk Drivers (Feature Importance)")
    top15 = feat_imp.head(15).reset_index()
    top15.columns = ['Feature', 'Importance']
    fig_fi = px.bar(
        top15.sort_values('Importance'), x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#378ADD','#EF9F27','#E24B4A']
    )
    fig_fi.update_layout(
        height=420, showlegend=False, coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', margin=dict(t=10,b=10,l=10,r=10)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ============================================================
# TAB 2 — ORDER-LEVEL RISK PREDICTION
# ============================================================
with tab2:
    st.markdown("### 🔍 Predict Risk for a New Order")
    st.markdown("Enter order details below to get an instant late delivery risk score.")

    col1, col2, col3 = st.columns(3)
    with col1:
        inp_shipping_mode = st.selectbox("Shipping Mode", ['Standard Class','Same Day','First Class','Second Class'])
        inp_scheduled_days = st.slider("Scheduled Shipping Days", 0, 4, 2)
        inp_quantity = st.slider("Order Item Quantity", 1, 5, 2)
        inp_segment = st.selectbox("Customer Segment", ['Consumer','Corporate','Home Office'])

    with col2:
        inp_market = st.selectbox("Market", ['Pacific Asia','LATAM','USCA','Europe','Africa'])
        inp_discount_rate = st.slider("Discount Rate", 0.0, 0.25, 0.05, step=0.01)
        inp_profit = st.number_input("Order Profit Per Order ($)", -500.0, 500.0, 50.0, step=10.0)
        inp_department = st.selectbox("Department", ['Footwear','Golf','Fan Shop','Apparel','Outdoors',
                                                      'Fitness','Book Shop','Technology','Health and Beauty ','Pet Shop'])

    with col3:
        inp_order_status = st.selectbox("Order Status", ['COMPLETE','ON_HOLD','PENDING_PAYMENT',
                                                          'PENDING','PROCESSING','SUSPECTED_FRAUD'])
        inp_benefit = st.number_input("Benefit per Order ($)", -500.0, 500.0, 40.0, step=10.0)
        inp_discount = st.number_input("Item Discount ($)", 0.0, 100.0, 5.0, step=1.0)
        inp_sales = st.number_input("Sales per Customer ($)", 0.0, 2000.0, 200.0, step=10.0)

    if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
        # Build input row matching training features
        input_dict = {col: 0 for col in feature_cols}

        # Numerical features
        input_dict['Days for shipment (scheduled)'] = inp_scheduled_days
        input_dict['Order Item Quantity']            = inp_quantity
        input_dict['Order Item Discount Rate']       = inp_discount_rate
        input_dict['Order Profit Per Order']         = inp_profit
        input_dict['Benefit per order']              = inp_benefit
        input_dict['Order Item Discount']            = inp_discount
        input_dict['Sales per customer']             = inp_sales
        input_dict['Sales']                          = inp_sales
        input_dict['Order Item Total']               = inp_sales - inp_discount
        input_dict['Order Item Profit Ratio']        = inp_profit / max(inp_sales, 1)

        # Engineered features
        input_dict['shipping_pressure']   = inp_scheduled_days / (inp_quantity + 1)
        input_dict['discount_flag']       = 1 if inp_discount > 0 else 0
        input_dict['profit_flag']         = 1 if inp_profit < 0 else 0
        input_dict['high_discount']       = 1 if inp_discount_rate > 0.15 else 0
        input_dict['low_scheduled_days']  = 1 if inp_scheduled_days <= 2 else 0

        # One-hot flags
        for mode in ['Standard Class', 'Second Class', 'Same Day']:
            key = f'Shipping Mode_{mode}'
            if key in input_dict:
                input_dict[key] = 1 if inp_shipping_mode == mode else 0

        for seg in ['Corporate', 'Home Office']:
            key = f'Customer Segment_{seg}'
            if key in input_dict:
                input_dict[key] = 1 if inp_segment == seg else 0

        for mkt in ['Europe','LATAM','Pacific Asia','USCA']:
            key = f'Market_{mkt}'
            if key in input_dict:
                input_dict[key] = 1 if inp_market == mkt else 0

        status_key = f'Order Status_{inp_order_status}'
        if status_key in input_dict:
            input_dict[status_key] = 1

        dept_key = f'Department Name_{inp_department}'
        if dept_key in input_dict:
            input_dict[dept_key] = 1

        # Scale and predict
        input_df = pd.DataFrame([input_dict])
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        prob = model.predict_proba(input_df)[0][1]

        if prob >= risk_threshold:
            tier, color, emoji = 'HIGH RISK', '#E24B4A', '🔴'
            action = "⚡ Immediate action required — contact customer, consider rerouting"
        elif prob >= 0.35:
            tier, color, emoji = 'MEDIUM RISK', '#EF9F27', '🟡'
            action = "👀 Monitor closely — flag for daily check-in"
        else:
            tier, color, emoji = 'LOW RISK', '#1D9E75', '🟢'
            action = "✅ Standard processing — no action needed"

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Late Delivery Probability", f"{prob*100:.1f}%")
        with col_r2:
            st.markdown(f"**Risk Category**")
            st.markdown(f"<span style='color:{color};font-size:22px;font-weight:700;'>{emoji} {tier}</span>",
                        unsafe_allow_html=True)
        with col_r3:
            st.metric("Scheduled Days", inp_scheduled_days)

        st.info(f"**Recommended Action:** {action}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Late Delivery Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 35],  'color': '#0f2e1e'},
                    {'range': [35, int(risk_threshold*100)], 'color': '#3a2e10'},
                    {'range': [int(risk_threshold*100), 100], 'color': '#3a1a1a'}
                ],
                'threshold': {'line': {'color': "white", 'width': 3},
                              'thickness': 0.8, 'value': prob * 100}
            }
        ))
        fig_gauge.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', margin=dict(t=40,b=10,l=20,r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# TAB 3 — REGION & MODE ANALYSIS
# ============================================================
with tab3:
    st.markdown("### 🗺️ Risk by Region & Shipping Mode")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Late Risk Rate by Shipping Mode")
        mode_risk = filtered.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        mode_risk.columns = ['Shipping Mode', 'Late Rate']
        mode_risk['Late Rate %'] = (mode_risk['Late Rate'] * 100).round(1)
        mode_risk = mode_risk.sort_values('Late Rate %', ascending=False)
        fig_mode = px.bar(
            mode_risk, x='Shipping Mode', y='Late Rate %',
            color='Late Rate %',
            color_continuous_scale=['#1D9E75','#EF9F27','#E24B4A'],
            text='Late Rate %'
        )
        fig_mode.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_mode.update_layout(
            height=320, showlegend=False, coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', yaxis=dict(range=[0,110]),
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with col_b:
        st.markdown("#### Late Risk Rate by Market")
        market_risk = filtered.groupby('Market')['Late_delivery_risk'].mean().reset_index()
        market_risk.columns = ['Market', 'Late Rate']
        market_risk['Late Rate %'] = (market_risk['Late Rate'] * 100).round(1)
        fig_market = px.bar(
            market_risk.sort_values('Late Rate %', ascending=True),
            x='Late Rate %', y='Market', orientation='h',
            color='Late Rate %',
            color_continuous_scale=['#1D9E75','#EF9F27','#E24B4A'],
            text='Late Rate %'
        )
        fig_market.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_market.update_layout(
            height=320, showlegend=False, coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', xaxis=dict(range=[0,70]),
            margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig_market, use_container_width=True)

    st.markdown("#### Average Risk Probability — Shipping Mode × Customer Segment")
    heatmap_data = filtered.groupby(['Shipping Mode','Customer Segment'])['late_delivery_probability'].mean().unstack()
    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale=['#1D9E75','#EF9F27','#E24B4A'],
        text_auto='.2f', aspect='auto'
    )
    fig_heat.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', margin=dict(t=10,b=10,l=10,r=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Risk Distribution by Department")
    dept_risk = filtered.groupby('Department Name').agg(
        Late_Rate=('Late_delivery_risk','mean'),
        Order_Count=('Late_delivery_risk','count'),
        Avg_Probability=('late_delivery_probability','mean')
    ).reset_index()
    dept_risk['Late_Rate_%'] = (dept_risk['Late_Rate']*100).round(1)
    dept_risk['Avg_Probability_%'] = (dept_risk['Avg_Probability']*100).round(1)
    dept_risk = dept_risk.sort_values('Late_Rate_%', ascending=False)
    fig_dept = px.scatter(
        dept_risk, x='Order_Count', y='Late_Rate_%',
        size='Avg_Probability_%', color='Late_Rate_%',
        color_continuous_scale=['#1D9E75','#EF9F27','#E24B4A'],
        hover_name='Department Name',
        text='Department Name',
        labels={'Order_Count':'Number of Orders','Late_Rate_%':'Late Rate (%)'}
    )
    fig_dept.update_traces(textposition='top center', textfont_size=10)
    fig_dept.update_layout(
        height=380, showlegend=False, coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', margin=dict(t=10,b=10,l=10,r=10)
    )
    st.plotly_chart(fig_dept, use_container_width=True)

# ============================================================
# TAB 4 — OPERATIONS ACTION PANEL
# ============================================================
with tab4:
    st.markdown("### ⚡ Operations Action Panel")
    st.markdown("High-risk orders requiring immediate attention, sorted by risk probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"🔴 **{n_high:,} HIGH RISK orders** need immediate action")
    with col2:
        st.warning(f"🟡 **{n_medium:,} MEDIUM RISK orders** need monitoring")
    with col3:
        st.success(f"🟢 **{n_low:,} LOW RISK orders** — process normally")

    st.markdown("---")

    # Filters for action panel
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        panel_tier = st.selectbox("Show risk tier", ['High', 'Medium', 'Low', 'All'])
    with col_f2:
        top_n = st.slider("Show top N orders", 10, 200, 50)

    action_df = filtered.copy()
    if panel_tier != 'All':
        action_df = action_df[action_df['risk_category'] == panel_tier]

    action_df = action_df.sort_values('late_delivery_probability', ascending=False).head(top_n)

    display_cols = [
        'Shipping Mode', 'Market', 'Customer Segment', 'Department Name',
        'Days for shipment (scheduled)', 'Order Item Quantity',
        'Order Profit Per Order', 'late_delivery_probability', 'risk_category',
        'Late_delivery_risk'
    ]
    display_df = action_df[[c for c in display_cols if c in action_df.columns]].copy()
    display_df['late_delivery_probability'] = (display_df['late_delivery_probability']*100).round(1).astype(str) + '%'
    display_df = display_df.rename(columns={
        'late_delivery_probability': 'Risk Probability',
        'risk_category': 'Risk Tier',
        'Late_delivery_risk': 'Actual Late',
        'Days for shipment (scheduled)': 'Sched Days'
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420
    )

    # Download button
    csv_export = action_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk Orders CSV",
        data=csv_export,
        file_name=f"apl_high_risk_orders_{panel_tier.lower()}.csv",
        mime='text/csv',
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("#### 📋 Recommended Actions by Tier")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.markdown("**🔴 High Risk**")
        st.markdown("""
- Proactively contact customer
- Consider rerouting shipment
- Escalate to ops manager
- Prioritize in fulfillment queue
- Offer proactive compensation
        """)
    with col_a2:
        st.markdown("**🟡 Medium Risk**")
        st.markdown("""
- Flag for daily check-in
- Set internal alerts
- Prepare contingency plan
- Monitor carrier updates
- Ready customer comms template
        """)
    with col_a3:
        st.markdown("**🟢 Low Risk**")
        st.markdown("""
- Standard processing
- No customer action needed
- Use freed capacity for High risk
- Regular shipment tracking
- Normal SLA applies
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>"
    "APL Logistics — Late Delivery Risk Intelligence Dashboard | "
    "Built with Streamlit + Random Forest | Phase 5 of 5"
    "</p>",
    unsafe_allow_html=True
)
