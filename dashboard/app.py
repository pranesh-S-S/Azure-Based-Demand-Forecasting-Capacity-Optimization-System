"""
Azure Capacity Intelligence Dashboard
Production-level Streamlit dashboard for Azure cloud forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Azure Capacity Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }

/* Global Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
div[data-testid="stVerticalBlock"] > div {
    animation: fadeIn 0.4s ease-out forwards;
}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 26, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,107,53,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { 
    background: linear-gradient(135deg, #FF6B35, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Glassmorphism KPI Cards */
.kpi-card {
    background: rgba(26, 31, 46, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 12px 14px; /* Reduced padding */
    margin: 6px 0;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    min-height: 100px; /* Ensure uniform height and prevent crushing */
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4), 0 0 15px rgba(255,107,53,0.15);
    border-color: rgba(255,107,53,0.4);
}
.kpi-label {
    font-size: 0.6rem; /* Reduced font size */
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-value {
    font-size: 1.25rem; /* Significantly reduced to prevent overlap in columns */
    font-weight: 800;
    color: #FAFAFA;
    line-height: 1.2;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    word-wrap: break-word; /* Allow wrapping if absolutely necessary */
}
.kpi-sub {
    font-size: 0.65rem; /* Reduced font size */
    color: #4ade80;
    margin-top: 4px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-sub.warn { color: #fbbf24; }
.kpi-sub.danger { color: #f87171; }

/* Section headers */
.section-header {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FF6B35;
    margin: 25px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,107,53,0.2);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Tab styling (Apple-like pills) */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 600;
    font-size: 0.8rem;
    color: #94a3b8;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF6B35, #e55a2b) !important;
    color: white !important;
    box-shadow: 0 4px 10px rgba(255,107,53,0.3);
}

/* Glassmorphism Plotly container */
[data-testid="stPlotlyChart"] {
    background: rgba(26, 31, 46, 0.4);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 10px;
    transition: all 0.3s;
    margin-bottom: 15px;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(255,255,255,0.15);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

/* Download button */
.stDownloadButton button {
    background: rgba(255,107,53,0.1) !important;
    border: 1px solid rgba(255,107,53,0.5) !important;
    color: #FF6B35 !important;
    transition: all 0.2s;
    font-size: 0.8rem !important; /* Smaller text on button */
    padding: 4px 12px !important;
}
.stDownloadButton button:hover {
    background: rgba(255,107,53,0.2) !important;
    box-shadow: 0 0 10px rgba(255,107,53,0.3) !important;
}

/* Hide default streamlit footer */
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Plotly Dark Template ─────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.5)",
        font=dict(family="Inter", color="#c9d1d9", size=10), # Reduced global plotly font
        title=dict(font=dict(size=14, color="#FAFAFA")), # Smaller titles
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=11), tickfont=dict(size=10)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=11), tickfont=dict(size=10)),
        colorway=["#FF6B35","#4ade80","#38bdf8","#a78bfa","#fb923c","#f472b6","#34d399","#fbbf24"],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), itemsizing="constant"),
        margin=dict(l=40, r=40, t=60, b=40), # Increased margins to prevent overlap
        hoverlabel=dict(bgcolor="#1a1f2e", font_size=11, font_color="#FAFAFA"),
        autosize=True
    )
)

ACCENT = "#FF6B35"
COLORS = ["#FF6B35","#4ade80","#38bdf8","#a78bfa","#fb923c","#f472b6","#34d399","#fbbf24"]


def kpi_card(label, value, sub="", sub_class=""):
    cls = f"kpi-sub {sub_class}" if sub_class else "kpi-sub"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="{cls}">{sub}</div>
    </div>"""


# ── Data Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    root = os.path.join(base, '..')
    models_dir = os.path.join(root, 'models')
    medians = joblib.load(os.path.join(models_dir, 'imputation_medians.pkl'))
    clips = joblib.load(os.path.join(models_dir, 'clip_bounds.pkl'))
    group_stats = joblib.load(os.path.join(models_dir, 'group_stats.pkl'))
    return medians, clips, group_stats

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    root = os.path.join(base, '..')
    csv_path = os.path.join(root, 'data', 'azure_dataset_missing_values.csv')
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Sync Formatting
    df['region'] = df['region'].str.lower().str.replace(" ", "-")
    df['service_type'] = df['service_type'].str.strip()

    medians, clips, group_stats = load_artifacts()

    # Sync Imputation
    num_cols = ['usage_units','provisioned_capacity','cost_usd',
                'availability_pct','economic_index','market_demand_index']
    for col in num_cols:
        df[col] = df[col].fillna(medians.get(col, 0))

    # Sync Clipping
    for col, bounds in clips.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds[0], upper=bounds[1])

    # Re-calculate derived based on clipped values
    rate = (df['cost_usd'] / df['usage_units']).median()
    df['cost_usd'] = df['cost_usd'].fillna(df['usage_units'] * rate)

    # Time Features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)

    # ---- NITRO: FOURIER TERMS ----
    df['fourier_weekly_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['fourier_weekly_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['fourier_monthly_sin'] = np.sin(2 * np.pi * df['day'] / 30.44)
    df['fourier_monthly_cos'] = np.cos(2 * np.pi * df['day'] / 30.44)

    # Derived
    df['capacity_utilization'] = df['usage_units'] / df['provisioned_capacity']
    df['headroom_units'] = df['provisioned_capacity'] - df['usage_units']
    df['wasted_capacity_cost'] = df['headroom_units'].clip(lower=0) * rate
    df['over_capacity_flag'] = (df['usage_units'] > df['provisioned_capacity']).astype(int)
    df['waste_pct'] = (df['headroom_units'] / df['provisioned_capacity']).clip(0, 1) * 100

    # Rolling & Momentum (Sync with training script)
    df = df.sort_values(['region','service_type','timestamp'])
    
    # Lags
    df['lag_1']  = df.groupby(['region', 'service_type'])['usage_units'].shift(1)
    df['lag_2']  = df.groupby(['region', 'service_type'])['usage_units'].shift(2)
    df['lag_3']  = df.groupby(['region', 'service_type'])['usage_units'].shift(3)
    df['lag_7']  = df.groupby(['region', 'service_type'])['usage_units'].shift(7)
    df['lag_14'] = df.groupby(['region', 'service_type'])['usage_units'].shift(14)
    df['lag_30'] = df.groupby(['region', 'service_type'])['usage_units'].shift(30)
    
    # Weekly momentum
    df['usage_trend_7'] = df['lag_1'] - df['lag_7']

    # Rolling
    df['rolling_mean_7'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_std_7'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    df['rolling_max_7'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).max())
    df['rolling_min_7'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).min())
    df['rolling_mean_14'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df['rolling_std_14'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
    df['rolling_mean_30'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df['rolling_std_30'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).std())

    # Momentum Ratios
    epsilon = 1e-6
    df['momentum_3_7'] = df['rolling_mean_7'] # placeholder to match length
    df['momentum_3_7'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) / (df['rolling_mean_7'] + epsilon)
    df['momentum_7_30'] = df['rolling_mean_7'] / (df['rolling_mean_30'] + epsilon)

    # EWMA (Exponentially Weighted Moving Average) — sync with training
    df['ewma_7'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).ewm(span=7, min_periods=1).mean())
    df['ewma_14'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).ewm(span=14, min_periods=1).mean())
    df['ewma_30'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.shift(1).ewm(span=30, min_periods=1).mean())

    # Interaction Features — sync with training
    df['usage_x_economic']   = df['usage_units'] * df['economic_index']
    df['capacity_x_demand']  = df['provisioned_capacity'] * df['market_demand_index']
    df['util_x_availability'] = df['capacity_utilization'] * df['availability_pct']

    # Historical Anchoring + Deviation
    df['group_historical_avg'] = df.apply(lambda row: group_stats.get((row['region'], row['service_type']), 0), axis=1)
    df['deviation_from_group'] = df['lag_1'] - df['group_historical_avg']

    # Usage spike flag — sync with training
    df['usage_spike_flag'] = (df['usage_units'] > df['rolling_mean_7'] + 2 * df['rolling_std_7']).astype(int)

    # Daily growth (clipped to valid range)
    df['daily_growth'] = df.groupby(['region','service_type'])['usage_units'].transform(lambda x: x.pct_change())
    df['daily_growth'] = df['daily_growth'].fillna(0).clip(-1, 1)

    return df


@st.cache_resource
def load_model():
    # Sync paths with root where training script saves artifacts
    base = os.path.dirname(__file__)
    root_dir = os.path.join(base, '..')
    models_dir = os.path.join(root_dir, 'models')
    model = joblib.load(os.path.join(models_dir, 'best_xgboost_model.pkl'))
    features = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
    return model, features


df_raw = load_data()
model, model_features = load_model()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 Azure Capacity Intel")
    st.markdown("---")

    st.markdown("**Regions**")
    all_regions = sorted(df_raw['region'].unique())
    sel_regions = st.multiselect("Select Regions", all_regions, default=all_regions, label_visibility="collapsed")

    st.markdown("**Service Type**")
    all_services = sorted(df_raw['service_type'].unique())
    sel_services = st.multiselect("Select Services", all_services, default=all_services, label_visibility="collapsed")

    st.markdown("**Year**")
    all_years = sorted(df_raw['year'].unique())
    sel_years = st.multiselect("Select Years", all_years, default=all_years, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Capacity Risk Threshold**")
    util_threshold = st.slider("Utilization % alert level", 0.5, 1.0, 0.85, 0.01)

    st.markdown("---")
    st.markdown("### 🧪 What-If Analysis")
    st.caption("Simulate adjusting global provisioned capacity.")
    capacity_adj = st.slider("Global Capacity Adjustment (%)", -30, 30, 0, 5)

    st.markdown("---")
    st.caption("Milestone 4 · Forecast Integration & Capacity Planning")

# ── Filter Data & Apply What-If ──────────────────────────────────────────
df = df_raw[
    (df_raw['region'].isin(sel_regions)) &
    (df_raw['service_type'].isin(sel_services)) &
    (df_raw['year'].isin(sel_years))
].copy()

if df.empty:
    st.warning("No data for selected filters. Adjust sidebar filters.")
    st.stop()

# Apply What-If Adjustments
if capacity_adj != 0:
    multiplier = 1 + (capacity_adj / 100.0)
    df['provisioned_capacity'] = df['provisioned_capacity'] * multiplier
    df['capacity_utilization'] = df['usage_units'] / df['provisioned_capacity']
    df['headroom_units'] = df['provisioned_capacity'] - df['usage_units']
    rate = (df['cost_usd'] / df['usage_units']).median()
    df['wasted_capacity_cost'] = df['headroom_units'].clip(lower=0) * rate
    df['over_capacity_flag'] = (df['usage_units'] > df['provisioned_capacity']).astype(int)
    df['waste_pct'] = (df['headroom_units'] / df['provisioned_capacity']).clip(0, 1) * 100

# ── Header ───────────────────────────────────────────────────────────────
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:800;letter-spacing:2px;
    color:#FAFAFA;margin-bottom:0;'>AZURE CAPACITY INTELLIGENCE</h1>
    <p style='color:#8892a4;font-size:0.9rem;margin-top:4px;'>
    Milestone 4 · Forecast Integration & Capacity Planning Dashboard</p>
    """, unsafe_allow_html=True)
    st.info("ℹ️ **Data Glossary:** 'Usage Units' is observed compute. 'Provisioned Capacity' is total allocated limits. 'Risk Events' trigger when Usage > Threshold (85% default).")

with head_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Export Executive Raw Data (CSV)",
        data=csv_data,
        file_name='azure_capacity_executive_report.csv',
        mime='text/csv',
    )
    st.caption("Last Refreshed: Just now")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Overview",
    "📈 Demand Trends",
    "🌍 Regional Analysis",
    "🤖 Model & Forecast",
    "⚠️ Risk Alerts"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — KPI OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">EXECUTIVE KPIS</div>', unsafe_allow_html=True)

    total_cost = df['cost_usd'].sum()
    wasted_cost = df['wasted_capacity_cost'].sum()
    avg_util = df['capacity_utilization'].mean() * 100
    risk_events = int((df['capacity_utilization'] >= util_threshold).sum())
    underutil = int((df['capacity_utilization'] < 0.3).sum())
    avg_headroom = df['headroom_units'].mean()
    avg_growth = df['daily_growth'].mean() * 100
    total_records = len(df)

    r1 = st.columns(4)
    r1[0].markdown(kpi_card("Total Cost (USD)", f"${total_cost:,.0f}", "Filtered period"), unsafe_allow_html=True)
    r1[1].markdown(kpi_card("Wasted Capacity Cost", f"${wasted_cost:,.0f}",
        f"▲ {wasted_cost/total_cost*100:.1f}% of total spend" if total_cost else "", "warn"), unsafe_allow_html=True)
    r1[2].markdown(kpi_card("Avg Utilization", f"{avg_util:.1f}%", "Across all services"), unsafe_allow_html=True)
    r1[3].markdown(kpi_card("Capacity Risk Events", f"{risk_events:,}",
        f"{risk_events/total_records*100:.1f}% of records", "danger"), unsafe_allow_html=True)

    r2 = st.columns(4)
    r2[0].markdown(kpi_card("Total Records", f"{total_records:,}", "After filtering"), unsafe_allow_html=True)
    r2[1].markdown(kpi_card("Underutilized Flags", f"{underutil:,}",
        f"{underutil/total_records*100:.1f}% of records", "warn"), unsafe_allow_html=True)
    r2[2].markdown(kpi_card("Avg Headroom (Units)", f"{avg_headroom:,.0f}", "Available buffer"), unsafe_allow_html=True)
    r2[3].markdown(kpi_card("Avg Daily Growth", f"{avg_growth:.3f}%", "Per day, all regions"), unsafe_allow_html=True)

    # ── Cost Composition ──
    st.markdown('<div class="section-header">COST COMPOSITION</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cost_by_svc = df.groupby('service_type')['cost_usd'].sum().reset_index()
        fig = px.pie(cost_by_svc, values='cost_usd', names='service_type',
                     title="Cost Efficiency Breakdown", hole=0.45,
                     color_discrete_sequence=COLORS)
        fig.update_layout(PLOTLY_TEMPLATE['layout'])
        fig.update_traces(textinfo='label+percent', textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        monthly = df.set_index('timestamp').resample('M').agg(
            total_cost=('cost_usd','sum'), wasted=('wasted_capacity_cost','sum')).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly['timestamp'], y=monthly['total_cost'],
                             name='Total Cost', marker_color='#38bdf8'))
        fig.add_trace(go.Bar(x=monthly['timestamp'], y=monthly['wasted'],
                             name='Wasted Capacity', marker_color='#f87171'))
        fig.update_layout(PLOTLY_TEMPLATE['layout'], title="Monthly Cost vs Wasted Capacity",
                          barmode='group', legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1))
        st.plotly_chart(fig, use_container_width=True)

    # ── Utilization Distribution ──
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='capacity_utilization', nbins=40,
                           title="Utilization Distribution",
                           color_discrete_sequence=[ACCENT])
        fig.add_vline(x=util_threshold, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Risk Threshold ({util_threshold:.0%})")
        fig.update_layout(PLOTLY_TEMPLATE['layout'])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cost_region = df.groupby('region')['cost_usd'].sum().sort_values(ascending=True).tail(10).reset_index()
        fig = px.bar(cost_region, x='cost_usd', y='region', orientation='h',
                     title="Top Regions by Total Cost",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(PLOTLY_TEMPLATE['layout'])
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMAND TRENDS
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">USAGE & DEMAND OVER TIME</div>', unsafe_allow_html=True)

    metric_options = {'Usage Units':'usage_units', 'Utilization Pct':'capacity_utilization',
                      'Cost USD':'cost_usd', 'Headroom Units':'headroom_units',
                      'Wasted Capacity Cost':'wasted_capacity_cost'}
    sel_metric_label = st.selectbox("Primary Metric", list(metric_options.keys()))
    sel_metric = metric_options[sel_metric_label]

    group_by = st.radio("Group by", ["service_type", "region"], horizontal=True)

    monthly_trend = df.set_index('timestamp').groupby([pd.Grouper(freq='M'), group_by])[sel_metric].mean().reset_index()
    fig = px.line(monthly_trend, x='timestamp', y=sel_metric, color=group_by,
                  title=f"Monthly Avg {sel_metric_label} by {group_by.replace('_',' ').title()}",
                  color_discrete_sequence=COLORS)
    fig.update_layout(PLOTLY_TEMPLATE['layout'], height=420)
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig, use_container_width=True)

    # ── Growth & Seasonality ──
    c1, c2 = st.columns(2)
    with c1:
        growth = df.set_index('timestamp').resample('M')['daily_growth'].mean().reset_index()
        fig = px.bar(growth, x='timestamp', y='daily_growth',
                     title="Avg Daily Growth Rate (%)",
                     color_discrete_sequence=['#38bdf8'])
        fig.update_layout(PLOTLY_TEMPLATE['layout'])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        weekly = df.groupby('day_of_week')['usage_units'].mean().reset_index()
        weekly['day_name'] = weekly['day_of_week'].map(lambda x: day_names[x])
        baseline = weekly['usage_units'].mean()
        weekly['index'] = weekly['usage_units'] / baseline
        fig = px.bar(weekly, x='day_name', y='index', title="Weekly Seasonality Index",
                     color='day_name', color_discrete_sequence=COLORS)
        fig.add_hline(y=1.0, line_dash="dash", line_color="#8892a4",
                      annotation_text="Baseline")
        fig.update_layout(PLOTLY_TEMPLATE['layout'], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Rolling Statistics ──
    st.markdown('<div class="section-header">ROLLING STATISTICS (30-DAY)</div>', unsafe_allow_html=True)
    ts_agg = df.groupby('timestamp').agg(
        actual=('usage_units','mean'), rm30=('rolling_mean_30','mean'),
        rstd=('rolling_std_7','mean')).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_agg['timestamp'], y=ts_agg['rm30']+ts_agg['rstd'],
        mode='lines', line=dict(width=0), showlegend=False, name='upper'))
    fig.add_trace(go.Scatter(x=ts_agg['timestamp'], y=ts_agg['rm30']-ts_agg['rstd'],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(56,189,248,0.15)',
        name='Confidence Band'))
    fig.add_trace(go.Scatter(x=ts_agg['timestamp'], y=ts_agg['rm30'],
        mode='lines', line=dict(color='#38bdf8', width=2.5), name='30-Day Rolling Mean'))
    fig.add_trace(go.Scatter(x=ts_agg['timestamp'], y=ts_agg['actual'],
        mode='lines', line=dict(color='#a78bfa', width=1), name='Actual Usage', opacity=0.6))
    fig.update_layout(PLOTLY_TEMPLATE['layout'],
        title="Usage Units: Actual vs 30-Day Rolling Mean (±1σ)", height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">REGIONAL CAPACITY BREAKDOWN</div>', unsafe_allow_html=True)

    # Aggregate per region
    reg = df.groupby('region').agg(
        avg_util=('capacity_utilization','mean'),
        waste_pct=('waste_pct','mean'),
        total_cost=('cost_usd','sum'),
        risk_events=('over_capacity_flag','sum'),
    ).reset_index()
    reg['avg_util'] = reg['avg_util'] * 100

    fig = px.scatter(reg, x='avg_util', y='waste_pct', size='total_cost',
                     color='risk_events', hover_name='region',
                     title="Regions: Utilization vs Waste % (bubble = cost, color = risk events)",
                     color_continuous_scale='YlOrRd', size_max=55,
                     labels={'avg_util':'Avg Utilization (%)','waste_pct':'Waste % of Total Cost'})
    fig.update_layout(PLOTLY_TEMPLATE['layout'], height=480)
    fig.add_hline(y=reg['waste_pct'].median(), line_dash="dot", line_color="rgba(255,255,255,0.2)",
                  annotation_text="Median Waste %")
    fig.add_vline(x=util_threshold*100, line_dash="dot", line_color="#f87171",
                  annotation_text="Risk threshold")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        wasted_reg = df.groupby('region')['wasted_capacity_cost'].sum().sort_values(ascending=True).tail(10).reset_index()
        fig = px.bar(wasted_reg, x='wasted_capacity_cost', y='region', orientation='h',
                     title="Top 10 Regions by Wasted Capacity ($)",
                     color_discrete_sequence=['#f87171'])
        fig.update_layout(PLOTLY_TEMPLATE['layout'])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_reg = df.groupby('region')['over_capacity_flag'].sum().sort_values(ascending=True).tail(10).reset_index()
        fig = px.bar(risk_reg, x='over_capacity_flag', y='region', orientation='h',
                     title="Top 10 Regions by Capacity Risk Events",
                     color_discrete_sequence=['#fb923c'])
        fig.update_layout(PLOTLY_TEMPLATE['layout'])
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: region × month utilization
    st.markdown('<div class="section-header">UTILIZATION HEATMAP</div>', unsafe_allow_html=True)
    heat = df.groupby(['region','month'])['capacity_utilization'].mean().reset_index()
    heat_pivot = heat.pivot(index='region', columns='month', values='capacity_utilization')
    fig = px.imshow(heat_pivot, title="Avg Utilization by Region × Month",
                    color_continuous_scale='YlOrRd', aspect='auto',
                    labels=dict(x="Month", y="Region", color="Utilization"))
    fig.update_layout(PLOTLY_TEMPLATE['layout'], height=400)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL & FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">MODEL ACCURACY & PERFORMANCE</div>', unsafe_allow_html=True)

    # --- Prepare train/test split & predictions ---
    split_date = '2024-07-01'
    df_model = df_raw.copy()
    df_model['region_lower'] = df_model['region'].str.lower().str.replace(" ", "-")

    # Build features matching the model
    train_df = df_model[df_model['timestamp'] < split_date]
    test_df = df_model[df_model['timestamp'] >= split_date]

    if len(test_df) > 0:
        # Encode categoricals to match model features
        df_encoded = pd.get_dummies(df_model, columns=['region','service_type'], drop_first=True)
        test_enc = df_encoded[df_encoded['timestamp'] >= split_date].copy()
        y_test_vals = test_enc['usage_units'].values

        # Align columns with model features
        for col in model_features:
            if col not in test_enc.columns:
                test_enc[col] = 0
        X_test_model = test_enc[model_features]

        # Predict with Raw-Scale Nitro Model
        preds = model.predict(X_test_model)

        # --- Accuracy Calculations ---
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        safe_actual = np.where(y_test_vals == 0, np.nan, y_test_vals)
        ape = np.abs((y_test_vals - preds) / safe_actual) * 100

        mae = mean_absolute_error(y_test_vals, preds)
        rmse = np.sqrt(mean_squared_error(y_test_vals, preds))
        mape = np.nanmean(ape)
        r2 = r2_score(y_test_vals, preds)
        accuracy_pct = max(0.0, 100.0 - mape)
        within_5 = np.nanmean(ape <= 5) * 100
        within_10 = np.nanmean(ape <= 10) * 100
        within_15 = np.nanmean(ape <= 15) * 100
        bias = np.mean(preds - y_test_vals)

        # NRMSE — normalized by range of actual values
        y_range = y_test_vals.max() - y_test_vals.min()
        nrmse = rmse / y_range if y_range > 0 else 0.0

        # Directional accuracy
        if len(y_test_vals) > 1:
            actual_dir = np.sign(np.diff(y_test_vals))
            pred_dir = np.sign(np.diff(preds))
            dir_acc = np.mean(actual_dir == pred_dir) * 100
        else:
            dir_acc = 0.0

        # --- KPI Cards for Accuracy ---
        r1 = st.columns(5)
        r1[0].markdown(kpi_card("Model Accuracy", f"{accuracy_pct:.1f}%",
            f"100% - MAPE ({mape:.2f}%)"), unsafe_allow_html=True)
        r1[1].markdown(kpi_card("R² Score", f"{r2:.4f}",
            "Variance explained" if r2 > 0.8 else "Needs improvement",
            "" if r2 > 0.8 else "warn"), unsafe_allow_html=True)
        r1[2].markdown(kpi_card("RMSE", f"{rmse:.2f}",
            f"MAE: {mae:.2f}"), unsafe_allow_html=True)
        # NRMSE quality label
        nrmse_label = "Excellent" if nrmse < 0.10 else ("Good" if nrmse < 0.20 else "Needs improvement")
        nrmse_cls = "" if nrmse < 0.10 else ("warn" if nrmse < 0.20 else "danger")
        r1[3].markdown(kpi_card("NRMSE", f"{nrmse:.4f}",
            f"{nrmse_label} · Normalized by range", nrmse_cls), unsafe_allow_html=True)
        r1[4].markdown(kpi_card("Directional Accuracy", f"{dir_acc:.1f}%",
            "Up/down prediction correctness"), unsafe_allow_html=True)

        r2c = st.columns(4)
        r2c[0].markdown(kpi_card("Within ±5%", f"{within_5:.1f}%",
            "Tight tolerance band"), unsafe_allow_html=True)
        r2c[1].markdown(kpi_card("Within ±10%", f"{within_10:.1f}%",
            "Medium tolerance band"), unsafe_allow_html=True)
        r2c[2].markdown(kpi_card("Within ±15%", f"{within_15:.1f}%",
            "Loose tolerance band"), unsafe_allow_html=True)
        r2c[3].markdown(kpi_card("Forecast Bias", f"{bias:+.2f}",
            "Over-predicting" if bias > 0 else "Under-predicting",
            "warn" if abs(bias) > 50 else ""), unsafe_allow_html=True)

        # --- Accuracy Visualizations ---
        st.markdown('<div class="section-header">ACCURACY VISUALIZATIONS</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            # Within-tolerance bar chart
            tol_data = pd.DataFrame({
                'Tolerance': ['±5%', '±10%', '±15%'],
                'Predictions (%)': [within_5, within_10, within_15]
            })
            fig = px.bar(tol_data, x='Tolerance', y='Predictions (%)',
                         title="Predictions Within Tolerance Bands",
                         color='Tolerance',
                         color_discrete_sequence=['#4ade80','#38bdf8','#fbbf24'])
            fig.update_layout(PLOTLY_TEMPLATE['layout'], showlegend=False)
            fig.update_traces(text=[f"{v:.1f}%" for v in tol_data['Predictions (%)']],
                              textposition='outside', textfont_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Error distribution histogram
            fig = px.histogram(x=ape[~np.isnan(ape)], nbins=50,
                               title="Absolute Percentage Error Distribution",
                               labels={'x': 'APE (%)', 'y': 'Count'},
                               color_discrete_sequence=[ACCENT])
            fig.add_vline(x=5, line_dash="dash", line_color="#4ade80", annotation_text="5%")
            fig.add_vline(x=10, line_dash="dash", line_color="#38bdf8", annotation_text="10%")
            fig.update_layout(PLOTLY_TEMPLATE['layout'])
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            # NRMSE Gauge Visualization
            nrmse_color = "#4ade80" if nrmse < 0.10 else ("#fbbf24" if nrmse < 0.20 else "#f87171")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=nrmse,
                number={'suffix': '', 'font': {'size': 36, 'color': '#FAFAFA'}},
                delta={'reference': 0.10, 'increasing': {'color': '#f87171'}, 'decreasing': {'color': '#4ade80'}},
                title={'text': 'NRMSE (Normalized RMSE)', 'font': {'size': 14, 'color': '#c9d1d9'}},
                gauge={
                    'axis': {'range': [0, 0.5], 'tickcolor': '#8892a4', 'tickfont': {'size': 10, 'color': '#8892a4'}},
                    'bar': {'color': nrmse_color, 'thickness': 0.6},
                    'bgcolor': 'rgba(26,31,46,0.5)',
                    'borderwidth': 1,
                    'bordercolor': 'rgba(255,255,255,0.08)',
                    'steps': [
                        {'range': [0, 0.10], 'color': 'rgba(74, 222, 128, 0.15)'},
                        {'range': [0.10, 0.20], 'color': 'rgba(251, 191, 36, 0.15)'},
                        {'range': [0.20, 0.50], 'color': 'rgba(248, 113, 113, 0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': '#FF6B35', 'width': 3},
                        'thickness': 0.8,
                        'value': nrmse
                    }
                }
            ))
            fig.update_layout(
                PLOTLY_TEMPLATE['layout'],
                title="NRMSE Quality Gauge",
                height=320,
                annotations=[
                    dict(x=0.15, y=-0.12, text="🟢 <0.10", showarrow=False, font=dict(size=10, color='#4ade80'), xref='paper', yref='paper'),
                    dict(x=0.50, y=-0.12, text="🟡 0.10-0.20", showarrow=False, font=dict(size=10, color='#fbbf24'), xref='paper', yref='paper'),
                    dict(x=0.85, y=-0.12, text="🔴 >0.20", showarrow=False, font=dict(size=10, color='#f87171'), xref='paper', yref='paper'),
                ]
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Forecast vs Actual ---
        st.markdown('<div class="section-header">FORECAST VS ACTUAL</div>', unsafe_allow_html=True)
        
        display_mode = st.radio(
            "Select Display Mode:", 
            ["Both Combined", "Actual Only", "XGBoost Prediction Only"], 
            horizontal=True
        )
        
        ts_test = test_enc['timestamp'].values
        fig = go.Figure()
        
        if display_mode in ["Both Combined", "Actual Only"]:
            fig.add_trace(go.Scatter(x=ts_test, y=y_test_vals,
                mode='lines', name='Actual Usage', line=dict(color='#38bdf8', width=2.5)))
                
        if display_mode in ["Both Combined", "XGBoost Prediction Only"]:
            fig.add_trace(go.Scatter(x=ts_test, y=preds,
                mode='lines', name='XGBoost Prediction', line=dict(color=ACCENT, width=2, dash='dash')))
                
        fig.update_layout(PLOTLY_TEMPLATE['layout'],
            title=f"Forecast vs Actual - Displaying: {display_mode}", height=420,
            legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # --- Feature Importance ---
        st.markdown('<div class="section-header">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=model_features)
        top20 = importances.nlargest(20).sort_values()
        fig = px.bar(x=top20.values, y=top20.index, orientation='h',
                     title="Top 20 Feature Importances (XGBoost)",
                     labels={'x':'Importance','y':'Feature'},
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(PLOTLY_TEMPLATE['layout'], height=500)
        st.plotly_chart(fig, use_container_width=True)

        # --- Residuals Scatter ---
        c1, c2 = st.columns(2)
        with c1:
            residuals = y_test_vals - preds
            fig = px.scatter(x=preds, y=residuals,
                             title="Residuals vs Predicted",
                             labels={'x':'Predicted','y':'Residual'},
                             color_discrete_sequence=['#a78bfa'], opacity=0.5)
            fig.add_hline(y=0, line_color='#f87171', line_dash='dash')
            fig.update_layout(PLOTLY_TEMPLATE['layout'])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(x=y_test_vals, y=preds,
                             title="Actual vs Predicted (Parity Plot)",
                             labels={'x':'Actual','y':'Predicted'},
                             color_discrete_sequence=['#4ade80'], opacity=0.5)
            min_v = min(y_test_vals.min(), preds.min())
            max_v = max(y_test_vals.max(), preds.max())
            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                mode='lines', line=dict(color='#f87171', dash='dash'), name='Perfect'))
            fig.update_layout(PLOTLY_TEMPLATE['layout'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No test data available after the split date for accuracy evaluation.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK ALERTS
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">CAPACITY RISK ALERTS</div>', unsafe_allow_html=True)

    at_risk = df[df['capacity_utilization'] >= util_threshold].copy()
    total_at_risk = len(at_risk)
    pct_at_risk = total_at_risk / len(df) * 100 if len(df) > 0 else 0

    r1 = st.columns(4)
    r1[0].markdown(kpi_card("🚨 Active Risk Events", f"{total_at_risk:,}",
        f"{pct_at_risk:.1f}% of all records", "danger"), unsafe_allow_html=True)
    r1[1].markdown(kpi_card("Risk Threshold", f"{util_threshold:.0%}",
        "Adjust in sidebar"), unsafe_allow_html=True)
    over_cap = int(df['over_capacity_flag'].sum())
    r1[2].markdown(kpi_card("Over-Capacity Events", f"{over_cap:,}",
        "Usage > Provisioned", "danger"), unsafe_allow_html=True)
    if len(at_risk) > 0:
        worst_region = at_risk.groupby('region')['capacity_utilization'].mean().idxmax()
        worst_util = at_risk.groupby('region')['capacity_utilization'].mean().max() * 100
        r1[3].markdown(kpi_card("Highest Risk Region", worst_region,
            f"Avg util: {worst_util:.1f}%", "danger"), unsafe_allow_html=True)
    else:
        r1[3].markdown(kpi_card("Highest Risk Region", "None", "All regions healthy"), unsafe_allow_html=True)

    # Risk timeline
    st.markdown('<div class="section-header">RISK EVENT TIMELINE</div>', unsafe_allow_html=True)
    risk_timeline = df.set_index('timestamp').resample('W')['over_capacity_flag'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=risk_timeline['timestamp'], y=risk_timeline['over_capacity_flag'],
                         name='Risk Events', marker_color='#f87171'))
    fig.update_layout(PLOTLY_TEMPLATE['layout'],
        title="Weekly Capacity Risk Events", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # At-risk regions table
    if len(at_risk) > 0:
        st.markdown('<div class="section-header">AT-RISK RECORDS (TOP 50)</div>', unsafe_allow_html=True)
        display_cols = ['timestamp','region','service_type','usage_units',
                        'provisioned_capacity','capacity_utilization','cost_usd']
        avail_cols = [c for c in display_cols if c in at_risk.columns]
        show_df = at_risk[avail_cols].sort_values('capacity_utilization', ascending=False).head(50)
        show_df['capacity_utilization'] = (show_df['capacity_utilization']*100).round(1).astype(str) + '%'
        st.dataframe(show_df, use_container_width=True, height=400)

        # Recommendations
        st.markdown('<div class="section-header">CAPACITY PLANNING RECOMMENDATIONS</div>', unsafe_allow_html=True)
        rec_df = at_risk.groupby('region').agg(
            events=('over_capacity_flag','sum'),
            avg_util=('capacity_utilization','mean'),
            avg_headroom=('headroom_units','mean')
        ).sort_values('avg_util', ascending=False).reset_index()
        rec_df['avg_util'] = (rec_df['avg_util'] * 100).round(1)
        rec_df['recommendation'] = rec_df['avg_util'].apply(
            lambda x: "🔴 Scale immediately" if x > 95 else
                      "🟠 Plan scaling within 2 weeks" if x > 90 else
                      "🟡 Monitor closely")
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.success("✅ No capacity risk events detected with current threshold. Infrastructure is healthy.")
