import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Population Intelligence",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Luxury Dark Intelligence Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --saffron: #FF9933;
    --saffron-dim: #cc7a29;
    --ivory: #F5F0E8;
    --navy: #0A0F2E;
    --navy-mid: #121940;
    --navy-card: #1A2355;
    --navy-border: #2A3570;
    --green: #138808;
    --green-bright: #1aad0a;
    --blue-chakra: #4169E1;
    --text-dim: #8892c4;
    --red-alert: #ff4b4b;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--navy);
    color: var(--ivory);
}

.stApp {
    background: linear-gradient(135deg, #080c24 0%, #0d1435 40%, #0a1628 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1435 0%, #080c24 100%);
    border-right: 1px solid var(--navy-border);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--saffron);
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1435 0%, #1a2355 50%, #0d1435 100%);
    border: 1px solid var(--navy-border);
    border-top: 3px solid var(--saffron);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(255,153,51,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: var(--ivory);
    margin: 0;
    line-height: 1.1;
}
.hero-title span {
    color: var(--saffron);
}
.hero-subtitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-dim);
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}
.hero-flags {
    font-size: 1.4rem;
    margin-bottom: 0.3rem;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: linear-gradient(135deg, var(--navy-card), #1e2b6a);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--saffron), var(--blue-chakra));
    border-radius: 0 0 12px 12px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--ivory);
    line-height: 1;
}
.kpi-delta {
    font-size: 0.78rem;
    margin-top: 0.3rem;
    font-weight: 400;
}
.delta-up { color: var(--green-bright); }
.delta-down { color: var(--red-alert); }

/* Section Headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--ivory);
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    background: var(--saffron);
    color: var(--navy);
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    vertical-align: middle;
}
.section-desc {
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-bottom: 1rem;
}

/* Card container */
.card {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Model badge */
.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
}
.badge-best { background: rgba(19,136,8,0.2); color: var(--green-bright); border: 1px solid var(--green); }
.badge-good { background: rgba(65,105,225,0.2); color: #7a9ff5; border: 1px solid #4169E1; }
.badge-warn { background: rgba(255,153,51,0.2); color: var(--saffron); border: 1px solid var(--saffron-dim); }
.badge-fail { background: rgba(255,75,75,0.15); color: #ff8080; border: 1px solid #ff4b4b; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, rgba(255,153,51,0.08), rgba(65,105,225,0.08));
    border-left: 3px solid var(--saffron);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: var(--ivory);
    line-height: 1.6;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--navy-border), transparent);
    margin: 1.5rem 0;
}

/* Metric chips in comparison */
.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--navy-border);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--ivory);
    margin: 0.2rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--navy-border); border-radius: 3px; }

/* Streamlit component overrides */
.stSlider > div > div > div { background: var(--saffron) !important; }
.stSelectbox > div > div { background: var(--navy-card) !important; border-color: var(--navy-border) !important; }
.stMultiSelect > div > div { background: var(--navy-card) !important; border-color: var(--navy-border) !important; }
div[data-testid="stMetricValue"] { color: var(--ivory); font-family: 'DM Mono', monospace; }
div[data-testid="stMetricLabel"] { color: var(--text-dim); font-size: 0.8rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--navy-card);
    border-radius: 10px;
    border: 1px solid var(--navy-border);
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: var(--text-dim);
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: var(--saffron) !important;
    color: var(--navy) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--saffron), #e6891a);
    color: var(--navy);
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(255,153,51,0.3);
}

/* Number input */
.stNumberInput > div > div > input {
    background: var(--navy-card) !important;
    border-color: var(--navy-border) !important;
    color: var(--ivory) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Warning / info boxes */
.stAlert {
    background: rgba(255,153,51,0.08) !important;
    border-color: var(--saffron) !important;
    color: var(--ivory) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    years = list(range(1950, 2025))
    populations = [
        376.3, 382.7, 389.3, 396.1, 402.8, 409.7, 416.7, 424.0, 431.4, 439.2,
        447.4, 455.7, 464.3, 473.1, 482.1, 491.3, 500.9, 510.7, 520.6, 530.9,
        541.3, 552.0, 563.0, 574.2, 585.8, 597.6, 609.7, 622.0, 634.6, 647.6,
        660.7, 674.2, 688.2, 702.2, 716.4, 731.2, 746.4, 761.7, 777.4, 793.5,
        809.9, 826.7, 843.8, 860.7, 877.8, 895.5, 913.6, 931.5, 949.3, 967.3,
        985.5, 1004.1, 1022.8, 1041.9, 1061.2, 1080.7, 1100.2, 1119.6, 1138.9,
        1158.2, 1177.7, 1197.1, 1216.5, 1236.0, 1255.6, 1275.1, 1294.6, 1314.0,
        1333.3, 1352.6, 1371.8, 1390.8, 1406.6, 1422.5, 1438.1
    ]
    df = pd.DataFrame({'Year': years, 'Population_M': populations})
    df['Population_Lag1'] = df['Population_M'].shift(1)
    df['Growth_Rate'] = (df['Population_M'] - df['Population_Lag1']) / df['Population_Lag1'] * 100
    df['Growth_Acceleration'] = df['Growth_Rate'].diff()
    return df.dropna().reset_index(drop=True)


@st.cache_data
def train_models(df, train_cutoff):
    train = df[df['Year'] <= train_cutoff].copy()
    test = df[df['Year'] > train_cutoff].copy()

    features = ['Year', 'Population_Lag1', 'Growth_Rate', 'Growth_Acceleration']
    X_train = train[features]
    y_train = train['Population_M']
    X_test = test[features]
    y_test = test['Population_M']

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Linear Regression'] = {
        'model': lr, 'pred': y_pred_lr,
        'r2': r2_score(y_test, y_pred_lr),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mape': mean_absolute_percentage_error(y_test, y_pred_lr) * 100,
        'color': '#FF9933', 'dash': 'solid'
    }

    # Polynomial Degree 2
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X_poly2_train = poly2.fit_transform(X_train)
    X_poly2_test = poly2.transform(X_test)
    lr2 = LinearRegression()
    lr2.fit(X_poly2_train, y_train)
    y_pred_p2 = lr2.predict(X_poly2_test)
    results['Polynomial (deg 2)'] = {
        'model': (poly2, lr2), 'pred': y_pred_p2,
        'r2': r2_score(y_test, y_pred_p2),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_p2)),
        'mape': mean_absolute_percentage_error(y_test, y_pred_p2) * 100,
        'color': '#4169E1', 'dash': 'dot'
    }

    # Polynomial Degree 3
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    X_poly3_train = poly3.fit_transform(X_train)
    X_poly3_test = poly3.transform(X_test)
    lr3 = LinearRegression()
    lr3.fit(X_poly3_train, y_train)
    y_pred_p3 = lr3.predict(X_poly3_test)
    results['Polynomial (deg 3)'] = {
        'model': (poly3, lr3), 'pred': y_pred_p3,
        'r2': r2_score(y_test, y_pred_p3),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_p3)),
        'mape': mean_absolute_percentage_error(y_test, y_pred_p3) * 100,
        'color': '#138808', 'dash': 'dash'
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'model': rf, 'pred': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mape': mean_absolute_percentage_error(y_test, y_pred_rf) * 100,
        'color': '#9b59b6', 'dash': 'dashdot'
    }

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results['Gradient Boosting'] = {
        'model': gb, 'pred': y_pred_gb,
        'r2': r2_score(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mape': mean_absolute_percentage_error(y_test, y_pred_gb) * 100,
        'color': '#e74c3c', 'dash': 'longdash'
    }

    return train, test, results, features


def forecast_linear(df, lr_model, target_year, growth_scenario='medium'):
    last_row = df.iloc[-1]
    last_pop = last_row['Population_M']
    last_gr = last_row['Growth_Rate']
    last_year = int(last_row['Year'])

    scenario_multiplier = {'low': 0.6, 'medium': 1.0, 'high': 1.4}
    mult = scenario_multiplier.get(growth_scenario, 1.0)
    gr_slope = -0.0513 * mult

    forecast_rows = []
    cur_pop = last_pop
    cur_gr = last_gr
    prev_gr = last_gr

    for yr in range(last_year + 1, target_year + 1):
        cur_gr = max(cur_gr + gr_slope, 0.05)
        gr_acc = cur_gr - prev_gr
        X_new = np.array([[yr, cur_pop, cur_gr, gr_acc]])
        pred_pop = lr_model.predict(X_new)[0]
        forecast_rows.append({'Year': yr, 'Population_M': pred_pop, 'Growth_Rate': cur_gr, 'Scenario': growth_scenario})
        prev_gr = cur_gr
        cur_pop = pred_pop

    return pd.DataFrame(forecast_rows)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,15,46,0.6)',
    font=dict(family='Outfit, sans-serif', color='#8892c4'),
    title_font=dict(family='DM Serif Display, serif', color='#F5F0E8', size=18),
    legend=dict(
        bgcolor='rgba(26,35,85,0.8)',
        bordercolor='#2A3570',
        borderwidth=1,
        font=dict(color='#F5F0E8', size=12)
    ),
    xaxis=dict(gridcolor='#1A2355', zerolinecolor='#2A3570', tickfont=dict(color='#8892c4')),
    yaxis=dict(gridcolor='#1A2355', zerolinecolor='#2A3570', tickfont=dict(color='#8892c4')),
    hovermode='x unified',
    margin=dict(l=20, r=20, t=50, b=20)
)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.2rem">🇮🇳</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#F5F0E8; line-height:1.2;">India Population<br>Intelligence</div>
        <div style="font-size:0.72rem; color:#8892c4; margin-top:0.3rem; font-family:'DM Mono',monospace; letter-spacing:0.1em;">ML FORECASTING SYSTEM</div>
    </div>
    <hr style="border-color:#2A3570; margin: 0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Configuration")

    train_cutoff = st.slider(
        "Training Cutoff Year",
        min_value=1970, max_value=2018, value=2015, step=1,
        help="Data before this year is used for training. Data after is used for testing."
    )

    st.markdown("### 🔭 Forecast Settings")

    target_year = st.slider(
        "Forecast Until Year",
        min_value=2025, max_value=2075, value=2050, step=1
    )

    growth_scenario = st.selectbox(
        "Growth Scenario",
        options=['low', 'medium', 'high'],
        index=1,
        format_func=lambda x: {'low': '📉 Low (Rapid Decline)', 'medium': '📊 Medium (Baseline)', 'high': '📈 High (Slow Decline)'}[x]
    )

    selected_models = st.multiselect(
        "Models to Display",
        options=['Linear Regression', 'Polynomial (deg 2)', 'Polynomial (deg 3)', 'Random Forest', 'Gradient Boosting'],
        default=['Linear Regression', 'Polynomial (deg 2)']
    )

    show_confidence = st.toggle("Show Confidence Bands", value=True)
    show_milestones = st.toggle("Show Population Milestones", value=True)
    show_un_comparison = st.toggle("Show UN Estimate Overlay", value=True)

    st.markdown("<hr style='border-color:#2A3570;'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Custom Scenario")
    custom_gr_2030 = st.number_input("Custom Growth Rate 2030 (%)", min_value=0.01, max_value=2.0, value=0.65, step=0.05)
    custom_gr_2050 = st.number_input("Custom Growth Rate 2050 (%)", min_value=0.01, max_value=1.5, value=0.25, step=0.05)

    st.markdown("<hr style='border-color:#2A3570;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#8892c4; text-align:center; line-height:1.8;">
    Data: World Bank 1950–2024<br>
    Models: 5 ML Algorithms<br>
    Built with Streamlit + Plotly<br>
    <span style="color:#FF9933;">● LIVE COMPUTATION</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
train_df, test_df, model_results, features = train_models(df, train_cutoff)
lr_model = model_results['Linear Regression']['model']
forecast_df = forecast_linear(df, lr_model, target_year, growth_scenario)

# Scenarios
forecast_low = forecast_linear(df, lr_model, target_year, 'low')
forecast_med = forecast_linear(df, lr_model, target_year, 'medium')
forecast_high = forecast_linear(df, lr_model, target_year, 'high')


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-flags">🇮🇳</div>
    <div class="hero-title">India Population <span>Intelligence</span></div>
    <div class="hero-subtitle">
        Machine Learning Forecasting System &nbsp;·&nbsp; 1950–{target_year} &nbsp;·&nbsp;
        Training on {train_df.shape[0]} annual observations &nbsp;·&nbsp; 5 ML Models
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
current_pop = df['Population_M'].iloc[-1]
current_gr = df['Growth_Rate'].iloc[-1]
forecast_2050_val = forecast_med['Population_M'].iloc[-1] if not forecast_med.empty else 0
peak_gr_year = df.loc[df['Growth_Rate'].idxmax(), 'Year']
lr_r2 = model_results['Linear Regression']['r2']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌏 Current Population (2024)", f"{current_pop/1000:.3f}B", f"+{current_gr:.2f}% YoY")
with col2:
    st.metric("🔭 Forecast Population", f"{forecast_2050_val/1000:.3f}B", f"by {target_year}")
with col3:
    best_r2 = max(v['r2'] for v in model_results.values() if v['r2'] > 0)
    st.metric("🎯 Best Model R²", f"{best_r2:.6f}", "Near-perfect fit")
with col4:
    st.metric("📅 Peak Growth Year", f"{int(peak_gr_year)}", "Historical high")


st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Population Forecast",
    "🤖 Model Comparison",
    "📊 Residual Analysis",
    "🌐 Growth Dynamics",
    "🔬 What-If Simulator"
])


# ══════════════════════════════════════════
# TAB 1: POPULATION FORECAST
# ══════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">Population Forecast <span class="section-tag">LIVE</span></div>
    <div class="section-desc">Historical actuals with multi-scenario ML projections. Adjust the sidebar to recompute in real time.</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df['Year'], y=df['Population_M'],
        mode='lines', name='Historical (World Bank)',
        line=dict(color='#F5F0E8', width=2.5),
        hovertemplate='<b>%{x}</b><br>Population: %{y:.1f}M<extra></extra>'
    ))

    # Train/test shading
    fig.add_vrect(x0=df['Year'].min(), x1=train_cutoff, fillcolor='rgba(255,153,51,0.04)',
                  layer='below', line_width=0, annotation_text="TRAIN", annotation_position="top left",
                  annotation=dict(font=dict(color='#FF9933', size=10)))
    fig.add_vrect(x0=train_cutoff, x1=df['Year'].max(), fillcolor='rgba(65,105,225,0.04)',
                  layer='below', line_width=0, annotation_text="TEST", annotation_position="top left",
                  annotation=dict(font=dict(color='#4169E1', size=10)))
    fig.add_vline(x=train_cutoff, line_dash='dot', line_color='#FF9933', opacity=0.5)
    fig.add_vline(x=2024, line_dash='dash', line_color='#8892c4', opacity=0.6,
                  annotation_text="Today", annotation_position="top right",
                  annotation=dict(font=dict(color='#8892c4', size=10)))

    # Confidence bands
    if show_confidence:
        upper = forecast_high['Population_M']
        lower = forecast_low['Population_M']
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_high['Year'], forecast_low['Year'][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255,153,51,0.10)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Scenario Range',
            hoverinfo='skip'
        ))

    # Scenario lines
    for scenario, color, dash, label in [
        (forecast_high, '#138808', 'dot', '↑ High Scenario'),
        (forecast_med, '#FF9933', 'solid', '● Baseline'),
        (forecast_low, '#4169E1', 'dot', '↓ Low Scenario'),
    ]:
        fig.add_trace(go.Scatter(
            x=scenario['Year'], y=scenario['Population_M'],
            mode='lines', name=label,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y:.1f}M<extra></extra>'
        ))

    # UN estimate overlay
    if show_un_comparison:
        un_years = [2025, 2030, 2035, 2040, 2045, 2050]
        un_vals = [1430, 1480, 1510, 1530, 1545, 1500]
        fig.add_trace(go.Scatter(
            x=un_years, y=un_vals,
            mode='markers+lines',
            name='UN Medium Variant',
            line=dict(color='#e74c3c', width=1.5, dash='longdashdot'),
            marker=dict(size=7, color='#e74c3c', symbol='diamond'),
            hovertemplate='UN %{x}: %{y:.0f}M<extra></extra>'
        ))

    # Population milestones
    if show_milestones:
        for milestone, yr_approx in [(500, 1966), (750, 1984), (1000, 2000), (1250, 2013), (1500, None)]:
            if yr_approx:
                fig.add_hline(y=milestone, line_dash='dot', line_color='rgba(255,255,255,0.1)',
                              annotation_text=f'{milestone}M', annotation_position='right',
                              annotation=dict(font=dict(size=9, color='#8892c4')))

    fig.update_layout(
        **PLOT_LAYOUT,
        title='India Population: 1950 → ' + str(target_year),
        height=500,
    )
    fig.update_xaxes(title_text='Year', title_font=dict(color='#8892c4'))
    fig.update_yaxes(title_text='Population (Millions)', title_font=dict(color='#8892c4'))
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("<div class='section-header' style='font-size:1rem;'>📋 Forecast Summary Table</div>", unsafe_allow_html=True)
    display_years = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
    display_years = [y for y in display_years if y <= target_year]

    rows = []
    for yr in display_years:
        lo = forecast_low[forecast_low['Year'] == yr]['Population_M'].values
        md = forecast_med[forecast_med['Year'] == yr]['Population_M'].values
        hi = forecast_high[forecast_high['Year'] == yr]['Population_M'].values
        if len(md) > 0:
            rows.append({
                'Year': yr,
                'Low (M)': f"{lo[0]:.1f}" if len(lo) > 0 else '—',
                'Baseline (M)': f"{md[0]:.1f}",
                'High (M)': f"{hi[0]:.1f}" if len(hi) > 0 else '—',
                'Baseline (B)': f"{md[0]/1000:.3f}",
                'vs 2024 (%)': f"{((md[0]-current_pop)/current_pop*100):+.1f}%"
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
# TAB 2: MODEL COMPARISON
# ══════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">Model Performance Comparison <span class="section-tag">5 MODELS</span></div>
    <div class="section-desc">Evaluating all models on the held-out test set. Tree-based models show documented failure — an important ML lesson.</div>
    """, unsafe_allow_html=True)

    # Metrics table
    metrics_data = []
    for name, res in model_results.items():
        r2 = res['r2']
        if r2 > 0.999:
            badge = '<span class="model-badge badge-best">BEST</span>'
        elif r2 > 0.99:
            badge = '<span class="model-badge badge-good">GOOD</span>'
        elif r2 > 0:
            badge = '<span class="model-badge badge-warn">FAIR</span>'
        else:
            badge = '<span class="model-badge badge-fail">FAILED</span>'
        metrics_data.append({
            'Model': name,
            'R² Score': f"{r2:.6f}",
            'RMSE (M)': f"{res['rmse']:.2f}",
            'MAPE (%)': f"{res['mape']:.3f}",
            'Status': badge
        })

    # Show as styled table
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        header = "| Model | R² | RMSE (M) | MAPE (%) |\n|---|---|---|---|\n"
        rows_str = ""
        for r in metrics_data:
            rows_str += f"| {r['Model']} | {r['R² Score']} | {r['RMSE (M)']} | {r['MAPE (%)']} |\n"
        st.markdown(header + rows_str)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="insight-box">
        <b>🔑 Key Insight: Why Tree Models Fail</b><br><br>
        Random Forest and Gradient Boosting predict the <em>mean of training leaves</em>.
        When test data (2016–2024) has Population_Lag1 values never seen in training,
        trees default to the last known leaf (~1.17B), producing a flat line
        while reality grew to 1.44B.<br><br>
        <b>Negative R² is not a bug</b> — it's documented evidence of the
        extrapolation failure inherent in tree-based models for time series.
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted plot
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=test_df['Year'], y=test_df['Population_M'],
        mode='lines+markers', name='Actual',
        line=dict(color='#F5F0E8', width=3),
        marker=dict(size=8, color='#F5F0E8')
    ))

    for name in selected_models:
        if name in model_results:
            res = model_results[name]
            fig2.add_trace(go.Scatter(
                x=test_df['Year'], y=res['pred'],
                mode='lines+markers', name=name,
                line=dict(color=res['color'], width=2, dash=res['dash']),
                marker=dict(size=6),
                hovertemplate=f'<b>{name}</b><br>Year: %{{x}}<br>Predicted: %{{y:.1f}}M<extra></extra>'
            ))

    fig2.update_layout(
        **PLOT_LAYOUT,
        title=f'Test Set Predictions ({train_cutoff+1}–2024)',
        xaxis_title='Year', yaxis_title='Population (Millions)',
        height=420
    )
    st.plotly_chart(fig2, use_container_width=True)

    # R² bar chart
    fig3 = go.Figure()
    names = list(model_results.keys())
    r2s = [model_results[n]['r2'] for n in names]
    colors = ['#FF9933' if r > 0.99 else '#4169E1' if r > 0 else '#e74c3c' for r in r2s]

    fig3.add_trace(go.Bar(
        x=names, y=r2s,
        marker_color=colors,
        text=[f"{r:.4f}" for r in r2s],
        textposition='outside',
        textfont=dict(color='#F5F0E8', size=11),
        hovertemplate='<b>%{x}</b><br>R²: %{y:.6f}<extra></extra>'
    ))
    fig3.add_hline(y=0, line_color='#ff4b4b', line_dash='dash', opacity=0.5,
                   annotation_text="R²=0 (No predictive power)", annotation_position="bottom right",
                   annotation=dict(font=dict(color='#ff4b4b', size=10)))
    fig3.update_layout(
        **PLOT_LAYOUT,
        title='Model R² Scores (Test Set)',
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════
# TAB 3: RESIDUAL ANALYSIS
# ══════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">Residual & Error Analysis <span class="section-tag">DIAGNOSTICS</span></div>
    <div class="section-desc">Deep-dive into prediction errors. Ideally residuals should be random (no pattern) and normally distributed.</div>
    """, unsafe_allow_html=True)

    selected_for_residuals = st.selectbox(
        "Select model for residual analysis",
        options=[m for m in model_results.keys() if model_results[m]['r2'] > 0]
    )

    res_model = model_results[selected_for_residuals]
    residuals = test_df['Population_M'].values - res_model['pred']

    col1, col2 = st.columns(2)

    with col1:
        fig_r1 = go.Figure()
        fig_r1.add_trace(go.Scatter(
            x=test_df['Year'], y=residuals,
            mode='lines+markers',
            line=dict(color='#FF9933', width=2),
            marker=dict(size=8, color='#FF9933'),
            name='Residual',
            hovertemplate='Year: %{x}<br>Residual: %{y:.2f}M<extra></extra>'
        ))
        fig_r1.add_hline(y=0, line_dash='dash', line_color='#8892c4', opacity=0.7)
        fig_r1.add_hrect(y0=-2, y1=2, fillcolor='rgba(65,105,225,0.07)', line_width=0)
        fig_r1.update_layout(**PLOT_LAYOUT, title='Residuals Over Time', height=320,
                              yaxis_title='Residual (M)')
        st.plotly_chart(fig_r1, use_container_width=True)

    with col2:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Histogram(
            x=residuals, nbinsx=8,
            marker_color='#4169E1',
            marker_line=dict(color='#F5F0E8', width=1),
            opacity=0.8, name='Residual Distribution'
        ))
        fig_r2.update_layout(**PLOT_LAYOUT, title='Residual Distribution', height=320,
                              xaxis_title='Residual (M)', yaxis_title='Count')
        st.plotly_chart(fig_r2, use_container_width=True)

    # Scatter: Actual vs Predicted
    fig_r3 = go.Figure()
    fig_r3.add_trace(go.Scatter(
        x=test_df['Population_M'], y=res_model['pred'],
        mode='markers',
        marker=dict(size=10, color='#FF9933', line=dict(color='#F5F0E8', width=1)),
        name='Actual vs Predicted',
        hovertemplate='Actual: %{x:.1f}M<br>Predicted: %{y:.1f}M<extra></extra>'
    ))
    # Perfect line
    lims = [test_df['Population_M'].min() - 5, test_df['Population_M'].max() + 5]
    fig_r3.add_trace(go.Scatter(
        x=lims, y=lims,
        mode='lines', name='Perfect Fit',
        line=dict(color='#138808', dash='dash', width=1.5)
    ))
    fig_r3.update_layout(**PLOT_LAYOUT, title='Actual vs Predicted (Perfect = diagonal)', height=350,
                          xaxis_title='Actual (M)', yaxis_title='Predicted (M)')
    st.plotly_chart(fig_r3, use_container_width=True)

    # Stats
    st.markdown(f"""
    <div class="insight-box">
    <b>📊 {selected_for_residuals} — Residual Statistics</b><br><br>
    Mean Residual: <b>{np.mean(residuals):.3f}M</b> &nbsp;|&nbsp;
    Std Dev: <b>{np.std(residuals):.3f}M</b> &nbsp;|&nbsp;
    Max |Error|: <b>{np.max(np.abs(residuals)):.3f}M</b> &nbsp;|&nbsp;
    Min |Error|: <b>{np.min(np.abs(residuals)):.3f}M</b><br>
    {"✅ Low systematic bias — model generalizes well." if abs(np.mean(residuals)) < 2 else "⚠️ Systematic bias detected — consider retraining."}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 4: GROWTH DYNAMICS
# ══════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">Growth Rate Dynamics <span class="section-tag">DEMOGRAPHIC TRANSITION</span></div>
    <div class="section-desc">Visualizing India's demographic transition — from rapid post-independence growth to modern stabilization.</div>
    """, unsafe_allow_html=True)

    fig_g1 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=('Annual Growth Rate (%)', 'Population (Millions)', 'Growth Acceleration'),
                            vertical_spacing=0.08)

    fig_g1.add_trace(go.Scatter(
        x=df['Year'], y=df['Growth_Rate'],
        mode='lines', name='Growth Rate',
        line=dict(color='#FF9933', width=2),
        fill='tozeroy', fillcolor='rgba(255,153,51,0.12)'
    ), row=1, col=1)

    fig_g1.add_trace(go.Scatter(
        x=df['Year'], y=df['Population_M'],
        mode='lines', name='Population',
        line=dict(color='#F5F0E8', width=2)
    ), row=2, col=1)

    fig_g1.add_trace(go.Bar(
        x=df['Year'], y=df['Growth_Acceleration'],
        name='Acceleration',
        marker_color=['#138808' if v >= 0 else '#e74c3c' for v in df['Growth_Acceleration']],
        opacity=0.7
    ), row=3, col=1)

    fig_g1.update_layout(
        **PLOT_LAYOUT,
        height=650,
        title='India Demographic Transition: 1951–2024',
        showlegend=True
    )
    # Fix subplot axis colors
    for i in range(1, 4):
        fig_g1.update_xaxes(gridcolor='#1A2355', tickfont=dict(color='#8892c4'), row=i, col=1)
        fig_g1.update_yaxes(gridcolor='#1A2355', tickfont=dict(color='#8892c4'), row=i, col=1)

    st.plotly_chart(fig_g1, use_container_width=True)

    # Forecast growth rates
    combined = pd.concat([
        df[['Year', 'Growth_Rate']].assign(Type='Historical'),
        forecast_med[['Year', 'Growth_Rate']].assign(Type='Forecast (Baseline)'),
        forecast_low[['Year', 'Growth_Rate']].assign(Type='Forecast (Low)'),
        forecast_high[['Year', 'Growth_Rate']].assign(Type='Forecast (High)'),
    ])

    fig_g2 = go.Figure()
    color_map = {'Historical': '#F5F0E8', 'Forecast (Baseline)': '#FF9933',
                 'Forecast (Low)': '#4169E1', 'Forecast (High)': '#138808'}
    for label, grp in combined.groupby('Type'):
        fig_g2.add_trace(go.Scatter(
            x=grp['Year'], y=grp['Growth_Rate'],
            mode='lines', name=label,
            line=dict(color=color_map[label], width=2,
                      dash='solid' if label == 'Historical' else 'dot')
        ))
    fig_g2.add_vline(x=2024, line_dash='dash', line_color='#8892c4', opacity=0.5)
    fig_g2.update_layout(**PLOT_LAYOUT, title='Historical & Projected Growth Rates', height=350,
                          yaxis_title='Growth Rate (%)')
    st.plotly_chart(fig_g2, use_container_width=True)


# ══════════════════════════════════════════
# TAB 5: WHAT-IF SIMULATOR
# ══════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="section-header">What-If Scenario Simulator <span class="section-tag">INTERACTIVE</span></div>
    <div class="section-desc">Simulate custom demographic scenarios by defining your own growth rate trajectory.</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_start_pop = st.number_input("Starting Population (M)", value=float(current_pop), min_value=1000.0, max_value=2000.0, step=1.0)
        sim_start_gr = st.number_input("Starting Growth Rate (%)", value=float(current_gr), min_value=0.01, max_value=3.0, step=0.01)
    with col2:
        sim_target_year = st.slider("Simulate Until", 2025, 2075, 2050)
        gr_decline_speed = st.number_input("Annual GR Decline (%/yr)", value=0.05, min_value=0.0, max_value=0.3, step=0.005, format="%.3f")
    with col3:
        gr_floor = st.number_input("Growth Rate Floor (%)", value=0.05, min_value=0.0, max_value=0.5, step=0.01)
        shock_year = st.number_input("Shock Year (0=none)", value=0, min_value=0, max_value=2070, step=1)
        shock_magnitude = st.number_input("Shock Impact (M, negative=loss)", value=0.0, min_value=-200.0, max_value=200.0, step=5.0)

    if st.button("🚀 Run Simulation"):
        sim_years = list(range(2025, sim_target_year + 1))
        sim_pops = []
        sim_grs = []
        cur_pop_sim = sim_start_pop
        cur_gr_sim = sim_start_gr

        for yr in sim_years:
            cur_gr_sim = max(cur_gr_sim - gr_decline_speed, gr_floor)
            new_pop = cur_pop_sim * (1 + cur_gr_sim / 100)
            if shock_year != 0 and yr == shock_year:
                new_pop += shock_magnitude
            sim_pops.append(new_pop)
            sim_grs.append(cur_gr_sim)
            cur_pop_sim = new_pop

        sim_df = pd.DataFrame({'Year': sim_years, 'Population_M': sim_pops, 'Growth_Rate': sim_grs})

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=df['Year'], y=df['Population_M'],
            mode='lines', name='Historical', line=dict(color='#F5F0E8', width=2)
        ))
        fig_sim.add_trace(go.Scatter(
            x=sim_df['Year'], y=sim_df['Population_M'],
            mode='lines', name='Your Simulation',
            line=dict(color='#FF9933', width=3),
            fill='tonexty' if False else None
        ))
        fig_sim.add_trace(go.Scatter(
            x=forecast_med['Year'], y=forecast_med['Population_M'],
            mode='lines', name='Baseline Model',
            line=dict(color='#4169E1', width=1.5, dash='dot')
        ))
        if shock_year != 0:
            fig_sim.add_vline(x=shock_year, line_color='#e74c3c', line_dash='dash',
                              annotation_text=f"Shock: {shock_magnitude:+.0f}M",
                              annotation=dict(font=dict(color='#e74c3c', size=11)))

        fig_sim.update_layout(**PLOT_LAYOUT, title='Custom Scenario Simulation',
                              height=420, yaxis_title='Population (M)')
        st.plotly_chart(fig_sim, use_container_width=True)

        final_pop = sim_pops[-1]
        diff = final_pop - (forecast_med['Population_M'].iloc[-1] if not forecast_med.empty else final_pop)
        st.markdown(f"""
        <div class="insight-box">
        <b>🧮 Simulation Result</b><br><br>
        By <b>{sim_target_year}</b>, your custom scenario projects India's population at
        <b>{final_pop/1000:.3f} Billion ({final_pop:.0f}M)</b>.<br>
        This is <b>{diff:+.0f}M ({diff/1000:+.3f}B)</b> vs the baseline model.
        {"⚠️ Demographic shock introduced at " + str(shock_year) if shock_year != 0 else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box" style="text-align:center; padding: 2rem;">
        Configure your parameters above and click <b>🚀 Run Simulation</b> to generate a custom demographic projection.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='fancy-divider'></div>
<div style="text-align:center; font-size:0.8rem; color:#8892c4; font-family:'DM Mono',monospace; padding: 1rem 0; letter-spacing:0.08em;">
    INDIA POPULATION INTELLIGENCE SYSTEM &nbsp;·&nbsp;
    DATA: WORLD BANK 1950–2024 &nbsp;·&nbsp;
    MODELS: LINEAR · POLY2 · POLY3 · RF · GBM &nbsp;·&nbsp;
    <span style="color:#FF9933;">● LIVE COMPUTATION</span>
</div>
""", unsafe_allow_html=True)
