# StartupLens — Streamlit demo: XGBoost + SHAP + Groq/Llama VC reports

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.explainer import StartupExplainer, GroqConnectionError

# Page config
st.set_page_config(
    page_title="StartupLens · VC Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud passes the key as a secret; the explainer reads os.environ and
# falls back to .env locally. Reading st.secrets with no secrets.toml raises and
# shows a warning banner, so check the file first.
_SECRETS_FILES = (Path.home() / ".streamlit" / "secrets.toml",
                  PROJECT_ROOT / ".streamlit" / "secrets.toml")
if any(p.exists() for p in _SECRETS_FILES) and "GROQ_API_KEY" in st.secrets:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])

# Inject CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #07080c;
    --surface:  #0e111a;
    --border:   #1e2535;
    --accent:   #00e5ff;
    --accent2:  #ff3d71;
    --accent3:  #a259ff;
    --text:     #c8d6e5;
    --muted:    #5a6a82;
    --success:  #00e5a0;
    --warn:     #ffb400;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
h1 { color: var(--accent) !important; letter-spacing: -1px; }
h2 { color: #ffffff !important; font-size: 1.2rem !important; margin-bottom: 0.5rem !important; }
h3 { color: var(--accent) !important; font-size: 1rem !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.7rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 18px rgba(0,229,255,0.35) !important;
}

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }
input, select, textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important; letter-spacing: 1px; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }

/* Expander */
.streamlit-expanderHeader { color: var(--accent) !important; background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; font-size: 0.78rem !important; }
.streamlit-expanderContent { background: var(--surface) !important; border: 1px solid var(--border) !important; }

/* Dataframes */
.stDataFrame { background: var(--surface) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Success/error banners */
.stSuccess { background: rgba(0,229,160,0.08) !important; border-left: 3px solid var(--success) !important; color: var(--success) !important; }
.stError   { background: rgba(255,61,113,0.08) !important; border-left: 3px solid var(--accent2) !important; color: var(--accent2) !important; }
.stWarning { background: rgba(255,180,0,0.08) !important; border-left: 3px solid var(--warn) !important; color: var(--warn) !important; }
.stInfo    { background: rgba(0,229,255,0.06) !important; border-left: 3px solid var(--accent) !important; color: var(--accent) !important; }

/* Tag pill */
.tag-success { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.7rem; background:rgba(0,229,160,0.12); border:1px solid #00e5a0; color:#00e5a0; font-family:'IBM Plex Mono'; }
.tag-fail    { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.7rem; background:rgba(255,61,113,0.12); border:1px solid #ff3d71; color:#ff3d71; font-family:'IBM Plex Mono'; }

/* Logo area */
.logo-area { display:flex; align-items:center; gap:10px; margin-bottom:1.5rem; }
.logo-hex  { font-size:2rem; color:#00e5ff; line-height:1; }
.logo-text { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800; color:#fff; letter-spacing:-0.5px; }
.logo-sub  { font-family:'IBM Plex Mono'; font-size:0.62rem; color:#5a6a82; letter-spacing:2px; text-transform:uppercase; }

/* Chart bg override */
.stPlotlyChart, .stImage { background: transparent !important; }

/* Section card */
.section-card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1.2rem 1.4rem; margin-bottom:1rem; }

/* Probability gauge number */
.prob-num { font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800; line-height:1; }
</style>
""", unsafe_allow_html=True)

# Constants
FEATURE_COLS = [
    'age_first_funding_year','age_last_funding_year','age_first_milestone_year',
    'age_last_milestone_year','relationships','funding_rounds','funding_total_usd',
    'milestones','is_CA','is_NY','is_MA','is_TX','is_otherstate',
    'is_software','is_web','is_mobile','is_enterprise','is_advertising',
    'is_gamesvideo','is_ecommerce','is_biotech','is_consulting','is_othercategory',
    'has_VC','has_angel','has_roundA','has_roundB','has_roundC','has_roundD',
    'avg_participants','is_top500'
]
COLS_SCALE = [
    'funding_total_usd','age_first_funding_year','age_last_funding_year',
    'age_first_milestone_year','age_last_milestone_year','avg_participants'
]
FEATURE_LABELS = {
    "age_first_funding_year":    "Edad primer financiamiento (años)",
    "age_last_funding_year":     "Edad último financiamiento (años)",
    "age_first_milestone_year":  "Edad primer hito (años)",
    "age_last_milestone_year":   "Edad último hito (años)",
    "relationships":             "Relaciones/contactos clave",
    "funding_rounds":            "Rondas de financiamiento",
    "funding_total_usd":         "Total financiado (USD)",
    "milestones":                "Hitos/milestones",
    "is_CA": "California","is_NY": "Nueva York","is_MA": "Massachusetts",
    "is_TX": "Texas","is_otherstate": "Otro estado",
    "is_software": "Software","is_web": "Web","is_mobile": "Mobile",
    "is_enterprise": "Enterprise","is_advertising": "Publicidad",
    "is_gamesvideo": "Games & Video","is_ecommerce": "E-commerce",
    "is_biotech": "Biotech","is_consulting": "Consultoría",
    "is_othercategory": "Otras industrias",
    "has_VC": "VC","has_angel": "Angel","has_roundA": "Ronda A",
    "has_roundB": "Ronda B","has_roundC": "Ronda C","has_roundD": "Ronda D",
    "avg_participants": "Participantes/ronda","is_top500": "Top 500",
}
SECTOR_MAP = {
    "Software": "is_software","Web": "is_web","Móvil (Mobile)": "is_mobile",
    "Empresarial (Enterprise)": "is_enterprise","Publicidad": "is_advertising",
    "Games & Video": "is_gamesvideo","E-commerce": "is_ecommerce",
    "Biotecnología": "is_biotech","Consultoría": "is_consulting","Otros": "is_othercategory",
}
STATE_MAP = {
    "California (CA)": "is_CA","Nueva York (NY)": "is_NY",
    "Massachusetts (MA)": "is_MA","Texas (TX)": "is_TX","Otro": "is_otherstate",
}
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"

# Load model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "No se encontro el modelo entrenado en "
            f"`{MODEL_PATH}`. Ejecuta `python scratch/recreate_model.py` "
            "desde la raiz del proyecto y recarga esta pagina."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_explainer() -> StartupExplainer:
    return StartupExplainer()

model = load_model()

# Helper functions
def build_row_from_inputs(inputs: dict) -> pd.DataFrame:
    # Build a properly structured DataFrame row from UI inputs
    row = {col: 0.0 for col in FEATURE_COLS}
    # Numeric
    for k in ['relationships','funding_rounds','funding_total_usd','milestones',
              'avg_participants','age_first_funding_year','age_last_funding_year',
              'age_first_milestone_year','age_last_milestone_year']:
        row[k] = float(inputs.get(k, 0))
    # Sector (one-hot)
    sector_col = SECTOR_MAP.get(inputs.get('sector',''), None)
    if sector_col: row[sector_col] = 1.0
    # State (one-hot)
    state_col = STATE_MAP.get(inputs.get('state',''), None)
    if state_col: row[state_col] = 1.0
    # Booleans
    for b in ['is_top500','has_VC','has_angel','has_roundA','has_roundB','has_roundC','has_roundD']:
        row[b] = 1.0 if inputs.get(b, False) else 0.0
    return pd.DataFrame([row], columns=FEATURE_COLS)

def scale_row(row_df: pd.DataFrame) -> pd.DataFrame:
    # Standardize continuous columns — approximate scaler stats from training distribution
    # Approximate scaler stats from typical training distribution
    MEANS = {
        'funding_total_usd': 14_000_000,
        'age_first_funding_year': 2.5,
        'age_last_funding_year': 4.5,
        'age_first_milestone_year': 2.0,
        'age_last_milestone_year': 4.0,
        'avg_participants': 3.2,
    }
    STDS = {
        'funding_total_usd': 28_000_000,
        'age_first_funding_year': 3.0,
        'age_last_funding_year': 3.5,
        'age_first_milestone_year': 2.8,
        'age_last_milestone_year': 3.5,
        'avg_participants': 2.2,
    }
    scaled = row_df.copy()
    for col in COLS_SCALE:
        scaled[col] = (scaled[col] - MEANS[col]) / STDS[col]
    return scaled

def run_prediction(row_df: pd.DataFrame):
    # Returns (probability, shap_values_dict, base_value)
    scaled = scale_row(row_df)
    prob = float(model.predict_proba(scaled)[0][1])
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(scaled)[0]
    base_val = float(explainer.expected_value)
    shap_dict = {col: float(shap_vals[i]) for i, col in enumerate(FEATURE_COLS)}
    return prob, shap_dict, base_val

def random_startup() -> dict:
    # Generate a random plausible startup input
    rng = np.random.default_rng()
    sector = rng.choice(list(SECTOR_MAP.keys()))
    state = rng.choice(list(STATE_MAP.keys()))
    rounds = int(rng.integers(1, 6))
    return {
        'relationships':          int(rng.integers(1, 40)),
        'funding_rounds':         rounds,
        'funding_total_usd':      float(rng.uniform(100_000, 60_000_000)),
        'milestones':             int(rng.integers(0, 7)),
        'avg_participants':       float(rng.uniform(1, 8)),
        'age_first_funding_year': float(rng.uniform(0, 5)),
        'age_last_funding_year':  float(rng.uniform(1, 10)),
        'age_first_milestone_year': float(rng.uniform(0, 5)),
        'age_last_milestone_year':  float(rng.uniform(1, 10)),
        'sector': sector,
        'state':  state,
        'is_top500':  bool(rng.choice([True, False], p=[0.3, 0.7])),
        'has_VC':     bool(rng.choice([True, False], p=[0.7, 0.3])),
        'has_angel':  bool(rng.choice([True, False], p=[0.4, 0.6])),
        'has_roundA': rounds >= 2,
        'has_roundB': rounds >= 3,
        'has_roundC': rounds >= 4,
        'has_roundD': rounds >= 5,
    }

# Plotting helpers
def plot_shap_waterfall(shap_dict: dict, base_value: float, prob: float, n=12):
    # Custom waterfall / horizontal bar chart for SHAP contributions
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    items = sorted(items, key=lambda x: x[1])
    names = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = ['#ff3d71' if v < 0 else '#00e5a0' for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(items))))
    fig.patch.set_facecolor('#0e111a')
    ax.set_facecolor('#0e111a')

    bars = ax.barh(names, values, color=colors, height=0.55, zorder=3)
    ax.axvline(0, color='#1e2535', linewidth=1.5, zorder=2)

    for bar, val in zip(bars, values):
        sign = '+' if val >= 0 else ''
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f'{sign}{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right',
            color='#c8d6e5', fontsize=7.5, family='monospace'
        )

    ax.set_xlabel('SHAP value', color='#5a6a82', fontsize=9)
    ax.tick_params(colors='#c8d6e5', labelsize=8.5)
    ax.spines[:].set_visible(False)
    ax.grid(axis='x', color='#1e2535', linestyle='--', linewidth=0.7, zorder=0)
    ax.set_title(f'SHAP — contribuciones locales  (base: {base_value:.3f})',
                 color='#5a6a82', fontsize=9, pad=10)
    plt.tight_layout()
    return fig

def plot_probability_gauge(prob: float):
    # Semicircular gauge for success probability
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={'aspect': 'equal'})
    fig.patch.set_facecolor('#07080c')
    ax.set_facecolor('#07080c')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.2, 1.3)
    ax.axis('off')

    theta_range = np.linspace(np.pi, 0, 200)
    xb = np.cos(theta_range); yb = np.sin(theta_range)
    ax.plot(xb, yb, color='#1e2535', linewidth=14, solid_capstyle='round', zorder=1)

    fill_end = np.pi - prob * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 200)
    xf = np.cos(theta_fill); yf = np.sin(theta_fill)
    color = '#00e5a0' if prob >= 0.5 else '#ff3d71'
    ax.plot(xf, yf, color=color, linewidth=14, solid_capstyle='round', zorder=2,
            alpha=0.9)

    ax.text(0, 0.22, f'{prob*100:.1f}%', ha='center', va='center',
            fontsize=26, fontweight='800', color=color, family='sans-serif')
    ax.text(0, -0.1, 'PROBABILIDAD DE ÉXITO', ha='center', va='center',
            fontsize=6.5, color='#5a6a82', family='monospace')
    plt.tight_layout()
    return fig

def plot_shap_dot(shap_dict: dict, n=15):
    # Dot plot showing top features by absolute SHAP impact
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    items_sorted = sorted(items, key=lambda x: x[1])
    names = [FEATURE_LABELS.get(k, k) for k, _ in items_sorted]
    vals = [v for _, v in items_sorted]
    colors = ['#ff3d71' if v < 0 else '#00e5a0' for v in vals]

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(items_sorted))))
    fig.patch.set_facecolor('#0e111a'); ax.set_facecolor('#0e111a')

    ax.hlines(range(len(names)), 0, vals, colors='#1e2535', linewidth=1.2, zorder=1)
    ax.scatter(vals, range(len(names)), c=colors, s=55, zorder=3)
    ax.axvline(0, color='#2a3448', linewidth=1, zorder=0)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8, color='#c8d6e5')
    ax.tick_params(axis='x', colors='#5a6a82', labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.grid(axis='x', color='#1a2030', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_xlabel('SHAP value', color='#5a6a82', fontsize=8)
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <span class="logo-hex">⬡</span>
        <div>
            <div class="logo-text">StartupLens</div>
            <div class="logo-sub">VC Intelligence · XGBoost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "MODO DE ENTRADA",
        ["🎲  Generación aleatoria", "✏️  Entrada manual"],
        label_visibility="visible"
    )

    st.markdown("---")

    # Random mode
    if mode == "🎲  Generación aleatoria":
        st.markdown("### 🎲 Startup aleatoria")
        st.caption("Genera un perfil sintético de startup con valores realistas y evalúalo al instante.")

        if st.button("⟳  GENERAR STARTUP", use_container_width=True):
            st.session_state['inputs'] = random_startup()
            st.session_state['run'] = True

        if 'inputs' in st.session_state and mode == "🎲  Generación aleatoria":
            inp = st.session_state['inputs']
            st.markdown("**Perfil generado:**")
            st.markdown(f"""
            <div style='font-size:0.72rem; color:#5a6a82; line-height:2'>
            <span style='color:#00e5ff'>Sector</span> {inp.get('sector','')} &nbsp;
            <span style='color:#00e5ff'>Estado</span> {inp.get('state','')}<br>
            <span style='color:#00e5ff'>Relaciones</span> {inp.get('relationships','')} &nbsp;
            <span style='color:#00e5ff'>Rondas</span> {inp.get('funding_rounds','')}<br>
            <span style='color:#00e5ff'>Fondeo</span> ${inp.get('funding_total_usd',0):,.0f}<br>
            <span style='color:#00e5ff'>Hitos</span> {inp.get('milestones','')} &nbsp;
            <span style='color:#00e5ff'>Top500</span> {'Sí' if inp.get('is_top500') else 'No'}
            </div>
            """, unsafe_allow_html=True)

    # Manual mode
    else:
        st.markdown("### ✏️ Ingresar startup")

        with st.expander("📍 Ubicación & Sector", expanded=True):
            sector = st.selectbox("Sector", list(SECTOR_MAP.keys()), key="m_sector")
            state = st.selectbox("Estado", list(STATE_MAP.keys()), key="m_state")

        with st.expander("💰 Financiamiento", expanded=True):
            funding_usd = st.number_input("Total financiado (USD)", 0.0, 500_000_000.0, 2_000_000.0, step=100_000.0, key="m_fund")
            funding_rounds = st.slider("Rondas de financiamiento", 1, 10, 2, key="m_rounds")
            avg_part = st.slider("Participantes/ronda (avg)", 1.0, 10.0, 3.0, 0.5, key="m_part")
            col1, col2 = st.columns(2)
            has_VC = col1.checkbox("VC", True, key="m_vc")
            has_angel = col2.checkbox("Angel", False, key="m_angel")
            col3, col4, col5, col6 = st.columns(4)
            has_rA = col3.checkbox("A", False, key="m_ra")
            has_rB = col4.checkbox("B", False, key="m_rb")
            has_rC = col5.checkbox("C", False, key="m_rc")
            has_rD = col6.checkbox("D", False, key="m_rd")

        with st.expander("🚀 Operación", expanded=True):
            relationships = st.slider("Relaciones clave", 0, 60, 8, key="m_rel")
            milestones = st.slider("Hitos alcanzados", 0, 10, 2, key="m_miles")
            is_top500 = st.checkbox("Listada en Top 500", False, key="m_top")

        with st.expander("📅 Tiempo de maduración", expanded=False):
            af = st.number_input("Edad primer financiamiento (años)", -5.0, 20.0, 1.0, 0.5, key="m_af")
            al = st.number_input("Edad último financiamiento (años)", -5.0, 20.0, 3.0, 0.5, key="m_al")
            amf = st.number_input("Edad primer hito (años)",           0.0,  20.0, 1.5, 0.5, key="m_amf")
            aml = st.number_input("Edad último hito (años)",           0.0,  20.0, 3.5, 0.5, key="m_aml")

        if st.button("▶  EVALUAR STARTUP", use_container_width=True):
            st.session_state['inputs'] = {
                'relationships': relationships,
                'funding_rounds': funding_rounds,
                'funding_total_usd': funding_usd,
                'milestones': milestones,
                'avg_participants': avg_part,
                'age_first_funding_year': af,
                'age_last_funding_year':  al,
                'age_first_milestone_year': amf,
                'age_last_milestone_year':  aml,
                'sector': sector,
                'state':  state,
                'is_top500':  is_top500,
                'has_VC':     has_VC,
                'has_angel':  has_angel,
                'has_roundA': has_rA,
                'has_roundB': has_rB,
                'has_roundC': has_rC,
                'has_roundD': has_rD,
            }
            st.session_state['run'] = True

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.62rem; color:#2a3448; text-align:center; letter-spacing:1px'>
    XGBoost · SHAP · Crunchbase · 923 startups<br>
    Accuracy 82% · ROC-AUC 0.89
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("""
<h1 style='margin-bottom:0'>⬡ StartupLens</h1>
<p style='color:#5a6a82; font-size:0.75rem; letter-spacing:2px; margin-top:0; margin-bottom:1.5rem'>
VENTURE CAPITAL INTELLIGENCE · PREDICTIVE ANALYTICS
</p>
""", unsafe_allow_html=True)

if 'run' not in st.session_state or not st.session_state.get('run'):
    # Welcome screen
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="section-card">
        <h3>🎲 Modo Aleatorio</h3>
        <p style='font-size:0.78rem; color:#5a6a82'>Genera una startup sintética y analiza su perfil de riesgo con un click.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="section-card">
        <h3>✏️ Modo Manual</h3>
        <p style='font-size:0.78rem; color:#5a6a82'>Ingresa los datos reales de tu startup y obtén un diagnóstico predictivo detallado.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="section-card">
        <h3>📊 SHAP + Informe</h3>
        <p style='font-size:0.78rem; color:#5a6a82'>Visualiza las contribuciones SHAP y consulta las métricas de evaluación del modelo.</p>
        </div>""", unsafe_allow_html=True)
    st.info("← Elige un modo en la barra lateral y evalúa una startup.")
    st.stop()

# Run prediction
inputs = st.session_state['inputs']
row_df = build_row_from_inputs(inputs)
prob, shap_dict, base_value = run_prediction(row_df)

verdict = "ÉXITO / ADQUISICIÓN" if prob >= 0.5 else "CIERRE / FRACASO"
verdict_color = "#00e5a0" if prob >= 0.5 else "#ff3d71"
verdict_tag = "tag-success" if prob >= 0.5 else "tag-fail"

# Header KPI row
kc1, kc2, kc3, kc4, kc5 = st.columns(5)
with kc1:
    st.metric("PROB. ÉXITO", f"{prob*100:.1f}%")
with kc2:
    st.metric("VEREDICTO", "ÉXITO ✓" if prob >= 0.5 else "CIERRE ✗")
with kc3:
    top_pos = max(shap_dict, key=lambda k: shap_dict[k])
    st.metric("TOP FACTOR +", FEATURE_LABELS.get(top_pos, top_pos)[:18])
with kc4:
    top_neg = min(shap_dict, key=lambda k: shap_dict[k])
    st.metric("TOP RIESGO", FEATURE_LABELS.get(top_neg, top_neg)[:18])
with kc5:
    n_pos = sum(1 for v in shap_dict.values() if v > 0.05)
    n_neg = sum(1 for v in shap_dict.values() if v < -0.05)
    st.metric("FACTORES ±", f"{n_pos}+ / {n_neg}−")

st.markdown("")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  SHAP · Explicabilidad  ",
    "  🎯  Predicción  ",
    "  📋  Perfil de Startup  ",
    "  📐  Métricas del Modelo  ",
    "  🤖  Informe IA  ",
])

# TAB 1 · SHAP
with tab1:
    st.markdown("## Análisis de Explicabilidad SHAP")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**Waterfall — contribuciones individuales**")
        fig_wf = plot_shap_waterfall(shap_dict, base_value, prob, n=14)
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

    with col_b:
        st.markdown("**Dot plot — magnitud de impacto**")
        fig_dot = plot_shap_dot(shap_dict, n=14)
        st.pyplot(fig_dot, use_container_width=True)
        plt.close(fig_dot)

    st.markdown("---")
    # Positive / Negative SHAP breakdown
    pos_items = sorted([(k,v) for k,v in shap_dict.items() if v > 0.02], key=lambda x: -x[1])[:6]
    neg_items = sorted([(k,v) for k,v in shap_dict.items() if v < -0.02], key=lambda x: x[1])[:6]

    c_pos, c_neg = st.columns(2)
    with c_pos:
        st.markdown("**✅ Factores positivos**")
        for feat, val in pos_items:
            label = FEATURE_LABELS.get(feat, feat)
            pct = val / (sum(v for v in shap_dict.values() if v > 0) + 1e-9)
            st.markdown(f"""
            <div style='margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;font-size:0.75rem'>
                <span style='color:#c8d6e5'>{label}</span>
                <span style='color:#00e5a0'>+{val:.3f}</span>
              </div>
              <div style='background:#1e2535;border-radius:2px;height:4px;margin-top:3px'>
                <div style='background:#00e5a0;height:4px;border-radius:2px;width:{min(pct*100*3,100):.0f}%'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with c_neg:
        st.markdown("**⚠️ Factores de riesgo**")
        for feat, val in neg_items:
            label = FEATURE_LABELS.get(feat, feat)
            pct = abs(val) / (sum(abs(v) for v in shap_dict.values() if v < 0) + 1e-9)
            st.markdown(f"""
            <div style='margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;font-size:0.75rem'>
                <span style='color:#c8d6e5'>{label}</span>
                <span style='color:#ff3d71'>{val:.3f}</span>
              </div>
              <div style='background:#1e2535;border-radius:2px;height:4px;margin-top:3px'>
                <div style='background:#ff3d71;height:4px;border-radius:2px;width:{min(pct*100*3,100):.0f}%'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🔢 Tabla completa de valores SHAP"):
        shap_df = pd.DataFrame([
            {"Feature": FEATURE_LABELS.get(k, k), "SHAP Value": round(v, 5), "Dirección": "↑ Positivo" if v > 0 else "↓ Negativo"}
            for k, v in sorted(shap_dict.items(), key=lambda x: -abs(x[1]))
        ])
        st.dataframe(shap_df, use_container_width=True, hide_index=True)


# TAB 2 · PREDICCIÓN
with tab2:
    st.markdown("## Resultado de Predicción")

    g1, g2 = st.columns([1.4, 2])
    with g1:
        fig_gauge = plot_probability_gauge(prob)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)
        st.markdown(f"""
        <div style='text-align:center; margin-top:-10px'>
          <span class='{verdict_tag}'>{verdict}</span>
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown(f"""
        <div class="section-card" style="border-color:{verdict_color}; margin-top:1rem">
        <h3 style="color:{verdict_color}">Veredicto del Modelo</h3>
        <p style="font-size:0.8rem; color:#c8d6e5; line-height:1.8">
        La startup analizada presenta una probabilidad estimada de éxito del
        <strong style="color:{verdict_color}">{prob*100:.1f}%</strong>, basada en
        31 variables operativas y financieras.
        </p>
        <p style="font-size:0.8rem; color:#c8d6e5; line-height:1.8">
        El factor que más impulsa esta predicción es
        <strong style="color:#00e5ff">{FEATURE_LABELS.get(max(shap_dict, key=lambda k: shap_dict[k]),'')} (+{shap_dict[max(shap_dict, key=lambda k: shap_dict[k])]:.3f})</strong>,
        mientras que el mayor riesgo identificado es
        <strong style="color:#ff3d71">{FEATURE_LABELS.get(min(shap_dict, key=lambda k: shap_dict[k]),'')} ({shap_dict[min(shap_dict, key=lambda k: shap_dict[k])]:.3f})</strong>.
        </p>
        <p style="font-size:0.75rem; color:#5a6a82">
        Umbral de decisión: 50% · Modelo: XGBoost · Dataset: Crunchbase 923 startups
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Distribución de contribuciones SHAP (suma → log-odds)**")
    total_pos = sum(v for v in shap_dict.values() if v > 0)
    total_neg = abs(sum(v for v in shap_dict.values() if v < 0))
    grand = total_pos + total_neg + 1e-9

    st.markdown(f"""
    <div style='margin-top:8px'>
      <div style='font-size:0.7rem; color:#5a6a82; margin-bottom:4px'>
        Positivos: <span style='color:#00e5a0'>+{total_pos:.3f}</span> &nbsp;|&nbsp;
        Negativos: <span style='color:#ff3d71'>-{total_neg:.3f}</span> &nbsp;|&nbsp;
        Base value: <span style='color:#a259ff'>{base_value:.3f}</span>
      </div>
      <div style='display:flex; height:16px; border-radius:4px; overflow:hidden; background:#1e2535'>
        <div style='background:#00e5a0;width:{total_pos/grand*100:.1f}%;opacity:0.85'></div>
        <div style='background:#ff3d71;width:{total_neg/grand*100:.1f}%;opacity:0.85'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# TAB 3 · PERFIL
with tab3:
    st.markdown("## Perfil de la Startup Evaluada")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
        <div class="section-card">
        <h3>📍 Ubicación & Categoría</h3>
        <table style='font-size:0.78rem; width:100%; border-collapse:collapse'>
        <tr><td style='color:#5a6a82; padding:4px 0'>Sector</td><td style='color:#00e5ff'>{inputs.get('sector','—')}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Estado</td><td style='color:#c8d6e5'>{inputs.get('state','—')}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Top 500</td><td style='color:#c8d6e5'>{'✓ Sí' if inputs.get('is_top500') else '✗ No'}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card" style="margin-top:0.5rem">
        <h3>💰 Financiamiento</h3>
        <table style='font-size:0.78rem; width:100%; border-collapse:collapse'>
        <tr><td style='color:#5a6a82; padding:4px 0'>Total USD</td><td style='color:#00e5ff'>${inputs.get('funding_total_usd',0):,.0f}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Rondas</td><td style='color:#c8d6e5'>{inputs.get('funding_rounds','—')}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Part./ronda</td><td style='color:#c8d6e5'>{inputs.get('avg_participants',0):.1f}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>VC</td><td style='color:#c8d6e5'>{'✓' if inputs.get('has_VC') else '✗'}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Angel</td><td style='color:#c8d6e5'>{'✓' if inputs.get('has_angel') else '✗'}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="section-card">
        <h3>🚀 Operación</h3>
        <table style='font-size:0.78rem; width:100%; border-collapse:collapse'>
        <tr><td style='color:#5a6a82; padding:4px 0'>Relaciones clave</td><td style='color:#00e5ff'>{inputs.get('relationships','—')}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Hitos</td><td style='color:#c8d6e5'>{inputs.get('milestones','—')}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Ronda A</td><td style='color:#c8d6e5'>{'✓' if inputs.get('has_roundA') else '✗'}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Ronda B</td><td style='color:#c8d6e5'>{'✓' if inputs.get('has_roundB') else '✗'}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Ronda C</td><td style='color:#c8d6e5'>{'✓' if inputs.get('has_roundC') else '✗'}</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Ronda D</td><td style='color:#c8d6e5'>{'✓' if inputs.get('has_roundD') else '✗'}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card" style="margin-top:0.5rem">
        <h3>📅 Maduración</h3>
        <table style='font-size:0.78rem; width:100%; border-collapse:collapse'>
        <tr><td style='color:#5a6a82; padding:4px 0'>Edad 1er financiamiento</td><td style='color:#c8d6e5'>{inputs.get('age_first_funding_year',0):.1f} años</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Edad último financiamiento</td><td style='color:#c8d6e5'>{inputs.get('age_last_funding_year',0):.1f} años</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Edad 1er hito</td><td style='color:#c8d6e5'>{inputs.get('age_first_milestone_year',0):.1f} años</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Edad último hito</td><td style='color:#c8d6e5'>{inputs.get('age_last_milestone_year',0):.1f} años</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # Full feature vector
    with st.expander("🔢 Vector de features (post-procesado)"):
        st.dataframe(
            row_df.T.rename(columns={0: 'Valor'}).assign(Feature=[FEATURE_LABELS.get(c,c) for c in FEATURE_COLS]),
            use_container_width=True
        )


# TAB 4 · MÉTRICAS DEL MODELO
with tab4:
    st.markdown("## Métricas del Modelo XGBoost")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  "82.2%")
    m2.metric("Precision", "84.5%")
    m3.metric("Recall",    "86.7%")
    m4.metric("F1-Score",  "85.6%")
    m5.metric("ROC-AUC",   "0.889")

    st.markdown("---")

    # Feature importance from model
    fi = dict(zip(FEATURE_COLS, model.feature_importances_))
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])[:15]

    fig_fi, ax_fi = plt.subplots(figsize=(8, 4.5))
    fig_fi.patch.set_facecolor('#0e111a'); ax_fi.set_facecolor('#0e111a')
    names_fi = [FEATURE_LABELS.get(k,k) for k,_ in fi_sorted]
    vals_fi = [v for _,v in fi_sorted]
    cmap_vals = np.linspace(0.3, 1.0, len(vals_fi))
    colors_fi = [cm.cool(c) for c in cmap_vals[::-1]]
    ax_fi.barh(names_fi, vals_fi, color=colors_fi, height=0.6)
    ax_fi.set_xlabel("Importancia (gain)", color='#5a6a82', fontsize=9)
    ax_fi.tick_params(colors='#c8d6e5', labelsize=8.5)
    ax_fi.spines[:].set_visible(False)
    ax_fi.grid(axis='x', color='#1e2535', linestyle='--', linewidth=0.6)
    ax_fi.invert_yaxis()
    ax_fi.set_title("Importancia global de features (XGBoost gain)", color='#5a6a82', fontsize=9, pad=10)
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)
    plt.close(fig_fi)

    st.markdown("---")

    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("""
        <div class="section-card">
        <h3>📖 Sobre el Dataset</h3>
        <table style='font-size:0.78rem; width:100%; border-collapse:collapse'>
        <tr><td style='color:#5a6a82; padding:4px 0'>Fuente</td><td style='color:#c8d6e5'>Crunchbase (Kaggle)</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Startups totales</td><td style='color:#00e5ff'>923</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Features originales</td><td style='color:#c8d6e5'>49</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Features usadas</td><td style='color:#c8d6e5'>31</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Train set (post-SMOTE)</td><td style='color:#c8d6e5'>954 muestras</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Test set</td><td style='color:#c8d6e5'>185 muestras</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Clase mayoritaria</td><td style='color:#c8d6e5'>Acquired (65%)</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with ci2:
        st.markdown("""
        <div class="section-card">
        <h3>⚙️ Pipeline de Preprocesamiento</h3>
        <table style='font-size:0.78rem; width:100%; border-collapse:collapse'>
        <tr><td style='color:#5a6a82; padding:4px 0'>Nulos</td><td style='color:#c8d6e5'>Imputados con 0 (milestones)</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Normalización</td><td style='color:#c8d6e5'>StandardScaler (6 cols)</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Balanceo</td><td style='color:#c8d6e5'>SMOTE (477 vs 261 → 477/477)</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Split</td><td style='color:#c8d6e5'>80/20 estratificado</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Modelo</td><td style='color:#c8d6e5'>XGBoost Classifier</td></tr>
        <tr><td style='color:#5a6a82; padding:4px 0'>Explicabilidad</td><td style='color:#c8d6e5'>SHAP TreeExplainer</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 Top features — tabla detallada"):
        fi_df = pd.DataFrame(
            [{"Feature": FEATURE_LABELS.get(k,k), "Importancia": round(v,5)}
             for k,v in fi_sorted],
        ).sort_values("Importancia", ascending=False)
        st.dataframe(fi_df, use_container_width=True, hide_index=True)


# TAB 5 · INFORME IA
with tab5:
    st.markdown("## Informe Ejecutivo — Llama-3.1 vía Groq")
    st.caption("El LLM actúa como analista senior de Venture Capital e interpreta la predicción y los valores SHAP en lenguaje ejecutivo.")

    explanation_key = f"llm_report_{hash(str(sorted(inputs.items())))}"

    if explanation_key in st.session_state:
        st.markdown(st.session_state[explanation_key])
        st.divider()
        if st.button("Regenerar informe"):
            del st.session_state[explanation_key]
            st.rerun()
    else:
        st.info(f"Startup evaluada con probabilidad de éxito **{prob*100:.1f}%** · veredicto: **{verdict}**")
        if st.button("Generar informe con IA", type="primary", use_container_width=True):
            with st.spinner("Consultando Llama-3.1 vía Groq API..."):
                try:
                    report = load_explainer().explain_startup(
                        row_df.iloc[0].to_dict(),
                        shap_dict,
                        prob,
                    )
                    st.session_state[explanation_key] = report
                    st.rerun()
                except GroqConnectionError as exc:
                    st.error(f"Error Groq: {exc}")
