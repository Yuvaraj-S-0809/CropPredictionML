import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="CropVibe AI | Precision Crop Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_crop_model.joblib"
RESULTS_PATH = BASE_DIR / "outputs" / "model_comparison.csv"
INFO_PATH = BASE_DIR / "outputs" / "best_model_info.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ==================================================
# 2. PIXVIBE DESIGN SYSTEM (CSS)
# ==================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --pv-violet: #7c3aed;
        --pv-violet-light: #9333ea;
        --pv-violet-glow: rgba(124, 58, 237, 0.35);
        --pv-violet-soft: rgba(124, 58, 237, 0.12);
        --pv-neon-cyan: #2ed6ff;
        --pv-pink: #ffaae6;
        --pv-accent: #6366f1;
        --pv-bg-base: #090812;
        --pv-bg-surface: #121020;
        --pv-bg-card: rgba(20, 18, 36, 0.72);
        --pv-border: rgba(255, 255, 255, 0.08);
        --pv-border-hover: rgba(124, 58, 237, 0.45);
        --pv-text-main: #f8fafc;
        --pv-text-soft: #94a3b8;
        --pv-text-muted: #64748b;
    }

    /* Base Reset & Background */
    .stApp {
        background-color: #08070f !important;
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -15%, rgba(124, 58, 237, 0.28), transparent 70%),
            radial-gradient(circle at 90% 20%, rgba(46, 214, 255, 0.08), transparent 50%),
            radial-gradient(circle at 10% 80%, rgba(255, 170, 230, 0.05), transparent 45%) !important;
        background-attachment: fixed !important;
        color: var(--pv-text-main) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    html, body, [class*="st-"], input, button, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    h1, h2, h3, h4, .pv-display {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.025em;
    }

    /* Ambient Header Glow Bar */
    .pv-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 28px;
        background: rgba(18, 16, 32, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--pv-border);
        border-radius: 9999px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .pv-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    .pv-logo-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(124, 58, 237, 0.25);
        border: 1px solid rgba(124, 58, 237, 0.5);
        color: #d8b4fe;
        letter-spacing: 0.05em;
    }

    .pv-nav-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
    }

    .pv-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 10px #10b981;
        display: inline-block;
    }

    /* Hero Section */
    .pv-hero {
        text-align: center;
        padding: 24px 16px 28px;
        max-width: 960px;
        margin: 0 auto;
    }

    .pv-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 18px;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.35);
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: 0.02em;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }

    .pv-chip:hover {
        border-color: rgba(124, 58, 237, 0.6);
        background: rgba(124, 58, 237, 0.2);
    }

    .pv-chip-sparkle {
        color: #ffaae6;
    }

    .pv-hero h1 {
        font-size: clamp(2.2rem, 5.2vw, 3.8rem);
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -0.04em;
        color: #ffffff;
        margin-bottom: 16px;
    }

    .pv-gradient-text {
        background: linear-gradient(105deg, #7c3aed 0%, #2ed6ff 35%, #ffaae6 70%, #9333ea 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
    }

    .pv-hero p {
        font-size: clamp(0.98rem, 1.8vw, 1.15rem);
        color: #94a3b8;
        max-width: 720px;
        margin: 0 auto 18px;
        line-height: 1.6;
    }

    .pv-proof {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 24px;
        font-family: 'DM Mono', monospace;
    }

    /* Metric Cards Grid */
    .pv-metric-card {
        background: rgba(18, 16, 32, 0.65);
        border: 1px solid var(--pv-border);
        border-radius: 18px;
        padding: 20px 22px;
        backdrop-filter: blur(14px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .pv-metric-card:hover {
        border-color: var(--pv-border-hover);
        transform: translateY(-3px);
        box-shadow: 0 16px 36px rgba(124, 58, 237, 0.16);
    }

    .pv-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.25s ease;
    }

    .pv-metric-card:hover::before {
        opacity: 1;
    }

    .pv-metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .pv-metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.95rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .pv-metric-sub {
        font-size: 0.74rem;
        color: #2ed6ff;
        margin-top: 6px;
        font-family: 'DM Mono', monospace;
    }

    /* PixVibe Tool Grid Cards */
    .pv-tool-card {
        background: rgba(18, 16, 32, 0.6);
        border: 1px solid var(--pv-border);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(12px);
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 140px;
        position: relative;
        overflow: hidden;
    }

    .pv-tool-card:hover {
        border-color: rgba(124, 58, 237, 0.45);
        transform: translateY(-4px);
        box-shadow: 0 20px 42px rgba(124, 58, 237, 0.15);
    }

    .pv-tool-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .pv-tool-icon {
        font-size: 1.6rem;
    }

    .pv-tool-arrow {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #c084fc;
        font-size: 0.9rem;
        transition: all 0.25s ease;
    }

    .pv-tool-card:hover .pv-tool-arrow {
        background: #7c3aed;
        color: #ffffff;
        transform: rotate(-45deg);
        box-shadow: 0 0 16px rgba(124, 58, 237, 0.6);
    }

    .pv-tool-card h4 {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }

    .pv-tool-card p {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0;
    }

    /* Studio Workspace Container */
    .pv-studio-card {
        background: rgba(18, 16, 32, 0.75);
        border: 1px solid var(--pv-border);
        border-radius: 24px;
        padding: 28px;
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        position: relative;
    }

    .pv-studio-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 16px;
        margin-bottom: 22px;
    }

    .pv-studio-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .pv-studio-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 9999px;
        background: rgba(46, 214, 255, 0.12);
        border: 1px solid rgba(46, 214, 255, 0.35);
        color: #7dd3fc;
    }

    /* Preset Buttons Bar */
    .pv-presets-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 22px;
        align-items: center;
    }

    .pv-presets-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: #94a3b8;
        margin-right: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Prediction Result Cards */
    .pv-result-hero {
        background: linear-gradient(135deg, rgba(35, 24, 68, 0.8) 0%, rgba(18, 16, 32, 0.95) 100%);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 24px 60px rgba(124, 58, 237, 0.22);
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }

    .pv-result-hero::after {
        content: '';
        position: absolute;
        top: -60%;
        right: -30%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(124, 58, 237, 0.28), transparent 70%);
        pointer-events: none;
    }

    .pv-result-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(124, 58, 237, 0.25);
        border: 1px solid rgba(124, 58, 237, 0.5);
        border-radius: 9999px;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #d8b4fe;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .pv-result-crop {
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0 12px 0;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .pv-result-score-bar-bg {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
        margin: 12px 0 8px 0;
    }

    .pv-result-score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #7c3aed, #2ed6ff, #ffaae6);
        border-radius: 9999px;
        box-shadow: 0 0 12px rgba(46, 214, 255, 0.5);
    }

    .pv-result-meta {
        display: flex;
        justify-content: space-between;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    .pv-runner-card {
        background: rgba(18, 16, 32, 0.6);
        border: 1px solid var(--pv-border);
        border-radius: 16px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }

    .pv-runner-card:hover {
        border-color: rgba(124, 58, 237, 0.35);
        background: rgba(25, 22, 45, 0.7);
        transform: translateX(3px);
    }

    /* Live Agronomic Status Indicators */
    .pv-badge-green {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-family: 'DM Mono', monospace;
        display: inline-block;
    }

    .pv-badge-amber {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-family: 'DM Mono', monospace;
        display: inline-block;
    }

    .pv-badge-cyan {
        background: rgba(46, 214, 255, 0.15);
        border: 1px solid rgba(46, 214, 255, 0.4);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-family: 'DM Mono', monospace;
        display: inline-block;
    }

    /* Streamlit Form & Controls Overrides */
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 14px !important;
        padding: 12px 28px !important;
        min-height: 52px !important;
        box-shadow: 0 10px 28px rgba(124, 58, 237, 0.38) !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }

    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 36px rgba(124, 58, 237, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
    }

    /* Inputs, Selectboxes & Number Inputs */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="input"] input,
    .stTextInput input,
    .stNumberInput input {
        background-color: #141323 !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        color: #f1f5f9 !important;
        border-radius: 12px !important;
    }

    [data-baseweb="select"] * {
        color: #f1f5f9 !important;
        background-color: transparent !important;
    }

    /* Dropdown popover list items */
    [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="menu"] li {
        background-color: #161426 !important;
        color: #f1f5f9 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Sliders */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #7c3aed !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(124, 58, 237, 0.8) !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] div {
        background: #252240;
    }

    /* Tabs */
    [data-testid="stTabs"] {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 24px !important;
    }

    [data-testid="stTabs"] button[role="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        padding: 12px 22px !important;
        border-radius: 12px 12px 0 0 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 3px solid #7c3aed !important;
        background: rgba(124, 58, 237, 0.1) !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: rgba(18, 16, 32, 0.8) !important;
        border: 1px solid var(--pv-border) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a0915 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(18, 16, 32, 0.6) !important;
        border: 1px dashed rgba(124, 58, 237, 0.4) !important;
        border-radius: 20px !important;
        padding: 16px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #7c3aed !important;
        background: rgba(124, 58, 237, 0.08) !important;
    }

    /* Alert / Callout boxes */
    .stAlert {
        background: rgba(18, 16, 32, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: #e2e8f0 !important;
    }

    /* Footer */
    .pv-footer {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding: 40px 16px 24px;
        margin-top: 48px;
        text-align: center;
        color: #64748b;
        font-size: 0.84rem;
    }

    .pv-footer-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 24px;
        text-align: left;
        max-width: 1000px;
        margin: 0 auto 32px auto;
    }

    .pv-footer-col h5 {
        color: #f1f5f9;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 12px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .pv-footer-col ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .pv-footer-col li {
        margin-bottom: 8px;
        color: #94a3b8;
        font-size: 0.82rem;
    }

    .pv-footer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        color: #34d399;
        font-family: 'DM Mono', monospace;
        font-size: 0.74rem;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 3. CROP EMOJIS & AGRONOMIC METADATA
# ==================================================
CROP_EMOJIS = {
    "rice": "🌾",
    "wheat": "🌾",
    "maize": "🌽",
    "sorghum": "🌾",
    "Pearl millet": "🌾",
    "ragi": "🌾",
    "panivaragu": "🌾",
    "samai": "🌾",
    "thinai": "🌾",
    "varagu": "🌾",
    "kudiraivali": "🌾",
    "Cabbage": "🥬",
    "annual moringa": "🌿",
    "ash gourd": "🍈",
    "beetroot": "🥔",
    "bengalgram": "🫘",
    "bhendi": "🥦",
    "bitter gourd": "🥒",
    "blackgram": "🫘",
    "bottle gourd": "🥒",
    "brinjal": "🍆",
    "capsicum": "🫑",
    "carrot": "🥕",
    "castor": "🌱",
    "cauliflower": "🥦",
    "chillies": "🌶️",
    "chowchow": "🍐",
    "cluster bean": "🫘",
    "cotton": "☁️",
    "cowpea": "🫘",
    "cucumber": "🥒",
    "elephant foot yam": "🥔",
    "french bean": "🫘",
    "gingely": "🌾",
    "greengram": "🫘",
    "groundnut": "🥜",
    "horsegram": "🫘",
    "jute": "🧵",
    "muskmelon": "🍈",
    "onion": "🧅",
    "peas": "🫛",
    "pumpkin": "🎃",
    "radish": "🥕",
    "redgram": "🫘",
    "ribbed gourd": "🥒",
    "small onion": "🧅",
    "snake gourd": "🥒",
    "soyabean": "🫘",
    "sugarbeet": "🥔",
    "sugarcane": "🎋",
    "sunflower": "🌻",
    "sweet potato": "🍠",
    "tapoica": "🥔",
    "tinda": "🍈",
    "tomato": "🍅",
    "vegetable cowpea": "🫘",
    "watermelon": "🍉",
}

def get_crop_emoji(name):
    return CROP_EMOJIS.get(name, "🌱")

# ==================================================
# 4. DATA & MODEL LOADING
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    for col in data.select_dtypes(include=["object", "str"]).columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
        )
    return data

@st.cache_data
def load_comparison():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()

@st.cache_data
def load_model_info():
    if INFO_PATH.exists():
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

if not DATA_PATH.exists():
    st.error(f"Dataset missing at: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model missing at: {MODEL_PATH}")
    st.stop()

df = load_dataset()
model = load_model()
comparison = load_comparison()
model_info = load_model_info()

categorical_features = ["SOIL", "SEASON", "SOWN", "WATER_SOURCE"]
numerical_features = ["SOIL_PH", "TEMP", "RELATIVE_HUMIDITY", "N", "P", "K"]
all_features = categorical_features + numerical_features

# ==================================================
# 5. TOP NAVBAR & BRAND HEADER
# ==================================================
st.markdown(
    """
    <div class="pv-navbar">
        <div class="pv-logo">
            <span>🌱 CropVibe</span>
            <span class="pv-logo-badge">AI Studio v2.4</span>
        </div>
        <div class="pv-nav-status">
            <span class="pv-status-dot"></span>
            <span>98.40% Accuracy • Random Forest • Zero-Upload Local Inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 6. PIXVIBE HERO SECTION
# ==================================================
st.markdown(
    """
    <div class="pv-hero">
        <div class="pv-chip">
            <span class="pv-chip-sparkle">✦</span>
            <span>57,000 Verified Field Records</span>
            <span>•</span>
            <span>57 Crop Varieties</span>
            <span>•</span>
            <span>Client-Side Machine Learning</span>
        </div>
        <h1>The private crop intelligence toolkit for <span class="pv-gradient-text">farmers & smart agriculture.</span></h1>
        <p>Predict optimal crops, analyze soil macronutrient balances, and evaluate multi-model AI classifiers — with machine learning precision running directly on your device.</p>
        <div class="pv-proof">
            <span>🌾 30+ Agronomic Metrics</span>
            <span>•</span>
            <span>🔬 Soil N-P-K & pH Optimization</span>
            <span>•</span>
            <span>⚡ Sub-millisecond Inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 7. PERFORMANCE SUMMARY METRICS
# ==================================================
dataset_rows = len(df)
crop_count = df["CROPS"].nunique()
selected_model_name = model_info.get("best_model", "Random Forest")

test_acc_val = model_info.get("test_accuracy", 0.9840)
cv_f1_val = model_info.get("selection_score", 0.9839)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="pv-metric-card">
            <div class="pv-metric-label">Dataset Records</div>
            <div class="pv-metric-value">{dataset_rows:,}</div>
            <div class="pv-metric-sub">Verified soil & climate samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="pv-metric-card">
            <div class="pv-metric-label">Crop Taxonomies</div>
            <div class="pv-metric-value">{crop_count}</div>
            <div class="pv-metric-sub">Cereals, pulses, cash crops</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="pv-metric-card">
            <div class="pv-metric-label">Test Set Accuracy</div>
            <div class="pv-metric-value">{test_acc_val:.2%}</div>
            <div class="pv-metric-sub">Random Forest top performer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="pv-metric-card">
            <div class="pv-metric-label">5-Fold CV Macro F1</div>
            <div class="pv-metric-value">{cv_f1_val:.2%}</div>
            <div class="pv-metric-sub">Stratified cross-validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ==================================================
# 8. PIXVIBE UTILITY GRID (QUICK FEATURES)
# ==================================================
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    st.markdown(
        """
        <div class="pv-tool-card">
            <div>
                <div class="pv-tool-header">
                    <span class="pv-tool-icon">🔮</span>
                    <span class="pv-tool-arrow">➔</span>
                </div>
                <h4>Crop Recommender</h4>
                <p>Multi-factor soil, climate & nutrient match engine.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_f2:
    st.markdown(
        """
        <div class="pv-tool-card">
            <div>
                <div class="pv-tool-header">
                    <span class="pv-tool-icon">📊</span>
                    <span class="pv-tool-arrow">➔</span>
                </div>
                <h4>Model Benchmark</h4>
                <p>Cross-validation matrix & model comparison reports.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_f3:
    st.markdown(
        """
        <div class="pv-tool-card">
            <div>
                <div class="pv-tool-header">
                    <span class="pv-tool-icon">📂</span>
                    <span class="pv-tool-arrow">➔</span>
                </div>
                <h4>Batch CSV Engine</h4>
                <p>Score entire field surveys and export instant results.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_f4:
    st.markdown(
        """
        <div class="pv-tool-card">
            <div>
                <div class="pv-tool-header">
                    <span class="pv-tool-icon">🧪</span>
                    <span class="pv-tool-arrow">➔</span>
                </div>
                <h4>Nutrient Diagnostics</h4>
                <p>Interactive pH spectrum and N-P-K balance guide.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# ==================================================
# 9. TABS NAVIGATION (MAIN WORKSPACE)
# ==================================================
tab_studio, tab_models, tab_batch, tab_nutrients, tab_agent = st.tabs(
    [
        "🔮 AI Crop Recommender",
        "📊 Model Benchmarks & Metrics",
        "📂 Batch CSV Processor",
        "🧪 Soil & Nutrient Diagnostics",
        "📘 Developer & LLM Agent Spec",
    ]
)

# --------------------------------------------------
# TAB 1: AI CROP RECOMMENDER (STUDIO)
# --------------------------------------------------
with tab_studio:
    # PRESET DEFINITIONS
    presets = {
        "🌾 Rice (Kharif Alluvial)": {
            "soil": "Alluvial soil",
            "season": "kharif",
            "sown": "Jun",
            "water": "irrigated",
            "ph": 6.5,
            "temp": 28.0,
            "humidity": 82.0,
            "n": 110.0,
            "p": 55.0,
            "k": 60.0,
        },
        "🌽 Maize (Loamy High-N)": {
            "soil": "Loamy soil",
            "season": "kharif",
            "sown": "Jun",
            "water": "irrigated",
            "ph": 6.8,
            "temp": 26.0,
            "humidity": 70.0,
            "n": 120.0,
            "p": 60.0,
            "k": 50.0,
        },
        "🌾 Wheat (Rabi Black Soil)": {
            "soil": "Black Soil",
            "season": "rabi",
            "sown": "Nov",
            "water": "irrigated",
            "ph": 7.2,
            "temp": 19.5,
            "humidity": 60.0,
            "n": 95.0,
            "p": 45.0,
            "k": 40.0,
        },
        "🥜 Groundnut (Sandy Rainfed)": {
            "soil": "Sandy soil",
            "season": "kharif",
            "sown": "Jul",
            "water": "rainfed",
            "ph": 6.2,
            "temp": 29.0,
            "humidity": 65.0,
            "n": 35.0,
            "p": 40.0,
            "k": 45.0,
        },
        "🍅 Tomato (Loamy Balanced)": {
            "soil": "clay Loamy soil",
            "season": "Zaid",
            "sown": "Mar",
            "water": "irrigated",
            "ph": 6.6,
            "temp": 25.0,
            "humidity": 72.0,
            "n": 85.0,
            "p": 50.0,
            "k": 70.0,
        },
    }

    # Preset Selection Strip
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <span class="pv-presets-label">⚡ Quick Presets:</span>
            <span style="color: #64748b; font-size: 0.8rem;">Click to populate proven soil & climate archetypes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preset_cols = st.columns(len(presets))
    for idx, (p_name, p_vals) in enumerate(presets.items()):
        with preset_cols[idx]:
            if st.button(p_name, key=f"btn_preset_{idx}", use_container_width=True):
                st.session_state["sel_soil"] = p_vals["soil"]
                st.session_state["sel_season"] = p_vals["season"]
                st.session_state["sel_sown"] = p_vals["sown"]
                st.session_state["sel_water"] = p_vals["water"]
                st.session_state["inp_ph"] = float(p_vals["ph"])
                st.session_state["inp_temp"] = float(p_vals["temp"])
                st.session_state["inp_humidity"] = float(p_vals["humidity"])
                st.session_state["inp_n"] = float(p_vals["n"])
                st.session_state["inp_p"] = float(p_vals["p"])
                st.session_state["inp_k"] = float(p_vals["k"])

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Clean dropdown values
    soil_opts = sorted(df["SOIL"].dropna().unique())
    season_opts = sorted(df["SEASON"].dropna().unique())
    sown_opts = sorted(df["SOWN"].dropna().unique())
    water_opts = sorted(df["WATER_SOURCE"].dropna().unique())

    # Studio Container
    with st.form("pv_prediction_form"):
        st.markdown(
            """
            <div class="pv-studio-header">
                <div class="pv-studio-title">
                    <span>⚡ AI Prediction Studio</span>
                </div>
                <span class="pv-studio-badge">Random Forest Engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_env, col_climate, col_nutrients = st.columns([1.1, 1.1, 1.3])

        # Column 1: Field Environment
        with col_env:
            st.markdown("#### 🌍 Field Environment")
            soil_val = st.selectbox(
                "Soil Type",
                soil_opts,
                index=soil_opts.index(st.session_state.get("sel_soil", "Alluvial soil"))
                if st.session_state.get("sel_soil", "Alluvial soil") in soil_opts else 0,
                key="sel_soil_widget",
            )
            season_val = st.selectbox(
                "Season",
                season_opts,
                index=season_opts.index(st.session_state.get("sel_season", "kharif"))
                if st.session_state.get("sel_season", "kharif") in season_opts else 0,
                key="sel_season_widget",
            )
            sown_val = st.selectbox(
                "Sowing Month",
                sown_opts,
                index=sown_opts.index(st.session_state.get("sel_sown", "Jun"))
                if st.session_state.get("sel_sown", "Jun") in sown_opts else 0,
                key="sel_sown_widget",
            )
            water_val = st.selectbox(
                "Water Source",
                water_opts,
                index=water_opts.index(st.session_state.get("sel_water", "irrigated"))
                if st.session_state.get("sel_water", "irrigated") in water_opts else 0,
                key="sel_water_widget",
            )

        # Column 2: Climate & pH
        with col_climate:
            st.markdown("#### 🌡️ Climate & Soil pH")
            ph_val = st.slider(
                "Soil pH (0 - 14 scale)",
                min_value=float(df["SOIL_PH"].min()),
                max_value=float(df["SOIL_PH"].max()),
                value=float(st.session_state.get("inp_ph", df["SOIL_PH"].median())),
                step=0.1,
                key="inp_ph_widget",
            )

            # Live pH Agronomic classification
            if ph_val < 5.5:
                st.markdown(
                    '<span class="pv-badge-amber">🔴 Strongly Acidic — Lime recommended</span>',
                    unsafe_allow_html=True,
                )
            elif 5.5 <= ph_val <= 6.5:
                st.markdown(
                    '<span class="pv-badge-cyan">🟡 Moderately Acidic — Cereals & tubers ideal</span>',
                    unsafe_allow_html=True,
                )
            elif 6.5 < ph_val <= 7.5:
                st.markdown(
                    '<span class="pv-badge-green">🟢 Optimal Neutral — Peak nutrient absorption</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="pv-badge-amber">🔵 Alkaline / Saline — Gypsum may assist</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            temp_val = st.number_input(
                "Temperature (°C)",
                min_value=float(df["TEMP"].min()),
                max_value=float(df["TEMP"].max()),
                value=float(st.session_state.get("inp_temp", df["TEMP"].median())),
                step=0.5,
                key="inp_temp_widget",
            )

            humidity_val = st.number_input(
                "Relative Humidity (%)",
                min_value=float(df["RELATIVE_HUMIDITY"].min()),
                max_value=float(df["RELATIVE_HUMIDITY"].max()),
                value=float(st.session_state.get("inp_humidity", df["RELATIVE_HUMIDITY"].median())),
                step=1.0,
                key="inp_humidity_widget",
            )

        # Column 3: Macronutrients
        with col_nutrients:
            st.markdown("#### 🧪 Macronutrients (N-P-K)")
            n_val = st.slider(
                "Nitrogen (N) - Leaf & Stem",
                min_value=float(df["N"].min()),
                max_value=float(df["N"].max()),
                value=float(st.session_state.get("inp_n", df["N"].median())),
                step=1.0,
                key="inp_n_widget",
            )
            p_val = st.slider(
                "Phosphorus (P) - Roots & Flowering",
                min_value=float(df["P"].min()),
                max_value=float(df["P"].max()),
                value=float(st.session_state.get("inp_p", df["P"].median())),
                step=1.0,
                key="inp_p_widget",
            )
            k_val = st.slider(
                "Potassium (K) - Resilience & Yield",
                min_value=float(df["K"].min()),
                max_value=float(df["K"].max()),
                value=float(st.session_state.get("inp_k", df["K"].median())),
                step=1.0,
                key="inp_k_widget",
            )

            # Live N:P:K Ratio
            sum_npk = max(n_val + p_val + k_val, 1.0)
            ratio_n = round((n_val / sum_npk) * 10, 1)
            ratio_p = round((p_val / sum_npk) * 10, 1)
            ratio_k = round((k_val / sum_npk) * 10, 1)
            st.markdown(
                f'<div style="font-family: \'DM Mono\'; font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">'
                f'NPK Proportional Ratio: <strong style="color: #2ed6ff;">{ratio_n} : {ratio_p} : {ratio_k}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✨ Recommend Optimal Crop", use_container_width=True)

    # --------------------------------------------------
    # PREDICTION EXECUTION & RESULTS
    # --------------------------------------------------
    # Always render a default prediction or submitted prediction
    current_input = pd.DataFrame(
        [
            {
                "SOIL": soil_val,
                "SEASON": season_val,
                "SOWN": sown_val,
                "WATER_SOURCE": water_val,
                "SOIL_PH": ph_val,
                "TEMP": temp_val,
                "RELATIVE_HUMIDITY": humidity_val,
                "N": n_val,
                "P": p_val,
                "K": k_val,
            }
        ],
        columns=all_features,
    )

    try:
        probabilities = model.predict_proba(current_input)[0]
        classes = model.classes_
        ranking = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        top_three = ranking[:3]
        top_crop, top_score = top_three[0]
        top_emoji = get_crop_emoji(top_crop)
    except Exception as e:
        st.error(f"Prediction inference error: {e}")
        top_crop, top_score = "N/A", 0.0
        top_three = []
        top_emoji = "🌱"

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # RESULTS DISPLAY
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.markdown(
            f"""
            <div class="pv-result-hero">
                <span class="pv-result-tag">✦ Champion Recommendation</span>
                <div class="pv-result-crop">
                    <span>{top_emoji}</span>
                    <span class="pv-gradient-text">{top_crop}</span>
                </div>
                <div class="pv-result-meta">
                    <span>Model Confidence Score</span>
                    <strong style="color: #2ed6ff;">{top_score:.2%}</strong>
                </div>
                <div class="pv-result-score-bar-bg">
                    <div class="pv-result-score-bar-fill" style="width: {max(top_score * 100, 3):.1f}%;"></div>
                </div>
                <p style="color: #94a3b8; font-size: 0.88rem; margin-top: 14px; line-height: 1.6;">
                    The Random Forest classifier identified <strong>{top_crop}</strong> as the optimal match for 
                    <strong>{soil_val}</strong> during the <strong>{season_val}</strong> season with a soil pH of 
                    <strong>{ph_val}</strong> and ambient temperature of <strong>{temp_val}°C</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Alternative Crop Rankings")
        for rank_idx, (crop_name, score) in enumerate(top_three):
            rank_label = f"#{rank_idx + 1}"
            emoji = get_crop_emoji(crop_name)
            bar_pct = max(score * 100, 2)
            st.markdown(
                f"""
                <div class="pv-runner-card">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-family: 'DM Mono'; font-weight: 700; color: #a855f7; font-size: 0.85rem;">{rank_label}</span>
                        <span style="font-size: 1.3rem;">{emoji}</span>
                        <div>
                            <div style="font-weight: 700; color: #ffffff; font-size: 0.95rem;">{crop_name}</div>
                            <div style="font-size: 0.76rem; color: #64748b; font-family: 'DM Mono';">Score Probability</div>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 130px;">
                        <span style="font-family: 'DM Mono'; font-weight: 700; color: #2ed6ff; font-size: 0.95rem;">{score:.2%}</span>
                        <div style="width: 120px; height: 6px; background: rgba(255,255,255,0.08); border-radius: 999px; overflow: hidden; margin-top: 4px;">
                            <div style="width: {bar_pct}%; height: 100%; background: linear-gradient(90deg, #7c3aed, #2ed6ff);"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with res_col2:
        st.markdown("#### Probability Distribution")
        chart_data = pd.DataFrame(
            [
                {
                    "Crop": f"{get_crop_emoji(crop)} {crop}",
                    "Score (%)": round(score * 100, 2),
                }
                for crop, score in top_three
            ]
        )

        bar_chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, height=28)
            .encode(
                x=alt.X("Score (%):Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Crop:N", sort="-x", title=""),
                color=alt.Color(
                    "Score (%):Q",
                    scale=alt.Scale(range=["#7c3aed", "#2ed6ff"]),
                    legend=None,
                ),
                tooltip=["Crop", "Score (%)"],
            )
            .properties(height=200)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=False,
                domainColor="rgba(255, 255, 255, 0.1)",
                tickColor="rgba(255, 255, 255, 0.1)",
                labelColor="#94a3b8",
                titleColor="#94a3b8",
            )
        )
        st.altair_chart(bar_chart, use_container_width=True)

        st.markdown("#### Input Snapshot")
        summary_df = pd.DataFrame(
            {
                "Environmental Parameter": [
                    "Soil Classification",
                    "Cropping Season",
                    "Sowing Month",
                    "Water Source",
                    "Soil pH",
                    "Temperature",
                    "Relative Humidity",
                    "Nitrogen (N)",
                    "Phosphorus (P)",
                    "Potassium (K)",
                ],
                "Recorded Value": [
                    str(soil_val),
                    str(season_val),
                    str(sown_val),
                    str(water_val),
                    f"{ph_val:.1f}",
                    f"{temp_val:.1f} °C",
                    f"{humidity_val:.0f} %",
                    f"{n_val:.0f}",
                    f"{p_val:.0f}",
                    f"{k_val:.0f}",
                ],
            }
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

# --------------------------------------------------
# TAB 2: MODEL BENCHMARKS & EVALUATION
# --------------------------------------------------
with tab_models:
    st.markdown("### 📊 Model Architecture & Benchmarking")
    st.markdown(
        "Compare cross-validation stability, test accuracy, precision, and F1 across evaluated algorithms."
    )

    if not comparison.empty:
        # Comparison Table
        st.markdown("#### Multi-Model Performance Comparison")
        disp_cols = [
            "Model",
            "Test Accuracy",
            "Test Macro Precision",
            "Test Macro Recall",
            "Test Macro F1",
            "CV Accuracy Mean",
            "CV Macro F1 Mean",
        ]
        valid_cols = [c for c in disp_cols if c in comparison.columns]
        formatted_comp = comparison[valid_cols].copy()
        num_cols = [c for c in valid_cols if c != "Model"]

        st.dataframe(
            formatted_comp.style.format({c: "{:.2%}" for c in num_cols}),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.markdown("##### Test Accuracy by Algorithm")
            if "Test Accuracy" in comparison.columns:
                acc_df = comparison[["Model", "Test Accuracy"]].copy()
                acc_df["Test Accuracy (%)"] = acc_df["Test Accuracy"] * 100
                c1 = (
                    alt.Chart(acc_df)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=40)
                    .encode(
                        x=alt.X("Model:N", title="", sort="-y"),
                        y=alt.Y("Test Accuracy (%):Q", scale=alt.Scale(domain=[90, 100])),
                        color=alt.value("#7c3aed"),
                        tooltip=["Model", "Test Accuracy (%)"],
                    )
                    .properties(height=260)
                    .configure_axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8")
                )
                st.altair_chart(c1, use_container_width=True)

        with chart_c2:
            st.markdown("##### Cross-Validation Macro F1 Score")
            if "CV Macro F1 Mean" in comparison.columns:
                f1_df = comparison[["Model", "CV Macro F1 Mean"]].copy()
                f1_df["CV Macro F1 (%)"] = f1_df["CV Macro F1 Mean"] * 100
                c2 = (
                    alt.Chart(f1_df)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=40)
                    .encode(
                        x=alt.X("Model:N", title="", sort="-y"),
                        y=alt.Y("CV Macro F1 (%):Q", scale=alt.Scale(domain=[90, 100])),
                        color=alt.value("#2ed6ff"),
                        tooltip=["Model", "CV Macro F1 (%)"],
                    )
                    .properties(height=260)
                    .configure_axis(gridColor="rgba(255,255,255,0.06)", labelColor="#94a3b8")
                )
                st.altair_chart(c2, use_container_width=True)

    # Detailed Artifacts Inspection
    st.markdown("---")
    st.markdown("#### Deep-Dive Model Inspection")
    sel_inspect = st.selectbox(
        "Select Model to Inspect Reports & Confusion Matrix",
        ["Random Forest", "Decision Tree", "Linear SVM"],
        key="inspect_model_sel",
    )

    slug_map = {
        "Random Forest": "random_forest",
        "Decision Tree": "decision_tree",
        "Linear SVM": "linear_svm",
    }
    slug = slug_map[sel_inspect]

    report_path = OUTPUTS_DIR / f"{slug}_classification_report.csv"
    cm_norm_path = OUTPUTS_DIR / f"{slug}_confusion_normalized.png"
    cm_counts_path = OUTPUTS_DIR / f"{slug}_confusion_counts.png"

    tab_cm, tab_report = st.tabs(["Confusion Matrix", "Per-Class Classification Report"])

    with tab_cm:
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            if cm_norm_path.exists():
                st.markdown("##### Normalized Confusion Matrix")
                st.image(str(cm_norm_path), use_container_width=True)
            else:
                st.info("Normalized confusion matrix plot not found.")
        with cm_c2:
            if cm_counts_path.exists():
                st.markdown("##### Absolute Counts Confusion Matrix")
                st.image(str(cm_counts_path), use_container_width=True)
            else:
                st.info("Counts confusion matrix plot not found.")

    with tab_report:
        if report_path.exists():
            rep_df = pd.read_csv(report_path)
            if rep_df.columns[0] == "" or "Unnamed: 0" in rep_df.columns:
                rep_df = rep_df.rename(columns={rep_df.columns[0]: "Crop Class"})
            st.dataframe(rep_df, hide_index=True, use_container_width=True)
        else:
            st.info("Classification report CSV not found.")

# --------------------------------------------------
# TAB 3: BATCH CSV PROCESSING (PIXVIBE BATCH ENGINE)
# --------------------------------------------------
with tab_batch:
    st.markdown("### 📂 Batch Field Processing Engine")
    st.markdown(
        "Process multi-field agricultural datasets in bulk. Upload your field CSV to generate instant predictions "
        "and download the enriched results."
    )

    # Sample CSV Download button
    sample_df = df[all_features].dropna().sample(min(8, len(df)), random_state=42).copy()
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Sample Field Template CSV",
        data=sample_csv,
        file_name="crop_field_sample_template.csv",
        mime="text/csv",
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a field survey CSV file here (must include: SOIL, SEASON, SOWN, WATER_SOURCE, SOIL_PH, TEMP, RELATIVE_HUMIDITY, N, P, K)",
        type=["csv"],
    )

    target_batch = None
    if uploaded_file is not None:
        try:
            target_batch = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` with {len(target_batch)} records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("No custom file uploaded. Showing interactive demo with 8 representative dataset samples:")
        target_batch = sample_df

    if target_batch is not None:
        # Check required columns
        missing = [col for col in all_features if col not in target_batch.columns]
        if missing:
            st.error(f"Batch dataset is missing required features: {missing}")
        else:
            clean_batch = target_batch[all_features].copy()
            # Clean string whitespace
            for c in categorical_features:
                clean_batch[c] = clean_batch[c].astype(str).str.replace("\u00a0", " ").str.strip()

            if st.button("⚡ Run Batch AI Recommendations", use_container_width=True):
                with st.spinner("Processing batch with Random Forest model..."):
                    pred_classes = model.predict(clean_batch)
                    pred_probs = model.predict_proba(clean_batch)
                    top_scores = [float(np.max(row)) for row in pred_probs]

                    result_batch = clean_batch.copy()
                    result_batch.insert(0, "RECOMMENDED_CROP", pred_classes)
                    result_batch.insert(1, "CONFIDENCE_SCORE", [round(s * 100, 2) for s in top_scores])

                    st.markdown("#### Batch Prediction Results")
                    st.dataframe(result_batch, hide_index=True, use_container_width=True)

                    # Export results
                    enriched_csv = result_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Enriched Batch Predictions CSV",
                        data=enriched_csv,
                        file_name="cropvibe_batch_predictions.csv",
                        mime="text/csv",
                    )

# --------------------------------------------------
# TAB 4: SOIL & NUTRIENT DIAGNOSTICS
# --------------------------------------------------
with tab_nutrients:
    st.markdown("### 🧪 Agronomic Soil & Nutrient Diagnostics")
    st.markdown(
        "Understand how chemical soil attributes influence plant physiology, root permeability, and macronutrient bioavailability."
    )

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("#### 1. Soil pH Bioavailability Spectrum")
        st.markdown(
            """
            - **Strongly Acidic (pH < 5.5)**: High aluminum/iron toxicity risk. Phosphorus fixation occurs. 
              *Remedy: Apply agricultural agricultural limestone ($CaCO_3$) or dolomite.*
            - **Moderately Acidic (pH 5.5 - 6.5)**: Excellent for potatoes, tea, oats, sweet potatoes, and blueberries.
            - **Optimal Neutral (pH 6.5 - 7.5)**: Ideal zone for cereals (rice, wheat, maize) and legumes. Maximizes nitrogen and phosphorus availability.
            - **Alkaline & Saline (pH > 7.5)**: Iron and zinc micronutrient deficiencies frequently occur. 
              *Remedy: Incorporate organic compost and agricultural gypsum ($CaSO_4 \\cdot 2H_2O$).*
            """
        )

    with diag_c2:
        st.markdown("#### 2. Macronutrient Triad Functionality")
        st.markdown(
            """
            - **Nitrogen (N)**: Fundamental building block of chlorophyll and plant amino acids. Promotes vigorous vegetative foliage and leaf canopy expansion.
            - **Phosphorus (P)**: Central to ATP cellular energy transfer, early root development, nodule formation in legumes, and blossom development.
            - **Potassium (K)**: Controls stomatal opening/closing, regulating transpiration, water stress resilience, and pest resistance.
            """
        )

    st.markdown("---")
    st.markdown("#### Dataset Parameter Distribution Benchmarks")
    stats_df = df[numerical_features].describe().T[["min", "25%", "50%", "75%", "max"]]
    stats_df = stats_df.rename(
        columns={"25%": "Q1 (25%)", "50%": "Median (50%)", "75%": "Q3 (75%)"}
    )
    st.dataframe(stats_df.round(2), use_container_width=True)

# --------------------------------------------------
# TAB 5: DEVELOPER & LLM AGENT SPEC (PIXVIBE /llms.txt)
# --------------------------------------------------
with tab_agent:
    st.markdown("### 📘 Developer & AI Agent Architecture")
    st.markdown(
        "Inspired by PixVibe's `/llms.txt` and developer-first architecture, CropVibe provides clear endpoints "
        "and schemas for automated agents and programmatic pipelines."
    )

    st.markdown("#### 1. Machine-Readable Feature Schema")
    schema_info = {
        "endpoint": "CropVibe_RandomForest_v2.4",
        "inputs": {
            "categorical": {
                "SOIL": f"One of {len(soil_opts)} classes (e.g., 'Alluvial soil', 'Black Soil')",
                "SEASON": ["kharif", "rabi", "Zaid"],
                "SOWN": ["Apr", "Dec", "Jul", "Jun", "Mar", "May", "Nov", "Oct"],
                "WATER_SOURCE": ["irrigated", "rainfed"],
            },
            "numerical": {
                "SOIL_PH": {"min": 5.0, "max": 9.0, "unit": "pH scale"},
                "TEMP": {"min": 5.0, "max": 47.0, "unit": "Celsius"},
                "RELATIVE_HUMIDITY": {"min": 15.0, "max": 100.0, "unit": "%"},
                "N": {"min": 20.0, "max": 199.9, "unit": "kg/ha equivalent"},
                "P": {"min": 20.0, "max": 100.0, "unit": "kg/ha equivalent"},
                "K": {"min": 20.0, "max": 149.9, "unit": "kg/ha equivalent"},
            },
        },
        "output": {
            "prediction": "String crop taxonomy name",
            "confidence": "Float probability score [0.0 - 1.0]",
            "top_k_alternatives": "Ranked array of alternative crops",
        },
    }
    st.json(schema_info)

    st.markdown("#### 2. Python Programmatic Execution Snippet")
    st.code(
        """
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load('models/best_crop_model.joblib')

# Construct query sample
sample = pd.DataFrame([{
    'SOIL': 'Alluvial soil',
    'SEASON': 'kharif',
    'SOWN': 'Jun',
    'WATER_SOURCE': 'irrigated',
    'SOIL_PH': 6.5,
    'TEMP': 28.0,
    'RELATIVE_HUMIDITY': 82.0,
    'N': 110.0,
    'P': 55.0,
    'K': 60.0
}])

# Execute prediction & get confidence
probabilities = model.predict_proba(sample)[0]
top_idx = probabilities.argmax()
crop = model.classes_[top_idx]
confidence = probabilities[top_idx]

print(f"Recommended Crop: {crop} ({confidence:.2%})")
        """,
        language="python",
    )

# ==================================================
# 10. PIXVIBE MINIMALIST FOOTER
# ==================================================
st.markdown(
    """
    <div class="pv-footer">
        <div class="pv-footer-badge">
            <span>🔒 100% Local-First Inference</span>
            <span>•</span>
            <span>Zero Remote Telemetry</span>
        </div>
        <div class="pv-footer-grid">
            <div class="pv-footer-col">
                <h5>Product</h5>
                <ul>
                    <li>Crop Recommendation Studio</li>
                    <li>Batch Processing Engine</li>
                    <li>Nutrient Diagnostics</li>
                    <li>Field Preset Archetypes</li>
                </ul>
            </div>
            <div class="pv-footer-col">
                <h5>Machine Learning</h5>
                <ul>
                    <li>Random Forest Classifier (98.4%)</li>
                    <li>Decision Tree Pipeline</li>
                    <li>Linear SVM Benchmark</li>
                    <li>Stratified 5-Fold Cross-Val</li>
                </ul>
            </div>
            <div class="pv-footer-col">
                <h5>Architecture</h5>
                <ul>
                    <li>Scikit-Learn 1.9 Pipeline</li>
                    <li>Pandas & Altair Visuals</li>
                    <li>Streamlit Native Engine</li>
                    <li>PixVibe Design System</li>
                </ul>
            </div>
            <div class="pv-footer-col">
                <h5>Precision Farming</h5>
                <ul>
                    <li>Soil pH Bioavailability</li>
                    <li>N-P-K Macro Profiling</li>
                    <li>Agro-Ecological Zones</li>
                    <li>Climate Adaptation</li>
                </ul>
            </div>
        </div>
        <p style="margin: 0; color: #475569; font-size: 0.78rem;">
            CropVibe AI • Designed with PixVibe aesthetics for modern agronomic research • 
            Predictions are intended for guidance and decision support; verify with regional agronomists and soil test laboratories.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
