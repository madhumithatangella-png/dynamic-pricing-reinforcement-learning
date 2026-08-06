"""Streamlit Dashboard Entrypoint Application with Custom Option Menu & Dark Theme styling."""

import streamlit as st
import requests
import os
from streamlit_option_menu import option_menu
from dashboard.pages import (
    dashboard,
    prediction,
    training,
    comparison,
    evaluation,
    system_status,
    about,
)

# Setup page config
st.set_page_config(
    page_title="Hotel Dynamic Pricing RL Studio",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom global CSS theme injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Dark Theme Background */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    /* Hide Streamlit Native Pages List from Sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Glassmorphic Cards Styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
        line-height: 1.1;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .metric-trend {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
    }
    
    .trend-up {
        color: #22C55E;
    }
    
    .trend-down {
        color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

# Connection Checks
api_connected = False
active_model = "N/A"
dqn_loaded = False
q_loaded = False
device = "CPU"

try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.5)
    if health_resp.status_code == 200:
        api_connected = True
        info_resp = requests.get(f"{API_URL}/model-info", timeout=1.5)
        if info_resp.status_code == 200:
            info_data = info_resp.json()
            active_model = info_data.get("active_model", "N/A").upper()
            dqn_loaded = info_data.get("dqn_model_loaded", False)
            q_loaded = info_data.get("q_learning_loaded", False)
        
        metrics_resp = requests.get(f"{API_URL}/metrics", timeout=1.5)
        if metrics_resp.status_code == 200:
            device = metrics_resp.json().get("device", "CPU")
except Exception:
    pass

# Redesign Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px; padding-top: 10px;">
            <span style="font-size: 2.2rem;">🏨</span>
            <h2 style="color: #3B82F6; margin: 5px 0 0 0; font-size: 1.4rem; font-weight: 700;">Hotel AI Pricing</h2>
            <span style="color: #64748B; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">MLOps Control Room</span>
        </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Live Prediction", "Training Metrics", "Model Comparison", "Evaluation Results", "System Status", "About Platform"],
        icons=["speedometer2", "cpu", "activity", "balance-scale", "journal-text", "hdd-network", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#8B5CF6", "font-size": "1rem"}, 
            "nav-link": {"font-size": "0.9rem", "text-align": "left", "margin":"2px 0px", "color": "#F8FAFC", "font-weight": "500", "border-radius": "6px"},
            "nav-link-selected": {"background-color": "#3B82F6", "font-weight": "600"},
        }
    )
    
    st.markdown("---")
    
    # Metadata Card in Sidebar
    status_color = "#22C55E" if api_connected else "#EF4444"
    status_text = "Connected" if api_connected else "Offline"
    
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 12px;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="height: 8px; width: 8px; background-color: {status_color}; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
                <span style="font-size: 0.8rem; font-weight: 600; color: #F8FAFC;">API Server: {status_text}</span>
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Active Model: <code style="color: #3B82F6; font-weight: 600;">{active_model}</code></div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Serving Node: <code style="color: #8B5CF6; font-weight: 600;">{device.upper()}</code></div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Platform Version: <code style="color: #64748B;">v1.4.0</code></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Deepmind AAC Project • Phase 4")

# Programmatic Routing
if selected == "Dashboard":
    dashboard.render(API_URL)
elif selected == "Live Prediction":
    prediction.render(API_URL)
elif selected == "Training Metrics":
    training.render(API_URL)
elif selected == "Model Comparison":
    comparison.render(API_URL)
elif selected == "Evaluation Results":
    evaluation.render(API_URL)
elif selected == "System Status":
    system_status.render(API_URL)
elif selected == "About Platform":
    about.render(API_URL)
