"""
ManufacturingIQ - Main Application
AI-Powered Predictive Maintenance Platform
Modern SaaS frontend with Predictive Intelligence and MLOps Center
"""

import streamlit as st
import sys
import os

# Ensure dashboard package is importable
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
if dashboard_dir not in sys.path:
    sys.path.insert(0, dashboard_dir)

from pages.predictive_intelligence import render as render_predictive
from pages.mlops_center import render as render_mlops
from utils.api import get_api_status


# Page configuration
st.set_page_config(
    page_title="ManufacturingIQ",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load custom CSS
def load_css():
    css_path = os.path.join(dashboard_dir, "assets", "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize session state
if "api_status" not in st.session_state:
    st.session_state.api_status = get_api_status()

if "page" not in st.session_state:
    st.session_state.page = "Predictive Intelligence"


def main():
    """Main application entry point"""
    load_css()

    # Navigation
    tab1, tab2 = st.tabs(["Predictive Intelligence", "MLOps Center"])

    with tab1:
        render_predictive()

    with tab2:
        render_mlops()


if __name__ == "__main__":
    main()