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


def check_authorization():
    """Check if the logged-in user is authorized."""
    try:
        allowed_emails = st.secrets.get("ALLOWED_EMAILS", "")
        allowed_domains = st.secrets.get("ALLOWED_EMAIL_DOMAINS", "")

        if not allowed_emails and not allowed_domains:
            return True

        email = getattr(st.user, "email", "").lower()

        if allowed_emails:
            allowed = [e.strip().lower() for e in allowed_emails.split(",") if e.strip()]
            if email in allowed:
                return True

        if allowed_domains:
            domains = [d.strip().lower() for d in allowed_domains.split(",") if d.strip()]
            domain = email.split("@")[-1] if "@" in email else ""
            if domain in domains:
                return True

        return False
    except Exception:
        return True


def is_user_logged_in() -> bool:
    """Check if the user is logged in via Google OAuth."""
    try:
        # In Streamlit 1.58.0+, user info is populated after login.
        # If email is accessible, the user is logged in.
        _ = st.user.email
        return True
    except (AttributeError, KeyError):
        return False


def main():
    """Main application entry point"""
    load_css()

    # Google OAuth login gate
    if not is_user_logged_in():
        st.markdown("## 🔒 ManufacturingIQ")
        st.write("Please sign in with Google to continue.")
        if st.button("Log in with Google"):
            st.login()
        st.stop()

    # Authorization check
    if not check_authorization():
        st.error("🚫 Access Denied")
        st.write("Your email is not authorized to access this application.")
        if st.button("Log out"):
            st.logout()
        st.stop()

    # Show user info in sidebar
    with st.sidebar:
        try:
            st.write(f"👤 **{st.user.name}**")
            st.write(f"📧 {st.user.email}")
        except (AttributeError, KeyError):
            pass
        if st.button("Log out"):
            st.logout()

    # Navigation
    tab1, tab2 = st.tabs(["Predictive Intelligence", "MLOps Center"])

    with tab1:
        render_predictive()

    with tab2:
        render_mlops()


if __name__ == "__main__":
    main()