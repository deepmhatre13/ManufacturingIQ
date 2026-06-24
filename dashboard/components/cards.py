"""
ManufacturingIQ - Card Components
Premium KPI cards and status badges
"""

import streamlit as st
from typing import Optional


def kpi_card(
    label: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    value_color: Optional[str] = None
):
    """Render a clean KPI card with no raw HTML leaks"""
    value_style = f"color: {value_color};" if value_color else ""
    
    footer = ""
    if delta:
        footer = f'<div class="kpi-delta">{delta}</div>'
    elif subtitle:
        footer = f'<div class="kpi-subtitle">{subtitle}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="{value_style}">{value}</div>
        {footer}
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str, size: str = "sm"):
    """Render a status badge"""
    status_lower = status.lower().replace(" ", "-")
    dot_color = {
        "healthy": "#10B981",
        "warning": "#F59E0B",
        "critical": "#EF4444",
        "info": "#0EA5E9",
        "success": "#10B981",
        "drift-detected": "#F59E0B",
        "stable": "#10B981",
        "approved": "#10B981",
        "pending-review": "#F59E0B",
        "rejected": "#EF4444",
        "production": "#0EA5E9",
        "active": "#10B981"
    }.get(status_lower, "#64748B")

    st.markdown(f"""
    <span class="status-badge {status_lower}">
        <span class="dot" style="background: {dot_color};"></span>
        {status}
    </span>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: Optional[str] = None):
    """Render a section header"""
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)