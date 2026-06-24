"""
ManufacturingIQ - Gauge Components
Compact health score gauge visualization using Plotly
"""

import plotly.graph_objects as go
from typing import Optional


def health_gauge(score: float, title: str = "Machine Health"):
    """
    Render a compact premium health score gauge.

    Zones:
    0-20: Critical
    20-50: High Risk
    50-80: Warning
    80-100: Healthy
    """
    if score is None:
        score = 0

    score = max(0, min(100, score))

    if score >= 80:
        status = "Healthy"
        color = "#10B981"
    elif score >= 50:
        status = "Warning"
        color = "#F59E0B"
    elif score >= 20:
        status = "High Risk"
        color = "#EF4444"
    else:
        status = "Critical"
        color = "#DC2626"

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font": {
                "family": "Inter, sans-serif",
                "size": 28,
                "color": color,
                "weight": 700
            },
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#CBD5E1",
                "tickfont": {
                    "family": "Inter, sans-serif",
                    "size": 8,
                    "color": "#94A3B8"
                },
                "dtick": 25,
                "showticklabels": True
            },
            "bar": {
                "color": color,
                "thickness": 0.15,
                "line": {
                    "color": color,
                    "width": 0
                }
            },
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "rgba(239, 68, 68, 0.12)"},
                {"range": [20, 50], "color": "rgba(239, 68, 68, 0.06)"},
                {"range": [50, 80], "color": "rgba(245, 158, 11, 0.08)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.08)"}
            ],
            "threshold": {
                "line": {
                    "color": color,
                    "width": 2
                },
                "thickness": 0.4,
                "value": score
            }
        },
        title={
            "text": f"{title}<br><span style='font-size:11px;color:{color};font-weight:600'>{status}</span>",
            "font": {
                "family": "Inter, sans-serif",
                "size": 12,
                "color": "#0F172A"
            }
        }
    ))

    fig.update_layout(
        height=240,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
        }
    )

    return fig


def mini_gauge(score: float, label: str = ""):
    """Small gauge for inline use"""
    score = max(0, min(100, score))

    if score >= 80:
        color = "#10B981"
    elif score >= 50:
        color = "#F59E0B"
    else:
        color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font": {
                "family": "Inter, sans-serif",
                "size": 18,
                "color": color,
                "weight": 700
            },
            "suffix": ""
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "visible": False
            },
            "bar": {
                "color": color,
                "thickness": 0.2
            },
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "rgba(239, 68, 68, 0.1)"},
                {"range": [20, 50], "color": "rgba(239, 68, 68, 0.05)"},
                {"range": [50, 80], "color": "rgba(245, 158, 11, 0.08)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.08)"}
            ]
        },
        title={
            "text": label,
            "font": {
                "family": "Inter, sans-serif",
                "size": 9,
                "color": "#64748B"
            }
        }
    ))

    fig.update_layout(
        height=120,
        margin={"l": 8, "r": 8, "t": 20, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig