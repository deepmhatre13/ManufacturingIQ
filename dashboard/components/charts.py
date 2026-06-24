"""
ManufacturingIQ - Chart Components
Plotly chart builders for analytics visualizations
"""

import plotly.graph_objects as go
from typing import Dict, List, Any


def _base_font() -> dict:
    """Base font configuration"""
    return {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color": "#64748B"
    }


def feature_importance_chart(features: List[Dict[str, Any]]):
    """
    Top 5 Feature Importance Horizontal Bar Chart using actual XGBoost values.
    Shows only the top 5 most important features for the XGBoost model.
    """
    if not features or len(features) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={"family": "Inter, sans-serif", "size": 13, "color": "#94A3B8"}
        )
        fig.update_layout(
            font=_base_font(),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 30, "r": 20, "t": 25, "b": 30},
            height=220,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    sorted_features = sorted(features, key=lambda x: x.get("importance", 0), reverse=True)[:5]
    sorted_features = sorted_features[::-1]

    names = [f.get("feature", "Unknown") for f in sorted_features]
    values = [f.get("importance", 0) for f in sorted_features]

    display_names = []
    for n in names:
        name = n.split("[")[0].strip() if "[" in n else n
        display_names.append(name)

    colors = [
        "#0F766E" if v >= 0.12 else "#14B8A6" if v >= 0.08 else "#99F6E4"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=display_names,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#64748B"),
        hoverinfo="x+y"
    ))

    fig.update_layout(
        font=_base_font(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 120, "r": 60, "t": 15, "b": 30},
        height=240,
        title=None,
        xaxis=dict(
            tickformat=".0%",
            gridcolor="#F1F5F9",
            zerolinecolor="#F1F5F9",
            tickfont={"size": 11, "color": "#94A3B8"},
            range=[0, max(values) * 1.3] if values else [0, 1]
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#F1F5F9",
            tickfont={"size": 12, "color": "#1E293B", "family": "Inter, sans-serif"},
            autorange="reversed"
        )
    )

    fig.update_xaxes(title_text="Importance Score", title_font={"size": 11, "color": "#94A3B8"})
    fig.update_yaxes(title_text=None)

    return fig


def model_evolution_chart(history: List[Dict[str, Any]]):
    """Version vs ROC-AUC Line Chart - rendered as operational model timeline"""
    if not history or len(history) < 2:
        # Return placeholder chart when insufficient data
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient history for trend visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={"family": "Inter, sans-serif", "size": 13, "color": "#94A3B8"}
        )
        fig.update_layout(
            font=_base_font(),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 30, "r": 20, "t": 25, "b": 30},
            height=220,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    versions = [h.get("version", f"v{i}") for i, h in enumerate(history)]
    roc_aucs = [h.get("roc_auc", 0) for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(versions))),
        y=roc_aucs,
        mode="lines+markers",
        line=dict(color="#0F766E", width=3, shape="spline"),
        marker=dict(color="#0F766E", size=8, line=dict(color="white", width=2)),
        fill="tozeroy",
        fillcolor="rgba(15, 118, 110, 0.08)",
        hovertemplate="<b>%{text}</b><br>AUC: %{y:.4f}<extra></extra>",
        text=versions
    ))

    annotations = []
    for i, (v, auc) in enumerate(zip(versions, roc_aucs)):
        annotations.append({
            "x": i,
            "y": auc,
            "text": f"{auc:.4f}",
            "showarrow": False,
            "font": {"family": "JetBrains Mono, monospace", "size": 9, "color": "#0F766E"},
            "yshift": 10
        })

    fig.update_layout(
        font=_base_font(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        height=220,
        title=None,
        xaxis=dict(
            tickvals=list(range(len(versions))),
            ticktext=versions,
            gridcolor="#F1F5F9",
            tickfont={"size": 10, "color": "#475569"}
        ),
        yaxis=dict(
            title_text="ROC-AUC",
            title_font={"size": 10, "color": "#94A3B8"},
            range=[0.90, 1.0],
            tickformat=".3f",
            gridcolor="#F1F5F9",
            tickfont={"size": 10, "color": "#94A3B8"}
        ),
        annotations=annotations
    )

    return fig